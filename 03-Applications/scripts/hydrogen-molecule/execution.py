"""Execution: run the variational loop for every (Hamiltonian, ansatz) pair.

The variational loop (VQE):

    1. A classical optimizer (COBYLA) proposes the ansatz angles theta.
    2. The Estimator runs the circuit and returns the energy <H>.
    3. The optimizer lowers <H> and goes back to step 1.

The loop is restarted from a few random points because the landscape has local
minima. Every pair of qubit mapping and ansatz gets the same budget and the
same random numbers, so the differences in the results come from the pair, not
from luck.

Called from main.py. With no backend name it runs on the Aer simulator, where
the energy is exact unless ``shots`` is set; with a backend name it runs on
IBM hardware inside a single Session.
"""

import time

import numpy as np
from encoding import ANSATZES, build_ansatz, circuit_cost, initial_point
from ops import MAPPINGS, build_hamiltonian, exact_energies
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from scipy.optimize import minimize


def get_primitives(backend_name: str | None):
    """Return (backend, estimator, session). The session is None on Aer."""
    if backend_name is None:
        return AerSimulator(), AerEstimator(), None
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService, Session

    backend = QiskitRuntimeService().backend(backend_name)
    # A Session keeps our place at the front of the queue for the whole loop.
    session = Session(backend=backend)
    return backend, EstimatorV2(mode=session), session


def minimize_energy(estimator, circuit, hamiltonian, x0, precision, maxiter):
    """Minimize <H>(theta) with COBYLA from the starting angles ``x0``.

    Returns the best angles and the list of every energy evaluated. An
    ansatz without angles (Hartree-Fock) is evaluated once.
    """
    history = []

    def energy(theta):
        job = estimator.run([(circuit, hamiltonian, theta, precision)])
        history.append(float(job.result()[0].data.evs))
        return history[-1]

    if len(x0) == 0:
        energy(x0)
        return x0, history
    result = minimize(
        energy,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter},
        tol=1e-6,
    )
    return result.x, history


def optimize(
    distance: float,
    mapping: str,
    ansatz_name: str,
    estimator,
    backend,
    reps: int = 2,
    starts: int = 3,
    shots: int = 0,
    maxiter: int = 300,
    seed: int = 1,
) -> dict:
    """Run the loop for one pair and return its record.

    ``shots = 0`` asks the simulator for exact energies. Otherwise the
    Estimator is asked for a standard error of 1 / sqrt(shots).
    """
    hamiltonian = build_hamiltonian(distance, mapping)
    ansatz = build_ansatz(ansatz_name, mapping, reps)
    depth, two_qubit = circuit_cost(ansatz)

    # Transpile once for the backend. The angles stay symbolic, so this
    # serves every step of the loop. The Hamiltonian follows the layout.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa = pm.run(ansatz)
    isa_h = hamiltonian.apply_layout(isa.layout)
    precision = 1 / np.sqrt(shots) if shots else 0.0

    rng = np.random.default_rng(seed)
    history, best_x, best_e = [], None, np.inf
    for _ in range(starts if ansatz.num_parameters else 1):
        x0 = initial_point(ansatz_name, ansatz.num_parameters, rng)
        x, hist = minimize_energy(estimator, isa, isa_h, x0, precision, maxiter)
        history.append(hist)
        if min(hist) < best_e:
            best_x, best_e = x, min(hist)
    return {
        "mapping": mapping,
        "ansatz": ansatz_name,
        "qubits": hamiltonian.num_qubits,
        "terms": len(hamiltonian),
        "params": ansatz.num_parameters,
        "depth": depth,
        "two_qubit": two_qubit,
        "history": history,
        "energy": best_e,
        "theta": [float(v) for v in best_x],
    }


def run(
    distance: float = 0.735,
    mappings=MAPPINGS,
    ansatzes=ANSATZES,
    backend_name=None,
    reps: int = 2,
    starts: int = 3,
    shots: int = 0,
    maxiter: int = 300,
    seed: int = 1,
) -> dict:
    """Optimize every pair and return a dict of results for analysis.py."""
    backend, estimator, session = get_primitives(backend_name)
    exact = exact_energies(distance)
    print(
        f"R = {distance} A: exact (FCI) energy {exact['fci']:.6f} Ha, "
        f"Hartree-Fock {exact['hf']:.6f} Ha"
    )
    t0 = time.perf_counter()
    runs = []
    try:
        for mapping in mappings:
            for name in ansatzes:
                rec = optimize(
                    distance, mapping, name, estimator, backend,
                    reps, starts, shots, maxiter, seed,
                )  # fmt: skip
                n_evals = sum(len(h) for h in rec["history"])
                print(
                    f"  {mapping:>8} {name:>7}: error "
                    f"{rec['energy'] - exact['fci']:+.2e} Ha "
                    f"({rec['params']} angles, {n_evals} evaluations)"
                )
                runs.append(rec)
    finally:
        if session is not None:
            session.close()
    return {
        "distance": distance,
        "backend": backend.name,
        "reps": reps,
        "starts": starts,
        "shots": shots,
        "seconds": time.perf_counter() - t0,
        "exact": exact,
        "runs": runs,
    }
