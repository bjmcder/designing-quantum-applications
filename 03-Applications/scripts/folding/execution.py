"""Execution: build the QAOA loop and run the lattice folding example.

The variational loop:

    1. A classical optimizer (COBYLA) proposes the QAOA angles (beta, gamma).
    2. The Estimator runs the circuit and returns the average energy <H>.
    3. The optimizer lowers <H> and goes back to step 1.

Because the landscape has many local minima, the loop is restarted from a
few random points. The best angles found are then used to sample the circuit,
and the counts go to analysis.py.

Called from main.py. With no backend name it runs on the Aer simulator;
with one it runs on IBM hardware inside a single Session.
"""

import time

import numpy as np
from encoding import build_ansatz
from ops import build_hamiltonian
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize


def get_primitives(backend_name: str | None):
    """Return (backend, estimator, sampler, session).

    The session is None on the simulator.
    """
    if backend_name is None:
        return AerSimulator(), AerEstimator(), AerSampler(), None
    from qiskit_ibm_runtime import (
        EstimatorV2,
        QiskitRuntimeService,
        SamplerV2,
        Session,
    )

    backend = QiskitRuntimeService().backend(backend_name)
    # A Session keeps our place at the front of the queue for the whole loop.
    # Without it, every step (which needs the previous result) would wait in
    # line again.
    session = Session(backend=backend)
    return backend, EstimatorV2(mode=session), SamplerV2(mode=session), session


def run_qaoa(estimator, circuit, hamiltonian, x0, shots: int, maxiter: int):
    """Minimize <H>(theta) with COBYLA from the starting angles ``x0``.

    Returns the optimizer result and the list of every energy it evaluated.
    """
    history = []
    # The Estimator takes a target standard error rather than a shot count.
    # For an observable with values near +-1, N shots give about 1 / sqrt(N).
    precision = 1 / np.sqrt(shots)

    def cost(theta):
        job = estimator.run([(circuit, hamiltonian, theta, precision)])
        history.append(float(job.result()[0].data.evs))
        return history[-1]

    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return result, history


def run(
    sequence,
    grid,
    encoding,
    backend_name=None,
    reps=2,
    starts=3,
    shots=4096,
    maxiter=150,
    seed=1,
) -> dict:
    """Optimize the QAOA circuit, sample it, and return a dict of results.

    The dict holds the settings, the energy history of every restart, the
    best angles, and the sampled counts (everything analysis.py needs).
    """
    hamiltonian, cards, _ = build_hamiltonian(encoding, sequence, grid)
    ansatz = build_ansatz(hamiltonian, cards, encoding, reps)
    backend, estimator, sampler, session = get_primitives(backend_name)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    # Transpile once: pick physical qubits and rewrite the gates into the
    # backend's native set. The angles stay symbolic, so this serves every
    # step of the loop. The Hamiltonian must follow the same qubit layout.
    isa = pm.run(ansatz)
    isa_h = hamiltonian.apply_layout(isa.layout)
    print(
        f"{encoding}: {isa.num_qubits} qubits, depth {isa.depth()}, "
        f"{isa.num_nonlocal_gates()} two-qubit gates on {backend.name}"
    )

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best, history = None, []
    try:
        for s in range(starts):
            # QAOAAnsatz orders its parameters beta_0..beta_(p-1), then
            # gamma_0..gamma_(p-1). Small gamma and moderate beta is a
            # common starting point.
            x0 = np.concatenate(
                [rng.uniform(0, np.pi / 2, reps), rng.uniform(0, 0.5, reps)]
            )
            result, hist = run_qaoa(estimator, isa, isa_h, x0, shots, maxiter)
            print(
                f"  start {s}: best <H> = {min(hist):+.3f} "
                f"({len(hist)} evaluations)"
            )
            history.append(hist)
            if best is None or result.fun < best.fun:
                best = result
        # Fix the best angles, add measurements, and sample the circuit.
        measured = ansatz.assign_parameters(best.x)
        measured.measure_all()
        counts = (
            sampler.run([pm.run(measured)], shots=shots)
            .result()[0]
            .data.meas.get_counts()
        )
    finally:
        if session is not None:
            session.close()

    return {
        "encoding": encoding,
        "backend": backend.name,
        "sequence": sequence,
        "grid": list(grid),
        "reps": reps,
        "shots": shots,
        "seconds": time.perf_counter() - t0,
        "theta": best.x.tolist(),
        "history": history,
        "counts": counts,
    }
