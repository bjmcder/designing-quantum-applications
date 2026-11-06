"""Execution: attach the Kraus operators to ``id`` and simulate.

Steps:
    1. ``relaxation_kraus(STEP)`` gives the four operators for one idle step.
    2. ``kraus_error`` wraps them in an error that Aer understands.
    3. ``NoiseModel.add_all_qubit_quantum_error`` says "apply this error after
       every id gate".
    4. The density-matrix simulator runs the circuit and returns rho.

We read rho itself instead of sampling shots. That gives exact numbers with
no shot noise, so any difference between the model and the formulas is
a real difference.
"""

import numpy as np
from encoding import build_circuit
from ops import STEP, T1, T2, relaxation_kraus
from qiskit import transpile
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, kraus_error


def make_noise_model(step: float = STEP, t1: float = T1, t2: float = T2) -> NoiseModel:
    """Noise model in which every id gate lasts ``step`` and has T1 and T2.

    Args:
        step: Duration of one idle step, in microseconds.
        t1: Relaxation time.
        t2: Dephasing time.

    Returns:
        The noise model.
    """
    error = kraus_error(relaxation_kraus(step, t1, t2))
    model = NoiseModel()
    model.add_all_qubit_quantum_error(error, ["id"])
    return model


def final_state(circuit, noise_model=None) -> np.ndarray:
    """Run a circuit and return the density matrix of the qubit at the end.

    Args:
        circuit: One-qubit circuit without measurement.
        noise_model: Aer noise model, or None for a perfect simulation.

    Returns:
        2x2 density matrix.
    """
    qc = circuit.copy()
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    # Keep id gates: with optimization_level=0 nothing is removed.
    qc = transpile(qc, sim, optimization_level=0)
    return np.array(sim.run(qc).result().data()["density_matrix"])


def expectation(rho: np.ndarray, pauli: str) -> float:
    """<O> = trace(rho O) for O = X, Y, or Z.

    Args:
        rho: 2x2 density matrix.
        pauli: "X", "Y", or "Z".

    Returns:
        The expectation value.
    """
    return float(np.real(np.trace(rho @ Pauli(pauli).to_matrix())))


def run(name: str, steps_list, step: float = STEP, t1: float = T1, t2: float = T2):
    """Simulate a test circuit for every wait length.

    Args:
        name: "t1" or "t2".
        steps_list: Numbers of idle steps to try.
        step: Duration of one idle step, in microseconds.
        t1: Relaxation time.
        t2: Dephasing time.

    Returns:
        Dict with ``times_us``, the density matrices ``rho`` (noisy), and
        ``rho_ideal`` (noiseless), one per wait.
    """
    model = make_noise_model(step, t1, t2)
    out = {"times_us": [], "rho": [], "rho_ideal": []}
    for n in steps_list:
        circuit = build_circuit(name, int(n))
        out["times_us"].append(float(n) * step)
        out["rho"].append(final_state(circuit, model))
        out["rho_ideal"].append(final_state(circuit))
    return out
