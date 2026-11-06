"""Encoding: the simple circuits we test the noise model on.

Each circuit prepares a state and then idles for n steps. An idle step is an
``id`` (identity) gate. Ideally it does nothing, so without noise every
circuit below keeps its state forever. All the change we see comes from the
noise model attached to ``id`` in execution.py.

t1   X, then n idle steps.   Prepares |1>.   We watch <Z> climb from -1 to +1.
t2   H, then n idle steps.   Prepares |+>.   We watch <X> fall from +1 to 0.

The wait is n * STEP microseconds long.
"""

from qiskit import QuantumCircuit


def build_circuit(name: str, steps: int) -> QuantumCircuit:
    """Build one test circuit.

    Args:
        name: "t1" or "t2".
        steps: Number of idle steps to wait.

    Returns:
        A one-qubit circuit with no measurement.
    """
    qc = QuantumCircuit(1)
    if name == "t1":
        qc.x(0)
    else:
        qc.h(0)
    for _ in range(steps):
        qc.id(0)
    return qc
