"""Encoding: the circuit of one random sequence.

Each Clifford is written out as its gates (H and S), one after the other,
and the whole sequence is measured at the end:

    |0>  --[C1]--[C2]-- ... --[Cm]--[C_inverse]--  measure

Without noise the gates cancel and the measurement always gives 0. A barrier
between Cliffords keeps the compiler from merging neighbors, which would
shorten the sequence and hide the errors we want to measure.
"""

import numpy as np
from ops import clifford_table
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Instructions that are not gates and so add no gate error.
NOT_GATES = {"measure", "barrier", "rz", "delay"}


def build_circuit(sequence: list[int]) -> QuantumCircuit:
    """Circuit for a sequence of Cliffords, with a measurement at the end.

    Args:
        sequence: Indices into ``ops.clifford_table()``.

    Returns:
        One-qubit circuit measured into one classical bit.
    """
    table = clifford_table()
    qc = QuantumCircuit(1)
    for index in sequence:
        for name in table[index][1]:
            getattr(qc, name)(0)
        qc.barrier()
    qc.measure_all()
    return qc


def transpile_for(circuit: QuantumCircuit, backend, qubit: int | None = None):
    """Compile a sequence without shortening it.

    Optimization is off. The optimizer would see that the whole sequence is
    the identity and delete it.

    Args:
        circuit: A circuit from ``build_circuit``.
        backend: Backend to compile for.
        qubit: Physical qubit to run on, or None to let the compiler choose.

    Returns:
        The compiled circuit.
    """
    layout = None if qubit is None else [qubit]
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=0, initial_layout=layout
    )
    return pm.run(circuit)


def gate_count(circuit: QuantumCircuit) -> int:
    """Number of gates that carry error (rz is a software phase, so it is free).

    Args:
        circuit: Compiled circuit.

    Returns:
        The count.
    """
    ops = circuit.count_ops()
    return int(sum(n for name, n in ops.items() if name not in NOT_GATES))
