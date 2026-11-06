"""Encoding: the circuit of one layer.

A layer circuit holds one random two-qubit sequence for each link of the
layer, side by side on separate qubits, so the links run at the same time:

    qubit 0 --[ sequence for link 0-1 ]--
    qubit 1 --[                       ]--     measure all
    qubit 2 --[ sequence for link 2-3 ]--
    qubit 3 --[                       ]--

Each link has its own random Cliffords. A qubit that is not in the layer does
nothing. Without noise every link returns to 00, so the survival probability
of a link is the fraction of shots in which both of its qubits read 0.
"""

import numpy as np
from ops import pair_sequence
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def build_layer_circuit(
    layer: list[tuple[int, int]], n_qubits: int, m: int, rng: np.random.Generator
) -> QuantumCircuit:
    """Circuit that runs a random sequence of m Cliffords on every link of a layer.

    Args:
        layer: Links of the layer, as (a, b) pairs of qubit numbers.
        n_qubits: Number of qubits in the chain.
        m: Number of random Cliffords per link.
        rng: Random number generator.

    Returns:
        Circuit on ``n_qubits`` qubits, measured at the end.
    """
    qc = QuantumCircuit(n_qubits)
    for a, b in layer:
        qc.compose(pair_sequence(m, rng), qubits=[a, b], inplace=True)
    qc.measure_all()
    return qc


def transpile_for(circuit: QuantumCircuit, backend, chain: list[int] | None = None):
    """Compile a layer circuit without shortening it.

    Optimization is off, because every sequence is the identity and an
    optimizer would delete it. On hardware, logical qubit i goes to physical
    qubit ``chain[i]``. On the simulator the circuit is only rewritten in
    CX, SX, and RZ gates, the gates that the noise model knows.

    Args:
        circuit: A circuit from ``build_layer_circuit``.
        backend: IBM backend, or None for the simulator.
        chain: Physical qubits of the chain, in order (hardware only).

    Returns:
        The compiled circuit.
    """
    if backend is None:
        return transpile(circuit, basis_gates=["cx", "sx", "rz"], optimization_level=0)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=0, initial_layout=chain
    )
    return pm.run(circuit)


def cx_count(circuit: QuantumCircuit) -> int:
    """Number of two-qubit gates in a compiled circuit."""
    return int(
        sum(n for name, n in circuit.count_ops().items() if name in ("cx", "ecr", "cz"))
    )
