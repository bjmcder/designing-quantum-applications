"""Encoding: the test circuit and the two ways to make its noise larger.

The test circuit (two qubits, no measurement; the Estimator measures ZZ):

    RY(THETA) on qubit 0, then N CX gates on (0, 1).

Ideally the CX gates cancel in pairs and <ZZ> = cos(THETA).

Gate folding raises the noise without changing the ideal answer. A gate G
followed by its inverse G^dagger does nothing, so replacing

    G  ->  G G^dagger G

keeps the circuit equivalent to the original and makes the gate run three
times. The noise of the gate triples. Here G is a CX, and a CX is its own
inverse, so the replacement is three CX in a row. Only two-qubit gates are
folded, because they are the noisiest. A noise factor between 1 and 3 folds
a fraction of the gates (each fold adds two gates, so a factor of 2 folds
half of them).

Folding the whole circuit does the same at the scale of the circuit:

    U  ->  U U^dagger U

U^dagger is U with the gates in reverse order, each one inverted. The noise
factor is 3, and every gate, including the single-qubit ones, is repeated.

The optimizer would delete both (a gate followed by its inverse is what it is
built to remove), so circuits are compiled with the optimizer off.
"""

from ops import THETA, folds_needed
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def build_circuit(n_cx: int, theta: float = THETA) -> QuantumCircuit:
    """The test circuit.

    Args:
        n_cx: Number of CX gates. Use an even number, so the ideal value is
            cos(theta).
        theta: Rotation angle of qubit 0 in radians.

    Returns:
        A two-qubit circuit with no measurement.
    """
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    for _ in range(n_cx):
        qc.cx(0, 1)
    return qc


def fold_gates(circuit: QuantumCircuit, factor: float) -> QuantumCircuit:
    """Raise the noise by folding two-qubit gates: G -> G G^dagger G.

    Args:
        circuit: A circuit from ``build_circuit``.
        factor: Noise factor, at least 1.

    Returns:
        A circuit with the same ideal result and about ``factor`` times as
        many two-qubit gates. When the factor needs only some of the gates
        folded once more, the first gates in the circuit are the ones.
    """
    two_qubit = sum(len(inst.qubits) == 2 for inst in circuit.data)
    each, extra = divmod(folds_needed(two_qubit, factor), max(two_qubit, 1))
    folded = circuit.copy_empty_like()
    seen = 0
    for inst in circuit.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if len(inst.qubits) == 2:
            for _ in range(each + (seen < extra)):
                folded.append(inst.operation.inverse(), inst.qubits, inst.clbits)
                folded.append(inst.operation, inst.qubits, inst.clbits)
            seen += 1
    return folded


def fold_circuit(circuit: QuantumCircuit, factor: int) -> QuantumCircuit:
    """Raise the noise by folding the whole circuit: U -> U U^dagger U.

    Args:
        circuit: A circuit from ``build_circuit``.
        factor: An odd whole number, 1, 3, 5, ...

    Returns:
        U followed by (U^dagger U) repeated (factor - 1) / 2 times.

    Raises:
        ValueError: If the factor is not an odd whole number.
    """
    if factor < 1 or factor != int(factor) or int(factor) % 2 == 0:
        raise ValueError(f"whole-circuit folding needs an odd factor, got {factor}")
    folded = circuit.copy()
    for _ in range((int(factor) - 1) // 2):
        folded.compose(circuit.inverse(), inplace=True)
        folded.compose(circuit, inplace=True)
    return folded


def transpile_for(circuit: QuantumCircuit, backend, qubits: list[int] | None = None):
    """Compile a circuit for a backend without changing it.

    Optimization is off. The optimizer would see that CX followed by CX does
    nothing and delete the very gates whose noise we want to measure.

    Args:
        circuit: A circuit from ``build_circuit``.
        backend: Backend to compile for.
        qubits: The two physical qubits to run on, or None to let the
            compiler choose.

    Returns:
        The compiled circuit.
    """
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=0, initial_layout=qubits
    )
    return pm.run(circuit)
