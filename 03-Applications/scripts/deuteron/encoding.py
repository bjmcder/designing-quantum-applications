"""Encoding: build the ansatzes for the deuteron example.

An ansatz is a circuit with free angles. It decides which states the
optimizer can reach. The ground state is a real vector over the N oscillator
states, so all four below use real rotations only.

ref       The state |0> (no angles). The baseline: no mixing of oscillator
          states, which is what the contact interaction alone would favor.
rot       |0> followed by a chain of rotations exp(theta_k G_k), where G_k
          moves amplitude from |k> to |k+1>. N - 1 angles. These are the
          spherical coordinates of a vector, so the chain reaches every real
          state: it is exact for this problem.
hea       Hardware-efficient: layers of R_Y rotations and CX gates, starting
          from |0...0>. Knows nothing about the basis.
hea_ref   The same circuit, started from the reference state |0>.

The rotation chain is built from the matrices in ops.py, so it works in every
qubit mapping. The hardware-efficient circuits are the same in every mapping,
only the meaning of the qubits changes.
"""

import numpy as np
from ops import n_qubits, ref_bits, rotation_generators, to_pauli
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate, real_amplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ANSATZES = ("ref", "rot", "hea", "hea_ref")

LABELS = {
    "ref": "Reference |0>",
    "rot": "Rotation chain",
    "hea": "HEA",
    "hea_ref": "HEA + |0> start",
}


def ref_state(n_states: int, mapping: str) -> QuantumCircuit:
    """Circuit that prepares oscillator state |0> in this mapping."""
    bits = ref_bits(n_states, mapping)  # qubit 0 is the rightmost character
    circuit = QuantumCircuit(n_qubits(n_states, mapping), name="ref")
    for qubit, bit in enumerate(reversed(bits)):
        if bit == "1":
            circuit.x(qubit)
    return circuit


def build_ansatz(
    name: str, n_states: int, mapping: str, reps: int = 2
) -> QuantumCircuit:
    """Parameterized circuit for ``name`` in the qubit ``mapping``.

    ``reps`` only applies to the hardware-efficient ansatzes.
    """
    if name not in ANSATZES:
        raise ValueError(f"unknown ansatz {name!r}; choose from {ANSATZES}")
    n = n_qubits(n_states, mapping)
    if name == "ref":
        return ref_state(n_states, mapping)
    if name.startswith("hea"):
        circuit = QuantumCircuit(n, name=name)
        if name == "hea_ref":
            circuit.compose(ref_state(n_states, mapping), inplace=True)
        circuit.compose(
            real_amplitudes(n, reps=reps, entanglement="linear"), inplace=True
        )
        return circuit

    circuit = ref_state(n_states, mapping)
    circuit.name = name
    for k, G in enumerate(rotation_generators(n_states)):
        # G is anti-Hermitian, so i G is Hermitian and exp(theta G) is the
        # evolution under it for "time" theta. The two Pauli terms of one
        # rotation commute, so a single Trotter step is exact. Different
        # rotations do not commute, so their order matters.
        op = to_pauli(1j * G, mapping)
        theta = Parameter(f"theta_{k:02d}")  # sorted order = chain order
        circuit.append(PauliEvolutionGate(op, time=theta), range(n))
    return circuit


def initial_point(name: str, n_params: int, rng: np.random.Generator):
    """Random starting angles.

    Rotation angles are small mixings away from |0>, so we start close to it.
    Hardware-efficient angles have no such meaning, so any angle is as good
    as any other.
    """
    if name == "rot":
        return rng.normal(0, 0.1, n_params)
    return rng.uniform(-np.pi, np.pi, n_params)


def circuit_cost(circuit: QuantumCircuit) -> tuple[int, int]:
    """(depth, two-qubit gates) after compiling to {RZ, SX, X, CX}.

    A fixed gate set makes the numbers comparable between ansatzes. It is not
    the cost on any one device, which also depends on its connectivity.
    """
    pm = generate_preset_pass_manager(
        basis_gates=["rz", "sx", "x", "cx"], optimization_level=3
    )
    compiled = pm.run(circuit.decompose(reps=2))
    return compiled.depth(), compiled.num_nonlocal_gates()
