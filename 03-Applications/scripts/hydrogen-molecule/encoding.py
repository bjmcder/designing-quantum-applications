"""Encoding: build the ansatzes for the H2 example.

An ansatz is a circuit with free angles. It decides which states the
optimizer can reach. The five below all aim at the H2 ground state, but they
differ in how much they know about it.

hf        Hartree-Fock state, no angles. The baseline: no electron correlation.
uccd      Hartree-Fock plus one double excitation, exp(theta G). One angle.
          For H2 in a minimal basis this is exact.
uccsd     Same plus the two single excitations. Three angles. The singles are
          forbidden by symmetry, so they cannot lower the energy.
hea       Hardware-efficient: layers of R_Y rotations and CX gates, starting
          from |0...0>. Knows nothing about electrons.
hea_hf    The same circuit, started from the Hartree-Fock state.

The UCC circuits are built from the fermionic generators in ops.py, so they
work in every qubit mapping. The hardware-efficient circuits are the same in
every mapping, only the meaning of the qubits changes.
"""

import numpy as np
from ops import excitation_generators, hf_bits, n_qubits, to_pauli
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate, real_amplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ANSATZES = ("hf", "uccd", "uccsd", "hea", "hea_hf")

LABELS = {
    "hf": "Hartree-Fock",
    "uccd": "UCCD",
    "uccsd": "UCCSD",
    "hea": "HEA",
    "hea_hf": "HEA + HF start",
}


def hf_state(mapping: str) -> QuantumCircuit:
    """Circuit that prepares the Hartree-Fock determinant in a qubit mapping.

    Args:
        mapping: Qubit mapping, one of ``ops.MAPPINGS``.

    Returns:
        The state-preparation circuit, which has no angles.
    """
    bits = hf_bits(mapping)  # qubit 0 is the rightmost character
    circuit = QuantumCircuit(n_qubits(mapping), name="HF")
    for qubit, bit in enumerate(reversed(bits)):
        if bit == "1":
            circuit.x(qubit)
    return circuit


def build_ansatz(name: str, mapping: str, reps: int = 2) -> QuantumCircuit:
    """Parameterized circuit for one ansatz in one qubit mapping.

    Args:
        name: Ansatz, one of ``ANSATZES``.
        mapping: Qubit mapping, one of ``ops.MAPPINGS``.
        reps: Layers of the hardware-efficient ansatzes. Ignored by the others.

    Returns:
        The ansatz with symbolic angles.

    Raises:
        ValueError: If the ansatz is unknown.
    """
    if name not in ANSATZES:
        raise ValueError(f"unknown ansatz {name!r}; choose from {ANSATZES}")
    n = n_qubits(mapping)
    if name == "hf":
        return hf_state(mapping)
    if name.startswith("hea"):
        circuit = QuantumCircuit(n, name=name)
        if name == "hea_hf":
            circuit.compose(hf_state(mapping), inplace=True)
        circuit.compose(
            real_amplitudes(n, reps=reps, entanglement="linear"), inplace=True
        )
        return circuit

    wanted = ["double"] if name == "uccd" else ["single_up", "single_down", "double"]
    generators = excitation_generators()
    circuit = hf_state(mapping)
    circuit.name = name
    for key in wanted:
        # G is anti-Hermitian, so i G is Hermitian and exp(theta G) is the
        # evolution under it for "time" theta. The Pauli terms of one
        # excitation commute with each other, so a single Trotter step is exact.
        op = to_pauli(1j * generators[key], mapping)
        theta = Parameter(f"theta_{key}")
        circuit.append(PauliEvolutionGate(op, time=theta), range(n))
    return circuit


def initial_point(name: str, n_params: int, rng: np.random.Generator):
    """Random starting angles.

    Excitation angles are rotations away from Hartree-Fock, so we start close
    to it. Hardware-efficient angles have no such meaning, so any angle is as
    good as any other.

    Args:
        name: Ansatz name.
        n_params: Number of angles.
        rng: Random number generator.

    Returns:
        The starting angles in radians.
    """
    if name.startswith("uc"):
        return rng.normal(0, 0.1, n_params)
    return rng.uniform(-np.pi, np.pi, n_params)


def circuit_cost(circuit: QuantumCircuit) -> tuple[int, int]:
    """Depth and two-qubit gate count after compiling to {RZ, SX, X, CX}.

    A fixed gate set makes the numbers comparable between ansatzes. It is not
    the cost on any one device, which also depends on its connectivity.

    Args:
        circuit: The ansatz.

    Returns:
        (depth, two-qubit gates).
    """
    pm = generate_preset_pass_manager(
        basis_gates=["rz", "sx", "x", "cx"], optimization_level=3
    )
    compiled = pm.run(circuit.decompose(reps=2))
    return compiled.depth(), compiled.num_nonlocal_gates()
