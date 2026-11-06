"""Encoding: build the ansatzes for a nucleus.

An ansatz is a circuit with free angles. It decides which states the
optimizer can reach. For a nucleus the state must also keep the right numbers
of protons and neutrons, so two of the four ansatzes below are built from
**excitations**: moves of nucleons from occupied orbitals to empty ones.

ref        The reference filling (no angles). Each nucleon sits in the lowest
           free oscillator state. The baseline: no mixing.
uccs       Reference followed by rotations exp(theta (a+_a a_i - a+_i a_a))
           that move one nucleon from orbital i to orbital a of the same
           kind. Each nucleon can then move to a different oscillator state,
           but the nucleons stay independent of each other (a mean-field
           state).
uccsd      The same plus pair rotations that move two nucleons at once
           (exp(theta (a+_a a+_b a_j a_i - h.c.)). The pairs let the
           nucleons correlate. This is the unitary coupled cluster ansatz of
           quantum chemistry, in its singles-and-doubles form.
hea_ref    Hardware-efficient: layers of R_Y rotations and CX gates, started
           from the reference filling. Knows nothing about nucleons, so it
           can leave the sector with the right numbers of them.

Every rotation conserves the number of nucleons of each kind, and so does the
product, so ``uccs`` and ``uccsd`` stay in the right sector by construction.
The excitations are built from fermion operators and mapped to Pauli strings,
so the same code works in every fermion-to-qubit mapping.
"""

from itertools import combinations

import numpy as np
from ops import (
    ladder,
    mapped_bits,
    n_modes,
    reference_occupation,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate, real_amplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ANSATZES = ("ref", "uccs", "uccsd", "hea_ref")

LABELS = {
    "ref": "Reference",
    "uccs": "UCC singles",
    "uccsd": "UCC singles + doubles",
    "hea_ref": "HEA + reference start",
}


def kind_of(j: int, n_orb: int) -> int:
    """Kind of nucleon (index into ``ops.KINDS``) that lives in orbital ``j``.

    Args:
        j: Orbital index.
        n_orb: Oscillator states per kind of nucleon.

    Returns:
        0 to 3.
    """
    return j // n_orb


def excitations(
    z: int, n: int, n_orb: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    """Single and double excitations that keep the number of each kind.

    Args:
        z: Number of protons.
        n: Number of neutrons.
        n_orb: Oscillator states per kind of nucleon.

    Returns:
        (singles, doubles). A single (i, a) moves a nucleon from occupied
        orbital i to empty orbital a of the same kind. A double (i, j, a, b)
        moves two nucleons from occupied i and j to empty a and b, so that
        every kind has as many nucleons afterwards as before.
    """
    occ = reference_occupation(z, n, n_orb)
    full = [j for j in range(n_modes(n_orb)) if occ[j]]
    empty = [j for j in range(n_modes(n_orb)) if not occ[j]]
    kind = lambda j: kind_of(j, n_orb)
    singles = [(i, a) for i in full for a in empty if kind(i) == kind(a)]
    doubles = [
        (i, j, a, b)
        for i, j in combinations(full, 2)
        for a, b in combinations(empty, 2)
        if sorted((kind(i), kind(j))) == sorted((kind(a), kind(b)))
    ]
    return singles, doubles


def reference_state(z: int, n: int, n_orb: int, mapping: str) -> QuantumCircuit:
    """Circuit that fills the reference orbitals in a mapping.

    Args:
        z: Number of protons.
        n: Number of neutrons.
        n_orb: Oscillator states per kind of nucleon.
        mapping: Fermion-to-qubit mapping, one of ``ops.MAPPINGS``.

    Returns:
        The state-preparation circuit, which has no angles.
    """
    bits = mapped_bits(reference_occupation(z, n, n_orb), mapping)
    circuit = QuantumCircuit(n_modes(n_orb), name="ref")
    for qubit, bit in enumerate(bits):
        if bit:
            circuit.x(qubit)
    return circuit


def _rotation(destroy: list[int], create: list[int], width: int, mapping: str):
    """Hermitian operator i (T - T+) for the excitation T = a+...a, as Paulis.

    Args:
        destroy: Orbitals to empty.
        create: Orbitals to fill.
        width: Total number of orbitals.
        mapping: Fermion-to-qubit mapping, one of ``ops.MAPPINGS``.

    Returns:
        The operator i (T - T+). exp(-i theta (that)) = exp(theta (T - T+))
        is a real rotation by the angle theta.
    """
    ops = [ladder(j, width, mapping) for j in range(width)]
    T = ops[create[0]][0]
    for j in create[1:]:
        T = T @ ops[j][0]
    for j in destroy:
        T = T @ ops[j][1]
    return (1j * (T - T.adjoint())).simplify(atol=1e-10)


def build_ansatz(
    name: str, z: int, n: int, n_orb: int, mapping: str, reps: int = 2
) -> QuantumCircuit:
    """Parameterized circuit for one ansatz in one mapping.

    Args:
        name: Ansatz, one of ``ANSATZES``.
        z: Number of protons.
        n: Number of neutrons.
        n_orb: Oscillator states per kind of nucleon.
        mapping: Fermion-to-qubit mapping, one of ``ops.MAPPINGS``.
        reps: Layers of the hardware-efficient ansatz. Ignored by the others.

    Returns:
        The ansatz with symbolic angles.

    Raises:
        ValueError: If the ansatz is unknown.
    """
    if name not in ANSATZES:
        raise ValueError(f"unknown ansatz {name!r}; choose from {ANSATZES}")
    width = n_modes(n_orb)
    circuit = reference_state(z, n, n_orb, mapping)
    circuit.name = name
    if name == "ref":
        return circuit
    if name == "hea_ref":
        circuit.compose(real_amplitudes(width, reps=reps, entanglement="linear"), inplace=True)
        return circuit

    singles, doubles = excitations(z, n, n_orb)
    moves = [([i], [a]) for i, a in singles]
    if name == "uccsd":
        moves += [([i, j], [a, b]) for i, j, a, b in doubles]
    for k, (destroy, create) in enumerate(moves):
        op = _rotation(destroy, create, width, mapping)
        # The Pauli strings of one excitation commute, so one Trotter step is
        # exact. Different excitations do not commute, so their order matters.
        theta = Parameter(f"theta_{k:03d}")  # sorted order = circuit order
        circuit.append(PauliEvolutionGate(op, time=theta), range(width))
    return circuit


def initial_point(name: str, n_params: int, rng: np.random.Generator):
    """Random starting angles.

    Excitation angles are small mixings away from the reference, so we start
    close to it. Hardware-efficient angles have no such meaning, so any angle
    is as good as any other.

    Args:
        name: Ansatz name.
        n_params: Number of angles.
        rng: Random number generator.

    Returns:
        The starting angles in radians.
    """
    if name.startswith("ucc"):
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
    pm = generate_preset_pass_manager(basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)
    compiled = pm.run(circuit)
    return compiled.depth(), compiled.num_nonlocal_gates()
