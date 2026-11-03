"""Ops: build the deuteron Hamiltonian in several qubit forms.

The problem: find the binding energy of the deuteron, a proton and a neutron
held together by the nuclear force. In the center-of-mass frame this is one
particle of reduced mass moving in a potential, in the S wave (no angular
momentum). We use the pionless effective field theory of Dumitrescu et al.,
PRL 120, 210501 (2018): at the low energies of the deuteron the force is so
short-ranged that a contact (delta-function) interaction is enough.

Steps:

1. basis: expand the wave function in the first N S-wave states |n> of a
   three-dimensional harmonic oscillator of frequency omega. The kinetic
   energy is tridiagonal in this basis and the contact interaction touches
   only |0>, so H is an N x N real matrix with a simple closed form.
2. qubit mapping: the same matrix written for qubits. The mappings differ in
   which qubit state stands for which basis state |n>, so they all have the
   same eigenvalues. They differ in the number of qubits and Pauli terms.

Energies are in MeV. The binding energy is minus the lowest eigenvalue.
"""

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

HBAR_OMEGA = 7.0  # MeV, oscillator spacing
V0 = 5.68658111  # MeV, strength of the contact interaction
EXPERIMENT = -2.224575  # MeV, measured deuteron binding energy

MAPPINGS = ("binary", "gray", "unary")


# ------------------------------------------------------------------- basis


def basis_hamiltonian(
    n_states: int, hbar_omega: float = HBAR_OMEGA, v0: float = V0
) -> np.ndarray:
    """Matrix of H = T + V in the first ``n_states`` S-wave oscillator states.

    T[n, n]   = (hw/2) (2n + 3/2)
    T[n, n+1] = -(hw/2) sqrt((n + 1)(n + 3/2))
    V[0, 0]   = -V0

    T couples each state only to its neighbors, so H is tridiagonal.

    Args:
        n_states: Number of oscillator states N.
        hbar_omega: Oscillator spacing in MeV.
        v0: Strength of the contact interaction in MeV.

    Returns:
        The N x N Hamiltonian in MeV.
    """
    H = np.zeros((n_states, n_states))
    for n in range(n_states):
        H[n, n] = 0.5 * hbar_omega * (2 * n + 1.5)
        if n + 1 < n_states:
            off = -0.5 * hbar_omega * np.sqrt((n + 1) * (n + 1.5))
            H[n, n + 1] = H[n + 1, n] = off
    H[0, 0] -= v0
    return H


def check_states(n_states: int) -> None:
    """Check that ``n_states`` is a power of two.

    The binary and Gray mappings need a power of two to fill the qubits.

    Args:
        n_states: Number of oscillator states.

    Raises:
        ValueError: If ``n_states`` is not 2, 4, 8, ...
    """
    if n_states < 2 or n_states & (n_states - 1):
        raise ValueError(f"n_states must be a power of two (2, 4, 8, ...), got {n_states}")


def ground_energy(n_states: int) -> float:
    """Lowest eigenvalue of the truncated Hamiltonian.

    Args:
        n_states: Number of oscillator states, any positive integer.

    Returns:
        The ground-state energy in MeV.
    """
    return float(np.linalg.eigvalsh(basis_hamiltonian(n_states))[0])


# ------------------------------------------------------------- mappings


def n_qubits(n_states: int, mapping: str) -> int:
    """Number of qubits a mapping needs.

    Args:
        n_states: Number of oscillator states.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        The qubit count.

    Raises:
        ValueError: If the mapping is unknown.
    """
    if mapping == "unary":
        return n_states
    if mapping in ("binary", "gray"):
        return int(np.log2(n_states))
    raise ValueError(f"unknown mapping {mapping!r}; choose from {MAPPINGS}")


def encode(n: int, mapping: str) -> int:
    """Index of the qubit basis state that stands for oscillator state |n>.

    binary: the bits of n.
    gray:   the Gray code of n, so neighbors n and n + 1 differ in one bit.
            The kinetic energy couples exactly such neighbors.
    unary:  one qubit per state, only qubit n is 1 ("one-hot").

    Args:
        n: Oscillator state index.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        The index of the qubit basis state.

    Raises:
        ValueError: If the mapping is unknown.
    """
    if mapping == "binary":
        return n
    if mapping == "gray":
        return n ^ (n >> 1)
    if mapping == "unary":
        return 1 << n
    raise ValueError(f"unknown mapping {mapping!r}; choose from {MAPPINGS}")


def _site(letters: dict[int, str], width: int) -> SparsePauliOp:
    """Pauli string with the given letters on some qubits and I elsewhere.

    Args:
        letters: Pauli letter for each qubit index, for example {0: "Z"}.
        width: Total number of qubits.

    Returns:
        The Pauli string as a one-term operator.
    """
    label = ["I"] * width
    for qubit, letter in letters.items():
        label[qubit] = letter
    return SparsePauliOp("".join(reversed(label)))  # qubit 0 is rightmost


def _unary_pauli(matrix: np.ndarray) -> SparsePauliOp:
    """Unary form of an operator on |0> .. |N-1>, written with Paulis directly.

    |n> is the state with only qubit n excited, so |n><m| becomes
    sigma+_n sigma-_m ("move the excitation from m to n") and |n><n| becomes
    the number operator (1 - Z_n) / 2. Both are products of two Paulis, which
    keeps the term count at about 2 N for a tridiagonal matrix. Building the
    dense matrix first would give hundreds of terms.

    The operator is only fixed on the N states with one excitation. The other
    states are not oscillator states, and this form gives them whatever the
    sum says, 0 for no excitation. For the Hamiltonian they are sums of
    oscillator energies, and all but the ground state are positive, so none
    of them goes below the ground state.

    Args:
        matrix: N x N operator on the oscillator states.

    Returns:
        The operator on N qubits.
    """
    n = len(matrix)
    total = SparsePauliOp("I" * n, 0)
    for p, q in np.ndindex(n, n):
        value = matrix[p, q]
        if value == 0:
            continue
        if p == q:
            total += value * (_site({}, n) - _site({p: "Z"}, n)) / 2
        else:
            # sigma+ = (X - iY) / 2 and sigma- = (X + iY) / 2
            raise_p = (_site({p: "X"}, n) - 1j * _site({p: "Y"}, n)) / 2
            lower_q = (_site({q: "X"}, n) + 1j * _site({q: "Y"}, n)) / 2
            total += value * (raise_p @ lower_q)
    return total


def to_pauli(matrix: np.ndarray, mapping: str) -> SparsePauliOp:
    """Pauli form of an operator on the oscillator states in a qubit mapping.

    A mapping is a relabeling of basis states: |n> becomes the qubit state
    encode(n). For binary and Gray every qubit state is some |n>, so moving
    the entries of the matrix to the new labels is the whole job.

    Args:
        matrix: N x N operator on the oscillator states.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        The operator as a sum of Pauli strings.
    """
    if mapping == "unary":
        return _unary_pauli(matrix).simplify(atol=1e-10)
    n_states = len(matrix)
    new = np.array([encode(n, mapping) for n in range(n_states)])
    size = 2 ** n_qubits(n_states, mapping)
    moved = np.zeros((size, size), dtype=matrix.dtype)
    moved[np.ix_(new, new)] = matrix
    return SparsePauliOp.from_operator(Operator(moved)).simplify(atol=1e-10)


def ref_bits(n_states: int, mapping: str) -> str:
    """Bitstring of the reference state |0>, in Qiskit order with qubit 0 last.

    Args:
        n_states: Number of oscillator states.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        One character per qubit.
    """
    width = n_qubits(n_states, mapping)
    return format(encode(0, mapping), f"0{width}b")


def build_hamiltonian(n_states: int, mapping: str) -> SparsePauliOp:
    """Qubit Hamiltonian of the deuteron in ``n_states`` oscillator states.

    Args:
        n_states: Number of oscillator states, a power of two.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        The Hamiltonian in MeV as a sum of Pauli strings.
    """
    check_states(n_states)
    return to_pauli(basis_hamiltonian(n_states), mapping)


def rotation_generators(n_states: int) -> list[np.ndarray]:
    """Anti-Hermitian generators G_k = |k+1><k| - |k><k+1|, k = 0 .. N-2.

    exp(theta G_k) rotates amplitude from |k> into |k+1> by the angle theta.

    Args:
        n_states: Number of oscillator states N.

    Returns:
        The N - 1 generators, each an N x N matrix.
    """
    gens = []
    for k in range(n_states - 1):
        G = np.zeros((n_states, n_states))
        G[k + 1, k], G[k, k + 1] = 1.0, -1.0
        gens.append(G)
    return gens


def exact_energies(n_states: int) -> dict[str, float]:
    """Reference energies for a basis of ``n_states`` states.

    Args:
        n_states: Number of oscillator states.

    Returns:
        Energies in MeV, keyed by

        exact:      lowest eigenvalue of the N x N matrix, the answer to find.
        ref:        energy of the state |0>, with no mixing of other states.
        converged:  the same problem in a very large basis, N = 256.
        experiment: the measured value.
    """
    H = basis_hamiltonian(n_states)
    return {
        "exact": float(np.linalg.eigvalsh(H)[0]),
        "ref": float(H[0, 0]),
        "converged": ground_energy(256),
        "experiment": EXPERIMENT,
    }
