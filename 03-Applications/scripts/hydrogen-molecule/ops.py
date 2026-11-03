"""Ops: build the H2 Hamiltonian in several qubit forms.

The problem: find the lowest energy of two electrons shared by two hydrogen
nuclei a distance R apart. We use the minimal (STO-3G) basis: each atom brings
one 1s orbital, and the two combine into a bonding orbital (sigma_g, low
energy) and an antibonding orbital (sigma_u, high energy). Each orbital holds
a spin-up and a spin-down electron, so there are 4 spin orbitals.

Steps:

1. integrals: the one- and two-electron integrals over the STO-3G Gaussians,
   computed here with a few lines of numpy (no chemistry package needed).
2. second quantization: H = sum h_pq a+_p a_q + 1/2 sum (pq|rs) a+_p a+_r a_s
   a_q + E_nuc, as a 16 x 16 matrix on the Fock space of 4 spin orbitals.
3. qubit mapping: the same operator written for qubits. The mappings differ
   only in how the occupation numbers n_0..n_3 are stored in the qubit bits,
   so they all have the same eigenvalues. They differ in the Pauli terms.

Spin orbitals are ordered [sigma_g up, sigma_u up, sigma_g down, sigma_u down].
Energies are in hartree, distances in angstrom.
"""

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.special import erf

BOHR = 0.529177210903  # angstrom per bohr

# STO-3G for hydrogen: three Gaussians fit to a 1s Slater orbital (zeta =
# 1.24). Each primitive is exp(-alpha r^2) with weight c.
ALPHA = np.array([3.42525091, 0.62391373, 0.16885540])
COEF = np.array([0.15432897, 0.53532814, 0.44463454])
NORM = (2 * ALPHA / np.pi) ** 0.75

MAPPINGS = ("jw", "parity", "bk", "parity2")
N_SPIN_ORBITALS = 4
HF_OCCUPATION = (1, 0, 1, 0)  # sigma_g up and sigma_g down are filled


# ---------------------------------------------------------------- integrals


def _boys0(t: np.ndarray) -> np.ndarray:
    """Boys function F0(t) = integral of exp(-t u^2) from 0 to 1, with F0(0) = 1.

    Args:
        t: Argument, a number or an array.

    Returns:
        F0(t), with the shape of ``t``.
    """
    t = np.asarray(t, dtype=float)
    safe = np.where(t < 1e-12, 1.0, t)
    return np.where(t < 1e-12, 1.0, 0.5 * np.sqrt(np.pi / safe) * erf(np.sqrt(safe)))


def _centers(distance: float) -> np.ndarray:
    """Positions of the two nuclei on the z axis.

    Args:
        distance: Bond length in angstrom.

    Returns:
        A 2 x 3 array of positions in bohr.
    """
    half = 0.5 * distance / BOHR
    return np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])


def atomic_integrals(distance: float):
    """Overlap, one-electron and two-electron integrals in the 1s basis.

    The basis is the two atomic orbitals phi_1, phi_2. Every integral is a sum
    over products of primitive Gaussians, each with a closed form for s-type
    functions (Szabo and Ostlund, appendix A). ``eri`` uses chemists'
    notation: (ab|cd) is the repulsion between the charge clouds
    phi_a phi_b and phi_c phi_d.

    Args:
        distance: Bond length in angstrom.

    Returns:
        (S, h, eri): the 2 x 2 overlap, the 2 x 2 one-electron integrals, and
        the 2 x 2 x 2 x 2 two-electron integrals, in hartree.
    """
    centers = _centers(distance)
    weight = COEF * NORM  # coefficient times normalization of each primitive
    n = 2
    S, T, V = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    eri = np.zeros((n, n, n, n))

    def gaussian_product(a, b, ca, cb):
        """Gaussian product theorem for two s-type primitives.

        exp(-a|r-ca|^2) exp(-b|r-cb|^2) = K exp(-p|r-P|^2). Returns (p, mu, P, K)
        with mu = a b / p.
        """
        p = a + b
        mu = a * b / p
        P = (a * ca + b * cb) / p
        K = np.exp(-mu * np.sum((ca - cb) ** 2))
        return p, mu, P, K

    for i in range(n):
        for j in range(n):
            for ka, a in enumerate(ALPHA):
                for kb, b in enumerate(ALPHA):
                    w = weight[ka] * weight[kb]
                    p, mu, P, K = gaussian_product(a, b, centers[i], centers[j])
                    r2 = np.sum((centers[i] - centers[j]) ** 2)
                    overlap = K * (np.pi / p) ** 1.5
                    S[i, j] += w * overlap
                    T[i, j] += w * mu * (3 - 2 * mu * r2) * overlap
                    for c in centers:  # each nucleus has charge +1
                        V[i, j] -= (
                            w * 2 * np.pi / p * K * _boys0(p * np.sum((P - c) ** 2))
                        )

    for i, j, k, l in np.ndindex(n, n, n, n):
        for ka, a in enumerate(ALPHA):
            for kb, b in enumerate(ALPHA):
                p, _, P, K1 = gaussian_product(a, b, centers[i], centers[j])
                for kc, c in enumerate(ALPHA):
                    for kd, d in enumerate(ALPHA):
                        q, _, Q, K2 = gaussian_product(c, d, centers[k], centers[l])
                        w = weight[ka] * weight[kb] * weight[kc] * weight[kd]
                        pref = 2 * np.pi**2.5 / (p * q * np.sqrt(p + q))
                        t = p * q / (p + q) * np.sum((P - Q) ** 2)
                        eri[i, j, k, l] += w * pref * K1 * K2 * _boys0(t)
    return S, T + V, eri


def molecular_integrals(distance: float):
    """Integrals over the bonding and antibonding molecular orbitals.

    By symmetry the orbitals are fixed without any iteration:
    sigma_g = (phi_1 + phi_2) / sqrt(2 + 2 S12) and
    sigma_u = (phi_1 - phi_2) / sqrt(2 - 2 S12).

    Args:
        distance: Bond length in angstrom.

    Returns:
        (h, eri, e_nuc): h[p, q], eri[p, q, r, s] = (pq|rs) over the orbitals
        sigma_g and sigma_u, and the nuclear repulsion 1/R, in hartree.
    """
    S, h_ao, eri_ao = atomic_integrals(distance)
    C = np.array([[1, 1], [1, -1]]) / np.sqrt(2 + 2 * np.array([S[0, 1], -S[0, 1]]))
    h = C.T @ h_ao @ C
    eri = np.einsum("ia,jb,kc,ld,ijkl->abcd", C, C, C, C, eri_ao)
    return h, eri, BOHR / distance  # nuclear repulsion 1/R, R in bohr


# ------------------------------------------------- second quantization


def _annihilation(j: int) -> np.ndarray:
    """Annihilation operator a_j on the 16-dimensional Fock space.

    The Jordan-Wigner signs make the a_j anticommute.

    Args:
        j: Spin-orbital index, 0 to 3.

    Returns:
        A 16 x 16 matrix.
    """
    n = N_SPIN_ORBITALS
    label = ["I"] * n
    label[j] = "X"
    x = SparsePauliOp("".join(reversed(label)))
    label[j] = "Y"
    y = SparsePauliOp("".join(reversed(label)))
    for k in range(j):  # Z string on the lower qubits keeps a_i, a_j anticommuting
        z = ["I"] * n
        z[k] = "Z"
        zk = SparsePauliOp("".join(reversed(z)))
        x, y = zk @ x, zk @ y
    return ((x + 1j * y) / 2).to_matrix()


def orbital(so: int) -> int:
    """Spatial orbital of a spin orbital.

    Args:
        so: Spin-orbital index, 0 to 3.

    Returns:
        0 for sigma_g, 1 for sigma_u.
    """
    return so % 2


def spin(so: int) -> int:
    """Spin of a spin orbital.

    Args:
        so: Spin-orbital index, 0 to 3.

    Returns:
        0 for spin up, 1 for spin down.
    """
    return so // 2


def fock_hamiltonian(distance: float) -> np.ndarray:
    """Molecular Hamiltonian on the 16-dimensional Fock space.

    Args:
        distance: Bond length in angstrom.

    Returns:
        A 16 x 16 complex matrix in hartree.
    """
    h, eri, e_nuc = molecular_integrals(distance)
    a = [_annihilation(j) for j in range(N_SPIN_ORBITALS)]
    ad = [x.conj().T for x in a]
    H = e_nuc * np.eye(16, dtype=complex)
    for P in range(4):
        for Q in range(4):
            if spin(P) == spin(Q):
                H += h[orbital(P), orbital(Q)] * ad[P] @ a[Q]
    for P, Q, R, S_ in np.ndindex(4, 4, 4, 4):
        if spin(P) == spin(Q) and spin(R) == spin(S_):
            g = eri[orbital(P), orbital(Q), orbital(R), orbital(S_)]
            H += 0.5 * g * ad[P] @ ad[R] @ a[S_] @ a[Q]
    return H


def excitation_generators() -> dict[str, np.ndarray]:
    """Anti-Hermitian generators G = T - T^dagger of the UCCSD excitations.

    exp(theta G) moves amplitude from the Hartree-Fock determinant to the
    excited one. Singles move one electron g -> u, the double moves both.

    Returns:
        16 x 16 matrices keyed "single_up", "single_down" and "double".
    """
    a = [_annihilation(j) for j in range(N_SPIN_ORBITALS)]
    ad = [x.conj().T for x in a]
    single_up = ad[1] @ a[0]
    single_down = ad[3] @ a[2]
    double = ad[1] @ ad[3] @ a[2] @ a[0]
    return {
        "single_up": single_up - single_up.conj().T,
        "single_down": single_down - single_down.conj().T,
        "double": double - double.conj().T,
    }


# ------------------------------------------------------------- mappings


def _bits(i: int) -> list[int]:
    """Occupation numbers of a Fock basis state.

    Args:
        i: Basis-state index, 0 to 15.

    Returns:
        Four occupation numbers, spin orbital 0 first.
    """
    return [(i >> k) & 1 for k in range(N_SPIN_ORBITALS)]


def encode(occupation, mapping: str) -> list[int]:
    """Qubit bits that store the occupations n_0..n_3.

    jw:     qubit k holds n_k.
    parity: qubit k holds n_0 + ... + n_k (mod 2).
    bk:     Bravyi-Kitaev, qubit k holds the sum of a block of occupations
            ending at k whose length is the lowest set bit of k + 1.

    Args:
        occupation: Occupation numbers n_0..n_3.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        Four bits, qubit 0 first.

    Raises:
        ValueError: If the mapping is unknown.
    """
    n = list(occupation)
    if mapping == "jw":
        return n
    if mapping in ("parity", "parity2"):
        return [sum(n[: k + 1]) % 2 for k in range(len(n))]
    if mapping == "bk":
        out = []
        for k in range(len(n)):
            size = (k + 1) & -(k + 1)
            out.append(sum(n[k + 1 - size : k + 1]) % 2)
        return out
    raise ValueError(f"unknown mapping {mapping!r}; choose from {MAPPINGS}")


def _index(bits) -> int:
    """Basis-state index of a list of bits.

    Args:
        bits: Bits with bit 0 first, the least significant.

    Returns:
        The index.
    """
    return sum(b << k for k, b in enumerate(bits))


def to_mapping(matrix: np.ndarray, mapping: str) -> np.ndarray:
    """Re-express a Fock-space operator in the basis of a qubit mapping.

    A mapping is a relabeling of basis states: the state with occupations n
    becomes the qubit basis state encode(n). Moving the entries of the matrix
    to the new labels gives the operator for that mapping.

    ``parity2`` also removes two qubits. In the parity mapping qubit 1 stores
    the parity of the spin-up electrons and qubit 3 the parity of all of them.
    Both are conserved, and for H2, with one up and one down electron, they
    are always 1 and 0. We keep only that block, which leaves qubits 0 and 2.

    Args:
        matrix: 16 x 16 operator on the Fock space.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        The matrix in the mapping's basis, 4 x 4 for ``parity2``.
    """
    new = np.array([_index(encode(_bits(i), mapping)) for i in range(16)])
    out = np.zeros_like(matrix)
    out[np.ix_(new, new)] = matrix
    if mapping == "parity2":
        keep = [q0 + 2 + 4 * q2 for q2 in (0, 1) for q0 in (0, 1)]
        out = out[np.ix_(keep, keep)]
    return out


def to_pauli(matrix: np.ndarray, mapping: str) -> SparsePauliOp:
    """Pauli decomposition of a Fock-space operator in a qubit mapping.

    Args:
        matrix: 16 x 16 operator on the Fock space.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        The operator as a sum of Pauli strings.
    """
    op = SparsePauliOp.from_operator(Operator(to_mapping(matrix, mapping)))
    return op.simplify(atol=1e-10)


def hf_bits(mapping: str) -> str:
    """Bitstring of the Hartree-Fock state, in Qiskit order with qubit 0 last.

    Args:
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        One character per qubit.
    """
    bits = encode(HF_OCCUPATION, mapping)
    if mapping == "parity2":
        bits = [bits[0], bits[2]]
    return "".join(str(b) for b in reversed(bits))


def n_qubits(mapping: str) -> int:
    """Number of qubits a mapping needs.

    Args:
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        2 for ``parity2``, otherwise 4.
    """
    return 2 if mapping == "parity2" else N_SPIN_ORBITALS


def build_hamiltonian(distance: float, mapping: str) -> SparsePauliOp:
    """Qubit Hamiltonian of H2.

    Args:
        distance: Bond length in angstrom.
        mapping: Qubit mapping, one of ``MAPPINGS``.

    Returns:
        The Hamiltonian in hartree as a sum of Pauli strings.
    """
    return to_pauli(fock_hamiltonian(distance), mapping)


def exact_energies(distance: float) -> dict[str, float]:
    """Reference energies from the 16 x 16 matrix.

    Args:
        distance: Bond length in angstrom.

    Returns:
        Energies in hartree, keyed by

        fci:      lowest energy with one up and one down electron, the answer.
        hf:       energy of the Hartree-Fock determinant.
        fock_min: lowest energy over every electron number, which is what an
                  ansatz that ignores particle number can fall into.
    """
    H = fock_hamiltonian(distance)
    reduced = to_mapping(H, "parity2").real
    hf = _index(HF_OCCUPATION)
    return {
        "fci": float(np.linalg.eigvalsh(reduced)[0]),
        "hf": float(H[hf, hf].real),
        "fock_min": float(np.linalg.eigvalsh(H)[0]),
    }
