"""Ops: build the Hamiltonian of a small nucleus as a qubit operator.

The deuteron example had one particle (the relative motion of a proton and a
neutron) in a basis of oscillator states, so the unknown was a vector with N
entries. A heavier nucleus has A nucleons, and they are identical fermions:
the wave function must change sign when two of them swap. The natural
language for that is **second quantization**. We list single-nucleon states
("orbitals") and ask which of them are occupied. Each orbital is one qubit,
|1> for occupied and |0> for empty.

The model (a schematic one, in the spirit of the pionless effective field
theory used for the deuteron):

- Orbitals: the first ``n_orb`` S-wave oscillator states |p> of the deuteron
  example, for each of four kinds of nucleon (proton or neutron, spin up or
  down). That is ``4 * n_orb`` orbitals, so ``4 * n_orb`` qubits.
- Kinetic energy: the same tridiagonal matrix t as the deuteron, for every
  nucleon.
- Interaction: a contact attraction -V0 between every pair of nucleons that
  are both in oscillator state |0>, the same strength and place as the
  deuteron's. All four kinds attract equally (the Wigner SU(4) limit of
  nuclear effective field theory), and the Coulomb force is left out.

A nucleus is a choice of the proton number Z and the neutron number N. The
Hamiltonian conserves the number of nucleons of each kind, so we look for the
lowest energy among states with the right numbers. Energies are in MeV.

Steps:

1. fermion to qubit mapping: how the creation and annihilation operators of
   the orbitals become Pauli strings. The mappings differ in which qubit
   stores what, so they have the same eigenvalues and different Pauli terms.
2. Hamiltonian: sum the one-body and two-body terms, plus a penalty that
   keeps the numbers of nucleons right for ansatzes that do not conserve them.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

HBAR_OMEGA = 7.0  # MeV, oscillator spacing
V0 = 5.68658111  # MeV, strength of the contact interaction
PENALTY = 30.0  # MeV, cost of each nucleon too many or too few of one kind

MAPPINGS = ("jordan_wigner", "parity")
KINDS = ("p_up", "p_down", "n_up", "n_down")

# (protons Z, neutrons N). Z and N need not be in this table.
NUCLEI = {
    "2n": (0, 2),
    "2H": (1, 1),
    "3H": (1, 2),
    "3He": (2, 1),
    "4He": (2, 2),
}


# ----------------------------------------------------------------- nucleus


def parse_nucleus(text: str) -> tuple[int, int]:
    """Proton and neutron numbers from a name or from "Z,N".

    Args:
        text: A key of ``NUCLEI`` such as "4He", or two numbers such as "2,2".

    Returns:
        (Z, N).

    Raises:
        ValueError: If the text is neither.
    """
    if text in NUCLEI:
        return NUCLEI[text]
    try:
        z, n = (int(v) for v in text.split(","))
    except ValueError:
        raise ValueError(f"unknown nucleus {text!r}; use a name from {list(NUCLEI)} or 'Z,N'") from None
    return z, n


def nucleus_name(z: int, n: int) -> str:
    """Short name of a nucleus, from ``NUCLEI`` if it is there.

    Args:
        z: Number of protons.
        n: Number of neutrons.

    Returns:
        The name, or "Z=..,N=..".
    """
    for name, zn in NUCLEI.items():
        if zn == (z, n):
            return name
    return f"Z={z},N={n}"


def n_modes(n_orb: int) -> int:
    """Number of orbitals, and of qubits, for ``n_orb`` oscillator states.

    Args:
        n_orb: Oscillator states per kind of nucleon.

    Returns:
        Four times ``n_orb``.
    """
    return 4 * n_orb


def mode(kind: int, p: int, n_orb: int) -> int:
    """Index of the orbital of oscillator state ``p`` for one kind of nucleon.

    Args:
        kind: Index into ``KINDS``.
        p: Oscillator state, 0 .. n_orb - 1.
        n_orb: Oscillator states per kind of nucleon.

    Returns:
        The orbital index, which is also the qubit it lives on.
    """
    return kind * n_orb + p


def reference_occupation(z: int, n: int, n_orb: int) -> list[int]:
    """Occupation of the lowest-energy filling: the reference state.

    Nucleons of each species fill oscillator states from |0> up, spin up and
    down alternately. In this schematic model two protons and two neutrons
    already fill |0>, so a nucleus with A > 4 puts nucleons in |1> and
    beyond, which a real nucleus would do with P-wave states.

    Args:
        z: Number of protons.
        n: Number of neutrons.
        n_orb: Oscillator states per kind of nucleon.

    Returns:
        A list of 0 and 1, one per orbital.

    Raises:
        ValueError: If the nucleons do not fit, or if there is no room left
            for any excitation.
    """
    occ = [0] * n_modes(n_orb)
    for count, first in ((z, 0), (n, 2)):
        for k in range(count):
            p = k // 2
            if p >= n_orb:
                raise ValueError(f"{count} nucleons need more than n_orb={n_orb} oscillator states")
            occ[mode(first + k % 2, p, n_orb)] = 1
    return occ


def kind_counts(z: int, n: int) -> list[int]:
    """Number of nucleons of each kind in the reference state.

    Args:
        z: Number of protons.
        n: Number of neutrons.

    Returns:
        Counts for ``KINDS``: spin up gets the extra nucleon when A is odd.
    """
    return [(z + 1) // 2, z // 2, (n + 1) // 2, n // 2]


# ----------------------------------------------------------------- mappings


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


def ladder(j: int, width: int, mapping: str) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Creation and annihilation operators of orbital ``j`` as Pauli sums.

    Fermion operators anticommute, a_i a_j = -a_j a_i, and qubit operators on
    different qubits commute. Each mapping repairs that with a string of Z
    gates that counts the occupied orbitals:

    jordan_wigner: qubit j stores the occupation of orbital j. The Z string
        runs over the orbitals below j, so it is as long as j.
    parity: qubit j stores the parity of the occupations of orbitals 0 .. j.
        Occupying orbital j flips the parity qubits j, j + 1, ..., which is
        an X string over the qubits above j. The sign needs only qubit j - 1.

    Args:
        j: Orbital index.
        width: Total number of orbitals.
        mapping: Fermion-to-qubit mapping, one of ``MAPPINGS``.

    Returns:
        (a_j^dagger, a_j).

    Raises:
        ValueError: If the mapping is unknown.
    """
    if mapping == "jordan_wigner":
        strings = {k: "Z" for k in range(j)}
        x, y = {**strings, j: "X"}, {**strings, j: "Y"}
    elif mapping == "parity":
        above = {k: "X" for k in range(j + 1, width)}
        below = {j - 1: "Z"} if j > 0 else {}
        x, y = {**above, **below, j: "X"}, {**above, j: "Y"}
    else:
        raise ValueError(f"unknown mapping {mapping!r}; choose from {MAPPINGS}")
    # |0><1| = (X + iY) / 2 removes a particle, |1><0| = (X - iY) / 2 adds one
    X, Y = _site(x, width), _site(y, width)
    return (X - 1j * Y) / 2, (X + 1j * Y) / 2


def mapped_bits(occupation: list[int], mapping: str) -> list[int]:
    """Qubit values that stand for an occupation pattern in a mapping.

    Args:
        occupation: Occupation of each orbital, 0 or 1.
        mapping: Fermion-to-qubit mapping, one of ``MAPPINGS``.

    Returns:
        One 0 or 1 per qubit. Jordan-Wigner copies the occupations; parity
        stores running sums of them, modulo 2.

    Raises:
        ValueError: If the mapping is unknown.
    """
    if mapping == "jordan_wigner":
        return list(occupation)
    if mapping == "parity":
        return [int(v) for v in np.cumsum(occupation) % 2]
    raise ValueError(f"unknown mapping {mapping!r}; choose from {MAPPINGS}")


# ------------------------------------------------------------- Hamiltonian


def one_body_matrix(n_orb: int, hbar_omega: float = HBAR_OMEGA) -> np.ndarray:
    """Kinetic energy t of one nucleon in ``n_orb`` S-wave oscillator states.

    t[p, p]   = (hw/2) (2p + 3/2)
    t[p, p+1] = -(hw/2) sqrt((p + 1)(p + 3/2))

    This is the kinetic part of the deuteron Hamiltonian, now for each nucleon.

    Args:
        n_orb: Number of oscillator states.
        hbar_omega: Oscillator spacing in MeV.

    Returns:
        The n_orb x n_orb matrix in MeV.
    """
    t = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        t[p, p] = 0.5 * hbar_omega * (2 * p + 1.5)
        if p + 1 < n_orb:
            t[p, p + 1] = t[p + 1, p] = -0.5 * hbar_omega * np.sqrt((p + 1) * (p + 1.5))
    return t


def build_hamiltonian(
    n_orb: int, mapping: str, z: int = 0, n: int = 0, penalty: float = 0.0, v0: float = V0
) -> SparsePauliOp:
    """Qubit Hamiltonian of the nucleons, with an optional number penalty.

    H = sum_kind sum_pq t[p, q] a+(kind,p) a(kind,q)
        - V0 sum_{kind < kind'} n(kind,0) n(kind',0)
        + penalty * sum_kind (N_kind - N_kind(Z, N))^2

    The penalty is zero on every state with the right numbers of nucleons, so
    it does not change the answer. It makes the other states expensive, for
    ansatzes that cannot be trusted to stay in the right sector.

    Args:
        n_orb: Oscillator states per kind of nucleon.
        mapping: Fermion-to-qubit mapping, one of ``MAPPINGS``.
        z: Number of protons, used only by the penalty.
        n: Number of neutrons, used only by the penalty.
        penalty: Penalty weight in MeV, 0 for none.
        v0: Strength of the contact interaction in MeV.

    Returns:
        The Hamiltonian in MeV as a sum of Pauli strings.
    """
    width = n_modes(n_orb)
    ops = [ladder(j, width, mapping) for j in range(width)]
    number = [ops[j][0] @ ops[j][1] for j in range(width)]
    t = one_body_matrix(n_orb)
    zero = SparsePauliOp("I" * width, 0)

    H = zero
    for kind in range(4):
        for p, q in np.ndindex(n_orb, n_orb):
            if t[p, q] != 0:
                H = H + t[p, q] * (ops[mode(kind, p, n_orb)][0] @ ops[mode(kind, q, n_orb)][1])
    for k1 in range(4):
        for k2 in range(k1 + 1, 4):
            H = H - v0 * (number[mode(k1, 0, n_orb)] @ number[mode(k2, 0, n_orb)])
    if penalty:
        identity = SparsePauliOp("I" * width)
        for kind, target in enumerate(kind_counts(z, n)):
            count = sum((number[mode(kind, p, n_orb)] for p in range(n_orb)), zero)
            off = count - target * identity
            H = H + penalty * (off @ off)
    return H.simplify(atol=1e-10)


# -------------------------------------------------------------- references


def sector_indices(n_orb: int, z: int, n: int) -> np.ndarray:
    """Basis states of the Jordan-Wigner register with the right nucleon numbers.

    Args:
        n_orb: Oscillator states per kind of nucleon.
        z: Number of protons.
        n: Number of neutrons.

    Returns:
        Indices of the computational basis states with N_kind nucleons of
        each kind, where bit j of the index is the occupation of orbital j.
    """
    index = np.arange(2 ** n_modes(n_orb))
    keep = np.ones_like(index, dtype=bool)
    for kind, target in enumerate(kind_counts(z, n)):
        block = (index >> (kind * n_orb)) & (2**n_orb - 1)
        count = np.zeros_like(index)
        for p in range(n_orb):
            count += (block >> p) & 1
        keep &= count == target
    return index[keep]


def exact_energies(n_orb: int, z: int, n: int) -> dict[str, float]:
    """Reference energies for a nucleus in a basis of ``n_orb`` states.

    Diagonalizes the Hamiltonian in the sector with the right numbers of
    nucleons of each kind, which has few states even when the register is big.

    Args:
        n_orb: Oscillator states per kind of nucleon.
        z: Number of protons.
        n: Number of neutrons.

    Returns:
        Energies in MeV, keyed by

        exact:     lowest eigenvalue in the sector, the answer to find.
        ref:       energy of the reference filling, with no mixing.
        free:      A times the lowest one-nucleon energy in this basis: the
                   energy of the same nucleons not bound to each other.
        dimension: number of states in the sector.
    """
    H = build_hamiltonian(n_orb, "jordan_wigner").to_matrix(sparse=True).tocsr()
    sector = sector_indices(n_orb, z, n)
    block = H[sector][:, sector].toarray()
    ref = reference_occupation(z, n, n_orb)
    ref_index = sum(bit << j for j, bit in enumerate(ref))
    return {
        "exact": float(np.linalg.eigvalsh(block)[0]),
        "ref": float(H[ref_index, ref_index].real),
        "free": float((z + n) * np.linalg.eigvalsh(one_body_matrix(n_orb))[0]),
        "dimension": len(sector),
    }
