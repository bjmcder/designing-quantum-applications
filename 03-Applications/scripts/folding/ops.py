"""Ops: build the Hamiltonian for the lattice protein folding example.

Model: the coordinate-based HP model on a small square grid (see encoding.py).
A sequence of H (hydrophobic) and P (polar) beads folds onto the grid. Every
pair of H beads that touch on the grid without being neighbors in the chain
lowers the energy by eps, so the best fold packs the H beads together.

The cost is a polynomial in the bits. Let y[i, r] = 1 when bead i takes
choice r, that is, when it sits on the r-th site of its color. For unary
codes y is a single bit. For binary and BU-binary codes y is a product of
bits (or their complements) that equals 1 only when the bead's whole block
spells the code for r. The cost is a sum of four kinds of term:

    contacts       -eps * y[i,r] * y[j,q]
                   H beads i and j with j - i odd and at least 3, on
                   adjacent sites r and q

    no overlap     +lam * y[i,r] * y[j,r]
                   two beads of the same color on the same site

    connectivity   +lam * y[i,r] * y[i+1,q]
                   consecutive beads on sites that are NOT adjacent

    valid code     +lam * (a bead's block is not a legal code)
                   binary and BU: add lam for every unused bit pattern
                   unary: add lam * (sum_r y[i,r] - 1)^2

The penalty weight lam is larger than any possible contact gain, so breaking
a rule never pays. Substituting x = (1 - Z) / 2 for every bit turns the
polynomial into an Ising Hamiltonian (a sum of Pauli-Z strings). No
auxiliary qubits are needed.
"""

from itertools import combinations, product

from encoding import (
    block_size,
    coordinate_cardinality,
    coordinate_sites,
    decode,
    encode,
    is_self_avoiding,
)
from qiskit.quantum_info import SparsePauliOp

SEQUENCE = "HPPHPH"
GRID = (2, 3)
EPS = 1.0  # energy gained per H-H contact
LAM = 3.0  # penalty weight, larger than any possible contact gain

# A polynomial in the bits, stored as {monomial: coefficient}. A monomial is a
# frozenset of bit indices, so frozenset({0, 2}) means x_0 * x_2 and the empty
# frozenset is the constant 1. Because x * x = x for a bit, a repeated bit
# collapses, and a set is all we need.
Poly = dict[frozenset, float]


# --- Polynomial helpers ------------------------------------------------------


def padd(*polys: Poly) -> Poly:
    """Sum of polynomials."""
    out: Poly = {}
    for p in polys:
        for mono, c in p.items():
            out[mono] = out.get(mono, 0.0) + c
    return out


def pmul(a: Poly, b: Poly) -> Poly:
    """Product of two polynomials (using x * x = x for each bit)."""
    out: Poly = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            out[ma | mb] = out.get(ma | mb, 0.0) + ca * cb
    return out


def pscale(p: Poly, s: float) -> Poly:
    """Polynomial times a number."""
    return {m: c * s for m, c in p.items()}


def evaluate(p: Poly, bits: str) -> float:
    """Classical value of the polynomial for a bitstring in qubit order."""
    return sum(c for m, c in p.items() if all(bits[k] == "1" for k in m))


def matches(offset: int, code: str) -> Poly:
    """Polynomial that is 1 when the bits starting at ``offset`` spell ``code``.

    Each position contributes x_k if the code has a 1 there, or (1 - x_k) if
    it has a 0. The product is 1 only when every bit agrees.
    """
    poly: Poly = {frozenset(): 1.0}
    for k, bit in enumerate(code):
        x = frozenset({offset + k})
        factor = {x: 1.0} if bit == "1" else {frozenset(): 1.0, x: -1.0}
        poly = pmul(poly, factor)
    return poly


def indicator(offset: int, r: int, c: int, encoding: str, g: int) -> Poly:
    """y = 1 exactly when the block starting at ``offset`` encodes choice r."""
    code = encode(r, c, encoding, g)
    if encoding == "unary":
        # One-hot: choice r is "bit r is set", whatever the other bits are.
        # The one-hot penalty below takes care of the other bits.
        return {frozenset({offset + code.index("1")}): 1.0}
    return matches(offset, code)


# --- Cost polynomial ---------------------------------------------------------


def adjacent(s: tuple, t: tuple) -> bool:
    """True if two lattice sites are nearest neighbors."""
    return sum(abs(a - b) for a, b in zip(s, t)) == 1


def fold_poly(sequence, grid, encoding, g=3, eps=EPS, lam=LAM) -> Poly:
    """The full cost of a bitstring as a polynomial in the bits."""
    n = len(sequence)
    cards = coordinate_cardinality(n, grid)
    sites = coordinate_sites(grid)
    # First qubit of each bead's block of bits.
    offsets = [
        sum(block_size(c, encoding, g) for c in cards[:i]) for i in range(n)
    ]
    # y[i][r] = "bead i sits on the r-th site of its color".
    y = [
        [indicator(offsets[i], r, c, encoding, g) for r in range(c)]
        for i, c in enumerate(cards)
    ]
    cost: list[Poly] = []

    for i, j in combinations(range(n), 2):
        # Contact reward: both H, not chain neighbors. Beads of opposite
        # color are an odd distance apart, and 1 apart means chain neighbors.
        both_h = sequence[i] == sequence[j] == "H"
        if both_h and (j - i) % 2 == 1 and j - i >= 3:
            for r, q in product(range(cards[i]), range(cards[j])):
                if adjacent(sites[i % 2][r], sites[j % 2][q]):
                    cost.append(pscale(pmul(y[i][r], y[j][q]), -eps))
        # No overlap: only beads of the same color can share a site, and
        # the r-th site of a color is the same site for both beads.
        if (j - i) % 2 == 0:
            for r in range(min(cards[i], cards[j])):
                cost.append(pscale(pmul(y[i][r], y[j][r]), lam))

    # Connectivity: consecutive beads must be on adjacent sites.
    for i in range(n - 1):
        for r, q in product(range(cards[i]), range(cards[i + 1])):
            if not adjacent(sites[i % 2][r], sites[(i + 1) % 2][q]):
                cost.append(pscale(pmul(y[i][r], y[i + 1][q]), lam))

    # Valid code: each bead's block must be a legal code.
    for i, c in enumerate(cards):
        if encoding == "unary":
            # (sum_r y_r - 1)^2 is 0 only when exactly one bit is set.
            total = padd(*y[i], {frozenset(): -1.0})
            cost.append(pscale(pmul(total, total), lam))
        else:
            # Penalize each bit pattern that decodes to nothing.
            n_bits = block_size(c, encoding, g)
            for bits in map("".join, product("01", repeat=n_bits)):
                if decode(bits, c, encoding, g) is None:
                    cost.append(pscale(matches(offsets[i], bits), lam))
    return padd(*cost)


def to_ising(poly: Poly, n_qubits: int) -> SparsePauliOp:
    """Substitute x = (1 - Z) / 2 into every monomial.

    A product of m bits expands into 2^m terms, one for each subset of the
    m bits (the Pauli-Z string on that subset). The expansion of
    x_a * x_b, for example, is (1 - Z_a - Z_b + Z_a Z_b) / 4.
    """
    terms: dict[frozenset, float] = {}
    for mono, c in poly.items():
        members = sorted(mono)
        scale = c / 2 ** len(members)
        for size in range(len(members) + 1):
            for subset in combinations(members, size):
                key = frozenset(subset)
                terms[key] = terms.get(key, 0.0) + scale * (-1) ** size
    labels = []
    for key, c in terms.items():
        if abs(c) > 1e-12:
            label = ["I"] * n_qubits
            for k in key:
                label[k] = "Z"
            # Qiskit labels put qubit 0 on the right, so reverse.
            labels.append(("".join(reversed(label)), c))
    return SparsePauliOp.from_list(labels).simplify()


def build_hamiltonian(encoding="binary", sequence=SEQUENCE, grid=GRID, g=3):
    """Return (H, cards, poly): the Hamiltonian, the cardinality vector,
    and the polynomial it came from (handy for checking H)."""
    cards = coordinate_cardinality(len(sequence), grid)
    n = sum(block_size(c, encoding, g) for c in cards)
    poly = fold_poly(sequence, grid, encoding, g)
    return to_ising(poly, n), cards, poly


# --- Classical references ----------------------------------------------------
# These work on decoded coordinates, not on qubits. They let us check the
# Hamiltonian and score the samples.


def is_valid_fold(coords) -> bool:
    """Every bead placed, no shared sites, chain neighbors adjacent."""
    return (
        coords is not None
        and is_self_avoiding(coords)
        and all(adjacent(a, b) for a, b in zip(coords, coords[1:]))
    )


def fold_energy(coords, sequence=SEQUENCE, eps=EPS) -> float:
    """HP energy: -eps for each H-H contact between non-chain-neighbors."""
    contacts = sum(
        1
        for i, j in combinations(range(len(sequence)), 2)
        if j - i >= 3
        and sequence[i] == sequence[j] == "H"
        and adjacent(coords[i], coords[j])
    )
    return -eps * contacts


def enumerate_folds(sequence=SEQUENCE, grid=GRID):
    """Every self-avoiding placement of the chain, by depth-first search.

    Yields lists of coordinates. Rotated and mirrored copies are yielded
    separately. This is the exact answer to compare the quantum samples
    against, and it avoids looping over all 2^n bitstrings.
    """
    sites = list(product(*(range(n) for n in grid)))

    def extend(path):
        if len(path) == len(sequence):
            yield list(path)
            return
        for site in sites:
            if site not in path and adjacent(path[-1], site):
                yield from extend([*path, site])

    for start in sites:
        yield from extend([start])


def auto_grid(n_beads: int) -> tuple[int, int]:
    """Smallest near-square grid with at least n_beads sites."""
    candidates = [
        (a, b)
        for a in range(1, n_beads + 1)
        for b in range(a, n_beads + 1)
        if a * b >= n_beads
    ]
    return min(candidates, key=lambda g: (g[0] * g[1], g[1] - g[0]))
