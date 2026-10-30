"""Encoding: build the ansatz for the lattice protein folding example.

Follows Linn et al. (2024); see 03-Applications/lattice-folding-encodings.md.

A conformation (a fold) is one choice per bead out of c_i options. The list
of option counts (c_1, ..., c_N) is the "cardinality vector". Each choice is
written in a bit code, the codes of all beads are concatenated, and each bit
is one qubit. Three codes are provided:

    unary   one-hot, c bits            (2-local terms, most qubits)
    binary  base 2, ceil(log2 c) bits  (fewest qubits, highest locality)
    bu      block-unary binary with block size g, a middle ground
            (g = 1 is unary)

Two lattice models supply the cardinality vector and a decoder from bits to
coordinates. The main example uses the first one:

    coordinate  each bead picks a lattice site (a checkerboard coloring of
                the grid halves the number of choices per bead)
    turn        each bead picks a step (a "turn") from the previous bead

Bitstrings in this file are written in qubit order (qubit 0 first). Qiskit
count keys put qubit 0 last, so reverse them with ``bits_from_key`` before
decoding.
"""

from math import ceil, log2

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

ENCODINGS = ("unary", "binary", "bu")


# --- One integer <-> bits ----------------------------------------------------


def block_size(c: int, encoding: str, g: int = 3) -> int:
    """Number of bits that encode one choice among c options.

    ``g`` is the block size for the "bu" code and is ignored otherwise.
    """
    if c == 1:
        return 0  # a forced choice needs no qubit
    if encoding == "unary":
        return c
    if encoding == "binary":
        return ceil(log2(c))
    return ceil(c / g) * ceil(log2(g + 1))


def encode(r: int, c: int, encoding: str, g: int = 3) -> str:
    """Bit string for choice r in range(c). Matches Table I of the paper.

    Example for c = 5, r = 3:  unary 00010, binary 011, bu (g=3) 0100.
    """
    n = block_size(c, encoding, g)
    if n == 0:
        return ""
    if encoding == "unary":
        return "".join("1" if k == r else "0" for k in range(c))
    if encoding == "binary":
        return format(r, f"0{n}b")
    # BU: the options are split into blocks of g. Block r // g (counted
    # from the right) holds the value (r % g) + 1; every other block is 0.
    w = ceil(log2(g + 1))
    blocks = ["0" * w] * ceil(c / g)
    blocks[-1 - r // g] = format(r % g + 1, f"0{w}b")
    return "".join(blocks)


def decode(bits: str, c: int, encoding: str, g: int = 3) -> int | None:
    """Choice r for a bit string, or None if the code is invalid.

    Invalid codes are the ones that pick no option or more than one, or
    that name an option beyond c (for example 101 in binary when c = 5).
    """
    if c == 1:
        return 0
    if encoding == "unary":
        return bits.index("1") if bits.count("1") == 1 else None
    if encoding == "binary":
        r = int(bits, 2)
        return r if r < c else None
    w = ceil(log2(g + 1))
    blocks = [int(bits[k : k + w], 2) for k in range(0, len(bits), w)]
    active = [(b, v) for b, v in enumerate(reversed(blocks)) if v]
    if len(active) != 1 or active[0][1] > g:
        return None
    r = active[0][0] * g + active[0][1] - 1
    return r if r < c else None


# --- Cardinality vectors -----------------------------------------------------


def coordinate_cardinality(n_beads: int, grid: tuple[int, ...]) -> list[int]:
    """Number of choices per bead: ceil(S/2), floor(S/2), ... (Eqs. 2-3).

    S is the number of grid sites. Consecutive beads sit on opposite colors
    of the checkerboard, so each bead can use only the sites of its own
    color. Bead 0 gets the larger color (the one containing site (0, 0)).
    """
    sites = 1
    for length in grid:
        sites *= length
    return [
        ceil(sites / 2) if i % 2 == 0 else sites // 2 for i in range(n_beads)
    ]


def turn_cardinality(n_beads: int, dim: int) -> list[int]:
    """Choices for turns 1..N-2 (Eqs. 4-5 with the symmetry fixing of App. C).

    Turn 0 is fixed to +x (no qubits). Turn 1 is +x or +y (c = 2), because
    every other direction is a rotation of these or doubles back. The
    remaining turns pick any of the 2*dim directions.
    """
    return [2] + [2 * dim] * (n_beads - 3) if n_beads >= 3 else []


def n_qubits(cards: list[int], encoding: str, g: int = 3) -> int:
    """Total conformation qubits (Table II)."""
    return sum(block_size(c, encoding, g) for c in cards)


# --- Bitstring -> conformation -----------------------------------------------


def bits_from_key(key: str) -> str:
    """Convert a Qiskit counts key (qubit 0 rightmost) to qubit order."""
    return key.replace(" ", "")[::-1]


def split_choices(bits: str, cards: list[int], encoding: str, g: int = 3):
    """Split a bitstring into one block per bead and decode each block.

    Returns a list with the choice r (or None if invalid) for every bead.
    """
    choices, pos = [], 0
    for c in cards:
        n = block_size(c, encoding, g)
        choices.append(decode(bits[pos : pos + n], c, encoding, g))
        pos += n
    return choices


def coordinate_sites(grid: tuple[int, ...]) -> list[list[tuple[int, ...]]]:
    """Lattice sites split by color, in row-major order.

    Returns [even_sites, odd_sites], where a site's color is the parity of
    the sum of its coordinates.
    """
    sites = [()]
    for length in grid:
        sites = [s + (k,) for s in sites for k in range(length)]
    return [[s for s in sites if sum(s) % 2 == p] for p in (0, 1)]


def decode_coordinate(bits, n_beads, grid, encoding, g=3):
    """Bead coordinates from a bitstring, or None if a bead's code is invalid.

    Each bead's choice r indexes the list of sites of that bead's color.
    """
    cards = coordinate_cardinality(n_beads, grid)
    parity_sites = coordinate_sites(grid)
    choices = split_choices(bits, cards, encoding, g)
    if None in choices:
        return None
    return [parity_sites[i % 2][r] for i, r in enumerate(choices)]


def turn_directions(dim: int) -> list[tuple[int, ...]]:
    """+x, -x, +y, -y, (+z, -z) as unit vectors."""
    return [
        tuple(s if k == axis else 0 for k in range(dim))
        for axis in range(dim)
        for s in (1, -1)
    ]


def decode_turn(bits, n_beads, dim, encoding, g=3):
    """Bead coordinates from a turn bitstring, or None if a code is invalid.

    Beads sit at the running sum of turns. Turn 0 is +x. Turn 1 uses
    r=0 -> +y and r=1 -> +x (so the single binary bit q0 = 1 means +x).
    """
    cards = turn_cardinality(n_beads, dim)
    choices = split_choices(bits, cards, encoding, g)
    if None in choices:
        return None
    dirs = turn_directions(dim)
    x, y = dirs[0], dirs[2]
    steps = (
        [x]
        + ([[y, x][choices[0]]] if choices else [])
        + [dirs[r] for r in choices[1:]]
    )
    pos = [(0,) * dim]
    for step in steps:
        pos.append(tuple(a + b for a, b in zip(pos[-1], step)))
    return pos


def is_self_avoiding(coords) -> bool:
    """True if no two beads share a lattice site."""
    return len(set(coords)) == len(coords)


# --- Ansatz ------------------------------------------------------------------


def one_hot_blocks(cards: list[int]) -> list[range]:
    """Qubit ranges of each unary block."""
    blocks, pos = [], 0
    for c in cards:
        blocks.append(range(pos, pos + block_size(c, "unary")))
        pos += block_size(c, "unary")
    return blocks


def xy_mixer(cards: list[int]) -> SparsePauliOp:
    """Mixer that swaps a "1" between neighboring qubits of each unary block.

    The terms (XX + YY) / 2 conserve the number of 1s in a block, so a state
    that starts one-hot (a single 1 per block) stays one-hot.
    """
    n, terms = n_qubits(cards, "unary"), []
    for block in one_hot_blocks(cards):
        for a, b in zip(block, list(block)[1:] + list(block)[:1]):
            if a == b:
                continue
            for pauli in "XY":
                label = ["I"] * n
                label[a] = label[b] = pauli
                # Qiskit labels put qubit 0 on the right, so reverse.
                terms.append(("".join(label)[::-1], 0.5))
    return SparsePauliOp.from_list(terms)


def one_hot_start(cards: list[int]) -> QuantumCircuit:
    """Initial state with the first option set in every unary block."""
    qc = QuantumCircuit(n_qubits(cards, "unary"))
    for block in one_hot_blocks(cards):
        qc.x(block[0])
    return qc


def build_ansatz(
    cost: SparsePauliOp, cards: list[int], encoding: str, reps: int = 1
):
    """QAOA ansatz for the folding cost operator.

    binary / bu: start in |+>^n and use the standard X mixer. Every
    bitstring is reachable, so every rule must be a penalty in the cost.
    unary: start in a one-hot state and use the XY mixer, so the circuit
    itself never leaves the "one site per bead" states. (The cost in ops.py
    still penalizes them, which does no harm.)
    """
    if encoding == "unary":
        return QAOAAnsatz(
            cost,
            reps=reps,
            initial_state=one_hot_start(cards),
            mixer_operator=xy_mixer(cards),
        )
    return QAOAAnsatz(cost, reps=reps)


if __name__ == "__main__":
    # Table I check: c = 5
    for r in range(5):
        print(r, *(encode(r, 5, e) for e in ENCODINGS))

    # N = 4 on a 2x2 grid, binary: 0110 is a U-shaped fold
    print(decode_coordinate("0110", 4, (2, 2), "binary"))

    # N = 5 cubic turn model, binary: 7 qubits.
    # The bits 0 001 001 mean +y, -x, -x after the fixed first step +x.
    cards = turn_cardinality(5, 3)
    print(n_qubits(cards, "binary"), decode_turn("0001001", 5, 3, "binary"))
