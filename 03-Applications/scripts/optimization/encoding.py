"""Encoding: build the ansatz for the traveling salesman example.

Bits: one bit, and so one qubit, for each pair (city c, step t). The bit
x[c, t] is 1 when city c is visited at step t. City 0 is fixed at step 0, so
n cities need (n - 1)^2 qubits. Written as a table with one row per city and
one column per step, a bitstring is a valid tour only when every row and
every column holds exactly one 1 (a "permutation matrix"). Every other
bitstring is discouraged by the penalty terms in ops.py.

Ansatz: QAOA. Start in |+>^n (an equal superposition of every bitstring) and
alternate the cost layer exp(-i gamma H) with the mixer exp(-i beta sum X).
"""

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


def n_qubits(n_cities: int) -> int:
    """Number of qubits for a tour: an (n - 1) x (n - 1) table of bits.

    Args:
        n_cities: Number of cities n.

    Returns:
        (n - 1)^2.
    """
    return (n_cities - 1) ** 2


def bits_from_key(key: str) -> str:
    """Convert a Qiskit counts key, with qubit 0 rightmost, to qubit order.

    Args:
        key: Bitstring from ``get_counts``.

    Returns:
        The bitstring with qubit 0 first.
    """
    return key.replace(" ", "")[::-1]


def decode_tour(bits: str, n_cities: int) -> list[int] | None:
    """Tour from a bitstring.

    Qubit (c - 1) * (n - 1) + (t - 1) holds x[c, t], so the table is stored
    row by row, one row per city.

    Args:
        bits: Bitstring in qubit order.
        n_cities: Number of cities n.

    Returns:
        The tour [0, c1, c2, ...], or None if the bits are not a tour.
    """
    m = n_cities - 1
    x = [
        [int(bits[(c - 1) * m + (t - 1)]) for t in range(1, n_cities)]
        for c in range(1, n_cities)
    ]
    rows_ok = all(sum(row) == 1 for row in x)
    cols_ok = all(sum(col) == 1 for col in zip(*x))
    if not (rows_ok and cols_ok):
        return None
    order = [0] * m
    for c, row in enumerate(x, start=1):
        order[row.index(1)] = c
    return [0, *order]


def tour_length(tour: list[int], dist) -> float:
    """Total cost of the closed tour, including the leg back to the start.

    Args:
        tour: Cities in visiting order.
        dist: Cost matrix, dist[a, b] = cost of going from city a to city b.

    Returns:
        The tour cost.
    """
    return float(sum(dist[a, b] for a, b in zip(tour, tour[1:] + tour[:1])))


def build_ansatz(cost: SparsePauliOp, reps: int = 2) -> QAOAAnsatz:
    """QAOA ansatz with one beta and one gamma per layer.

    Args:
        cost: Cost Hamiltonian H_C.
        reps: Number of QAOA layers p.

    Returns:
        The QAOA circuit with 2 * reps symbolic angles.
    """
    return QAOAAnsatz(cost, reps=reps)
