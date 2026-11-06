"""Ops: the Clifford group and the decay model of randomized benchmarking.

The problem: how good is a gate? We could apply it once and look at the
output, but one application changes the state by an amount too small to see
against the errors in preparing and reading the qubit. We could repeat it
many times, but then a small systematic error (a rotation slightly too
long) adds up in a way that does not represent typical use.

Randomized benchmarking (RB) avoids both problems:

    1. Draw m gates at random from a set of 24 single-qubit gates, the
       Cliffords.
    2. Append the one Clifford that undoes all of them. The ideal circuit
       is then the identity, so an ideal qubit started in |0> ends in |0>.
    3. Run, and record the probability of reading 0, the survival
       probability.
    4. Repeat for many random sequences and for several lengths m.

Every gate adds a little error, so the survival probability falls as m
grows. The random choice makes the average error look like a simple
depolarizing process: with each Clifford, a fraction (1 - p) of the signal is
lost to a fair coin flip. After m Cliffords the survival probability is

    F(m) = A p^m + B

A and B absorb the errors of preparation and read-out, which do not grow
with m, so the decay rate p is blind to them. From p we get the error per
Clifford for one qubit (dimension d = 2):

    r = (1 - p) (d - 1) / d = (1 - p) / 2

and the average gate fidelity of a Clifford is 1 - r.

The 24 Cliffords are the gates that map each of the six states |0>, |1>,
|+>, |->, |+i>, |-i> to another of the six. We build them from two gates, H
and S, by repeatedly applying either one until nothing new appears.
"""

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

LENGTHS = (1, 5, 10, 20, 40, 80, 120, 160, 240, 320)  # Cliffords per sequence

# The simulated qubit: a depolarizing error on every H and S gate, and a
# read-out error that flips 0 to 1 with probability SIM_READOUT[0] and 1 to 0
# with probability SIM_READOUT[1].
SIM_GATE_ERROR = 0.004
SIM_READOUT = (0.02, 0.04)


def _gate(name: str) -> Clifford:
    """The Clifford of a single H or S gate."""
    qc = QuantumCircuit(1)
    getattr(qc, name)(0)
    return Clifford(qc)


@lru_cache(maxsize=None)
def clifford_table() -> tuple:
    """All 24 single-qubit Cliffords, each with a gate sequence that makes it.

    Starting from the identity, apply H and S to every gate found so far and
    keep each result that is new. The search ends after 24 gates.

    Returns:
        Tuple of (Clifford, gate names), the first entry being the identity
        with an empty sequence. Each sequence is the shortest one found.
    """
    table = [(Clifford(QuantumCircuit(1)), ())]
    i = 0
    while i < len(table):
        cliff, gates = table[i]
        for name in ("h", "s"):
            new = cliff.compose(_gate(name))  # apply cliff, then the gate
            if not any(new == known for known, _ in table):
                table.append((new, gates + (name,)))
        i += 1
    return tuple(table)


def random_sequence(m: int, rng: np.random.Generator) -> list[int]:
    """Random sequence of m Cliffords followed by the one that undoes them.

    Args:
        m: Number of random Cliffords.
        rng: Random number generator.

    Returns:
        List of m + 1 indices into ``clifford_table()``.
    """
    table = clifford_table()
    picks = [int(i) for i in rng.integers(len(table), size=m)]
    total = table[0][0]  # start from the identity
    for i in picks:
        total = total.compose(table[i][0])
    inverse = total.adjoint()
    last = next(i for i, (c, _) in enumerate(table) if c == inverse)
    return picks + [last]


def decay(m, amp, p, offset):
    """RB model: F(m) = A p^m + B.

    Args:
        m: Sequence length (number of Cliffords), scalar or array.
        amp: A, the size of the decaying part.
        p: Decay rate per Clifford, between 0 and 1.
        offset: B, the value the curve approaches for long sequences.

    Returns:
        The survival probability.
    """
    return amp * p ** np.asarray(m, dtype=float) + offset


def initial_guess(m: np.ndarray, y: np.ndarray) -> list[float]:
    """Starting values for the fit, read off the data.

    Args:
        m: Sequence lengths.
        y: Mean survival at each length.

    Returns:
        [A, p, B]. B is the last value, A the swing from it, and p the rate
        that would bring the first point down to the last over the full range.
    """
    offset = float(y[-1])
    amp = float(y[0] - offset)
    p = float(np.clip(1 - 1 / m[-1], 0.5, 0.9999))
    return [amp, p, offset]


def error_per_clifford(p: float) -> float:
    """r = (1 - p) / 2 for one qubit."""
    return (1 - p) / 2


def simulator_truth(gate_error: float = SIM_GATE_ERROR) -> dict:
    """The decay rate that the simulator's noise gives, from first principles.

    A depolarizing error of size e on a gate multiplies the signal by
    (1 - e). A Clifford made of k gates multiplies it by (1 - e)^k, and a
    random Clifford is any of the 24 with equal probability, so

        p = average over the 24 of (1 - e)^k.

    Args:
        gate_error: Depolarizing parameter of each H and S gate.

    Returns:
        Dict with ``p``, ``epc`` (error per Clifford), and ``gates`` (average
        number of gates per Clifford).
    """
    counts = np.array([len(g) for _, g in clifford_table()])
    p = float(np.mean((1 - gate_error) ** counts))
    return {"p": p, "epc": error_per_clifford(p), "gates": float(counts.mean())}
