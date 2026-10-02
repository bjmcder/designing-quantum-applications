#!/usr/bin/env python3
"""Grover's algorithm: search an unsorted list with fewer queries.

The problem
-----------
There are N = 2^n possible items, and a black box (the "oracle") that can
tell us whether a given item is one of the M "marked" ones. Find a marked
item. With no structure to exploit, a classical search checks about N/2 items
on average. Grover's algorithm needs only about (pi/4) * sqrt(N/M) oracle
calls, a quadratic speedup.

How it works
------------
Start in an equal superposition of all N items (Hadamards). Then repeat a
two-step "Grover iteration":

1. Oracle: flip the sign (phase) of each marked state. The probabilities
   don't change yet, but the marked states now point the opposite way from
   the rest.
2. Diffuser: reflect every amplitude about the average amplitude. Because
   the marked states are below average after step 1, this boosts them.

Geometrically, each iteration rotates the state by an angle 2*theta toward
the marked states, where sin(theta) = sqrt(M/N). After k iterations,

    P(success) = sin^2((2k + 1) * theta).

This peaks near k = (pi / (4*theta)) - 1/2. Running *more* iterations rotates
past the target and the success probability falls again. Use --iterations to
see that rise and fall, for example ``--iterations 0 1 2 3``.

Circuit:  H^n, then k times [ oracle, diffuser ], then measure.

How the reflections are built
-----------------------------
Both steps need a gate that flips the sign of one specific basis state. We
build it from a multi-controlled Z: X gates turn the target pattern into all
ones, the multi-controlled Z flips the sign of the all-ones state, and the X
gates are then undone. The diffuser is this same gate for the state
|00...0>, sandwiched between Hadamard layers.

Run with no arguments for n = 3 and marked state 0b101. Pass --skip-hardware
to run only the simulator.
"""

import math
import sys

from _common import make_parser, run_experiments
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def _flip_phase_of(circuit: QuantumCircuit, qubits: list, state: int) -> None:
    """Multiply the amplitude of one basis state by -1, leaving others alone.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to add the gates to.
    qubits : list
        The qubits to act on. Qubit i is bit i of ``state``.
    state : int
        The basis state to flip, as an integer.
    """
    # X on every qubit where state has a 0, so the target becomes 11...1.
    zeros = [q for i, q in enumerate(qubits) if not (state >> i) & 1]
    if zeros:
        circuit.x(zeros)
    # Multi-controlled Z, written as H - multi-controlled X - H on the last qubit.
    circuit.h(qubits[-1])
    circuit.mcx(qubits[:-1], qubits[-1])
    circuit.h(qubits[-1])
    if zeros:
        circuit.x(zeros)


def build_oracle(n: int, marked: list[int]) -> QuantumCircuit:
    """Build a phase oracle that flips the sign of every marked state.

    Parameters
    ----------
    n : int
        Number of qubits.
    marked : list[int]
        The marked items, as integers.

    Returns
    -------
    QuantumCircuit
        The oracle, acting on n qubits.
    """
    oracle = QuantumCircuit(n, name="oracle")
    for w in marked:
        _flip_phase_of(oracle, list(oracle.qubits), w)
    return oracle


def build_diffuser(n: int) -> QuantumCircuit:
    """Build the diffuser, which reflects amplitudes about their average.

    It flips the sign of |00...0> between two layers of Hadamards. That equals
    "inversion about the mean," up to a global phase that has no effect.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    QuantumCircuit
        The diffuser, acting on n qubits.
    """
    diffuser = QuantumCircuit(n, name="diffuser")
    diffuser.h(range(n))
    _flip_phase_of(diffuser, list(diffuser.qubits), 0)
    diffuser.h(range(n))
    return diffuser


def optimal_iterations(n: int, num_marked: int) -> int:
    """Return the number of iterations that maximizes the success probability.

    Parameters
    ----------
    n : int
        Number of qubits (N = 2^n items).
    num_marked : int
        Number of marked items, M.

    Returns
    -------
    int
        The best iteration count. It is 0 when more than about half the items
        are marked, because the starting superposition then already beats any
        number of Grover iterations.
    """
    theta = math.asin(math.sqrt(num_marked / 2**n))
    return max(0, round(math.pi / (4 * theta) - 0.5))


def build_grover_circuit(n: int, marked: list[int], iterations: int) -> QuantumCircuit:
    """Build the full Grover circuit.

    Parameters
    ----------
    n : int
        Number of qubits.
    marked : list[int]
        The marked items, as integers.
    iterations : int
        How many (oracle, diffuser) pairs to apply.

    Returns
    -------
    QuantumCircuit
        A circuit that measures all n qubits into register "meas".
    """
    qubits = QuantumRegister(n, name="q")
    bits = ClassicalRegister(n, name="meas")
    circuit = QuantumCircuit(qubits, bits, name=f"grover_k={iterations}")
    circuit.h(qubits)  # equal superposition of all items
    oracle, diffuser = build_oracle(n, marked), build_diffuser(n)
    for _ in range(iterations):
        circuit.barrier()
        circuit.compose(oracle, inplace=True)
        circuit.compose(diffuser, inplace=True)
    circuit.barrier()
    circuit.measure(qubits, bits)
    return circuit


def main() -> int:
    """Search for the marked state(s) on Aer, then on hardware."""
    parser = make_parser("Grover search on Aer and on IBM hardware.")
    parser.add_argument(
        "-n",
        "--qubits",
        type=int,
        default=3,
        help="Number of qubits; the search space has 2^n items (default: 3).",
    )
    parser.add_argument(
        "--marked",
        type=lambda s: int(s, 0),
        nargs="+",
        default=[0b101],
        help="Marked item(s), e.g. 0b101 (default: 0b101).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=None,
        help="Iteration count(s) to run (default: the optimal count).",
    )
    args = parser.parse_args()
    n = args.qubits
    if n < 2 or any(not 0 <= w < 2**n for w in args.marked):
        parser.error(f"need n >= 2 and --marked values that fit in {n} bits")
    marked = sorted(set(args.marked))
    marked_strings = {f"{w:0{n}b}" for w in marked}
    ks = args.iterations or [optimal_iterations(n, len(marked))]
    print(
        f"Grover, n={n} (N={2**n}), marked = {sorted(marked_strings)}, iterations = {ks}"
    )

    circuits = {f"k={k}": build_grover_circuit(n, marked, k) for k in ks}

    def report(prefix: str, name: str, counts: dict) -> None:
        total = sum(counts.values())
        p = sum(c for s, c in counts.items() if s in marked_strings) / total
        k = int(name.split("=")[1])
        theta = math.asin(math.sqrt(len(marked) / 2**n))
        ideal = math.sin((2 * k + 1) * theta) ** 2
        print(
            f"[{prefix}] {name:<5} P(marked) = {p:.3f} (ideal {ideal:.3f}; "
            f"random guess {len(marked) / 2**n:.3f})"
        )
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        print(f"[{prefix}] top counts: {dict(top)}")

    return run_experiments(circuits, args, report)


if __name__ == "__main__":
    sys.exit(main())
