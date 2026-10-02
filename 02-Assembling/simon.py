#!/usr/bin/env python3
"""Simon's algorithm: find a hidden "period" of a function using XOR.

The problem
-----------
A black box (the "oracle") computes a function f that maps n-bit strings to
n-bit strings. We are promised there is a hidden nonzero string s such that

    f(x) = f(y)   exactly when   y = x   or   y = x XOR s.

In other words, f is 2-to-1, and every pair of inputs that give the same
output differ by s. What is s?

Classically you must find two inputs with the same output (a "collision"),
which takes about 2^(n/2) queries. Simon's algorithm needs only about n
queries. It was the first exponentially faster quantum algorithm of this
kind, and it inspired Shor's algorithm.

How it works
------------
1. Hadamards put the n input qubits into an equal superposition of all x.
2. The oracle writes f(x) into n output qubits.
3. Measuring the outputs would pick some value f(x0), leaving the inputs in
   the superposition of the pair {x0, x0 XOR s}. (We never actually measure
   the outputs; the result is the same.)
4. Hadamards on the inputs, then measure them. Interference leaves only
   strings y with y . s = 0 (mod 2), each equally likely.

Every shot therefore gives one random y that is "orthogonal" to s, which is
one linear equation on the bits of s. After about n different y values, s is
the only nonzero string that fits them all. That last step is ordinary
classical computing.

Circuit (n input qubits, n output qubits):

    inputs  : |0>^n --H^n--[ U_f ]--H^n--measure -> y
    outputs : |0>^n --------[     ]------

The oracle used here
--------------------
Copy x into the output register (n CX gates). Then, if bit j of x is 1,
where j is the lowest 1 bit of s, XOR s into the output. This makes
f(x) = f(x XOR s), and f is 2-to-1.

Finding s from noisy data
-------------------------
Hardware noise produces some y values that don't satisfy y . s = 0. Instead
of solving the equations exactly, we try every nonzero candidate s and score
it by the fraction of shots with y . s = 0. The true s scores about 1.0 and
wrong candidates score about 0.5, so the best score wins even with noise.

Run with no arguments for n = 3 and s = 0b110. Pass --skip-hardware to run
only the simulator. This algorithm uses 2n qubits, so keep n small on
hardware.
"""

import sys

from _common import make_parser, run_experiments
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

DEFAULT_SECRET = 0b110


def build_oracle(n: int, secret: int) -> QuantumCircuit:
    """Build a 2-to-1 oracle with hidden XOR mask ``secret``.

    Qubits 0 to n-1 are the inputs and qubits n to 2n-1 are the outputs.

    Parameters
    ----------
    n : int
        Number of input bits (and output bits).
    secret : int
        The hidden nonzero string s, as an integer.

    Returns
    -------
    QuantumCircuit
        The oracle, acting on 2n qubits.
    """
    oracle = QuantumCircuit(2 * n, name=f"U_f[s={secret:0{n}b}]")
    for i in range(n):  # copy: |x>|0> -> |x>|x>
        oracle.cx(i, n + i)
    j = (secret & -secret).bit_length() - 1  # position of the lowest 1 bit of s
    for i in range(n):  # if x_j = 1, XOR s into the output
        if (secret >> i) & 1:
            oracle.cx(j, n + i)
    return oracle


def build_simon_circuit(n: int, secret: int) -> QuantumCircuit:
    """Build the full Simon circuit.

    Parameters
    ----------
    n : int
        Number of input bits (and output bits).
    secret : int
        The hidden nonzero string s, as an integer.

    Returns
    -------
    QuantumCircuit
        A circuit that measures only the n input qubits into register "meas".
    """
    inputs = QuantumRegister(n, name="x")
    outputs = QuantumRegister(n, name="f")
    bits = ClassicalRegister(n, name="meas")
    circuit = QuantumCircuit(inputs, outputs, bits, name="simon")

    circuit.h(inputs)  # Step 1: superposition of every input
    circuit.barrier()
    circuit.compose(  # Step 2: the oracle
        build_oracle(n, secret), qubits=list(inputs) + list(outputs), inplace=True
    )
    circuit.barrier()
    circuit.h(inputs)  # Step 4: interfere, then measure the inputs
    circuit.measure(inputs, bits)
    return circuit


def dot(a: int, b: int) -> int:
    """Return the bitwise inner product of a and b, mod 2.

    This counts the positions where both numbers have a 1 and keeps only
    whether that count is odd (1) or even (0).
    """
    return (a & b).bit_count() & 1


def recover_secret(counts: dict, n: int) -> tuple[int, float]:
    """Find the hidden string that best fits the measured y values.

    Parameters
    ----------
    counts : dict
        Maps each measured y (as a bitstring) to its count.
    n : int
        Number of input bits.

    Returns
    -------
    tuple[int, float]
        The best candidate s, and the fraction of shots with y . s = 0 for
        that candidate. Without noise the true s scores 1.0 and every wrong
        candidate scores about 0.5.
    """
    total = sum(counts.values())
    best = max(
        range(1, 1 << n),
        key=lambda s: sum(c for y, c in counts.items() if dot(int(y, 2), s) == 0),
    )
    score = sum(c for y, c in counts.items() if dot(int(y, 2), best) == 0) / total
    return best, score


def main() -> int:
    """Recover the hidden XOR mask on Aer, then on hardware."""
    parser = make_parser("Simon's algorithm on Aer and on IBM hardware.")
    parser.add_argument(
        "-n",
        "--qubits",
        type=int,
        default=3,
        help="Bits in the input register (default: 3; the circuit uses 2n qubits).",
    )
    parser.add_argument(
        "--secret",
        type=lambda s: int(s, 0),
        default=DEFAULT_SECRET,
        help="Hidden nonzero string, e.g. 0b110 (default: 0b110).",
    )
    args = parser.parse_args()
    n = args.qubits
    if not 0 < args.secret < (1 << n):
        parser.error(f"--secret must be a nonzero {n}-bit value")
    print(f"Simon's algorithm, n={n} ({2 * n} qubits), hidden s = {args.secret:0{n}b}")

    def report(prefix: str, name: str, counts: dict) -> None:
        s, score = recover_secret(counts, n)
        status = "correct" if s == args.secret else "WRONG"
        print(
            f"[{prefix}] best s = {s:0{n}b} ({score:.3f} of shots satisfy y.s=0) "
            f"(expected {args.secret:0{n}b}: {status})"
        )
        print(f"[{prefix}] counts: {counts}")

    return run_experiments({"simon": build_simon_circuit(n, args.secret)}, args, report)


if __name__ == "__main__":
    sys.exit(main())
