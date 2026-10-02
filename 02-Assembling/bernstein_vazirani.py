#!/usr/bin/env python3
"""Bernstein-Vazirani algorithm: find a hidden bit string in one query.

The problem
-----------
A black box (the "oracle") hides an n-bit string s. Given an n-bit input x,
it returns the single bit

    f(x) = s . x = (s_0 x_0 + s_1 x_1 + ... + s_(n-1) x_(n-1)) mod 2,

the parity of the bits where s and x are both 1. What is s?

Classically you need n queries: ask about x = 100...0, 010...0, and so on,
which reveals one bit of s per query. Bernstein-Vazirani finds all of s with
a single query.

How it works
------------
The circuit is the same as Deutsch-Jozsa (see deutsch_jozsa.py):

1. Hadamards put the input qubits into an equal superposition of all x, and
   the ancilla into |->.
2. The oracle applies a CX from input qubit i onto the ancilla for every bit
   s_i = 1. Phase kickback turns this into a phase (-1)^(s . x) on each |x>.
3. Hadamards again. A state with phase pattern (-1)^(s . x) is exactly what a
   Hadamard layer turns into the single basis state |s>.
4. Measure the inputs. The result is s, on every shot (in the ideal case).

Circuit (n input qubits, 1 ancilla):

    inputs  : |0>^n --H^n--[ oracle U_f ]--H^n--measure  -> s
    ancilla : |1>   --H----[            ]

Run with no arguments to recover the secret 0b101 using 3 qubits. Pass
--skip-hardware to run only the simulator.
"""

import sys

from _common import make_parser, run_experiments
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

DEFAULT_SECRET = 0b101


def build_oracle(n: int, secret: int) -> QuantumCircuit:
    """Build the oracle for f(x) = secret . x.

    Qubits 0 to n-1 are the inputs. Qubit n is the ancilla, which receives
    f(x) by XOR.

    Parameters
    ----------
    n : int
        Number of input qubits.
    secret : int
        The hidden string s, as an integer (bit i of s is bit i of this
        number).

    Returns
    -------
    QuantumCircuit
        The oracle, acting on n + 1 qubits.
    """
    oracle = QuantumCircuit(n + 1, name=f"U_f[s={secret:0{n}b}]")
    for i in range(n):
        if (secret >> i) & 1:
            oracle.cx(i, n)
    return oracle


def build_bv_circuit(n: int, secret: int) -> QuantumCircuit:
    """Build the full Bernstein-Vazirani circuit.

    Parameters
    ----------
    n : int
        Number of input qubits.
    secret : int
        The hidden string s, as an integer.

    Returns
    -------
    QuantumCircuit
        A circuit that measures the n input qubits into register "meas".
    """
    inputs = QuantumRegister(n, name="x")
    ancilla = QuantumRegister(1, name="anc")
    bits = ClassicalRegister(n, name="meas")
    circuit = QuantumCircuit(inputs, ancilla, bits, name="bernstein_vazirani")

    # Step 1: |+>^n on the inputs, |-> on the ancilla (so CX kicks back a phase).
    circuit.x(ancilla)
    circuit.h(ancilla)
    circuit.h(inputs)
    circuit.barrier()

    # Step 2: the single oracle call.
    circuit.compose(
        build_oracle(n, secret), qubits=list(inputs) + list(ancilla), inplace=True
    )
    circuit.barrier()

    # Steps 3-4: Hadamards turn the phase pattern into |s>; measure it.
    circuit.h(inputs)
    circuit.measure(inputs, bits)
    return circuit


def main() -> int:
    """Recover a hidden bit string on Aer, then on hardware."""
    parser = make_parser("Bernstein-Vazirani on Aer and on IBM hardware.")
    parser.add_argument(
        "-n",
        "--qubits",
        type=int,
        default=3,
        help="Number of input qubits (default: 3).",
    )
    parser.add_argument(
        "--secret",
        type=lambda s: int(s, 0),
        default=DEFAULT_SECRET,
        help="Hidden string, e.g. 0b101 (default: 0b101).",
    )
    args = parser.parse_args()
    n = args.qubits
    if not 0 <= args.secret < (1 << n):
        parser.error(f"--secret must fit in {n} bits")
    expected = f"{args.secret:0{n}b}"
    print(f"Bernstein-Vazirani, n={n}, hidden string s = {expected}")

    def report(prefix: str, name: str, counts: dict) -> None:
        total = sum(counts.values())
        found, hits = max(counts.items(), key=lambda kv: kv[1])
        status = "correct" if found == expected else "WRONG"
        print(
            f"[{prefix}] most frequent = {found} ({hits / total:.3f}) "
            f"-> s = {found} (expected {expected}: {status})"
        )
        print(f"[{prefix}] counts: {counts}")

    return run_experiments(
        {"bernstein_vazirani": build_bv_circuit(n, args.secret)}, args, report
    )


if __name__ == "__main__":
    sys.exit(main())
