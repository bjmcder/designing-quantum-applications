#!/usr/bin/env python3
"""Deutsch-Jozsa algorithm: is a hidden function constant or balanced?

The problem
-----------
A black box (the "oracle") computes a function f. It takes an n-bit string x
and returns one bit, f(x). We are promised that f is one of two kinds:

* constant: f(x) is the same for every x.
* balanced: f(x) = 0 for exactly half of the inputs and 1 for the other half.

Which kind is it? Classically, you may need 2^(n-1) + 1 calls to the oracle
in the worst case. Deutsch-Jozsa answers with a single call.

How it works
------------
1. Hadamards put the n input qubits into an equal superposition of all 2^n
   strings, so one oracle call touches every input at once. (Measuring that state
   right away would reveal only one random x. The advantage comes from the
   interference in step 3.)
2. The ancilla qubit starts in the state |->. With the ancilla in |->, the
   oracle's "XOR f(x) into the ancilla" becomes a phase of (-1)^f(x) on each
   input |x>. This trick is phase kickback (see phase_kickback.py).
3. Hadamards again make the amplitudes interfere. The amplitude of |00...0>
   is the average of (-1)^f(x) over all x. That average is +1 or -1 if f is
   constant, and exactly 0 if f is balanced.
4. Measure the inputs. All zeros means constant; anything else means balanced.

Circuit (n input qubits, 1 ancilla):

    inputs  : |0>^n --H^n--[ oracle U_f ]--H^n--measure
    ancilla : |1>   --H----[            ]

Oracles built here
------------------
    constant0 : f(x) = 0 (do nothing).
    constant1 : f(x) = 1 (flip the ancilla with X).
    balanced  : f(x) = (s . x) XOR b, where s is a nonzero n-bit mask and
                s . x is the parity of the bits of x selected by s. Any
                nonzero s gives a balanced function. It uses one CX from each
                input qubit where s has a 1 (onto the ancilla), plus an X on
                the ancilla if b = 1.

On a balanced oracle the output register equals the mask s exactly, not
just "something nonzero". That is because this oracle is linear (a parity
function), and Hadamards turn a parity phase pattern into the single state
|s>. Parity functions are only a small subset of all balanced functions,
but they preview Bernstein-Vazirani (see bernstein_vazirani.py).

Run with no arguments to test all three oracles with 3 qubits (the default
balanced mask is 0b011). Qiskit prints bitstrings with qubit 0 on the right,
so that mask appears as 011. Pass
--skip-hardware to run only the simulator.
"""

import sys

from _common import make_parser, run_experiments
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

DEFAULT_QUBITS = 3
ORACLE_KINDS = ("constant0", "constant1", "balanced")
DEFAULT_MASK = 0b011  # not a palindrome, so it checks the bit ordering


def build_oracle(
    n: int, kind: str, mask: int | None = None, flip: bool = False
) -> QuantumCircuit:
    """Build the Deutsch-Jozsa oracle U_f.

    Qubits 0 to n-1 are the inputs. Qubit n is the ancilla, which receives
    f(x) by XOR.

    Parameters
    ----------
    n : int
        Number of input qubits.
    kind : str
        One of "constant0", "constant1", or "balanced".
    mask : int, optional
        For "balanced" only: the nonzero n-bit mask s that picks which input
        bits are included in the parity (default: 0b011, cut down to n bits).
    flip : bool, optional
        For "balanced" only: also XOR the output with 1 (default: False).

    Returns
    -------
    QuantumCircuit
        The oracle, acting on n + 1 qubits.
    """
    if kind not in ORACLE_KINDS:
        raise ValueError(f"kind must be one of {ORACLE_KINDS}, got {kind!r}")

    oracle = QuantumCircuit(n + 1, name=f"U_f[{kind}]")
    if kind == "constant1":
        oracle.x(n)
    elif kind == "balanced":
        mask = DEFAULT_MASK & ((1 << n) - 1) if mask is None else mask
        if not 0 < mask < (1 << n):
            raise ValueError(f"mask must be a nonzero {n}-bit value, got {mask}")
        for i in range(n):
            if (mask >> i) & 1:
                oracle.cx(i, n)
        if flip:
            oracle.x(n)
    return oracle


def build_deutsch_jozsa_circuit(n: int, oracle: QuantumCircuit) -> QuantumCircuit:
    """Wrap an oracle in the Deutsch-Jozsa circuit.

    Parameters
    ----------
    n : int
        Number of input qubits.
    oracle : QuantumCircuit
        The (n + 1)-qubit oracle from ``build_oracle``.

    Returns
    -------
    QuantumCircuit
        A circuit that measures the n input qubits into register "meas".
    """
    inputs = QuantumRegister(n, name="x")
    ancilla = QuantumRegister(1, name="anc")
    bits = ClassicalRegister(n, name="meas")
    circuit = QuantumCircuit(inputs, ancilla, bits, name=f"deutsch_jozsa_{oracle.name}")

    # Step 1-2: |+>^n on the inputs, |-> on the ancilla. Because the ancilla
    # is in |->, the oracle gives U_f |x>|-> = (-1)^f(x) |x>|->.
    circuit.x(ancilla)
    circuit.h(ancilla)
    circuit.h(inputs)
    circuit.barrier()

    # The single oracle call.
    circuit.compose(oracle, qubits=list(inputs) + list(ancilla), inplace=True)
    circuit.barrier()

    # Step 3-4: interfere, then measure only the input qubits.
    circuit.h(inputs)
    circuit.measure(inputs, bits)
    return circuit


def interpret(counts: dict, n: int) -> str:
    """Decide whether f is constant or balanced from the counts.

    Parameters
    ----------
    counts : dict
        Maps each measured bitstring to its count.
    n : int
        Number of input qubits.

    Returns
    -------
    str
        "constant" if all zeros is the majority outcome, else "balanced".
        In theory a constant f gives all zeros on every shot. Real hardware
        noise sends some shots elsewhere, so we use a majority vote instead
        of demanding a perfect result.
    """
    zeros = counts.get("0" * n, 0)
    return "constant" if zeros > sum(counts.values()) / 2 else "balanced"


def main() -> int:
    """Run Deutsch-Jozsa for the chosen oracle(s) on Aer, then on hardware."""
    parser = make_parser("Deutsch-Jozsa on Aer and on IBM hardware.")
    parser.add_argument(
        "-n",
        "--qubits",
        type=int,
        default=DEFAULT_QUBITS,
        help=f"Number of input qubits (default: {DEFAULT_QUBITS}).",
    )
    parser.add_argument(
        "--oracle",
        choices=(*ORACLE_KINDS, "all"),
        default="all",
        help="Which oracle to test (default: all).",
    )
    parser.add_argument(
        "--mask",
        type=lambda s: int(s, 0),
        default=None,
        help="Balanced oracle: nonzero bitmask choosing which input bits are "
        "in the parity, e.g. 0b101 (default: 0b011).",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Balanced oracle: also XOR the output with 1.",
    )
    args = parser.parse_args()
    n = args.qubits
    if args.mask is not None and not 0 < args.mask < (1 << n):
        parser.error(f"--mask must be a nonzero {n}-bit value")
    kinds = ORACLE_KINDS if args.oracle == "all" else (args.oracle,)

    circuits = {
        kind: build_deutsch_jozsa_circuit(
            n, build_oracle(n, kind, mask=args.mask, flip=args.flip)
        )
        for kind in kinds
    }
    expected = {k: ("balanced" if k == "balanced" else "constant") for k in kinds}
    print(f"Deutsch-Jozsa, n={n} input qubits")

    def report(prefix: str, name: str, counts: dict) -> None:
        total = sum(counts.values())
        p0 = counts.get("0" * n, 0) / total
        verdict = interpret(counts, n)
        status = "correct" if verdict == expected[name] else "WRONG"
        print(
            f"[{prefix}] {name:<10} P(all zeros) = {p0:.3f} -> {verdict} "
            f"(expected {expected[name]}: {status})"
        )
        print(f"[{prefix}] counts: {counts}")

    return run_experiments(circuits, args, report)


if __name__ == "__main__":
    sys.exit(main())
