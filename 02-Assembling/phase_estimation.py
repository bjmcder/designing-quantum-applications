#!/usr/bin/env python3
"""Quantum phase estimation (QPE): measure the eigenvalue of a unitary.

The problem
-----------
A unitary gate U has an eigenstate |psi>, meaning

    U |psi> = e^(2*pi*i*theta) |psi>

for some number theta between 0 and 1 (a fraction of a full turn). QPE
estimates theta to t bits of precision. It is the core of Shor's algorithm and
of many quantum chemistry methods.

How it works
------------
QPE is phase kickback (see phase_kickback.py) used at t different scales,
followed by an inverse quantum Fourier transform (QFT).

1. Put t "counting" qubits in |+>, and the target qubit in |psi>.
2. Counting qubit k controls U applied 2^k times. Phase kickback leaves the
   phase 2*pi*theta*2^k on counting qubit k. Together, the counting register
   holds a state whose phases encode the binary digits of theta.
3. The inverse QFT converts those phases into a basis state, the integer
   2^t * theta written in binary.
4. Measure the counting qubits and divide by 2^t to get theta.

If 2^t * theta is a whole number, the answer is exact and every shot agrees.
Otherwise the counts peak at the nearest whole numbers, with a smaller tail
that shrinks as t grows.

Circuit (t counting qubits, 1 target qubit):

    counting k : H --*(U^(2^k))--\\
                                 [ inverse QFT ]--measure
    target     : |psi> ----------/

The unitary used here
---------------------
U = P(2*pi*theta), the phase gate that multiplies |1> by e^(2*pi*i*theta).
The target is |1>, which is an eigenstate. Since U^(2^k) = P(2*pi*theta*2^k),
each controlled-U^(2^k) is one controlled-phase gate, with no repeated
applications needed.

Run with no arguments for theta = 5/16 and t = 4. Here 2^t * theta = 5, so
the answer is exact and the outcome is always 0101. Try --theta 0.3 to see the
inexact case. Pass --skip-hardware to run only the simulator.
"""

import math
import sys

from _common import make_parser, run_experiments
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate


def _inverse_qft(t: int) -> QuantumCircuit:
    """Return the inverse QFT on ``t`` qubits as basic H, CP, and SWAP gates."""
    iqft = QuantumCircuit(t, name="IQFT")
    iqft.append(QFTGate(t).inverse(), range(t))
    return iqft.decompose(reps=1)


def build_qpe_circuit(t: int, theta: float) -> QuantumCircuit:
    """Build QPE for U = P(2*pi*theta) with eigenstate |1>.

    Parameters
    ----------
    t : int
        Number of counting qubits (bits of precision).
    theta : float
        The phase to estimate, in turns (0 <= theta < 1).

    Returns
    -------
    QuantumCircuit
        A circuit that measures the t counting qubits into register "meas".
    """
    counting = QuantumRegister(t, name="cnt")
    target = QuantumRegister(1, name="targ")
    bits = ClassicalRegister(t, name="meas")
    circuit = QuantumCircuit(counting, target, bits, name=f"qpe_t={t}")

    # Step 1: target in the eigenstate |1>, counting qubits in |+>.
    circuit.x(target)
    circuit.h(counting)
    circuit.barrier()

    # Step 2: counting qubit k kicks back the phase 2*pi*theta*2^k.
    for k in range(t):
        circuit.cp(2 * math.pi * theta * 2**k, counting[k], target[0])
    circuit.barrier()

    # Step 3: inverse QFT. It is decomposed into basic gates because the
    # Aer simulator doesn't accept the QFT as a single block.
    iqft = _inverse_qft(t)
    circuit.compose(iqft, qubits=counting, inplace=True)

    # Step 4: measure the counting qubits.
    circuit.measure(counting, bits)
    return circuit


def main() -> int:
    """Estimate theta on Aer, then on hardware."""
    parser = make_parser("Quantum phase estimation on Aer and on IBM hardware.")
    parser.add_argument(
        "-t",
        "--counting",
        type=int,
        default=4,
        help="Number of counting qubits, i.e. bits of precision (default: 4).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=5 / 16,
        help="Phase to estimate, in turns, 0 <= theta < 1 (default: 0.3125).",
    )
    args = parser.parse_args()
    t, theta = args.counting, args.theta
    if not 0 <= theta < 1:
        parser.error("--theta must be in [0, 1)")
    nearest = round(theta * 2**t) % 2**t
    print(
        f"QPE, t={t} counting qubits, theta = {theta} "
        f"(2^t*theta = {theta * 2**t:.3f}, nearest integer {nearest} = {nearest:0{t}b})"
    )

    def report(prefix: str, name: str, counts: dict) -> None:
        total = sum(counts.values())
        best, hits = max(counts.items(), key=lambda kv: kv[1])
        estimate = int(best, 2) / 2**t
        status = "correct" if int(best, 2) == nearest else "WRONG"
        print(
            f"[{prefix}] most frequent = {best} ({hits / total:.3f}) -> "
            f"theta ~ {estimate:.4f} (true {theta}: {status})"
        )
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        print(f"[{prefix}] top counts: {dict(top)}")

    return run_experiments({f"qpe_t={t}": build_qpe_circuit(t, theta)}, args, report)


if __name__ == "__main__":
    sys.exit(main())
