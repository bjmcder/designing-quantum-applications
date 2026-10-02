#!/usr/bin/env python3
"""Phase kickback: a controlled gate that changes its *control* qubit.

The idea
--------
A controlled-U gate is usually described as "apply U to the target if the
control is 1." But if the target is in an eigenstate of U, meaning

    U |psi> = e^(i*phi) |psi>,

then applying U just multiplies the target by a phase and leaves it in the
same state. By linearity, the controlled-U gives

    |0>|psi>  ->  |0>|psi>
    |1>|psi>  ->  e^(i*phi) |1>|psi>

The target is unchanged, but the phase e^(i*phi) now sits on the |1> part of
the control. The phase has "kicked back" onto the control qubit.

This is the engine behind Deutsch-Jozsa, Bernstein-Vazirani, and phase
estimation. It is also the usual way to turn a bit oracle into a phase oracle
for Grover's algorithm. It is worth seeing on its own.

How we measure it
-----------------
A phase on |1> is invisible if we measure right away. So we start the control
in |+> = (|0> + |1>)/sqrt(2). After the kickback it is
(|0> + e^(i*phi)|1>)/sqrt(2), and a final Hadamard turns the phase into a
probability:

    P(control measures 1) = sin^2(phi / 2).

Circuit:

    control : --H--*--H--measure
    target  : ----[U]---      (target starts in an eigenstate of U)

Cases run here
--------------
U is the phase gate P(phi), which multiplies |1> by e^(i*phi) and leaves |0>
alone. So |1> is an eigenstate with eigenvalue e^(i*phi), and |0> is an
eigenstate with eigenvalue 1.

    kick_phi=...    : target |1>. The phase phi kicks back, and P(1) is
                      sin^2(phi/2). Try phi = 0, pi/2, pi: P(1) = 0, 0.5, 1.
    no_kick_phi=... : target |0>, whose eigenvalue is 1. Nothing kicks back,
                      so P(1) = 0 for every phi. This is the control
                      experiment.
    cx_on_minus     : a CX with target |->. Since X|-> = -|->, the eigenvalue
                      is -1, so CX acts like a Z on the control and P(1) = 1.
                      This special case is the one used in Deutsch-Jozsa and
                      Bernstein-Vazirani.

Run with no arguments. Pass --skip-hardware to run only the simulator.
"""

import math
import sys

from _common import make_parser, run_experiments
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

# How far a measured P(1) may be from the ideal value and still be marked "ok".
# This leaves room for shot noise and hardware errors.
TOLERANCE = 0.1
DEFAULT_PHASES = (0.0, 0.5, 1.0)  # in units of pi


def _skeleton(name: str) -> tuple[QuantumCircuit, QuantumRegister, QuantumRegister]:
    """Make an empty circuit with one control qubit, one target qubit, and one bit."""
    ctrl = QuantumRegister(1, name="ctrl")
    targ = QuantumRegister(1, name="targ")
    bits = ClassicalRegister(1, name="meas")
    return QuantumCircuit(ctrl, targ, bits, name=name), ctrl, targ


def build_kickback(phi: float, eigenstate: bool) -> QuantumCircuit:
    """Build a controlled-P(phi) circuit that reads out the control's phase.

    Parameters
    ----------
    phi : float
        Phase angle of the gate, in radians.
    eigenstate : bool
        If True, the target is |1> (eigenvalue e^(i*phi)), so the phase kicks
        back. If False, the target is |0> (eigenvalue 1), so nothing does.

    Returns
    -------
    QuantumCircuit
        A circuit that measures the control qubit.
    """
    label = "kick" if eigenstate else "no_kick"
    circuit, ctrl, targ = _skeleton(f"{label}_phi={phi / math.pi:g}pi")
    if eigenstate:
        circuit.x(targ)
    circuit.h(ctrl)  # control in |+>
    circuit.cp(phi, ctrl, targ)  # controlled-P(phi)
    circuit.h(ctrl)  # turn the kicked-back phase into a probability
    circuit.measure(ctrl, circuit.cregs[0])
    return circuit


def build_cx_on_minus() -> QuantumCircuit:
    """Build the CX-on-|-> circuit, where CX acts like a Z on the control.

    Returns
    -------
    QuantumCircuit
        A circuit that measures the control qubit. It should give 1 every time.
    """
    circuit, ctrl, targ = _skeleton("cx_on_minus")
    circuit.x(targ)
    circuit.h(targ)  # target in |->
    circuit.h(ctrl)  # control in |+>
    circuit.cx(ctrl, targ)
    circuit.h(ctrl)
    circuit.measure(ctrl, circuit.cregs[0])
    return circuit


def main() -> int:
    """Run the kickback cases on Aer, then on hardware."""
    parser = make_parser("Phase kickback on Aer and on IBM hardware.")
    parser.add_argument(
        "--phases",
        type=float,
        nargs="+",
        default=list(DEFAULT_PHASES),
        help="Phases phi to kick back, in units of pi (default: 0 0.5 1).",
    )
    args = parser.parse_args()

    circuits: dict[str, QuantumCircuit] = {}
    expected_p1: dict[str, float] = {}  # ideal P(1) for each circuit
    for p in args.phases:
        phi = p * math.pi
        for eigenstate in (True, False):
            circuit = build_kickback(phi, eigenstate)
            circuits[circuit.name] = circuit
            expected_p1[circuit.name] = math.sin(phi / 2) ** 2 if eigenstate else 0.0
    circuits["cx_on_minus"] = build_cx_on_minus()
    expected_p1["cx_on_minus"] = 1.0

    def report(prefix: str, name: str, counts: dict) -> None:
        total = sum(counts.values())
        p1 = counts.get("1", 0) / total
        exp = expected_p1[name]
        status = "ok" if abs(p1 - exp) <= TOLERANCE else "OFF"
        print(f"[{prefix}] {name:<22} P(1) = {p1:.3f} (expected {exp:.3f}: {status})")

    return run_experiments(circuits, args, report)


if __name__ == "__main__":
    sys.exit(main())
