"""Ops: the observable, the read-out error model, and what each fix predicts.

The problem: the last step of every circuit is a measurement, and the
measurement misreads bits. A qubit that is in |1> is read as 0 more often
than a qubit in |0> is read as 1, so every expectation value <Z> is pushed
away from its true value, and the push depends on the value.

Two numbers describe the error of one qubit:

    p01   probability to read 1 when the bit is 0
    p10   probability to read 0 when the bit is 1

The confusion matrix collects them. Column j is the state that was prepared,
row i is the answer that was read, so each column adds up to 1:

    C = [[1 - p01, p10    ],
         [p01,     1 - p10]]

The expectation value is <Z> = P(0) - P(1). If the true value is z, the
measured value is

    z_measured = a z + b,   a = 1 - p01 - p10,   b = p10 - p01.

The factor a (attenuation) shrinks <Z> toward 0. The offset b shifts it, and
it is not zero when the error is asymmetric.

Two fixes are compared:

inversion   Measure C with two calibration circuits and undo it:
            z = (z_measured - b) / a. It needs both p01 and p10.
twirling    Flip the bit with an X gate before the measurement, in a random
            half of the runs, and flip the answer back in software. The
            offset b averages away, so z_measured = a z. One calibration
            state with a known answer gives a, and z = z_measured / a.

On the simulator the error is built in (``SIM_READOUT``), so the corrected
values can be checked against the truth.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

SIM_READOUT = (0.02, 0.06)  # (p01, p10) of the simulated qubit
ANGLES = 13  # default number of RY angles in the sweep


def observable() -> SparsePauliOp:
    """The operator every circuit measures.

    Returns:
        The single-qubit Pauli operator Z.
    """
    return SparsePauliOp("Z")


def angle_grid(points: int = ANGLES) -> np.ndarray:
    """Evenly spaced RY angles from 0 to pi.

    The state RY(theta)|0> has <Z> = cos(theta), so the sweep covers every
    true value from +1 to -1.

    Args:
        points: Number of angles.

    Returns:
        The angles in radians.
    """
    return np.linspace(0.0, np.pi, points)


def ideal_z(theta) -> np.ndarray:
    """True expectation value of Z after RY(theta) on |0>.

    Args:
        theta: Angle or array of angles in radians.

    Returns:
        cos(theta).
    """
    return np.cos(theta)


def confusion_matrix(p01: float, p10: float) -> np.ndarray:
    """The 2 x 2 confusion matrix, columns summing to 1.

    Args:
        p01: Probability to read 1 when the bit is 0.
        p10: Probability to read 0 when the bit is 1.

    Returns:
        ``[[1 - p01, p10], [p01, 1 - p10]]``.
    """
    return np.array([[1 - p01, p10], [p01, 1 - p10]])


def attenuation(p01: float, p10: float) -> float:
    """The factor a = 1 - p01 - p10 that shrinks <Z>."""
    return 1 - p01 - p10


def offset(p01: float, p10: float) -> float:
    """The shift b = p10 - p01 that an asymmetric error adds to <Z>."""
    return p10 - p01


def average_error(p01: float, p10: float) -> float:
    """The error rate after twirling, (p01 + p10) / 2."""
    return (p01 + p10) / 2


def measured_z(z, p01: float, p10: float, twirled: bool = False):
    """What the noisy read-out reports for a true value z.

    Args:
        z: True expectation value or array of values.
        p01: Probability to read 1 when the bit is 0.
        p10: Probability to read 0 when the bit is 1.
        twirled: If True, the offset is averaged away.

    Returns:
        ``a z + b``, or ``a z`` if twirled.
    """
    shift = 0.0 if twirled else offset(p01, p10)
    return attenuation(p01, p10) * np.asarray(z) + shift


def simulator_truth() -> dict:
    """The read-out error the simulator was built with.

    Returns:
        p01, p10, and the a, b, and average error that follow from them.
    """
    p01, p10 = SIM_READOUT
    return {
        "p01": p01,
        "p10": p10,
        "a": attenuation(p01, p10),
        "b": offset(p01, p10),
        "average_error": average_error(p01, p10),
    }
