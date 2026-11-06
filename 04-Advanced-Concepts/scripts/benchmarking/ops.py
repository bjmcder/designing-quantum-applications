"""Ops: the observables and decay models for T1, T2*, and T2.

The problem: a qubit does not hold its state forever. Three times say how fast
it forgets, and they are the first numbers to read off a device before
running anything on it. Each is measured the same way: prepare a state, wait
for a delay t, and measure one observable. Repeating for many delays traces a
curve, and the time is read off by fitting the curve.

T1       Energy relaxation. Prepare |1>, wait, and measure Z. The expectation
         <Z> falls from -1 (the bit is 1) to +1 (the bit is 0) as the qubit
         leaks to |0>. This is a bit flip, and the curve is exp(-t / T1).
T2*      Dephasing seen by a Ramsey experiment. Prepare |+> = (|0> + |1>)/sqrt(2),
         wait, and measure X. The relative phase between |0> and |1> drifts
         and scrambles, so <X> falls from +1 to 0. Every cause of dephasing
         counts, including slow drifts of the qubit frequency. We add a known
         detuning on purpose, so the curve oscillates inside the decay
         envelope exp(-t / T2*).
T2       The same decay with the slow drifts removed. A single X gate in the
         middle of the wait (a Hahn echo) flips the sign of the phase that has
         built up so far, so whatever is constant during the wait cancels.
         What remains is exp(-t / T2).

The three are ordered T2* <= T2 <= 2 T1. The first inequality is the echo
removing slow noise. The second says that a qubit that flips cannot keep a
phase either.

Times are in microseconds and frequencies in MHz, so 2 pi f t is an angle in
radians.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

EXPERIMENTS = ("t1", "t2star", "t2")

NAMES = {
    "t1": "T1 (relaxation)",
    "t2star": "T2* (Ramsey)",
    "t2": "T2 (Hahn echo)",
}

# The Pauli operator each experiment measures.
OBSERVABLES = {"t1": "Z", "t2star": "X", "t2": "X"}

# Longest delay of each experiment, in microseconds, if none is requested.
# A good range runs to about three decay times.
DEFAULT_T_MAX = {"t1": 300.0, "t2star": 120.0, "t2": 200.0}

DETUNING = 0.05  # MHz, the programmed Ramsey detuning (period 20 us)

# What the simulator is told. T2* is not an input: it follows from T2 and the
# width of the slow frequency noise. The ensemble average of cos(delta t) over
# a Lorentzian spread of half-width GAMMA_SLOW is exp(-GAMMA_SLOW t).
SIM_T1 = 100.0  # us
SIM_T2 = 70.0  # us (the echo time, not the Ramsey time)
GAMMA_SLOW = 1 / 100.0  # 1/us, half-width of the slow frequency noise
SIM_READOUT = (0.02, 0.04)  # P(read 1 | state 0), P(read 0 | state 1)
SIM_SHIFTS = 256  # frequency offsets in the slow-noise ensemble


def observable(name: str) -> SparsePauliOp:
    """The operator an experiment measures.

    Args:
        name: Experiment, one of ``EXPERIMENTS``.

    Returns:
        A single-qubit Pauli operator.

    Raises:
        ValueError: If the experiment is unknown.
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}; choose from {EXPERIMENTS}")
    return SparsePauliOp(OBSERVABLES[name])


def delay_grid(t_max: float, points: int) -> np.ndarray:
    """Evenly spaced delays from 0 to ``t_max`` microseconds.

    Args:
        t_max: Longest delay in microseconds.
        points: Number of delays.

    Returns:
        The delays, starting at 0.
    """
    return np.linspace(0.0, t_max, points)


def simulator_truth() -> dict:
    """The times the simulator was built with, in microseconds.

    Returns:
        T1, T2, and the T2* that follows from T2 and the slow noise.
    """
    return {
        "t1": SIM_T1,
        "t2": SIM_T2,
        "t2star": 1 / (1 / SIM_T2 + GAMMA_SLOW),
    }


def slow_shifts(count: int = SIM_SHIFTS) -> np.ndarray:
    """Frequency offsets that stand in for slow noise, in radians per us.

    The offsets are the evenly spaced quantiles of a Lorentzian distribution.
    Each one is constant during a run, as slow noise is, and different from
    one member of the ensemble to the next.

    Args:
        count: Number of offsets.

    Returns:
        The offsets.
    """
    k = np.arange(count)
    return GAMMA_SLOW * np.tan(np.pi * ((k + 0.5) / count - 0.5))


# ------------------------------------------------------------------ models


def decay(t, amp, tau, offset):
    """Exponential decay: ``amp exp(-t / tau) + offset``.

    The amplitude and offset absorb the errors of preparing and reading the
    state, which are not part of the decay. Used by T1 and T2.
    """
    return amp * np.exp(-t / tau) + offset


def ramsey(t, amp, tau, freq, phase, offset):
    """Decaying oscillation: ``amp exp(-t / tau) cos(2 pi freq t + phase) + offset``.

    Used by T2*. ``freq`` is the detuning in MHz.
    """
    return amp * np.exp(-t / tau) * np.cos(2 * np.pi * freq * t + phase) + offset


MODELS = {"t1": decay, "t2star": ramsey, "t2": decay}

PARAMETERS = {
    "t1": ("amp", "tau", "offset"),
    "t2star": ("amp", "tau", "freq", "phase", "offset"),
    "t2": ("amp", "tau", "offset"),
}


def initial_guess(name: str, times: np.ndarray, values: np.ndarray) -> list:
    """Starting parameters for the fit, read off the data.

    The amplitude is the drop between the first points and the tail, the time
    constant starts at a third of the longest delay, and the Ramsey frequency
    starts at the detuning we programmed.

    Args:
        name: Experiment, one of ``EXPERIMENTS``.
        times: Delays in microseconds.
        values: Measured expectation values.

    Returns:
        Starting values in the order of ``PARAMETERS[name]``.
    """
    tail = float(np.mean(values[-3:]))
    amp = float(values[0] - tail)
    tau = float(times[-1]) / 3
    if name == "t2star":
        return [amp, tau, DETUNING, 0.0, tail]
    return [amp, tau, tail]
