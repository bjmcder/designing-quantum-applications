"""Ops: the observable, the pulse sequences, and the noise model of the idle qubit.

The problem: a qubit that waits is not safe. Prepare |+> = (|0> + |1>)/sqrt(2),
do nothing for a time t, and measure X. The expectation <X> starts at +1 and
falls toward 0, because the qubit frequency is not steady. It drifts, a little
differently in every run, and the relative phase between |0> and |1> picks up
an error that grows with t. Idle time costs coherence.

Dynamical decoupling (DD) puts pulses into the idle time. A pulse is a fast
bit flip, X or Y, and it exchanges |0> and |1>. After a flip the phase error
that is building up changes sign, so what was collected before the flip is
undone by what is collected after it. A frequency offset that is constant
during the run cancels completely. One that changes during the run cancels
only as well as the pulses are fast compared with the change.

echo   One X pulse in the middle of the wait (the Hahn echo of the
       benchmarking example). Cancels offsets that are constant for the
       whole wait.
xy4    The repeating pattern X, Y, X, Y. Four pulses per cycle, so offsets that
       change on the scale of a cycle cancel too.
cpmg   A train of n equal X pulses. Used to count pulses one at a time.

The pulses sit at the centers of n equal slices of the wait, so the first and
last free waits are half as long as the ones between pulses.

A pulse is never perfect. Each one adds a small error, and every pulse adds
its own. More pulses cancel more noise and add more error, so there is a best
number. The simulator models a pulse error as depolarizing noise: with
probability p the qubit state is replaced by a fully mixed state, which shrinks
<X> by a factor (1 - p).

Times are in microseconds and frequencies in MHz or in radians per microsecond
(2 pi times MHz), so a frequency times a time is a phase in radians.
"""

from itertools import pairwise

import numpy as np
from qiskit.quantum_info import SparsePauliOp

ARMS = ("none", "echo", "xy4")

NAMES = {
    "none": "no pulses",
    "echo": "1 pulse (echo)",
    "xy4": "XY4",
    "runtime_xpxm": "runtime XpXm",
    "runtime_xy4": "runtime XY4",
}

DEFAULT_T_MAX = 200.0  # us, the longest idle time of the decay curves
XY4_CYCLE = 8.0  # us, the target length of one X, Y, X, Y cycle

# What the simulator is told. The coherence time of the qubit without any
# frequency noise is SIM_T2. Two kinds of frequency noise are added on top.
SIM_T1 = 100.0  # us
SIM_T2 = 70.0  # us
SIM_MEMBERS = 256  # noise realizations that are averaged

# Slow noise: one frequency offset per run, constant during the run. The
# offsets follow a Lorentzian distribution of half-width GAMMA_SLOW, as in
# the benchmarking example.
GAMMA_SLOW = 1 / 100.0  # rad/us

# Fast noise: the offset is redrawn every CORRELATION microseconds, from a
# normal distribution of standard deviation FAST_SIGMA. It is constant inside
# each block and independent between blocks. Without pulses, a block of
# length c adds a random phase of standard deviation FAST_SIGMA * c, and the
# average of cos(phase) over many runs falls as exp(-FAST_RATE t).
CORRELATION = 20.0  # us
FAST_SIGMA = 0.055  # rad/us
FAST_RATE = FAST_SIGMA**2 * CORRELATION / 2  # 1/us
MAX_BLOCKS = 64  # blocks of noise that are drawn, enough for 1280 us

PULSE_ERROR = 0.001  # depolarizing probability of one pulse
SEED = 1  # fixes the noise realizations, so every run sees the same ones


def observable() -> SparsePauliOp:
    """The operator every experiment measures: X, which reads the phase."""
    return SparsePauliOp("X")


def delay_grid(t_max: float, points: int) -> np.ndarray:
    """Evenly spaced idle times from 0 to ``t_max`` microseconds.

    Args:
        t_max: Longest idle time in microseconds.
        points: Number of idle times.

    Returns:
        The idle times, starting at 0.
    """
    return np.linspace(0.0, t_max, points)


def simulator_truth() -> dict:
    """The rates the simulator was built with.

    Returns:
        The coherence time without frequency noise, and the decay rates (per
        microsecond) of the three parts that add up when there are no pulses.
    """
    return {
        "t2": SIM_T2,
        "relaxation_rate": 1 / SIM_T2,
        "slow_rate": GAMMA_SLOW,
        "fast_rate": FAST_RATE,
        "t2star": 1 / (1 / SIM_T2 + GAMMA_SLOW + FAST_RATE),
    }


# ----------------------------------------------------------- pulse sequences


def pulse_train(kind: str, total: float, count: int | None = None) -> list:
    """The pulses of a sequence during an idle time.

    Args:
        kind: ``"none"``, ``"echo"``, ``"xy4"``, or ``"cpmg"``.
        total: Idle time in microseconds.
        count: Number of pulses. XY4 uses a whole number of cycles of four
            and picks about one cycle per ``XY4_CYCLE`` if not given. CPMG
            needs it. The echo always has one.

    Returns:
        A list of (time in microseconds, axis), axis being ``"x"`` or ``"y"``.

    Raises:
        ValueError: If the kind is unknown or the count does not fit the kind.
    """
    if kind not in ("none", "echo", "xy4", "cpmg"):
        raise ValueError(f"unknown sequence {kind!r}")
    if kind == "none" or total <= 0:
        return []
    if kind == "echo":
        count = 1
    elif kind == "xy4":
        count = count or 4 * max(1, round(total / XY4_CYCLE))
        if count % 4:
            raise ValueError("XY4 needs a multiple of 4 pulses")
    elif count is None:
        raise ValueError("cpmg needs a pulse count")
    axes = ["x", "y"] * (count // 2 + 1) if kind == "xy4" else ["x"] * count
    times = (np.arange(count) + 0.5) * total / count
    return [(float(t), axes[i]) for i, t in enumerate(times)]


def segments(total: float, pulses: list) -> list:
    """Cut an idle time at every pulse and every change of the fast noise.

    Inside one piece the noise offset is constant and no pulse acts. A pulse
    acts at the end of its piece.

    Args:
        total: Idle time in microseconds.
        pulses: Output of ``pulse_train``.

    Returns:
        A list of (start, stop, block, sign, axis). ``block`` numbers the
        piece of fast noise, ``sign`` is +1 or -1 and says whether an odd
        number of pulses came before, and ``axis`` is the pulse that acts at
        ``stop`` or None.
    """
    pulse_at = {round(t, 9): axis for t, axis in pulses}
    cuts = {0.0, round(total, 9)} | set(pulse_at)
    j = 1
    while j * CORRELATION < total - 1e-9:
        cuts.add(round(j * CORRELATION, 9))
        j += 1
    edges = sorted(cuts)
    sign, pieces = 1, []
    for start, stop in pairwise(edges):
        block = int(np.floor((start + 1e-9) / CORRELATION))
        axis = pulse_at.get(stop)
        pieces.append((start, stop, block, sign, axis))
        if axis:
            sign = -sign
    return pieces


# ------------------------------------------------------------------ noise


def slow_offsets(members: int = SIM_MEMBERS) -> np.ndarray:
    """Slow-noise offsets, one per member, in radians per microsecond.

    The offsets are evenly spaced quantiles of a Lorentzian distribution,
    shuffled so that they pair with the fast noise at random.

    Args:
        members: Number of noise realizations.

    Returns:
        The offsets.
    """
    k = np.arange(members)
    quantiles = GAMMA_SLOW * np.tan(np.pi * ((k + 0.5) / members - 0.5))
    return np.random.default_rng(SEED).permutation(quantiles)


def fast_offsets(members: int = SIM_MEMBERS, blocks: int = MAX_BLOCKS) -> np.ndarray:
    """Fast-noise offsets, one per member and block, in radians per microsecond.

    Args:
        members: Number of noise realizations.
        blocks: Number of blocks of ``CORRELATION`` microseconds.

    Returns:
        An array of shape (members, blocks).
    """
    rng = np.random.default_rng(SEED + 1)
    return rng.normal(0.0, FAST_SIGMA, (members, blocks))


def noise_offsets(members: int = SIM_MEMBERS) -> np.ndarray:
    """The total offset of every member in every block (slow plus fast).

    Args:
        members: Number of noise realizations.

    Returns:
        An array of shape (members, MAX_BLOCKS), in radians per microsecond.
    """
    return slow_offsets(members)[:, None] + fast_offsets(members)


def final_phase(offsets: np.ndarray, total: float, pulses: list) -> np.ndarray:
    """The phase error at the end of the idle time, for every member.

    Each piece adds its offset times its length, with the sign that the
    pulses before it have set.

    Args:
        offsets: Array of shape (members, blocks) in radians per microsecond.
        total: Idle time in microseconds.
        pulses: Output of ``pulse_train``.

    Returns:
        The phase of each member in radians.
    """
    phase = np.zeros(offsets.shape[0])
    for start, stop, block, sign, _ in segments(total, pulses):
        phase += sign * offsets[:, block] * (stop - start)
    return phase


def predicted(
    kind: str,
    total: float,
    pulse_error: float = PULSE_ERROR,
    count: int | None = None,
    members: int = SIM_MEMBERS,
) -> float:
    """The <X> that the simulator should return, without shot noise.

    It is the average of cos(phase) over the noise members, times the decay
    exp(-t / T2) of the qubit itself, times (1 - p) for each pulse. The
    read-out error is not included, because the fit takes care of it.

    Args:
        kind: Sequence, see ``pulse_train``.
        total: Idle time in microseconds.
        pulse_error: Depolarizing probability of one pulse.
        count: Number of pulses, see ``pulse_train``.
        members: Number of noise realizations.

    Returns:
        The expectation value of X.
    """
    pulses = pulse_train(kind, total, count)
    phase = final_phase(noise_offsets(members), total, pulses)
    return float(
        np.mean(np.cos(phase))
        * np.exp(-total / SIM_T2)
        * (1 - pulse_error) ** len(pulses)
    )


# ------------------------------------------------------------------ models


def decay(t, amp, tau, power, offset):
    """Stretched exponential: ``amp exp(-(t / tau)^power) + offset``.

    ``power`` is 1 for an ordinary exponential. Noise that changes slowly
    compared with the idle time gives a larger power, a decay that starts
    flat and then falls faster. ``tau`` is the time at which the decaying part
    has fallen to 1/e of its start, whatever the power.

    The amplitude and offset absorb the errors of preparing and reading the
    state, which are not part of the decay.
    """
    return amp * np.exp(-((t / tau) ** power)) + offset


PARAMETERS = ("amp", "tau", "power", "offset")


def initial_guess(times: np.ndarray, values: np.ndarray) -> list:
    """Starting parameters for the fit, read off the data.

    The amplitude is the drop between the first points and the tail, the time
    constant starts at a third of the longest idle time, and the power starts
    at 1.

    Args:
        times: Idle times in microseconds.
        values: Measured expectation values.

    Returns:
        Starting values in the order of ``PARAMETERS``.
    """
    tail = float(np.mean(values[-3:]))
    return [float(values[0] - tail), float(times[-1]) / 3, 1.0, tail]
