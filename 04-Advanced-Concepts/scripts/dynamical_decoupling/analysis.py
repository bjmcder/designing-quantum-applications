"""Analysis: fit the decay curves and compare the pulse sequences.

Each sequence gives a curve <X>(t). We fit the stretched exponential of
ops.py to it with least squares (``scipy.optimize.curve_fit``). The fit gives
the decay time tau (the time at which the decaying part has fallen to 1/e)
and its standard error, which is the square root of the matching diagonal
element of the covariance matrix.

Prints one table with a row per sequence and draws a figure with all curves
and fits, and below them the residuals (data minus fit). A fit that captures
the physics leaves residuals that scatter around zero with no pattern.

On the simulator the table also shows the decay time that the noise model
implies for the same idle times. It comes from fitting the noise-free
prediction of ops.predicted with the same model, so it includes everything
the fit sees except shot noise.

Two more figures explain the result:

- ``plot_time_domain``: the phase error that builds up during one idle time,
  with and without pulses, for a constant frequency offset and for one that
  changes during the run.
- ``plot_sweep``: <X> at one idle time against the number of pulses, for
  several pulse errors.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from ops import (
    CORRELATION,
    NAMES,
    PARAMETERS,
    decay,
    fast_offsets,
    initial_guess,
    predicted,
    pulse_train,
    segments,
)
from scipy.optimize import curve_fit

# Color-blind-safe colors, one per sequence (used in every figure).
COLORS = {
    "none": "#D55E00",
    "echo": "#0072B2",
    "xy4": "#009E73",
    "runtime_xpxm": "#CC79A7",
    "runtime_xy4": "#56B4E9",
}
BOUNDS = ([-3, 0.5, 0.3, -1], [3, 5000, 8, 1])  # amp, tau, power, offset


def fit(record: dict) -> dict:
    """Fit one curve.

    Args:
        record: One entry of ``data["curves"]`` from ``execution.run``.

    Returns:
        The fitted parameters by name under ``"params"``, their standard
        errors under ``"errors"``, and the residuals (data minus fit).
    """
    t = np.array(record["times_us"])
    y = np.array(record["evs"])
    popt, pcov = curve_fit(
        decay, t, y, p0=initial_guess(t, y), bounds=BOUNDS, maxfev=20000
    )
    errors = np.sqrt(np.diag(pcov))
    return {
        "params": dict(zip(PARAMETERS, map(float, popt))),
        "errors": dict(zip(PARAMETERS, map(float, errors))),
        "residuals": (y - decay(t, *popt)).tolist(),
    }


def fit_all(data: dict) -> dict:
    """Fit every curve in the data.

    Args:
        data: Output of ``execution.run``.

    Returns:
        The result of ``fit`` for each sequence.
    """
    return {name: fit(record) for name, record in data["curves"].items()}


def curve(fitted: dict, t: np.ndarray) -> np.ndarray:
    """The fitted model evaluated at the times ``t``."""
    return decay(t, *fitted["params"].values())


def model_tau(name: str, times: np.ndarray, pulse_error: float) -> float:
    """The decay time that the noise model implies, in microseconds.

    Args:
        name: Sequence, one of ``ops.ARMS``.
        times: Idle times in microseconds.
        pulse_error: Depolarizing probability of one pulse.

    Returns:
        The tau of the stretched exponential fitted to the noise-free
        prediction.
    """
    y = np.array([predicted(name, t, pulse_error) for t in times])
    popt, _ = curve_fit(
        decay, times, y, p0=initial_guess(times, y), bounds=BOUNDS, maxfev=20000
    )
    return float(popt[1])


def print_table(data: dict, fits: dict) -> None:
    """Print the fitted times, with the model's values if known.

    Args:
        data: Output of ``execution.run``.
        fits: Output of ``fit_all``.
    """
    print(
        f"{'':<16} {'fit tau (us)':>17} {'power':>6} {'model tau (us)':>15} "
        f"{'amplitude':>10} {'offset':>7}"
    )
    for name, result in fits.items():
        p, err = result["params"], result["errors"]
        built = f"{'-':>15}"
        if data["simulated"]:
            times = np.array(data["curves"][name]["times_us"])
            built = f"{model_tau(name, times, data['pulse_error']):>15.1f}"
        print(
            f"{NAMES[name]:<16} {p['tau']:>10.1f} +/- {err['tau']:<4.1f} "
            f"{p['power']:>6.2f} {built} {p['amp']:>10.2f} {p['offset']:>7.2f}"
        )


def plot_fits(data: dict, fits: dict):
    """The data and fits of every sequence, with the residuals below.

    Args:
        data: Output of ``execution.run``.
        fits: Output of ``fit_all``.

    Returns:
        The figure.
    """
    fig, (ax, res) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
        sharex=True,
    )
    for name, record in data["curves"].items():
        t = np.array(record["times_us"])
        color = COLORS[name]
        fine = np.linspace(t[0], t[-1], 600)
        tau, err = fits[name]["params"]["tau"], fits[name]["errors"]["tau"]
        ax.plot(
            t,
            record["evs"],
            "o",
            color=color,
            ms=4,
            label=f"{NAMES[name]}: {tau:.0f} $\\pm$ {err:.0f} $\\mu$s",
        )
        ax.plot(fine, curve(fits[name], fine), "-", color=color, lw=1.5)
        res.plot(t, fits[name]["residuals"], "o", color=color, ms=3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("$\\langle X \\rangle$")
    ax.set_title(f"Phase coherence versus idle time ({data['backend']})")
    ax.legend(fontsize=8, title="decay time", loc="upper right")
    res.axhline(0, color="black", lw=0.8)
    res.set_xlabel("idle time ($\\mu$s)")
    res.set_ylabel("data $-$ fit")
    return fig


def plot_time_domain(total: float = 40.0, member: int = 3):
    """The phase error during one idle time, with and without pulses.

    Two columns: a frequency offset that is constant during the run, and one
    that changes every ``ops.CORRELATION`` microseconds. The top row shows the
    offset. The bottom row shows the phase error that has built up by each
    moment, for no pulses, one pulse, and XY4. The phase error at the end is
    what turns <X> from 1 into cos(phase).

    Args:
        total: Idle time in microseconds.
        member: Which noise realization to draw for the changing offset.

    Returns:
        The figure.
    """
    changing = fast_offsets()[member]
    cases = {
        "Constant frequency offset": np.full_like(changing, 0.06),
        f"Frequency offset changing every {CORRELATION:.0f} $\\mu$s": changing,
    }
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 6.0),
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.12},
        sharex=True,
    )
    for col, (title, offsets) in enumerate(cases.items()):
        top, bottom = axes[0, col], axes[1, col]
        for start, stop, block, _, _ in segments(total, []):
            top.plot(
                [start, stop],
                [offsets[block] / (2 * np.pi)] * 2,
                "-",
                color="black",
                lw=2,
            )
        top.set_title(title)
        for name in ("none", "echo", "xy4"):
            pulses = pulse_train(name, total)
            pieces = segments(total, pulses)
            time, phase = [0.0], [0.0]
            for start, stop, block, sign, _ in pieces:
                time.append(stop)
                phase.append(phase[-1] + sign * offsets[block] * (stop - start))
            bottom.plot(
                time,
                phase,
                "-",
                color=COLORS[name],
                lw=1.5,
                label=f"{NAMES[name]}: final {phase[-1]:+.2f} rad",
            )
        bottom.axhline(0, color="gray", lw=0.5)
        bottom.set_xlabel("time in idle period ($\\mu$s)")
        bottom.legend(fontsize=8, loc="best")
    limit = 1.1 * max(np.abs(o).max() for o in cases.values()) / (2 * np.pi)
    for top in axes[0]:
        top.set_ylim(-limit, limit)
        top.axhline(0, color="gray", lw=0.5)
    axes[0, 0].set_ylabel("frequency offset (MHz)")
    axes[1, 0].set_ylabel("phase error (rad)")
    return fig


def plot_sweep(sweep: dict):
    """<X> at one idle time against the number of pulses.

    Args:
        sweep: Output of ``execution.run_sweep``.

    Returns:
        The figure.
    """
    counts = sweep["counts"]
    x = np.arange(len(counts))
    palette = ["#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for color, (key, record) in zip(palette, sweep["curves"].items()):
        label = "hardware" if key == "None" else f"pulse error {float(key):g}"
        ax.errorbar(
            x,
            record["evs"],
            yerr=record["stds"],
            fmt="o",
            color=color,
            ms=5,
            capsize=2,
            label=label,
        )
        if sweep["simulated"]:
            total, p = sweep["total_us"], float(key)
            model = [
                predicted("cpmg" if n else "none", total, p, n or None) for n in counts
            ]
            ax.plot(x, model, "-", color=color, lw=1)
    ax.set_xticks(x, [str(n) for n in counts])
    ax.set_xlabel("number of pulses")
    ax.set_ylabel("$\\langle X \\rangle$")
    ax.set_title(f"Phase coherence at {sweep['total_us']:.0f} $\\mu$s idle time")
    ax.legend(fontsize=8, loc="best")
    return fig


def analyze(data: dict, out: Path) -> dict:
    """Fit every curve, print the table, and save the figure.

    Args:
        data: Output of ``execution.run``.
        out: Path of the PNG to write.

    Returns:
        The fits from ``fit_all``.
    """
    fits = fit_all(data)
    print_table(data, fits)
    fig = plot_fits(data, fits)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")
    return fits


def analyze_sweep(sweep: dict, out: Path) -> None:
    """Print the sweep and save its figure.

    Args:
        sweep: Output of ``execution.run_sweep``.
        out: Path of the PNG to write.
    """
    print(f"<X> at {sweep['total_us']:.0f} us against the number of pulses")
    print(f"{'pulse error':<12}" + "".join(f"{n:>7}" for n in sweep["counts"]))
    for key, record in sweep["curves"].items():
        print(f"{key:<12}" + "".join(f"{v:>7.2f}" for v in record["evs"]))
    fig = plot_sweep(sweep)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")
