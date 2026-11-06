"""Analysis: fit the decay curves and read off T1, T2*, and T2.

Each experiment gives a curve <O>(t). We fit the model of ops.py to it with
least squares (``scipy.optimize.curve_fit``) and the fit gives the time
constant and its standard error, which is the square root of the matching
diagonal element of the covariance matrix.

Prints one table with a row per experiment and draws a figure with one
column per experiment:

- top row: the measured points, the fitted curve, and for Ramsey the decay
  envelope that the oscillation sits inside
- bottom row: the residuals (data minus fit). A fit that captures the
  physics leaves residuals that scatter around zero with no pattern.

On the simulator the table also shows the times that were built into the
noise model, so you can check the fit against the truth.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from ops import EXPERIMENTS, MODELS, NAMES, PARAMETERS, initial_guess
from scipy.optimize import curve_fit

# Color-blind-safe colors, one per experiment (used in every panel).
COLORS = {"t1": "#0072B2", "t2star": "#D55E00", "t2": "#009E73"}
SYMBOLS = {"t1": "T1", "t2star": "T2*", "t2": "T2"}


def fit(name: str, record: dict) -> dict:
    """Fit one experiment's curve.

    Args:
        name: Experiment, one of ``ops.EXPERIMENTS``.
        record: Its entry in the data from ``execution.run``.

    Returns:
        The fitted parameters by name under ``"params"``, their standard
        errors under ``"errors"``, and the residuals (data minus fit).
    """
    t = np.array(record["times_us"])
    y = np.array(record["evs"])
    p0 = initial_guess(name, t, y)
    popt, pcov = curve_fit(MODELS[name], t, y, p0=p0, maxfev=20000)
    names = PARAMETERS[name]
    errors = np.sqrt(np.diag(pcov))
    return {
        "params": dict(zip(names, map(float, popt))),
        "errors": dict(zip(names, map(float, errors))),
        "residuals": (y - MODELS[name](t, *popt)).tolist(),
    }


def fit_all(data: dict) -> dict:
    """Fit every experiment in the data.

    Args:
        data: Output of ``execution.run``.

    Returns:
        The result of ``fit`` for each experiment.
    """
    return {name: fit(name, data["experiments"][name]) for name in data["experiments"]}


def curve(name: str, fitted: dict, t: np.ndarray) -> np.ndarray:
    """The fitted model evaluated at the times ``t``."""
    return MODELS[name](t, *fitted["params"].values())


def print_table(data: dict, fits: dict) -> None:
    """Print the fitted times, with the simulator's true values if known.

    Args:
        data: Output of ``execution.run``.
        fits: Output of ``fit_all``.
    """
    truth = data["truth"]
    print(f"{'':<16} {'fit (us)':>16} {'built in (us)':>14} {'amplitude':>10}")
    for name, result in fits.items():
        tau = result["params"]["tau"]
        err = result["errors"]["tau"]
        built = f"{truth[name]:>14.1f}" if truth else f"{'-':>14}"
        amp = result["params"]["amp"]
        print(f"{NAMES[name]:<16} {tau:>9.1f} +/- {err:<4.1f} {built} {amp:>10.2f}")
    if "t2star" in fits:
        f = fits["t2star"]["params"]["freq"]
        print(f"\nRamsey fringe frequency: {f * 1000:.1f} kHz (programmed: 50.0 kHz)")


def plot_panel(ax_top, ax_res, name: str, record: dict, result: dict) -> None:
    """Draw one experiment: data and fit above, residuals below.

    Args:
        ax_top: Axes for the data and the fit.
        ax_res: Axes for the residuals.
        name: Experiment, one of ``ops.EXPERIMENTS``.
        record: Its entry in the data from ``execution.run``.
        result: Its entry in the output of ``fit_all``.
    """
    t = np.array(record["times_us"])
    y = np.array(record["evs"])
    color = COLORS[name]
    fine = np.linspace(t[0], t[-1], 600)
    p = result["params"]

    ax_top.plot(t, y, "o", color=color, ms=4, label="measured")
    ax_top.plot(
        fine, curve(name, result, fine), "-", color="black", lw=1.5, label="fit"
    )
    if name == "t2star":
        # The decay envelope that the oscillation sits inside.
        env = p["amp"] * np.exp(-fine / p["tau"])
        ax_top.plot(fine, p["offset"] + env, "--", color="gray", lw=1)
        ax_top.plot(fine, p["offset"] - env, "--", color="gray", lw=1, label="envelope")
    else:
        # The time constant: the curve has covered 1 - 1/e of its drop.
        reach = p["offset"] + p["amp"] / np.e
        ax_top.plot(
            [p["tau"], p["tau"]],
            [ax_top.get_ylim()[0], reach],
            ":",
            color=color,
            lw=1,
        )
    ax_top.axhline(p["offset"], color="gray", lw=0.5)
    ax_top.set_title(
        f"{NAMES[name]}\n{SYMBOLS[name]} = {p['tau']:.1f} $\\pm$ "
        f"{result['errors']['tau']:.1f} $\\mu$s"
    )
    ax_top.set_ylabel(f"$\\langle {'Z' if name == 't1' else 'X'} \\rangle$")
    ax_top.legend(fontsize=8, loc="best")

    ax_res.axhline(0, color="black", lw=0.8)
    ax_res.plot(t, result["residuals"], "o", color=color, ms=3)
    ax_res.set_xlabel("delay ($\\mu$s)")
    ax_res.set_ylabel("data $-$ fit")


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


def plot_fits(data: dict, fits: dict):
    """The data and fits of every experiment, one column each.

    Args:
        data: Output of ``execution.run``.
        fits: Output of ``fit_all``.

    Returns:
        The figure.
    """
    names = [n for n in EXPERIMENTS if n in fits]
    fig, axes = plt.subplots(
        2,
        len(names),
        figsize=(4.6 * len(names), 5.6),
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1, "wspace": 0.3},
        sharex="col",
    )
    for col, name in enumerate(names):
        plot_panel(
            axes[0, col], axes[1, col], name, data["experiments"][name], fits[name]
        )
        axes[0, col].tick_params(labelbottom=False)
    fig.suptitle(f"Decay fits ({data['backend']})", y=1.02)
    return fig


def plot_single(name: str, record: dict, result: dict):
    """One experiment alone, for the step-by-step view in the notebook.

    Args:
        name: Experiment, one of ``ops.EXPERIMENTS``.
        record: Its entry in the data from ``execution.run``.
        result: Its entry in the output of ``fit_all``.

    Returns:
        The figure.
    """
    fig, (ax, res) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
        sharex=True,
    )
    plot_panel(ax, res, name, record, result)
    ax.tick_params(labelbottom=False)
    return fig


def plot_checks(name: str, record: dict, result: dict):
    """Two views of a single-exponential fit, for T1 and T2.

    Left: take the fitted offset and amplitude out of the data. What is left
    is exp(-t / tau), so its logarithm is a straight line of slope -1 / tau.
    Points where the decay has nearly finished are left out, because their
    logarithm is mostly noise.

    Right: the sum of squared residuals as the time constant is varied. For
    each trial value the amplitude and offset are set to their best values.
    The fit is the trial value with the smallest sum, and the width of the
    minimum shows the uncertainty of the fit.

    Args:
        name: ``"t1"`` or ``"t2"``.
        record: Its entry in the data from ``execution.run``.
        result: Its entry in the output of ``fit_all``.

    Returns:
        The figure.
    """
    t = np.array(record["times_us"])
    y = np.array(record["evs"])
    p, err = result["params"], result["errors"]
    color = COLORS[name]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    scaled = (y - p["offset"]) / p["amp"]
    keep = scaled > 0.1
    left.plot(t[keep], np.log(scaled[keep]), "o", color=color, ms=4)
    left.plot(t, -t / p["tau"], "-", color="black", lw=1.2)
    left.set_xlabel("delay ($\\mu$s)")
    left.set_ylabel("$\\ln[(\\langle O\\rangle - B)\\,/\\,A]$")
    left.set_title("Log of scaled decay, slope $-1/\\tau$")

    taus = p["tau"] * np.linspace(0.6, 1.5, 300)
    cost = []
    for tau in taus:
        basis = np.column_stack([np.exp(-t / tau), np.ones_like(t)])
        coef = np.linalg.lstsq(basis, y, rcond=None)[0]
        cost.append(np.sum((y - basis @ coef) ** 2))
    right.plot(taus, cost, "-", color=color)
    right.axvspan(
        p["tau"] - err["tau"], p["tau"] + err["tau"], color="gray", alpha=0.25
    )
    right.axvline(p["tau"], color="black", lw=1)
    right.set_xlabel("trial time constant ($\\mu$s)")
    right.set_ylabel("sum of squared residuals")
    right.set_title("Squared residuals vs. trial $\\tau$")
    fig.tight_layout()
    return fig
