"""Analysis: extrapolate <ZZ> to zero noise and weigh the cost.

For every measurement we have <ZZ> at a few noise factors. We fit each model
of ops.py (``scipy.optimize.curve_fit``) and read the fit at noise factor 0.
That number is the mitigated value. The value at noise factor 1 is the
unmitigated one, the device as it is.

Prints one table with a row per extrapolator: the mean mitigated value over
the repeated runs, its standard deviation (the cost), its bias (the mean minus
the ideal value), and the root-mean-square error, which combines the two.
On the simulator the table also shows the exact ideal value.

Draws figures for:

- the data, each fit extended to noise factor 0, and the ideal value
- the mitigated values of all repeated runs next to the unmitigated ones
- the unmitigated and mitigated values against circuit length
- the standard deviation against the error of one measurement, with the analytic line
  for the two extrapolators that are weighted sums of the data
- bias and standard deviation for different sets of noise factors
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from ops import METHODS, MODELS, NAMES, exact_value, initial_guess
from scipy.optimize import OptimizeWarning, curve_fit

# Color-blind-safe colors: the unmitigated value and one per extrapolator.
COLORS = {
    "unmitigated": "#555555",
    "linear": "#0072B2",
    "quadratic": "#D55E00",
    "exponential": "#009E73",
}
LABELS = {"unmitigated": "Unmitigated", **NAMES}
MARKERS = ["o", "s", "^", "D"]
YLABEL = r"$\langle ZZ\rangle$"


def fit_model(method: str, factors, values) -> np.ndarray:
    """Fit one model to the data.

    With as many noise factors as parameters the fit passes through every
    point and curve_fit cannot estimate a covariance. We do not use it, so the
    warning is silenced.

    Args:
        method: One of ``ops.METHODS``.
        factors: Noise factors.
        values: Expectation values measured at those factors.

    Returns:
        The fitted parameters. The first one is the value at noise factor 0.

    Raises:
        RuntimeError: If the fit does not converge.
    """
    x, y = np.asarray(factors, float), np.asarray(values, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(
            MODELS[method], x, y, p0=initial_guess(method, x, y), maxfev=20000
        )
    return popt


def extrapolate(method: str, factors, values) -> float:
    """Fit one model and read it at noise factor 0.

    Args:
        method: One of ``ops.METHODS``.
        factors: Noise factors.
        values: Expectation values measured at those factors.

    Returns:
        The fitted value at noise factor 0, or NaN if the fit failed.
    """
    try:
        popt = fit_model(method, factors, values)
    except (RuntimeError, ValueError):
        return float("nan")
    return float(MODELS[method](0.0, *popt))


def select(data: dict, factors) -> np.ndarray:
    """The measured values at some of the noise factors.

    Args:
        data: Output of ``execution.run``.
        factors: Noise factors to keep. Each must have been measured.

    Returns:
        Array of shape (circuit lengths, repeats, len(factors)).
    """
    have = list(data["factors"])
    columns = [have.index(float(f)) for f in factors]
    return np.asarray(data["evs"], dtype=float)[:, :, columns]


def mitigate(data: dict, factors=None) -> dict:
    """Unmitigated and mitigated values for every circuit and repeat.

    Args:
        data: Output of ``execution.run``.
        factors: Noise factors to fit. Default: all that were measured.

    Returns:
        Dict from ``"unmitigated"`` and each method to an array of shape
        (circuit lengths, repeats).
    """
    factors = data["factors"] if factors is None else factors
    evs = select(data, factors)
    out = {"unmitigated": evs[:, :, 0]}
    for method in METHODS:
        out[method] = np.array(
            [[extrapolate(method, factors, run) for run in block] for block in evs]
        )
    return out


def weights(method: str, factors) -> np.ndarray:
    """The weights c_i with value at 0 = sum c_i y_i, for a polynomial fit.

    A least-squares polynomial of degree d is a weighted sum of the data. The
    weights are the first row of the pseudo-inverse of the design matrix.

    Args:
        method: ``"linear"`` or ``"quadratic"``.
        factors: Noise factors.

    Returns:
        One weight per noise factor. They add to 1.
    """
    degree = {"linear": 1, "quadratic": 2}[method]
    design = np.vander(np.asarray(factors, float), degree + 1, increasing=True)
    return np.linalg.pinv(design)[0]


def analytic_std(method: str, factors, precision: float) -> float:
    """Standard deviation of a polynomial extrapolation from measurement noise.

    Each measured value has variance precision^2. For a weighted sum the
    variances add with the weights squared, so the standard deviation of the
    result is precision times the square root of the sum of the squared
    weights.

    Args:
        method: ``"linear"`` or ``"quadratic"``.
        factors: Noise factors.
        precision: Standard error of each measured value.

    Returns:
        The standard deviation of the mitigated value.
    """
    c = weights(method, factors)
    return float(precision * np.sqrt(np.sum(c**2)))


def statistics(values: np.ndarray, ideal: float) -> dict:
    """Mean, spread, bias, and total error of repeated estimates.

    Args:
        values: Estimates from repeated runs (NaN fits are ignored).
        ideal: The ideal value.

    Returns:
        Dict with ``mean``, ``std``, ``bias`` (mean minus ideal), and ``rmse``
        (root-mean-square distance to the ideal).
    """
    v = values[np.isfinite(values)]
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
        "bias": float(v.mean() - ideal),
        "rmse": float(np.sqrt(np.mean((v - ideal) ** 2))),
    }


def print_table(data: dict, factors=None, depth: int = 0) -> dict:
    """Print the unmitigated and mitigated values of one circuit length.

    Args:
        data: Output of ``execution.run``.
        factors: Noise factors to fit. Default: all that were measured.
        depth: Index into ``data["n_cx"]``.

    Returns:
        The statistics of each row, by name.
    """
    factors = data["factors"] if factors is None else factors
    values = mitigate(data, factors)
    ideal = data["ideal"]
    runs = values["unmitigated"].shape[1]
    print(
        f"{data['n_cx'][depth]} CX gates, noise factors {list(map(int, factors))}, "
        f"{runs} run(s), standard error {data['precision']} each, ideal {ideal:.4f}"
    )
    print(f"{'':<24} {'mean':>8} {'std':>8} {'bias':>8} {'rmse':>8}")
    stats = {}
    for name in ["unmitigated", *METHODS]:
        s = stats[name] = statistics(values[name][depth], ideal)
        print(
            f"{LABELS[name]:<24} {s['mean']:>8.4f} {s['std']:>8.4f} "
            f"{s['bias']:>+8.4f} {s['rmse']:>8.4f}"
        )
    return stats


def plot_extrapolation(data: dict, depth: int = 0, run: int = 0, factors=None, ax=None):
    """The data of one run, each fit extended to noise factor 0.

    Args:
        data: Output of ``execution.run``.
        depth: Index into ``data["n_cx"]``.
        run: Which repeat to draw.
        factors: Noise factors to fit. Default: all that were measured.
        ax: Axes to draw on, or None for a new figure.

    Returns:
        The figure.
    """
    factors = np.asarray(data["factors"] if factors is None else factors, float)
    y = select(data, factors)[depth, run]
    n_cx = data["n_cx"][depth]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
    else:
        fig = ax.figure
    x = np.linspace(0, factors.max() + 0.3, 200)

    if data["exact"] is not None:
        ax.plot(
            x,
            [exact_value(n_cx, f) for f in x],
            "-",
            color="lightgray",
            lw=3,
            label="exact",
        )
    ax.errorbar(
        factors,
        y,
        yerr=data["precision"],
        fmt="o",
        color="black",
        ms=5,
        capsize=3,
        label="measured",
        zorder=5,
    )
    for method in METHODS:
        popt = fit_model(method, factors, y)
        ax.plot(x, MODELS[method](x, *popt), "--", color=COLORS[method], lw=1.5)
        ax.plot(
            [0],
            [MODELS[method](0.0, *popt)],
            MARKERS[METHODS.index(method)],
            color=COLORS[method],
            ms=8,
            label=f"{LABELS[method]} fit at 0",
            zorder=6,
        )
    ax.plot(
        [0],
        [data["ideal"]],
        "*",
        color="black",
        ms=13,
        mfc="white",
        label="ideal",
        zorder=7,
    )
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-0.4, factors.max() + 0.4)
    ax.set_xlabel("noise factor")
    ax.set_ylabel(YLABEL)
    ax.set_title(f"Zero-noise extrapolation, {n_cx} CX gates")
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_estimates(data: dict, factors=None, depth: int = 0):
    """Every repeated estimate, by method, with the ideal value.

    Args:
        data: Output of ``execution.run``.
        factors: Noise factors to fit. Default: all that were measured.
        depth: Index into ``data["n_cx"]``.

    Returns:
        The figure.
    """
    values = mitigate(data, factors)
    names = ["unmitigated", *METHODS]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for k, name in enumerate(names):
        v = values[name][depth]
        v = v[np.isfinite(v)]
        ax.plot(
            k + rng.uniform(-0.2, 0.2, len(v)),
            v,
            "o",
            color=COLORS[name],
            ms=3,
            alpha=0.4,
        )
        ax.errorbar(
            [k],
            [v.mean()],
            yerr=v.std(ddof=1),
            fmt="D",
            color="black",
            mfc=COLORS[name],
            ms=8,
            capsize=5,
            zorder=5,
        )
    ax.axhline(data["ideal"], color="black", ls="--", lw=1, label="ideal")
    ax.set_xticks(range(len(names)), [LABELS[n] for n in names], fontsize=8)
    ax.set_ylabel(YLABEL)
    runs = values["unmitigated"].shape[1]
    ax.set_title(f"Estimates from {runs} repeated runs")
    ax.legend(fontsize=8)
    return fig


def plot_depth_sweep(data: dict, factors=None):
    """Mean and spread of every estimate against circuit length.

    Args:
        data: Output of ``execution.run`` with several circuit lengths.
        factors: Noise factors to fit. Default: all that were measured.

    Returns:
        The figure.
    """
    values = mitigate(data, factors)
    n = np.array(data["n_cx"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(data["ideal"], color="black", ls="--", lw=1, label="ideal")
    for k, name in enumerate(["unmitigated", *METHODS]):
        v = values[name]
        ax.errorbar(
            n + (k - 1.5) * 0.4,
            np.nanmean(v, axis=1),
            yerr=np.nanstd(v, axis=1, ddof=1),
            fmt=MARKERS[k] + "-",
            color=COLORS[name],
            ms=5,
            capsize=3,
            lw=1,
            label=LABELS[name],
        )
    ax.set_xlabel("number of CX gates")
    ax.set_ylabel(YLABEL)
    ax.set_title(r"$\langle ZZ\rangle$ against circuit length")
    ax.legend(fontsize=8)
    return fig


def plot_std_vs_precision(runs: list, factors=None):
    """Standard deviation of every estimate against the error of one measurement.

    Args:
        runs: List of outputs of ``execution.run``, one per precision, all
            with one circuit length, the same noise factors, and many repeats.
        factors: Noise factors to fit. Default: all that were measured.

    Returns:
        The figure.
    """
    factors = runs[0]["factors"] if factors is None else factors
    precision = np.array([r["precision"] for r in runs])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name in ["unmitigated", *METHODS]:
        std = [
            statistics(mitigate(r, factors)[name][0], r["ideal"])["std"] for r in runs
        ]
        ax.plot(
            precision,
            std,
            MARKERS[(["unmitigated", *METHODS]).index(name)] + "-",
            color=COLORS[name],
            ms=5,
            label=LABELS[name],
        )
        if name in ("linear", "quadratic"):
            line = [analytic_std(name, factors, p) for p in precision]
            ax.plot(precision, line, ":", color=COLORS[name], lw=1.5)
    ax.plot([], [], ":", color="gray", label="analytic")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(precision, [f"{p:g}" for p in precision])
    ax.minorticks_off()
    ax.set_xlabel("standard error of one measured value")
    ax.set_ylabel("standard deviation of the estimate")
    ax.set_title("Standard deviation against measurement error")
    ax.legend(fontsize=8)
    return fig


def bias_variance_table(data: dict, sets: list) -> dict:
    """Bias and spread of every extrapolator for several sets of noise factors.

    Args:
        data: Output of ``execution.run`` with one circuit length and several
            repeats.
        sets: List of noise-factor sets, each a subset of the measured ones.

    Returns:
        Dict from (set as a tuple, method) to the dict of ``statistics``.
    """
    out = {}
    for fs in sets:
        values = mitigate(data, fs)
        for method in METHODS:
            out[(tuple(fs), method)] = statistics(values[method][0], data["ideal"])
    return out


def print_bias_variance(table: dict) -> None:
    """Print the output of ``bias_variance_table``."""
    print(
        f"{'noise factors':<16} {'extrapolator':<24} {'bias':>8} {'std':>8} {'rmse':>8}"
    )
    for (fs, method), s in table.items():
        label = ",".join(str(int(f)) for f in fs)
        print(
            f"{label:<16} {LABELS[method]:<24} {s['bias']:>+8.4f} "
            f"{s['std']:>8.4f} {s['rmse']:>8.4f}"
        )


def plot_bias_variance(table: dict):
    """Bias against standard deviation, one marker per extrapolator and set.

    Color is the extrapolator and the marker shape is the set of noise factors.

    Args:
        table: Output of ``bias_variance_table``.

    Returns:
        The figure.
    """
    sets = list(dict.fromkeys(fs for fs, _ in table))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (fs, method), s in table.items():
        ax.plot(
            s["std"], s["bias"], MARKERS[sets.index(fs)], color=COLORS[method], ms=9
        )
    ax.axhline(0, color="black", lw=0.8)
    for k, fs in enumerate(sets):  # legend entries: shapes for sets, colors for methods
        ax.plot(
            [],
            [],
            MARKERS[k],
            color="gray",
            label="noise factors " + ",".join(str(int(f)) for f in fs),
        )
    for method in METHODS:
        ax.plot([], [], "s", color=COLORS[method], label=LABELS[method])
    ax.set_xlabel("standard deviation of the estimate")
    ax.set_ylabel("bias (mean minus ideal)")
    ax.set_title("Bias and standard deviation")
    ax.legend(fontsize=8)
    return fig


def analyze(data: dict, out: Path) -> None:
    """Print the table and save the extrapolation figure.

    Args:
        data: Output of ``execution.run``.
        out: Path of the PNG to write.
    """
    print_table(data)
    if data["runtime"]:
        print("\nZero-noise value from the Runtime Estimator:")
        for method, value in data["runtime"][0].items():
            print(f"  {LABELS[method]:<24} {value:>8.4f}")
    fig = plot_extrapolation(data)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")
