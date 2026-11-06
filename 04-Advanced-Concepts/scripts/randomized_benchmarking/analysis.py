"""Analysis: fit the decay and read off the gate fidelity.

The mean survival probability at each length is fitted with

    F(m) = A p^m + B

using least squares (``scipy.optimize.curve_fit``). The decay rate p gives

    error per Clifford   r = (1 - p) / 2
    fidelity             1 - r
    error per gate       r divided by the average number of gates per Clifford

Prints a table and draws a figure with the individual sequences, their mean,
the fit, and the residuals (data minus fit).

On the simulator the table also shows the values that follow from the noise
that was built in, so you can check the fit against the truth.
"""

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from ops import decay, error_per_clifford, initial_guess
from scipy.optimize import curve_fit

COLOR = "#0072B2"


def fit(data: dict) -> dict:
    """Fit the mean survival probability.

    Args:
        data: Output of ``execution.run``.

    Returns:
        Dict with the lengths ``m``, mean survival ``mean``, standard error of
        the mean ``sem``, the fitted ``params`` (amp, p, offset) and their
        ``errors``, the ``residuals``, and the derived ``epc``, ``epc_error``,
        ``fidelity``, and ``epg`` (error per gate).
    """
    m = np.array(data["lengths"], dtype=float)
    runs = [np.array(data["survival"][int(k)]) for k in m]
    mean = np.array([r.mean() for r in runs])
    sem = np.array([r.std(ddof=1) / np.sqrt(len(r)) for r in runs])

    popt, pcov = curve_fit(decay, m, mean, p0=initial_guess(m, mean), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    p, p_err = popt[1], perr[1]
    epc = error_per_clifford(p)
    return {
        "m": m,
        "mean": mean,
        "sem": sem,
        "params": dict(zip(("amp", "p", "offset"), popt)),
        "errors": dict(zip(("amp", "p", "offset"), perr)),
        "residuals": mean - decay(m, *popt),
        "epc": epc,
        "epc_error": p_err / 2,
        "fidelity": 1 - epc,
        "epg": epc / data["gates_per_clifford"],
    }


def print_table(data: dict, result: dict) -> None:
    """Print the fit, with the simulator's true values if known.

    Args:
        data: Output of ``execution.run``.
        result: Output of ``fit``.
    """
    truth = data["truth"]
    p, pe = result["params"]["p"], result["errors"]["p"]
    print(f"{'':<28} {'fit':>20} {'built in':>10}")
    built = (lambda x, f: f"{x:>10{f}}") if truth else (lambda x, f: f"{'-':>10}")
    print(
        f"{'decay rate p':<28} {p:>12.5f} +/- {pe:<6.5f} {built(truth['p'] if truth else 0, '.5f')}"
    )
    print(
        f"{'error per Clifford r':<28} {result['epc']:>12.5f} +/- "
        f"{result['epc_error']:<6.5f} {built(truth['epc'] if truth else 0, '.5f')}"
    )
    print(f"{'Clifford fidelity 1 - r':<28} {result['fidelity']:>12.5f}")
    print(
        f"{'error per gate':<28} {result['epg']:>12.5f} "
        f"({data['gates_per_clifford']:.2f} gates per Clifford)"
    )
    print(
        f"{'A, B':<28} {result['params']['amp']:>12.3f}, "
        f"{result['params']['offset']:.3f}"
    )


def plot_fit(
    data: dict,
    result: dict,
    ax=None,
    res_ax=None,
    label=None,
    color=COLOR,
    show_sequences=True,
):
    """Draw the sequences, their mean, the fit, and the residuals.

    Args:
        data: Output of ``execution.run``.
        result: Output of ``fit``.
        ax: Axes for the data. If None, a new figure with residuals is made.
        res_ax: Axes for the residuals, used with ``ax``.
        label: Legend text of the mean, used when the sequences are not drawn.
        color: Color of the mean and the fit.
        show_sequences: Draw every sequence as a gray dot.

    Returns:
        The figure.
    """
    if ax is None:
        fig, (ax, res_ax) = plt.subplots(
            2,
            1,
            figsize=(7, 5.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
        )
    else:
        fig = ax.figure
    m, p = result["m"], result["params"]
    if show_sequences:
        for k, runs in data["survival"].items():
            ax.plot([k] * len(runs), runs, ".", color="gray", alpha=0.35, ms=4)
        ax.plot([], [], ".", color="gray", alpha=0.6, label="one sequence")
    ax.errorbar(
        m,
        result["mean"],
        result["sem"],
        fmt="o",
        color=color,
        ms=5,
        label="mean" if show_sequences else label,
    )
    fine = np.linspace(0, m.max(), 300)
    ax.plot(
        fine,
        decay(fine, p["amp"], p["p"], p["offset"]),
        "-",
        color="black" if show_sequences else color,
        lw=1.5,
        label=f"fit, $p$ = {p['p']:.4f}" if show_sequences else None,
    )
    ax.set_ylabel("survival probability")
    ax.legend()
    if res_ax is not None:
        res_ax.axhline(0, color="black", lw=0.8)
        res_ax.plot(m, result["residuals"], "o", color=color, ms=4)
        res_ax.set_xlabel("sequence length (Cliffords)")
        res_ax.set_ylabel("data $-$ fit")
    return fig


def plot_spam(runs: dict):
    """One curve per read-out error, on the same axes.

    Args:
        runs: Dict {label: (data, result)}.

    Returns:
        The figure.
    """
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (label, (data, result)), color in zip(runs.items(), colors):
        plot_fit(
            data,
            result,
            ax=ax,
            label=f"{label}: $p$ = {result['params']['p']:.4f}",
            color=color,
            show_sequences=False,
        )
    ax.set_xlabel("sequence length (Cliffords)")
    ax.set_title("Survival with two read-out errors")
    return fig


def analyze(data: dict, path=None) -> dict:
    """Fit, print the table, and draw the figure.

    Args:
        data: Output of ``execution.run``.
        path: Where to save the figure.

    Returns:
        The fit result.
    """
    result = fit(data)
    print_table(data, result)
    fig = plot_fit(data, result)
    fig.axes[0].set_title(f"Survival against sequence length, {data['backend']}")
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return result
