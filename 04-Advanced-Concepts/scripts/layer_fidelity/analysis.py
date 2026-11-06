"""Analysis: fit each link, multiply the fidelities, and read off the layer fidelity.

For each link the mean survival probability at each length is fitted with

    F(m) = A p^m + B

using least squares (``scipy.optimize.curve_fit``). The decay rate p gives the
process fidelity of the link, PF = p + (1 - p) / 16. The layer fidelity is
the product of the link fidelities, and the error per layered gate follows
from it (see ops.py).

Prints a table of the links and the layer, and draws two figures:

- one panel per link with its sequences, their mean, and the fit
- the process fidelity of each link, and the layer fidelity of the first k
  links as the chain grows

On the simulator the table also shows the values that follow from the noise
that was built in.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from ops import decay, eplg, initial_guess, layer_fidelity, process_fidelity
from scipy.optimize import curve_fit

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def link_names(data: dict) -> list[str]:
    """Links of the chain in order, as names like ``"0-1"``."""
    return sorted(data["survival"], key=lambda name: int(name.split("-")[0]))


def fit_link(data: dict, name: str) -> dict:
    """Fit the survival probability of one link.

    Args:
        data: Output of ``execution.run``.
        name: Link name such as ``"0-1"``.

    Returns:
        Dict with ``m``, ``mean``, ``sem`` (standard error of the mean),
        ``params`` (amp, p, offset), ``errors``, ``residuals``, ``pf`` (process
        fidelity), and ``pf_error``.
    """
    m = np.array(data["lengths"], dtype=float)
    runs = [np.array(data["survival"][name][int(k)]) for k in m]
    mean = np.array([r.mean() for r in runs])
    sem = np.array([r.std(ddof=1) / np.sqrt(len(r)) for r in runs])
    popt, pcov = curve_fit(decay, m, mean, p0=initial_guess(m, mean), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return {
        "m": m,
        "mean": mean,
        "sem": sem,
        "params": dict(zip(("amp", "p", "offset"), popt)),
        "errors": dict(zip(("amp", "p", "offset"), perr)),
        "residuals": mean - decay(m, *popt),
        "pf": process_fidelity(popt[1]),
        "pf_error": perr[1] * (1 - 1 / 16),  # d(PF)/dp = 1 - 1/d^2
    }


def fit_layer(data: dict) -> dict:
    """Fit every link and combine them.

    Args:
        data: Output of ``execution.run``.

    Returns:
        Dict with ``links`` (fit of each link in chain order), ``lf``,
        ``lf_error``, ``eplg``, and ``eplg_error``.
    """
    names = link_names(data)
    links = {name: fit_link(data, name) for name in names}
    pf = np.array([links[n]["pf"] for n in names])
    pf_err = np.array([links[n]["pf_error"] for n in names])
    lf = layer_fidelity(pf)
    # Independent errors: the relative errors add in quadrature.
    lf_err = lf * float(np.sqrt(np.sum((pf_err / pf) ** 2)))
    n = len(names)
    value = eplg(lf, n)
    # d(EPLG)/d(LF) = -(4/5) (1/n) LF^(1/n - 1)
    value_err = 0.8 / n * lf ** (1 / n - 1) * lf_err
    return {"links": links, "lf": lf, "lf_error": lf_err, "eplg": value, "eplg_error": value_err}


def print_table(data: dict, result: dict) -> None:
    """Print the process fidelity of each link and the layer fidelity.

    Args:
        data: Output of ``execution.run``.
        result: Output of ``fit_layer``.
    """
    truth = data["truth"]
    print(f"{'link':<8} {'decay rate p':>20} {'process fidelity':>22} {'built in':>9}")
    for name, link in result["links"].items():
        p, pe = link["params"]["p"], link["errors"]["p"]
        built = f"{truth[name]['pf']:>9.4f}" if truth else f"{'-':>9}"
        print(
            f"{name:<8} {p:>11.4f} +/- {pe:<6.4f} {link['pf']:>11.4f} +/- "
            f"{link['pf_error']:<6.4f} {built}"
        )
    built_lf = f"{truth['lf']:.4f}" if truth else "-"
    built_eplg = f"{truth['eplg']:.4f}" if truth else "-"
    print(f"\nlayer fidelity LF = {result['lf']:.4f} +/- {result['lf_error']:.4f}   (built in: {built_lf})")
    print(f"EPLG              = {result['eplg']:.4f} +/- {result['eplg_error']:.4f}   (built in: {built_eplg})")


def plot_links(data: dict, result: dict):
    """One panel per link: every sequence, the mean, and the fit.

    Args:
        data: Output of ``execution.run``.
        result: Output of ``fit_layer``.

    Returns:
        The figure.
    """
    names = list(result["links"])
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.6 * len(names), 4), sharey=True,
        gridspec_kw={"wspace": 0.08}, squeeze=False,
    )
    for ax, name, color in zip(axes[0], names, COLORS):
        link = result["links"][name]
        for m, runs in data["survival"][name].items():
            ax.plot([m] * len(runs), runs, ".", color="gray", alpha=0.35, ms=4)
        ax.errorbar(link["m"], link["mean"], link["sem"], fmt="o", color=color, ms=5)
        fine = np.linspace(0, link["m"].max(), 300)
        p = link["params"]
        ax.plot(fine, decay(fine, p["amp"], p["p"], p["offset"]), "k-", lw=1.5)
        ax.set_title(f"Link {name}, $p$ = {p['p']:.4f}")
        ax.set_xlabel("sequence length (Cliffords)")
    axes[0, 0].set_ylabel("survival probability")
    return fig


def plot_layer(data: dict, result: dict):
    """Left: process fidelity of each link. Right: layer fidelity versus chain length.

    Args:
        data: Output of ``execution.run``.
        result: Output of ``fit_layer``.

    Returns:
        The figure.
    """
    names = list(result["links"])
    pf = np.array([result["links"][n]["pf"] for n in names])
    pf_err = np.array([result["links"][n]["pf_error"] for n in names])
    truth = data["truth"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.3})
    x = np.arange(len(names))
    left.bar(x, pf, yerr=pf_err, color=COLORS[: len(names)], capsize=4)
    if truth:
        left.plot(
            x, [truth[n]["pf"] for n in names], "D", mfc="white", mec="black", ms=8, label="built in"
        )
        left.legend()
    left.set_xticks(x, names)
    left.set_ylim(max(0.0, pf.min() - 0.08), 1.0)
    left.set_xlabel("link")
    left.set_ylabel("process fidelity")
    left.set_title("Process fidelity of each link")

    k = np.arange(1, len(names) + 1)
    right.plot(k, np.cumprod(pf), "o-", color="black")
    if truth:
        right.plot(k, np.cumprod([truth[n]["pf"] for n in names]), "s--", color="gray", label="built in")
        right.legend()
    right.set_xticks(k)
    right.set_xlabel("number of links in the chain")
    right.set_ylabel("layer fidelity")
    right.set_title("Layer fidelity of the first $k$ links")
    return fig


def analyze(data: dict, path=None) -> dict:
    """Fit, print the table, and draw the figures.

    Args:
        data: Output of ``execution.run``.
        path: Where to save the second figure. The first goes next to it with
            ``_links`` added to the name.

    Returns:
        The result of ``fit_layer``.
    """
    result = fit_layer(data)
    print_table(data, result)
    fig1 = plot_links(data, result)
    fig2 = plot_layer(data, result)
    if path is not None:
        path = Path(path)
        fig1.savefig(path.with_name(path.stem + "_links.png"), dpi=150, bbox_inches="tight")
        fig2.savefig(path, dpi=150, bbox_inches="tight")
    return result
