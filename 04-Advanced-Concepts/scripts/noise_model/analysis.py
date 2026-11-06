"""Analysis: compare the Kraus noise model with the formulas.

For each test circuit we have three curves of the same expectation value:

- ideal      the noiseless circuit. It never changes, because id does nothing.
- noise model  the simulation with the Kraus operators attached to id.
- formula    the decay curve from the benchmarking example, with the same
             T1 and T2.

If the Kraus operators are right, the noise model sits on the formula.
The bottom row of the figure shows their difference, which should be
at the level of rounding error.
"""

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from execution import expectation
from ops import T1, T2, expected

OBSERVABLE = {"t1": "Z", "t2": "X"}
TITLE = {
    "t1": "T1 circuit: X, then idle",
    "t2": "T2 circuit: H, then idle",
}
COLORS = {"t1": "#0072B2", "t2": "#009E73"}


def curves(name: str, data: dict, t1: float = T1, t2: float = T2) -> dict:
    """The three curves for one test circuit.

    Args:
        name: "t1" or "t2".
        data: Output of ``execution.run`` for that circuit.
        t1: Relaxation time used in the noise model.
        t2: Dephasing time used in the noise model.

    Returns:
        Dict of arrays: ``t``, ``model``, ``ideal``, ``formula``.
    """
    obs = OBSERVABLE[name]
    t = np.array(data["times_us"])
    return {
        "t": t,
        "model": np.array([expectation(r, obs) for r in data["rho"]]),
        "ideal": np.array([expectation(r, obs) for r in data["rho_ideal"]]),
        "formula": expected(name, t, t1, t2),
    }


def print_table(results: dict, t1: float = T1, t2: float = T2) -> None:
    """Print the model against the formula at a few waits.

    Args:
        results: Dict of curves by circuit name, from ``curves``.
        t1: Relaxation time used in the noise model.
        t2: Dephasing time used in the noise model.
    """
    print(f"T1 = {t1:g} us, T2 = {t2:g} us")
    print(f"{'circuit':8} {'t (us)':>8} {'ideal':>9} {'model':>9} {'formula':>9}")
    for name, c in results.items():
        for i in np.linspace(0, len(c["t"]) - 1, 5).astype(int):
            print(
                f"{name:8} {c['t'][i]:8.1f} {c['ideal'][i]:9.4f} "
                f"{c['model'][i]:9.4f} {c['formula'][i]:9.4f}"
            )
        worst = np.abs(c["model"] - c["formula"]).max()
        print(f"{name:8} largest |model - formula| = {worst:.1e}")


def plot_compare(results: dict, path=None):
    """Figure: noise model against formula, and their difference.

    Args:
        results: Dict of curves by circuit name, from ``curves``.
        path: Where to save the figure, or None to only return it.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(
        2,
        len(results),
        figsize=(5.5 * len(results), 6),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1, "wspace": 0.3},
        squeeze=False,
    )
    for col, (name, c) in enumerate(results.items()):
        top, bottom = axes[0, col], axes[1, col]
        top.plot(c["t"], c["ideal"], "--", color="gray", label="ideal (no noise)")
        top.plot(c["t"], c["formula"], "-", color="black", lw=2, label="formula")
        top.plot(
            c["t"],
            c["model"],
            "o",
            color=COLORS[name],
            ms=5,
            label="Kraus noise model",
        )
        top.set_title(TITLE[name])
        top.set_ylabel(f"$\\langle {OBSERVABLE[name]} \\rangle$")
        top.legend()
        bottom.axhline(0, color="gray", lw=1)
        bottom.plot(c["t"], c["model"] - c["formula"], "o", color=COLORS[name])
        bottom.set_xlabel("wait ($\\mu$s)")
        bottom.set_ylabel("model $-$ formula")
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def analyze(data: dict, path=None, t1: float = T1, t2: float = T2) -> dict:
    """Make the curves, print the table, and draw the figure.

    Args:
        data: Dict of ``execution.run`` outputs by circuit name.
        path: Where to save the figure.
        t1: Relaxation time used in the noise model.
        t2: Dephasing time used in the noise model.

    Returns:
        Dict of curves by circuit name.
    """
    results = {name: curves(name, d, t1, t2) for name, d in data.items()}
    print_table(results, t1, t2)
    plot_compare(results, path)
    return results
