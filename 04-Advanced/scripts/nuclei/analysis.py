"""Analysis: compare the ansatzes and mappings and plot them.

Prints one table with a row per (fermion mapping, ansatz) pair and draws:

- convergence: the energy error of the best restart against the number of
  energy evaluations, one panel per mapping
- final error: the energy error each pair reached, against the 10 keV
  tolerance

The error is the energy minus the exact ground-state energy of the nucleus in
this basis. The variational principle says it cannot be negative, apart from
shot noise.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from encoding import LABELS
from matplotlib.ticker import NullLocator

TOLERANCE = 1e-2  # MeV (10 keV)
FLOOR = 1e-9  # errors below this are drawn at the floor
YMIN = 1e-10  # bottom of the axes, so floor-level bars stay visible

# Color-blind-safe colors, one per ansatz (used in every panel).
COLORS = {
    "ref": "#999999",
    "uccs": "#009E73",
    "uccsd": "#0072B2",
    "hea_ref": "#D55E00",
}
MAPPING_NAMES = {"jordan_wigner": "Jordan-Wigner", "parity": "Parity"}

# Color and line style of each curve in the nucleus scan.
SCAN_STYLE = {
    "exact": ("black", "-"),
    "ref": (COLORS["ref"], ":"),
    "uccs": (COLORS["uccs"], "-."),
    "uccsd": (COLORS["uccsd"], "--"),
}


def mapping_label(rows: list[dict], mapping: str) -> str:
    """Axis label with the mapping name and the qubits it needs.

    Args:
        rows: Run records with an error field, from ``errors``.
        mapping: Fermion mapping to label.

    Returns:
        Two lines of label text.
    """
    qubits = next(r["qubits"] for r in rows if r["mapping"] == mapping)
    return f"{MAPPING_NAMES[mapping]}\n{qubits} qubits"


def errors(data: dict) -> list[dict]:
    """Add the error against the exact energy to every run record.

    Args:
        data: Results from ``execution.run``.

    Returns:
        Copies of the run records with an ``error`` field in MeV.
    """
    exact = data["exact"]["exact"]
    return [{**r, "error": r["energy"] - exact} for r in data["runs"]]


def print_table(data: dict, rows: list[dict]) -> None:
    """Print the run summary and one line per (mapping, ansatz) pair.

    Args:
        data: Results from ``execution.run``.
        rows: Run records with an error field, from ``errors``.
    """
    ex = data["exact"]
    shots = f"{data['shots']} shots" if data["shots"] else "exact energies"
    print(f"\n{data['nucleus']} (Z={data['z']}, N={data['n']}) in {data['n_orb']} oscillator states "
          f"per nucleon on {data['backend']}, {shots}")
    print(f"Exact energy in this basis:   {ex['exact']:.6f} MeV ({ex['dimension']} states in the sector)")
    print(f"Reference energy:             {ex['ref']:.6f} MeV")
    print(f"Unbound nucleons:             {ex['free']:.6f} MeV")
    print(f"Binding energy in this model: {ex['free'] - ex['exact']:.3f} MeV\n")
    head = f"{'mapping':>13} {'ansatz':>7} {'qubits':>6} {'terms':>5} "
    head += f"{'angles':>6} {'CX':>5} {'depth':>6} {'evals':>6} "
    print(head + f"{'error (keV)':>12}")
    for r in rows:
        evals = sum(len(h) for h in r["history"])
        ok = "*" if abs(r["error"]) < TOLERANCE else " "
        print(
            f"{r['mapping']:>13} {r['ansatz']:>7} {r['qubits']:>6} "
            f"{r['terms']:>5} {r['params']:>6} {r['two_qubit']:>5} "
            f"{r['depth']:>6} {evals:>6} {1e3 * r['error']:>11.3f}{ok}"
        )
    print("* = within 10 keV of the exact energy")


def best_trace(run: dict, exact: float) -> np.ndarray:
    """Running minimum of the error for the restart that ended lowest.

    Args:
        run: One run record.
        exact: Exact ground-state energy in MeV.

    Returns:
        The lowest error so far after each evaluation, in MeV.
    """
    hist = min(run["history"], key=min)
    return np.minimum.accumulate(np.array(hist) - exact)


def plot_convergence(axes, data: dict, rows: list[dict]) -> None:
    """Plot the error against evaluations, one panel per mapping.

    Args:
        axes: One matplotlib axes per mapping.
        data: Results from ``execution.run``.
        rows: Run records with an error field, from ``errors``.
    """
    exact = data["exact"]["exact"]
    for ax, mapping in zip(axes, dict.fromkeys(r["mapping"] for r in rows)):
        for r in (r for r in rows if r["mapping"] == mapping):
            color = COLORS[r["ansatz"]]
            if r["params"] == 0:
                ax.axhline(r["error"], color=color, ls=":", lw=2, label=LABELS["ref"])
                continue
            trace = np.maximum(best_trace(r, exact), FLOOR)
            ax.plot(range(1, len(trace) + 1), trace, color=color, lw=2, label=LABELS[r["ansatz"]])
        ax.axhline(TOLERANCE, color="black", ls="--", lw=1)
        ax.set(title=mapping_label(rows, mapping), xlabel="energy evaluation", yscale="log", ylim=(YMIN, 1e3))
    axes[0].set_ylabel("energy error (MeV)")
    axes[0].legend(fontsize=8, loc="lower left")


def plot_final(ax, rows: list[dict]) -> None:
    """Plot the final error as grouped bars, a group per mapping.

    Args:
        ax: Matplotlib axes to draw on.
        rows: Run records with an error field, from ``errors``.
    """
    mappings = list(dict.fromkeys(r["mapping"] for r in rows))
    names = list(dict.fromkeys(r["ansatz"] for r in rows))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        heights = []
        for mapping in mappings:
            r = next(r for r in rows if r["mapping"] == mapping and r["ansatz"] == name)
            heights.append(max(r["error"], FLOOR))
        x = np.arange(len(mappings)) + (k - (len(names) - 1) / 2) * width
        ax.bar(x, heights, width, color=COLORS[name], label=LABELS[name])
    ax.axhline(TOLERANCE, color="black", ls="--", lw=1)
    ax.text(len(mappings) - 0.5, TOLERANCE * 1.3, "10 keV", ha="right", fontsize=8)
    ax.set(
        title="Final energy error, best restart",
        yscale="log",
        ylim=(YMIN, 1e3),
        ylabel="energy error (MeV)",
        xticks=range(len(mappings)),
        xticklabels=[mapping_label(rows, m) for m in mappings],
    )
    ax.legend(fontsize=8, ncols=len(names), loc="upper center")


def plot_results(data: dict, rows: list[dict], out: Path) -> None:
    """Save the convergence panels and the final-error bars.

    Args:
        data: Results from ``execution.run``.
        rows: Run records with an error field, from ``errors``.
        out: Path of the image file to write.
    """
    n = len(dict.fromkeys(r["mapping"] for r in rows))
    fig = plt.figure(figsize=(5 * max(n, 2), 8), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1])
    axes = top.subplots(1, n, sharey=True)
    plot_convergence(np.atleast_1d(axes), data, rows)
    plot_final(bottom.subplots(), rows)
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


def analyze(data: dict, out: Path) -> None:
    """Print the table and save the figure.

    Args:
        data: Results from ``execution.run``.
        out: Path of the image file to write.
    """
    rows = errors(data)
    print_table(data, rows)
    plot_results(data, rows, out)


def plot_nuclei_scan(names, curves, experiment=None):
    """Plot the binding energy and the VQE error for a set of nuclei.

    Args:
        names: Nucleus names in plotting order.
        curves: Binding energies in MeV for each nucleus, keyed "exact" and
            by ansatz. Binding energy is the energy of the unbound nucleons
            minus the energy found, so larger means more bound.
        experiment: Optional measured binding energies in MeV, one per name.

    Returns:
        The matplotlib figure.
    """
    fig, (energy_ax, error_ax) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(names))
    exact = np.array(curves["exact"])
    for key, values in curves.items():
        color, ls = SCAN_STYLE[key]
        label = "exact in this model" if key == "exact" else LABELS[key]
        style = {"color": color, "ls": ls, "lw": 2, "marker": "o", "label": label}
        energy_ax.plot(x, values, **style)
        if key != "exact":
            error_ax.plot(x, np.maximum(exact - np.array(values), FLOOR), **style)
    if experiment is not None:
        energy_ax.plot(x, experiment, "s", color="#E69F00", ms=8, label="measured")
    energy_ax.set(ylabel="binding energy (MeV)", title="Binding energy", xticks=x, xticklabels=names)
    energy_ax.legend()
    error_ax.axhline(TOLERANCE, color="black", ls="--", lw=1)
    error_ax.set(
        yscale="log",
        ylabel="error vs exact in model (MeV)",
        title="VQE error",
        xticks=x,
        xticklabels=names,
    )
    error_ax.legend()
    for ax in (energy_ax, error_ax):
        ax.xaxis.set_minor_locator(NullLocator())
    return fig
