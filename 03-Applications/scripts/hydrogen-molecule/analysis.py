"""Analysis: compare the ansatzes and Hamiltonians and plot them.

Prints one table with a row per (qubit mapping, ansatz) pair and draws:

- convergence: the energy error of the best restart against the number of
  energy evaluations, one panel per qubit mapping
- final error: the energy error each pair reached, against chemical accuracy

The error is the energy minus the exact ground-state energy (FCI). The
variational principle says it cannot be negative, apart from shot noise.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from encoding import LABELS

CHEMICAL_ACCURACY = 1.6e-3  # hartree (about 1 kcal/mol)
FLOOR = 1e-11  # errors below this are drawn at the floor
YMIN = 1e-12  # bottom of the axes, so floor-level bars stay visible

# Color-blind-safe colors, one per ansatz (used in every panel).
COLORS = {
    "hf": "#999999",
    "uccd": "#0072B2",
    "uccsd": "#56B4E9",
    "hea": "#D55E00",
    "hea_hf": "#E69F00",
}
MAPPING_LABELS = {
    "jw": "Jordan-Wigner\n(4 qubits)",
    "parity": "Parity\n(4 qubits)",
    "bk": "Bravyi-Kitaev\n(4 qubits)",
    "parity2": "Parity, reduced\n(2 qubits)",
}


def errors(data: dict) -> list[dict]:
    """Add the error against the exact energy to every run record."""
    fci = data["exact"]["fci"]
    return [{**r, "error": r["energy"] - fci} for r in data["runs"]]


def print_table(data: dict, rows: list[dict]) -> None:
    """Print the run summary and one line per (mapping, ansatz) pair."""
    ex = data["exact"]
    shots = f"{data['shots']} shots" if data["shots"] else "exact energies"
    print(f"\nH2 at R = {data['distance']} A on {data['backend']}, {shots}")
    print(f"Exact (FCI) energy:  {ex['fci']:.6f} Ha")
    print(f"Hartree-Fock energy: {ex['hf']:.6f} Ha")
    print(f"Correlation energy:  {ex['fci'] - ex['hf']:+.6f} Ha\n")
    head = f"{'mapping':>8} {'ansatz':>7} {'qubits':>6} {'terms':>5} "
    head += f"{'angles':>6} {'CX':>4} {'depth':>5} {'evals':>6} "
    print(head + f"{'error (mHa)':>12}")
    for r in rows:
        evals = sum(len(h) for h in r["history"])
        ok = "*" if abs(r["error"]) < CHEMICAL_ACCURACY else " "
        print(
            f"{r['mapping']:>8} {r['ansatz']:>7} {r['qubits']:>6} "
            f"{r['terms']:>5} {r['params']:>6} {r['two_qubit']:>4} "
            f"{r['depth']:>5} {evals:>6} {1e3 * r['error']:>11.4f}{ok}"
        )
    print("* = within chemical accuracy (1.6 mHa)")


def best_trace(run: dict, fci: float) -> np.ndarray:
    """Running minimum of the error for the restart that ended lowest."""
    hist = min(run["history"], key=min)
    return np.minimum.accumulate(np.array(hist) - fci)


def plot_convergence(axes, data: dict, rows: list[dict]) -> None:
    """One panel per mapping: error against evaluations, a line per ansatz."""
    fci = data["exact"]["fci"]
    for ax, mapping in zip(axes, dict.fromkeys(r["mapping"] for r in rows)):
        for r in (r for r in rows if r["mapping"] == mapping):
            color = COLORS[r["ansatz"]]
            if r["params"] == 0:
                ax.axhline(
                    r["error"], color=color, ls=":", lw=2, label=LABELS["hf"]
                )
                continue
            trace = np.maximum(best_trace(r, fci), FLOOR)
            ax.plot(
                range(1, len(trace) + 1),
                trace,
                color=color,
                lw=2,
                label=LABELS[r["ansatz"]],
            )
        ax.axhline(CHEMICAL_ACCURACY, color="black", ls="--", lw=1)
        ax.set(
            title=MAPPING_LABELS[mapping],
            xlabel="energy evaluation",
            yscale="log",
            ylim=(YMIN, 1),
        )
    axes[0].set_ylabel("energy error (Ha)")
    axes[0].legend(fontsize=8, loc="lower left")


def plot_final(ax, data: dict, rows: list[dict]) -> None:
    """Grouped bars: final error for each mapping (group) and ansatz (color)."""
    mappings = list(dict.fromkeys(r["mapping"] for r in rows))
    names = list(dict.fromkeys(r["ansatz"] for r in rows))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        heights = []
        for mapping in mappings:
            r = next(
                r for r in rows if r["mapping"] == mapping and r["ansatz"] == name
            )
            heights.append(max(r["error"], FLOOR))
        x = np.arange(len(mappings)) + (k - (len(names) - 1) / 2) * width
        ax.bar(x, heights, width, color=COLORS[name], label=LABELS[name])
    ax.axhline(CHEMICAL_ACCURACY, color="black", ls="--", lw=1)
    ax.text(
        len(mappings) - 0.5,
        CHEMICAL_ACCURACY * 1.3,
        "chemical accuracy",
        ha="right",
        fontsize=8,
    )
    ax.set(
        title="Final energy error (best of the restarts)",
        yscale="log",
        ylim=(YMIN, 1),
        ylabel="energy error (Ha)",
        xticks=range(len(mappings)),
        xticklabels=[MAPPING_LABELS[m] for m in mappings],
    )
    ax.legend(fontsize=8, ncols=len(names), loc="upper center")


def plot_results(data: dict, rows: list[dict], out: Path) -> None:
    """Save the convergence panels and the final-error bars to ``out``."""
    n = len(dict.fromkeys(r["mapping"] for r in rows))
    fig = plt.figure(figsize=(3.4 * n, 8), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1])
    axes = top.subplots(1, n, sharey=True)
    plot_convergence(np.atleast_1d(axes), data, rows)
    plot_final(bottom.subplots(), data, rows)
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


def analyze(data: dict, out: Path) -> None:
    """Print the table and save the figure."""
    rows = errors(data)
    print_table(data, rows)
    plot_results(data, rows, out)
