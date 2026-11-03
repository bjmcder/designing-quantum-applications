"""Analysis: compare the ansatzes and mappings and plot them.

Prints one table with a row per (qubit mapping, ansatz) pair and draws:

- convergence: the energy error of the best restart against the number of
  energy evaluations, one panel per qubit mapping
- final error: the energy error each pair reached, against chemical accuracy

The error is the energy minus the exact ground-state energy of the truncated
Hamiltonian. The
variational principle says it cannot be negative, apart from shot noise.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from encoding import LABELS

TOLERANCE = 1e-2  # MeV (10 keV), half a percent of the binding energy
FLOOR = 1e-9  # errors below this are drawn at the floor
YMIN = 1e-10  # bottom of the axes, so floor-level bars stay visible

# Color-blind-safe colors, one per ansatz (used in every panel).
COLORS = {
    "ref": "#999999",
    "rot": "#0072B2",
    "hea": "#D55E00",
    "hea_ref": "#E69F00",
}
MAPPING_NAMES = {"binary": "Binary", "gray": "Gray", "unary": "Unary"}


def mapping_label(rows: list[dict], mapping: str) -> str:
    """Axis label: the mapping name and the qubits it needs."""
    qubits = next(r["qubits"] for r in rows if r["mapping"] == mapping)
    return f"{MAPPING_NAMES[mapping]}\n({qubits} qubits)"


def errors(data: dict) -> list[dict]:
    """Add the error against the exact energy to every run record."""
    exact = data["exact"]["exact"]
    return [{**r, "error": r["energy"] - exact} for r in data["runs"]]


def print_table(data: dict, rows: list[dict]) -> None:
    """Print the run summary and one line per (mapping, ansatz) pair."""
    ex = data["exact"]
    shots = f"{data['shots']} shots" if data["shots"] else "exact energies"
    print(f"\nDeuteron in {data['n_states']} oscillator states on {data['backend']}, {shots}")
    print(f"Exact energy in this basis:  {ex['exact']:.6f} MeV")
    print(f"Reference |0> energy:        {ex['ref']:.6f} MeV")
    print(f"Infinite-basis energy:       {ex['converged']:.6f} MeV")
    print(f"Measured binding energy:     {ex['experiment']:.6f} MeV\n")
    head = f"{'mapping':>8} {'ansatz':>7} {'qubits':>6} {'terms':>5} "
    head += f"{'angles':>6} {'CX':>4} {'depth':>5} {'evals':>6} "
    print(head + f"{'error (keV)':>12}")
    for r in rows:
        evals = sum(len(h) for h in r["history"])
        ok = "*" if abs(r["error"]) < TOLERANCE else " "
        print(
            f"{r['mapping']:>8} {r['ansatz']:>7} {r['qubits']:>6} "
            f"{r['terms']:>5} {r['params']:>6} {r['two_qubit']:>4} "
            f"{r['depth']:>5} {evals:>6} {1e3 * r['error']:>11.3f}{ok}"
        )
    print("* = within 10 keV of the exact energy")


def best_trace(run: dict, exact: float) -> np.ndarray:
    """Running minimum of the error for the restart that ended lowest."""
    hist = min(run["history"], key=min)
    return np.minimum.accumulate(np.array(hist) - exact)


def plot_convergence(axes, data: dict, rows: list[dict]) -> None:
    """One panel per mapping: error against evaluations, a line per ansatz."""
    exact = data["exact"]["exact"]
    for ax, mapping in zip(axes, dict.fromkeys(r["mapping"] for r in rows)):
        for r in (r for r in rows if r["mapping"] == mapping):
            color = COLORS[r["ansatz"]]
            if r["params"] == 0:
                ax.axhline(
                    r["error"], color=color, ls=":", lw=2, label=LABELS["ref"]
                )
                continue
            trace = np.maximum(best_trace(r, exact), FLOOR)
            ax.plot(
                range(1, len(trace) + 1),
                trace,
                color=color,
                lw=2,
                label=LABELS[r["ansatz"]],
            )
        ax.axhline(TOLERANCE, color="black", ls="--", lw=1)
        ax.set(
            title=mapping_label(rows, mapping),
            xlabel="energy evaluation",
            yscale="log",
            ylim=(YMIN, 10),
        )
    axes[0].set_ylabel("energy error (MeV)")
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
    ax.axhline(TOLERANCE, color="black", ls="--", lw=1)
    ax.text(
        len(mappings) - 0.5,
        TOLERANCE * 1.3,
        "10 keV",
        ha="right",
        fontsize=8,
    )
    ax.set(
        title="Final energy error (best of the restarts)",
        yscale="log",
        ylim=(YMIN, 10),
        ylabel="energy error (MeV)",
        xticks=range(len(mappings)),
        xticklabels=[mapping_label(rows, m) for m in mappings],
    )
    ax.legend(fontsize=8, ncols=len(names), loc="upper center")


def plot_results(data: dict, rows: list[dict], out: Path) -> None:
    """Save the convergence panels and the final-error bars to ``out``."""
    n = len(dict.fromkeys(r["mapping"] for r in rows))
    fig = plt.figure(figsize=(4.2 * n, 8), constrained_layout=True)
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
