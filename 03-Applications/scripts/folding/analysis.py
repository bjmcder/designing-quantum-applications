"""Analysis: post-process the lattice folding results and plot them.

Decodes every sampled bitstring into a conformation, ranks the valid ones by
energy, and draws two panels in one image:

- the lowest-energy conformation found, drawn on its grid
- the energy spectrum: for each energy level, how often the circuit sampled
  it, next to how many reachable conformations exist at that level (found by
  exact enumeration)
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
from encoding import bits_from_key, decode_coordinate
from ops import enumerate_folds, fold_energy, is_valid_fold

# A color-blind-safe pair: orange for H beads and the best result, blue for
# P beads and everything else, gray for invalid samples.
ORANGE, BLUE, GRAY = "#D55E00", "#0072B2", "#999999"


def rank_conformations(data: dict) -> tuple[list[dict], float]:
    """Decode the counts and rank the valid conformations.

    Returns (ranked, p_invalid). ``ranked`` is sorted by energy, then by
    sampling probability. ``p_invalid`` is the probability of a bitstring
    that is not a valid fold.
    """
    seq, grid, enc = data["sequence"], tuple(data["grid"]), data["encoding"]
    shots = sum(data["counts"].values())
    ranked, p_invalid = [], 0.0
    for key, n in data["counts"].items():
        bits = bits_from_key(key)
        coords = decode_coordinate(bits, len(seq), grid, enc)
        if is_valid_fold(coords):
            ranked.append(
                {
                    "bits": bits,
                    "coords": coords,
                    "probability": n / shots,
                    "energy": fold_energy(coords, seq),
                }
            )
        else:
            p_invalid += n / shots
    ranked.sort(key=lambda r: (r["energy"], -r["probability"]))
    return ranked, p_invalid


def print_ranking(
    data: dict, ranked: list[dict], p_invalid: float, top: int
) -> None:
    """Print the run summary and the table of the best ``top`` folds."""
    rows, cols = data["grid"]
    print(
        f"\n{data['sequence']} on a {rows}x{cols} grid, "
        f"{data['encoding']} encoding, {data['backend']}, "
        f"{data['shots']} shots"
    )
    print(f"Invalid samples: {p_invalid:.1%}")
    print(f"Distinct valid conformations sampled: {len(ranked)}\n")
    if not ranked:
        print("No valid conformation was sampled.")
        print("Try more --shots, --starts or --reps.")
        return
    width = len(ranked[0]["bits"])
    print(f"{'rank':>4}  {'energy':>6}  {'prob':>7}  {'bits':<{width}}  path")
    for k, r in enumerate(ranked[:top], start=1):
        path = " ".join(f"({x},{y})" for x, y in r["coords"])
        energy = r["energy"] + 0  # + 0 turns -0.0 into 0.0
        print(
            f"{k:>4}  {energy:>+6.0f}  {r['probability']:>7.2%}  "
            f"{r['bits']}  {path}"
        )
    if len(ranked) > top:
        print(f"... {len(ranked) - top} more (use --top to show more)")


def draw_fold(ax, coords, seq, grid, title) -> None:
    """Draw a fold: the grid sites, the chain, and the beads (H orange)."""
    gx, gy = zip(*[(x, y) for x in range(grid[0]) for y in range(grid[1])])
    ax.scatter(gx, gy, s=60, color="#dddddd", zorder=0)
    xs, ys = zip(*coords)
    ax.plot(xs, ys, color="black", lw=2, zorder=1)
    colors = [ORANGE if b == "H" else BLUE for b in seq]
    ax.scatter(xs, ys, s=500, zorder=2, c=colors)
    for i, (x, y) in enumerate(coords):
        label = f"{seq[i]}{i}"
        ax.annotate(
            label, (x, y), ha="center", va="center", color="white", zorder=3
        )
    ax.set(title=title, aspect="equal", xticks=[], yticks=[])
    ax.margins(0.2)


def plot_results(
    data: dict, ranked: list[dict], p_invalid: float, out: Path
) -> None:
    """Save the best fold and the energy spectrum to ``out``."""
    seq, grid = data["sequence"], tuple(data["grid"])

    # Exact count of folds at each energy. The encodings put bead 0 on an
    # even site, so only placements that start there are reachable.
    exact = Counter(
        fold_energy(c, seq)
        for c in enumerate_folds(seq, grid)
        if sum(c[0]) % 2 == 0
    )
    levels = sorted(exact)
    sampled = {
        e: sum(r["probability"] for r in ranked if r["energy"] == e)
        for e in levels
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )
    if ranked:
        best = ranked[0]
        title = f"Best sampled conformation, E = {best['energy'] + 0:+.0f}"
        draw_fold(ax1, best["coords"], seq, grid, title)
    else:
        ax1.set(title="No valid conformation sampled", xticks=[], yticks=[])

    labels = [f"{e + 0:+.0f}\n({exact[e]} reachable)" for e in levels]
    heights = [sampled[e] for e in levels]
    colors = [ORANGE if e == levels[0] else BLUE for e in levels]
    ax2.bar(labels + ["invalid"], heights + [p_invalid], color=colors + [GRAY])
    ax2.set(
        title="Energy spectrum (orange = lowest level)",
        xlabel="conformation energy",
        ylabel="sampled probability",
    )
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


def analyze(data: dict, out: Path, top: int = 10) -> None:
    """Rank the samples, print the table, and save the figure."""
    ranked, p_invalid = rank_conformations(data)
    print_ranking(data, ranked, p_invalid, top)
    plot_results(data, ranked, p_invalid, out)
