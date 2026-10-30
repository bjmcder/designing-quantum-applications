"""Analysis: post-process the traveling salesman results and plot them.

Decodes every sampled bitstring into a tour, ranks the valid ones by cost,
and draws two panels in one image:

- the cheapest tour found, drawn as a loop over the cities
- the tour spectrum: every possible tour sorted by cost, with the probability
  the circuit sampled it, plus the probability of an invalid sample
"""

from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; notebooks switch back to inline
import matplotlib.pyplot as plt
import numpy as np
from encoding import bits_from_key, decode_tour, tour_length

# A color-blind-safe pair: orange marks the start city and the cheapest
# tours, blue is everything else, and gray is invalid samples.
ORANGE, BLUE, GRAY = "#D55E00", "#0072B2", "#999999"


def all_tours(n: int) -> list[list[int]]:
    """Every tour starting at city 0 (a reversed tour is listed separately)."""
    return [[0, *p] for p in permutations(range(1, n))]


def rank_tours(data: dict) -> tuple[list[dict], float]:
    """Decode the counts and rank the valid tours.

    Returns (ranked, p_invalid). ``ranked`` is sorted by cost, then by
    sampling probability. ``p_invalid`` is the probability of a bitstring
    that is not a tour.
    """
    dist = np.array(data["costs"])
    shots = sum(data["counts"].values())
    ranked, p_invalid = [], 0.0
    for key, n in data["counts"].items():
        bits = bits_from_key(key)
        tour = decode_tour(bits, len(dist))
        if tour is None:
            p_invalid += n / shots
        else:
            ranked.append(
                {
                    "bits": bits,
                    "tour": tour,
                    "probability": n / shots,
                    "length": tour_length(tour, dist),
                }
            )
    ranked.sort(key=lambda r: (r["length"], -r["probability"]))
    return ranked, p_invalid


def tour_names(tour: list[int], names: list[str]) -> str:
    """Write a tour with city names, for example "A -> B -> C -> A"."""
    return " -> ".join(names[c] for c in [*tour, tour[0]])


def print_ranking(
    data: dict, ranked: list[dict], p_invalid: float, top: int
) -> None:
    """Print the run summary and the table of the cheapest ``top`` tours."""
    print(
        f"\n{len(data['cities'])} cities, {data['backend']}, "
        f"{data['shots']} shots"
    )
    print(f"Invalid samples: {p_invalid:.1%}")
    print(f"Distinct valid tours sampled: {len(ranked)}\n")
    if not ranked:
        print("No valid tour was sampled.")
        print("Try more --shots, --starts or --reps.")
        return
    print(f"{'rank':>4}  {'cost':>7}  {'prob':>7}  tour")
    for k, r in enumerate(ranked[:top], start=1):
        route = tour_names(r["tour"], data["cities"])
        print(f"{k:>4}  {r['length']:>7.2f}  {r['probability']:>7.2%}  {route}")
    if len(ranked) > top:
        print(f"... {len(ranked) - top} more (use --top to show more)")


def draw_tour(ax, tour: list[int], names: list[str], dist, title: str) -> None:
    """Draw the cities on a circle and the tour as labeled arrows.

    Each arrow carries the cost of that leg. The start city is orange.
    """
    angles = 2 * np.pi * np.arange(len(names)) / len(names) + np.pi / 2
    pos = np.c_[np.cos(angles), np.sin(angles)]
    arrow = {"arrowstyle": "-|>", "lw": 2, "color": "black"}
    arrow |= {"shrinkA": 32, "shrinkB": 32}  # stop at the edge of the circles
    for a, b in zip(tour, [*tour[1:], tour[0]]):
        ax.annotate("", xy=pos[b], xytext=pos[a], zorder=1, arrowprops=arrow)
        middle = (pos[a] + pos[b]) / 2 * 1.08
        ax.text(
            *middle,
            f"{dist[a, b]:g}",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"fc": "white", "ec": "none", "pad": 1},
        )
    colors = [ORANGE if c == 0 else BLUE for c in range(len(names))]
    ax.scatter(*pos.T, s=3200, c=colors, zorder=2)
    for name, (x, y) in zip(names, pos):
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=3,
        )
    ax.set(title=title, aspect="equal", xticks=[], yticks=[])
    ax.margins(0.25)


def plot_results(
    data: dict, ranked: list[dict], p_invalid: float, out: Path
) -> None:
    """Save the cheapest tour and the tour spectrum to ``out``."""
    names, dist = data["cities"], np.array(data["costs"])
    tours = sorted(all_tours(len(names)), key=lambda t: tour_length(t, dist))
    lengths = [tour_length(t, dist) for t in tours]
    probs = {tuple(r["tour"]): r["probability"] for r in ranked}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )
    if ranked:
        title = f"Cheapest sampled tour, cost {ranked[0]['length']:g}"
        draw_tour(ax1, ranked[0]["tour"], names, dist, title)
    else:
        ax1.set(title="No valid tour sampled", xticks=[], yticks=[])

    # With few tours, label each bar with its city order. With many, the
    # labels would overlap, so use the cost instead.
    few = len(tours) <= 12
    labels = [
        "-".join(map(str, t)) if few else f"{c:g}"
        for t, c in zip(tours, lengths)
    ]
    colors = [ORANGE if abs(c - lengths[0]) < 1e-9 else BLUE for c in lengths]
    heights = [probs.get(tuple(t), 0.0) for t in tours]
    ax2.bar(labels + ["invalid"], heights + [p_invalid], color=colors + [GRAY])
    ax2.set(
        title="Tour spectrum (orange = cheapest)",
        xlabel="tour (cities in visiting order), sorted by cost"
        if few
        else "tour cost",
        ylabel="sampled probability",
    )
    ax2.tick_params(axis="x", rotation=60, labelsize=8)
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


def analyze(data: dict, out: Path, top: int = 10) -> None:
    """Rank the samples, print the table, and save the figure."""
    ranked, p_invalid = rank_tours(data)
    print_ranking(data, ranked, p_invalid, top)
    plot_results(data, ranked, p_invalid, out)
