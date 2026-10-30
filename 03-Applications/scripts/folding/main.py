"""Entry point for the lattice protein folding example.

    python folding/main.py "HPPHHP" --encoding=binary --backend=ibm_xxx

Runs QAOA on the HP lattice model for the given sequence of H (hydrophobic)
and P (polar) beads. It prints the sampled conformations ranked by energy and
saves a figure with the best conformation and the energy spectrum.

Without --backend the circuit runs on the Aer simulator. The grid defaults to
the smallest near-square grid that fits the chain (for example 2x3 for 6
beads). The number of qubits depends on the encoding, and the script prints
it before running.
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze
from encoding import ENCODINGS, coordinate_cardinality, n_qubits
from execution import run
from ops import auto_grid

RESULTS = Path(__file__).parent / "results"
MAX_SIM_QUBITS = 22


def parse_grid(text: str) -> tuple[int, int]:
    """Turn a command-line grid size such as "2x3" into (2, 3)."""
    a, b = text.lower().split("x")
    return int(a), int(b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sequence", help='Chain of H and P beads, for example "HPPHHP".'
    )
    parser.add_argument("--encoding", choices=ENCODINGS, default="binary")
    parser.add_argument(
        "--backend", help="IBM backend name (default: Aer simulator)."
    )
    parser.add_argument(
        "--grid",
        type=parse_grid,
        help="Lattice size such as 2x3 (default: automatic).",
    )
    parser.add_argument("--reps", type=int, default=2, help="QAOA layers.")
    parser.add_argument(
        "--starts",
        type=int,
        default=3,
        help="Random restarts of the optimizer.",
    )
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--top", type=int, default=10, help="Conformations to list."
    )
    args = parser.parse_args()

    sequence = args.sequence.upper()
    if len(sequence) < 4 or set(sequence) - {"H", "P"}:
        parser.error("sequence must be at least 4 beads, using only H and P")
    grid = args.grid or auto_grid(len(sequence))
    if grid[0] * grid[1] < len(sequence):
        parser.error(f"a {grid[0]}x{grid[1]} grid has fewer sites than beads")

    qubits = n_qubits(
        coordinate_cardinality(len(sequence), grid), args.encoding
    )
    print(
        f"{sequence}: {len(sequence)} beads on a {grid[0]}x{grid[1]} grid, "
        f"{args.encoding} encoding needs {qubits} qubits"
    )
    if args.backend is None and qubits > MAX_SIM_QUBITS:
        parser.error(
            f"{qubits} qubits is too many for the simulator; use a denser "
            "encoding, a smaller grid, or --backend"
        )

    data = run(
        sequence,
        grid,
        args.encoding,
        args.backend,
        args.reps,
        args.starts,
        args.shots,
        args.maxiter,
        args.seed,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"{sequence}_{args.encoding}_{data['backend']}"
    stem.with_suffix(".json").write_text(json.dumps(data, indent=2))
    analyze(data, stem.with_suffix(".png"), args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
