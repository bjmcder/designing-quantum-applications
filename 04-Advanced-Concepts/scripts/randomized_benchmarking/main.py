"""Entry point for the randomized benchmarking example.

    python randomized_benchmarking/main.py --backend=ibm_rensselaer --qubit=0

Runs random sequences of single-qubit Clifford gates of several lengths,
fits the decay of the survival probability, and prints the error per
Clifford and the fidelity. It saves the data and a figure.

Without --backend the circuits run on the Aer simulator with the noise of
ops.py and execution.py, so the fit can be checked against the built-in
error. On hardware choose the longest sequence so that the survival
probability falls most of the way to its floor (0.5 for one qubit).
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze
from execution import run
from ops import LENGTHS

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    """Parse the command line, run the sequences, and save the results.

    Returns:
        Exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", help="IBM backend name (default: Aer simulator).")
    parser.add_argument("--qubit", type=int, default=0, help="Physical qubit on hardware.")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=list(LENGTHS), help="Sequence lengths."
    )
    parser.add_argument(
        "--sequences", type=int, default=20, help="Random sequences per length."
    )
    parser.add_argument("--shots", type=int, default=1000, help="Shots per sequence.")
    parser.add_argument("--seed", type=int, default=1, help="Seed of the random sequences.")
    args = parser.parse_args()

    data = run(
        args.lengths,
        args.sequences,
        args.shots,
        args.backend,
        args.qubit if args.backend else None,
        args.seed,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = f"rb_{data['backend']}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(data, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
