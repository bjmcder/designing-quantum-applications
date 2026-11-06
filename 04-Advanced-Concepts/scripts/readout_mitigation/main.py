"""Entry point for the readout mitigation example.

    python readout_mitigation/main.py --backend=ibm_rensselaer --qubit=0

Measures the read-out error of a qubit with two calibration circuits, sweeps
the state RY(theta)|0> so that the true <Z> runs from +1 to -1, and corrects
the measured <Z> in two ways: by inverting the confusion matrix, and by
twirling the measurement (a random bit flip before the measurement, undone in
the counts) followed by a rescaling. Prints the read-out error and the bias
of each method, and saves the data and a figure.

Without --backend the circuits run on the Aer simulator with the read-out
error of ops.py, so the corrections can be checked against the truth. With a
backend the Estimator with IBM's measurement twirling and mitigation runs as
well, and its result is added to the table.
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze
from execution import run

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    """Parse the command line, run the circuits, and save the results.

    Returns:
        Exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", help="IBM backend name (default: Aer simulator).")
    parser.add_argument(
        "--qubit", type=int, default=0, help="Physical qubit on hardware."
    )
    parser.add_argument(
        "--points", type=int, default=13, help="RY angles in the sweep."
    )
    parser.add_argument("--shots", type=int, default=4000, help="Shots per angle.")
    parser.add_argument(
        "--randomizations",
        type=int,
        default=20,
        help="Randomizations R per twirled run.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed of the coin flips."
    )
    args = parser.parse_args()

    data = run(
        args.points,
        args.shots,
        args.randomizations,
        args.backend,
        args.qubit if args.backend else None,
        args.seed,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = f"readout_{data['backend']}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(data, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
