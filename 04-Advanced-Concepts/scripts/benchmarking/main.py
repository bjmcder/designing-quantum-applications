"""Entry point for the benchmarking example.

    python benchmarking/main.py --backend=ibm_rensselaer --qubit=0

Measures the three coherence times of a qubit, T1, T2* (Ramsey), and T2
(Hahn echo), fits the decay curve of each, and prints one table with the
fitted times. It saves the data and a figure with every curve and its fit.

Without --backend the circuits run on the Aer simulator with the noise of
ops.py, so the fits can be checked against the times that were built in.
On hardware the device decides the times, so choose the longest delays with
--t1-max, --t2star-max, and --t2-max (microseconds) to cover about three decay
times. If the curve has barely started to decay, the fit cannot tell.
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze
from execution import run
from ops import DEFAULT_T_MAX, EXPERIMENTS

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    """Parse the command line, run the experiments, and save the results.

    Returns:
        Exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS)
    )
    parser.add_argument("--backend", help="IBM backend name (default: Aer simulator).")
    parser.add_argument(
        "--qubit", type=int, default=0, help="Physical qubit on hardware."
    )
    parser.add_argument("--points", type=int, default=41, help="Delays per experiment.")
    parser.add_argument(
        "--precision",
        type=float,
        default=0.02,
        help="Target standard error of each expectation value.",
    )
    for name in EXPERIMENTS:
        parser.add_argument(
            f"--{name}-max",
            type=float,
            default=DEFAULT_T_MAX[name],
            help=f"Longest {name} delay in microseconds.",
        )
    args = parser.parse_args()

    t_max = {name: getattr(args, f"{name}_max") for name in EXPERIMENTS}
    data = run(
        args.experiments,
        t_max,
        args.points,
        args.backend,
        args.qubit,
        args.precision,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = f"coherence_{data['backend']}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(data, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
