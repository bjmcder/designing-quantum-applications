"""Entry point for the zero-noise extrapolation example.

    python zero_noise_extrapolation/main.py --backend=ibm_rensselaer --qubits 0 1

Runs a two-qubit circuit whose ideal <ZZ> is known, at several noise factors,
extrapolates to zero noise with three models, and prints one table that
compares the unmitigated value, the mitigated values, and the ideal value. It
saves the data and a figure with the data, the fits, and the ideal value.

Without --backend the circuit runs on the Aer simulator with a depolarizing
error on every CX gate (ops.SIM_CX_ERROR). The measurement is repeated
--repeats times, so the table shows the spread of each estimate as well as its
mean. On hardware the IBM Runtime Estimator folds the gates and extrapolates
(options.resilience.zne_mitigation), the measurement runs once, and the table
shows that one result. The hardware path has not been tested.
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze
from execution import run
from ops import FACTORS, N_CX

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    """Parse the command line, run the measurement, and save the results.

    Returns:
        Exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", help="IBM backend name (default: Aer simulator).")
    parser.add_argument(
        "--qubits",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Two connected physical qubits on hardware.",
    )
    parser.add_argument(
        "--n-cx", type=int, default=N_CX, help="CX gates in the circuit (even)."
    )
    parser.add_argument(
        "--factors", type=float, nargs="+", default=list(FACTORS), help="Noise factors."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=200,
        help="Repeats of the measurement (simulator).",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.02,
        help="Target standard error of each expectation value.",
    )
    args = parser.parse_args()

    data = run(
        (args.n_cx,),
        args.factors,
        args.repeats,
        args.precision,
        args.backend,
        args.qubits,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = f"zne_{data['backend']}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(data, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
