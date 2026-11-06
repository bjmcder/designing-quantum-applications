"""Entry point for the dynamical decoupling example.

    python dynamical_decoupling/main.py --backend=ibm_rensselaer --qubit=0

Measures how long a qubit in |+> keeps its phase during an idle time, with no
pulses, with one pulse (the Hahn echo), and with XY4 pulses. Fits the decay
of each curve, prints the decay times, and sweeps the number of pulses at one
idle time to show that more pulses stop helping. It saves the data and two
figures.

Without --backend the circuits run on the Aer simulator with the noise of
ops.py, so the fits can be checked against the noise model. --pulse-error sets
the error of each pulse there. With --backend the device supplies the noise.
Add --runtime-dd to also run the service's own dynamical decoupling
(Estimator option dynamical_decoupling.enable, sequence types XpXm and XY4).
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze, analyze_sweep
from execution import run, run_sweep
from ops import DEFAULT_T_MAX, PULSE_ERROR

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    """Parse the command line, run the experiments, and save the results.

    Returns:
        Exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", help="IBM backend name (default: Aer simulator).")
    parser.add_argument(
        "--qubit", type=int, default=0, help="Physical qubit on hardware."
    )
    parser.add_argument("--points", type=int, default=41, help="Idle times per curve.")
    parser.add_argument(
        "--t-max",
        type=float,
        default=DEFAULT_T_MAX,
        help="Longest idle time in microseconds.",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.02,
        help="Target standard error of each expectation value.",
    )
    parser.add_argument(
        "--pulse-error",
        type=float,
        default=PULSE_ERROR,
        help="Depolarizing probability of one pulse (simulator only).",
    )
    parser.add_argument(
        "--runtime-dd",
        action="store_true",
        help="Hardware only: also run the runtime's dynamical decoupling.",
    )
    parser.add_argument(
        "--sweep-time",
        type=float,
        default=50.0,
        help="Idle time of the pulse-count sweep in microseconds.",
    )
    args = parser.parse_args()

    data = run(
        t_max=args.t_max,
        points=args.points,
        backend_name=args.backend,
        qubit=args.qubit,
        precision=args.precision,
        pulse_error=args.pulse_error,
        runtime_dd=args.runtime_dd,
    )
    sweep = run_sweep(
        total=args.sweep_time,
        backend_name=args.backend,
        qubit=args.qubit,
        precision=args.precision,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = f"dd_{data['backend']}"
    saved = {"curves": data, "sweep": sweep}
    (RESULTS / f"{stem}.json").write_text(json.dumps(saved, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    print()
    analyze_sweep(sweep, RESULTS / f"{stem}_sweep.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
