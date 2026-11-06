"""Entry point for the noise-model example.

    python noise_model/main.py --t1=100 --t2=70

Builds Kraus operators from T1 and T2, attaches them to the identity gate
of an Aer noise model, runs two simple circuits (prepare |1> and wait,
prepare |+> and wait), and compares the result with the decay formulas
fitted in the benchmarking example. Saves the data and a figure.

Use the T1 and T2 you measured with benchmarking/main.py. The defaults are
the values built into that example's simulator.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from analysis import analyze
from execution import run
from ops import STEP, T1, T2

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    """Parse the command line, run the circuits, and save the results.

    Returns:
        Exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--t1", type=float, default=T1, help="T1 in microseconds.")
    parser.add_argument("--t2", type=float, default=T2, help="T2 in microseconds.")
    parser.add_argument(
        "--step", type=float, default=STEP, help="Duration of one idle step (us)."
    )
    parser.add_argument(
        "--max-steps", type=int, default=300, help="Longest wait, in steps."
    )
    parser.add_argument(
        "--points", type=int, default=31, help="Number of waits to simulate."
    )
    args = parser.parse_args()

    steps = np.unique(np.linspace(0, args.max_steps, args.points).astype(int))
    data = {
        name: run(name, steps, args.step, args.t1, args.t2) for name in ("t1", "t2")
    }

    RESULTS.mkdir(exist_ok=True)
    analyze(data, RESULTS / "noise_model.png", args.t1, args.t2)
    saved = {
        name: {
            "times_us": d["times_us"],
            "rho_real": np.real(d["rho"]).tolist(),
            "rho_imag": np.imag(d["rho"]).tolist(),
        }
        for name, d in data.items()
    }
    (RESULTS / "noise_model.json").write_text(json.dumps(saved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
