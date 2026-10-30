"""Entry point for the traveling salesman example.

    python optimization/main.py cities.json --backend=ibm_xxx

The input is a JSON file with city names and a table of travel costs (see
example_cities.json). The script runs QAOA to find a cheap tour that visits
every city once and returns to the first city. It prints the sampled tours
ranked by cost and saves a figure with the best tour and the tour spectrum.

Without --backend the circuit runs on the Aer simulator. A problem with n
cities needs (n - 1)^2 qubits, so the simulator handles up to 5 cities.
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze
from encoding import n_qubits
from execution import run
from ops import load_problem

RESULTS = Path(__file__).parent / "results"
MAX_SIM_QUBITS = 22


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "cities", type=Path, help="JSON file with 'cities' and 'costs'."
    )
    parser.add_argument(
        "--backend", help="IBM backend name (default: Aer simulator)."
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
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--top", type=int, default=10, help="Tours to list.")
    args = parser.parse_args()

    try:
        names, dist = load_problem(args.cities)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(f"could not read {args.cities}: {exc}")

    qubits = n_qubits(len(names))
    print(f"{len(names)} cities ({', '.join(names)}) need {qubits} qubits")
    if args.backend is None and qubits > MAX_SIM_QUBITS:
        parser.error(
            f"{qubits} qubits is too many for the simulator; "
            "use fewer cities or --backend"
        )

    data = run(
        names,
        dist,
        args.backend,
        args.reps,
        args.starts,
        args.shots,
        args.maxiter,
        args.seed,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"{args.cities.stem}_{data['backend']}"
    stem.with_suffix(".json").write_text(json.dumps(data, indent=2))
    analyze(data, stem.with_suffix(".png"), args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
