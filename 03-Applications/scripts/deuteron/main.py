"""Entry point for the deuteron example.

    python deuteron/main.py --states=8 --backend=ibm_xxx

Finds the ground-state energy of the deuteron with a variational quantum
eigensolver (VQE) for every combination of qubit mapping (the form of the
Hamiltonian) and ansatz (the trial circuit). It prints one table comparing them and saves
a figure with the convergence curves and the final errors.

Without --backend the circuits run on the Aer simulator with exact energies
(set --shots for sampling noise). Hardware runs every pair in turn, which
takes many jobs; select a few pairs with --mappings and --ansatzes first.
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze
from encoding import ANSATZES
from execution import run
from ops import MAPPINGS, check_states

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--states",
        type=int,
        default=8,
        help="Oscillator states in the basis (a power of two).",
    )
    parser.add_argument(
        "--mappings", nargs="+", choices=MAPPINGS, default=list(MAPPINGS)
    )
    parser.add_argument(
        "--ansatzes", nargs="+", choices=ANSATZES, default=list(ANSATZES)
    )
    parser.add_argument(
        "--backend", help="IBM backend name (default: Aer simulator)."
    )
    parser.add_argument(
        "--reps", type=int, default=2, help="Layers of the HEA ansatzes."
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=3,
        help="Random restarts of the optimizer.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Shots per energy (default 0: exact on the simulator).",
    )
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    try:
        check_states(args.states)
    except ValueError as err:
        parser.error(str(err))
    if args.backend is not None and args.shots == 0:
        args.shots = 4096  # hardware always samples

    data = run(
        args.states,
        args.mappings,
        args.ansatzes,
        args.backend,
        args.reps,
        args.starts,
        args.shots,
        args.maxiter,
        args.seed,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = f"deuteron_{args.states}states_{data['backend']}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(data, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
