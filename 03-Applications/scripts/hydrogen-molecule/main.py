"""Entry point for the hydrogen molecule example.

    python hydrogen-molecule/main.py --distance=0.735 --backend=ibm_xxx

Finds the ground-state energy of H2 with a variational quantum eigensolver
(VQE) for every combination of qubit mapping (the form of the Hamiltonian)
and ansatz (the trial circuit). It prints one table comparing them and saves
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
from ops import MAPPINGS

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--distance",
        type=float,
        default=0.735,
        help="H-H bond length in angstrom (equilibrium is about 0.735).",
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
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.distance <= 0:
        parser.error("--distance must be positive")
    if args.backend is not None and args.shots == 0:
        args.shots = 4096  # hardware always samples

    data = run(
        args.distance,
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
    # No dots in the stem: with_suffix would treat ".735" as an extension.
    stem = f"h2_{args.distance:g}A_{data['backend']}".replace(".", "p")
    (RESULTS / f"{stem}.json").write_text(json.dumps(data, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
