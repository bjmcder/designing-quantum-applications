"""Entry point for the nuclei example.

    python nuclei/main.py --nucleus=4He --orbitals=2 --backend=ibm_xxx

Finds the ground-state energy of a small nucleus with a variational quantum
eigensolver (VQE) for every combination of fermion-to-qubit mapping and
ansatz, the trial circuit. It prints one table comparing them and saves a
figure with the convergence curves and the final errors.

The nucleus is a name from ops.NUCLEI (2H, 3H, 3He, 4He, 2n) or "Z,N" for the
numbers of protons and neutrons. Without --backend the circuits run on the
Aer simulator with exact energies (set --shots for sampling noise). Hardware
runs every pair in turn, which takes many jobs; select a few pairs with
--mappings and --ansatzes first.
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import analyze
from encoding import ANSATZES
from execution import run
from ops import MAPPINGS, nucleus_name, parse_nucleus, reference_occupation

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    """Parse the command line, run every pair, and save the results.

    Returns:
        Exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nucleus", default="2H", help="Name such as 4He, or 'Z,N'.")
    parser.add_argument(
        "--orbitals",
        type=int,
        default=2,
        help="Oscillator states per kind of nucleon (4 times this many qubits).",
    )
    parser.add_argument("--mappings", nargs="+", choices=MAPPINGS, default=list(MAPPINGS))
    parser.add_argument("--ansatzes", nargs="+", choices=ANSATZES, default=list(ANSATZES))
    parser.add_argument("--backend", help="IBM backend name (default: Aer simulator).")
    parser.add_argument("--reps", type=int, default=2, help="Layers of the HEA ansatz.")
    parser.add_argument("--starts", type=int, default=3, help="Random restarts of the optimizer.")
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
        z, n = parse_nucleus(args.nucleus)
        reference_occupation(z, n, args.orbitals)
    except ValueError as err:
        parser.error(str(err))
    if args.backend is not None and args.shots == 0:
        args.shots = 4096  # hardware always samples

    data = run(
        z, n, args.orbitals, args.mappings, args.ansatzes, args.backend,
        args.reps, args.starts, args.shots, args.maxiter, args.seed,
    )  # fmt: skip

    RESULTS.mkdir(exist_ok=True)
    stem = f"{nucleus_name(z, n)}_{args.orbitals}orbitals_{data['backend']}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(data, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
