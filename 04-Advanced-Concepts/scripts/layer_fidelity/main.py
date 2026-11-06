"""Entry point for the layer fidelity example.

    python layer_fidelity/main.py --backend=ibm_rensselaer --chain 0 1 2 3

Runs randomized benchmarking on every link of a chain of qubits, with the
links of each layer running at the same time. Fits each link, multiplies the
process fidelities into the layer fidelity, and prints it with the error per
layered gate. It saves the data and two figures.

--chain gives physical qubits in order, and consecutive qubits must be
connected on the device. Without --backend the circuits run on the Aer
simulator, with the noise of ops.py, so the fit can be checked against the
built-in values.
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
    """Parse the command line, run the layers, and save the results.

    Returns:
        Exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", help="IBM backend name (default: Aer simulator).")
    parser.add_argument(
        "--chain", type=int, nargs="+", help="Physical qubits of the chain, in order."
    )
    parser.add_argument(
        "--n-qubits", type=int, default=4, help="Chain length on the simulator."
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(LENGTHS),
        help="Sequence lengths.",
    )
    parser.add_argument(
        "--sequences", type=int, default=20, help="Random sequences per length."
    )
    parser.add_argument("--shots", type=int, default=500, help="Shots per circuit.")
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed of the random sequences."
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Run each link alone instead of together with the rest of its layer.",
    )
    args = parser.parse_args()

    if args.backend and not args.chain:
        parser.error("--chain is required with --backend")
    n_qubits = len(args.chain) if args.chain else args.n_qubits

    data = run(
        n_qubits,
        args.lengths,
        args.sequences,
        args.shots,
        args.backend,
        args.chain,
        args.seed,
        simultaneous=not args.isolated,
    )

    RESULTS.mkdir(exist_ok=True)
    stem = f"layer_{data['backend']}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(data, indent=2))
    analyze(data, RESULTS / f"{stem}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
