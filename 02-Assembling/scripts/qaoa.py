#!/usr/bin/env python3
"""Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut.

The problem
-----------
Given a graph, split its nodes into two groups so that as many edges as
possible connect one group to the other. An edge that crosses between the
groups is "cut". This is Max-Cut. Checking every split costs 2^n, and no
known classical algorithm solves Max-Cut exactly in time that grows only
polynomially with the graph size. QAOA is a quantum heuristic that looks for
a good (not always perfect) split.

From a graph to a Hamiltonian
-----------------------------
Give each node a qubit. Qubit i in |0> or |1> means node i is in group 0 or
group 1. The edge (i, j) is cut when the two qubits differ, which is when
Z_i Z_j = -1. So the number of cut edges is

    cut(z) = sum over edges of (1 - Z_i Z_j) / 2.

QAOA minimizes an energy, so we use the cost Hamiltonian H_C = -cut:

    H_C = sum over edges of (Z_i Z_j - 1) / 2.

H_C is diagonal: every bitstring is an eigenstate whose energy is minus its cut
size. The ground state is therefore the best cut, and E_0 = -(max cut). Unlike
the VQE example, we are not after the energy itself: we want the *bitstring*.

How it works
------------
QAOA is a variational algorithm (see vqa.py) with a fixed, problem-shaped
ansatz. Start in the uniform superposition |+>^n, then repeat p times:

1. Cost layer exp(-i gamma H_C). Gives each bitstring a phase proportional to
   its energy. On hardware it is one RZZ(gamma) gate per edge.
2. Mixer layer exp(-i beta sum_i X_i). Moves amplitude between bitstrings. It
   is one RX(2 beta) gate per qubit.

The 2p angles (gamma_1..gamma_p, beta_1..beta_p) are the variational
parameters. A classical optimizer tunes them to minimize <H_C>, exactly as in
VQE. Then we *sample* the final circuit: the bitstrings it returns are
candidate cuts, and we keep the best one.

Loop (quantum on the left, classical on the right):

    (gamma, beta) --> [ H, then p x (cost, mixer) ] --> [ measure <H_C> ]
          ^                                                    |
          +--------------- optimizer updates angles <----------+

          best angles --> [ same circuit + measure all ] --> bitstrings

Things to keep in mind
----------------------
- p sets the quality. At p = 0 the circuit only makes uniform noise, whose
  average cut is half the edges. Raising p can only help in the ideal case (as
  p grows, QAOA can approach the exact answer), but each layer adds
  two-qubit gates, which are the noisiest operations on hardware.
- The approximation ratio <cut> / max cut measures how good the average
  sample is. The probability of sampling an optimal cut measures how good the
  best sample is. They are different: you only need the optimum to appear
  among your shots.
- The angle landscape has many local minima, so the starting point matters.
  The script runs the optimizer from several random starts and keeps the best.
- The mixer and cost layers do not commute, so the circuit cannot be
  simplified into a single step.

Execution modes on IBM hardware
-------------------------------
The optimizer loop is a chain of dependent Estimator jobs, so on hardware it
belongs in a session (a single wait in the queue). Job mode queues every
evaluation separately. See vqa.py for the full comparison, including batch
mode. The final sampling step is one extra Sampler job inside the same mode.

Run with no arguments for the Aer simulator. Use --mode job or session to run
on hardware (default backend ibm_rensselaer).
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from _common import connect, make_parser
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime import Session
from scipy.optimize import minimize


OPTIMIZATION_LEVEL = 3

# Each graph is (number of nodes, list of edges).
GRAPHS = {
    # A square: bipartite, so every edge can be cut (max cut 4).
    "ring": (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    # A square with a triangular roof: the triangle can never be fully cut,
    # so the best cut leaves one edge uncut (max cut 5 of 6).
    "house": (5, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)]),
}


def maxcut_hamiltonian(num_nodes: int, edges: list[tuple[int, int]]) -> SparsePauliOp:
    """Cost Hamiltonian H_C = sum over edges of (Z_i Z_j - 1) / 2.

    The constant -1/2 per edge is the identity term. Its eigenvalue for a
    bitstring is minus the number of cut edges.
    """
    terms = []
    for i, j in edges:
        label = ["I"] * num_nodes
        label[i] = label[j] = "Z"
        # Qiskit labels put qubit 0 on the right, so reverse the string.
        terms.append(("".join(reversed(label)), 0.5))
    terms.append(("I" * num_nodes, -0.5 * len(edges)))
    return SparsePauliOp.from_list(terms)


def cut_value(bits: str, edges: list[tuple[int, int]]) -> int:
    """Number of edges cut by a bitstring (node i is the i-th bit from the right)."""
    group = bits[::-1]
    return sum(group[i] != group[j] for i, j in edges)


def exact_max_cut(num_nodes: int, edges: list[tuple[int, int]]) -> int:
    """Largest cut by brute force over all 2^n splits (fine for a few nodes).

    This is the reference QAOA is trying to reach. The loop is exactly the
    exponential cost that QAOA hopes to avoid for large graphs.
    """
    return max(
        cut_value("".join(map(str, bits)), edges)
        for bits in itertools.product([0, 1], repeat=num_nodes)
    )


def qaoa_circuit(
    num_nodes: int, edges: list[tuple[int, int]], reps: int
) -> QuantumCircuit:
    """Build the QAOA circuit with ``reps`` (the p) cost/mixer layers.

    Has ``2 * reps`` parameters. The first ``reps`` are the cost angles gamma,
    the last ``reps`` are the mixer angles beta. No measurements are added
    here, because the Estimator measures H_C itself.
    """
    theta = ParameterVector("theta", 2 * reps)
    gamma, beta = theta[:reps], theta[reps:]
    circuit = QuantumCircuit(num_nodes, name=f"qaoa_p{reps}")
    circuit.h(range(num_nodes))
    for layer in range(reps):
        circuit.barrier()
        for i, j in edges:
            # RZZ(g) = exp(-i g Z Z / 2), which matches exp(-i gamma H_C)
            # up to a global phase when H_C has coefficient 1/2 on Z Z.
            circuit.rzz(gamma[layer], i, j)
        for qubit in range(num_nodes):
            # RX(2 beta) = exp(-i beta X)
            circuit.rx(2 * beta[layer], qubit)
    return circuit


def transpile_problem(
    circuit: QuantumCircuit, hamiltonian: SparsePauliOp, pass_manager
) -> tuple[QuantumCircuit, SparsePauliOp]:
    """Transpile the circuit and remap the Hamiltonian to match.

    The Hamiltonian acts on *logical* qubits, so it must be mapped onto the
    same physical qubits the transpiler chose, or the Estimator would measure
    the wrong qubits. The angles survive transpiling, so this only needs to
    be done once, not once per optimizer step.
    """
    isa_circuit = pass_manager.run(circuit)
    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)
    print(
        f"  {circuit.name}: depth {isa_circuit.depth()}, "
        f"{isa_circuit.num_nonlocal_gates()} two-qubit gates"
    )
    return isa_circuit, isa_hamiltonian


def make_primitives(backend, session: Session | None = None):
    """Return (estimator, sampler) that run on Aer or on IBM hardware.

    Both kinds are called the same way. ``session`` is an open Runtime
    ``Session`` on ``backend``; with None, every job is queued independently
    ("job mode"). It is ignored for Aer.
    """
    if isinstance(backend, AerSimulator):
        return AerEstimator(), AerSampler()
    mode = session if session is not None else backend
    return RuntimeEstimator(mode=mode), RuntimeSampler(mode=mode)


def energy(estimator, circuit, hamiltonian, values, shots: int) -> float:
    """Submit one job and return <H_C> for the angles ``values``."""
    # The Estimator takes a target standard error, not a shot count. The error
    # on a +/-1 observable after N shots is about 1/sqrt(N).
    precision = 1 / np.sqrt(shots)
    job = estimator.run([(circuit, hamiltonian, values, precision)])
    return float(job.result()[0].data.evs)


def run_qaoa(estimator, circuit, hamiltonian, x0, shots: int, maxiter: int) -> dict:
    """Minimize <H_C>(angles) with COBYLA and record every energy it saw.

    Each evaluation is one Estimator job, and the next angles depend on the
    last result, which is why hardware runs belong in a session.

    Returns
    -------
    dict
        ``theta`` (best angles found), ``history`` (energy per evaluation),
        and ``seconds`` (wall-clock time of the whole loop).
    """
    history: list[float] = []

    def cost(theta):
        history.append(energy(estimator, circuit, hamiltonian, theta, shots))
        return history[-1]

    start = time.perf_counter()
    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return {
        "theta": result.x,
        "history": history,
        "seconds": time.perf_counter() - start,
    }


def best_of_starts(
    estimator, circuit, hamiltonian, reps, rng, shots: int, maxiter: int, starts: int
) -> dict:
    """Run the optimizer from several random starts and keep the lowest energy.

    The angle landscape has several valleys, so one start can settle in a poor
    one. Cost angles gamma come first, then mixer angles beta. The returned
    dict is the best run's, plus ``start_energies``: the best energy of every
    start, so you can see how much the starting point mattered.
    """
    runs = []
    for k in range(starts):
        x0 = np.concatenate(
            [rng.uniform(0, np.pi, reps), rng.uniform(0, np.pi / 2, reps)]
        )
        runs.append(run_qaoa(estimator, circuit, hamiltonian, x0, shots, maxiter))
        print(f"  start {k}: best <H_C> = {min(runs[-1]['history']):+.4f}")
    best = min(runs, key=lambda r: min(r["history"]))
    return {**best, "start_energies": [min(r["history"]) for r in runs]}


def sample_counts(sampler, circuit, theta, shots: int, pass_manager) -> dict:
    """Fix the angles, measure every qubit, and return the bitstring counts.

    The circuit is measured and transpiled again here, because the Estimator
    version has no measurements.
    """
    measured = circuit.assign_parameters(theta)
    measured.measure_all()
    job = sampler.run([pass_manager.run(measured)], shots=shots)
    return job.result()[0].data.meas.get_counts()


def summarize_counts(counts: dict, edges: list, max_cut: int) -> dict:
    """Reduce counts to the numbers that show how good the samples are."""
    shots = sum(counts.values())
    cuts = {bits: cut_value(bits, edges) for bits in counts}
    mean_cut = sum(n * cuts[b] for b, n in counts.items()) / shots
    optimal = sum(n for b, n in counts.items() if cuts[b] == max_cut) / shots
    best_bits = max(counts, key=lambda b: (cuts[b], counts[b]))
    return {
        "mean_cut": mean_cut,
        "approx_ratio": mean_cut / max_cut,
        "p_optimal": optimal,
        "best_bits": best_bits,
        "best_cut": cuts[best_bits],
    }


def convergence_summary(history: list[float], max_cut: int) -> str:
    """Describe the best energy the loop saw against the exact ``-max_cut``.

    A shallow QAOA circuit is not expected to reach E_0, so the gap is the
    quality measure here, not a pass/fail test.
    """
    return f"best <H_C> = {min(history):+.4f} (exact E_0 = {-max_cut:+.4f})"


# Colour = number of layers p (colour-blind-safe pair).
REPS_COLOR = {1: "#0072B2", 2: "#D55E00"}


def save_results(outdir: Path, args, max_cut: int, records: dict) -> Path:
    """Write one run's results to JSON so runs can be plotted together later."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"qaoa_{args.graph}_{args.mode}.json"
    payload = {
        "graph": args.graph,
        "mode": args.mode,
        "backend": "aer_simulator" if args.mode == "aer" else args.backend,
        "shots": args.shots,
        "max_cut": max_cut,
        "layers": records,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {path}")
    return path


def plot_results(outdir: Path, show: bool = False) -> int:
    """Compare every saved run: convergence and the sampled cut distribution.

    Reads all ``qaoa_*.json`` files in ``outdir``, so simulator and hardware
    runs launched separately appear on the same figure.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.loads(f.read_text()) for f in sorted(outdir.glob("qaoa_*.json"))]
    if not runs:
        print(f"No qaoa_*.json files in {outdir}; run the script first.")
        return 1

    fig, axes = plt.subplots(
        2, len(runs), figsize=(6 * len(runs), 8), squeeze=False, constrained_layout=True
    )
    for col, run in enumerate(runs):
        title = f"{run['graph']}, {run['mode']}"
        ax = axes[0, col]
        ax.axhline(-run["max_cut"], color="black", lw=1, label="exact $E_0$")
        for reps, rec in run["layers"].items():
            ax.plot(
                range(1, len(rec["history"]) + 1),
                rec["history"],
                color=REPS_COLOR[int(reps)],
                lw=1.5,
                label=f"p = {reps}",
            )
        ax.set(
            title=f"{title}: convergence",
            xlabel="energy evaluation (one job each)",
            ylabel="$\\langle H_C \\rangle$ = $-$ mean cut",
        )
        ax.legend(fontsize=8)

        # Bottom row: how often each cut size shows up in the final samples.
        ax = axes[1, col]
        width = 0.38
        sizes = np.arange(run["max_cut"] + 1)
        for k, (reps, rec) in enumerate(run["layers"].items()):
            freq = np.zeros(len(sizes))
            for bits, n in rec["counts"].items():
                freq[rec["cuts"][bits]] += n
            ax.bar(
                sizes + (k - 0.5) * width,
                freq / freq.sum(),
                width,
                color=REPS_COLOR[int(reps)],
                label=f"p = {reps}",
            )
        ax.set(
            title=f"{title}: sampled cut sizes",
            xlabel="edges cut",
            ylabel="fraction of shots",
            xticks=sizes,
        )
        ax.legend(fontsize=8)

    out = outdir / "qaoa_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    if show:
        plt.show()
    return 0


def main() -> int:
    parser = make_parser("QAOA: Max-Cut on a small graph, comparing p = 1 and p = 2.")
    parser.add_argument(
        "--graph",
        choices=list(GRAPHS),
        default="ring",
        help="ring: every edge can be cut (default); house: one edge must stay uncut.",
    )
    parser.add_argument(
        "--mode",
        choices=["aer", "job", "session"],
        default="aer",
        help="Where/how to run (default: aer, the noiseless simulator).",
    )
    parser.add_argument("--maxiter", type=int, default=100, help="COBYLA iterations.")
    parser.add_argument(
        "--starts", type=int, default=3, help="Random starting angles to try."
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--outdir", type=Path, default=Path("qaoa_results"), help="Results folder."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Don't run anything; plot all saved results in --outdir and exit.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open the plot window."
    )
    args = parser.parse_args()
    if args.plot:
        return plot_results(args.outdir, args.show)

    num_nodes, edges = GRAPHS[args.graph]
    hamiltonian = maxcut_hamiltonian(num_nodes, edges)
    max_cut = exact_max_cut(num_nodes, edges)
    print(
        f"{args.graph} graph: {num_nodes} nodes, {len(edges)} edges, "
        f"max cut = {max_cut} (E_0 = {-max_cut:+.4f})"
    )

    rng = np.random.default_rng(args.seed)
    records: dict[int, dict] = {}

    session = None
    if args.mode == "aer":
        backend = AerSimulator()
    else:
        if args.skip_hardware:
            parser.error(f"--mode {args.mode} needs hardware; drop --skip-hardware")
        backend = connect(args.backend, args.name)
        if backend is None:
            return 0
        if args.mode == "session":
            session = Session(backend=backend)

    try:
        estimator, sampler = make_primitives(backend, session)
        pass_manager = generate_preset_pass_manager(
            backend=backend, optimization_level=OPTIMIZATION_LEVEL
        )
        for reps in (1, 2):
            circuit = qaoa_circuit(num_nodes, edges, reps)
            print(f"\n--- p = {reps}: {circuit.num_parameters} parameters ---")
            isa_circuit, isa_hamiltonian = transpile_problem(
                circuit, hamiltonian, pass_manager
            )
            out = best_of_starts(
                estimator,
                isa_circuit,
                isa_hamiltonian,
                reps,
                rng,
                args.shots,
                args.maxiter,
                args.starts,
            )
            print(
                f"best start: {len(out['history'])} jobs in {out['seconds']:.1f} s; "
                + convergence_summary(out["history"], max_cut)
            )

            counts = sample_counts(
                sampler, circuit, out["theta"], args.shots, pass_manager
            )
            stats = summarize_counts(counts, edges, max_cut)
            print(
                f"sampled mean cut {stats['mean_cut']:.3f} "
                f"(approximation ratio {stats['approx_ratio']:.3f}), "
                f"P(optimal cut) = {stats['p_optimal']:.3f}, "
                f"best sample {stats['best_bits']} cuts {stats['best_cut']} edges"
            )
            records[reps] = {
                "parameters": circuit.num_parameters,
                "history": out["history"],
                "seconds": out["seconds"],
                "start_energies": out["start_energies"],
                "theta": out["theta"].tolist(),
                "counts": counts,
                "cuts": {b: cut_value(b, edges) for b in counts},
                **stats,
            }
    finally:
        if session is not None:
            session.close()
    save_results(args.outdir, args, max_cut, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
