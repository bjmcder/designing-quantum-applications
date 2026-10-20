#!/usr/bin/env python3
"""Variational quantum algorithm (VQA) built on the VQE idea.

The problem
-----------
Given a Hamiltonian H (a Hermitian matrix, written as a sum of Pauli strings),
find its lowest eigenvalue E_0, the "ground-state energy". Solving this system
exactly with exact diagonalization costs memory that grows as 2^n, which is
hopeless for large systems. The Variational Quantum Eigensolver (VQE) instead
lets a quantum computer do the part that is hard classically and a classical
optimizer do the rest.

The variational principle
-------------------------
For ANY normalized state |psi>, the energy expectation value is bounded
below by the ground-state energy:

    E(psi) = <psi|H|psi>  >=  E_0.

So minimizing E over a family of states gets us closer to E_0 from above, and
never below it (in the ideal, noiseless case). The better the family, the
tighter the bound.

How it works
------------
1. Ansatz: a parameterized circuit U(theta) acting on |0...0> prepares a
   trial state |psi(theta)>. The parameters theta are rotation angles.
2. Measure: write H = sum_k c_k P_k with Pauli strings P_k. The quantum
   computer estimates each <P_k> from repeated shots (the Estimator
   primitive does this), and we combine them as E(theta) = sum_k c_k <P_k>.
3. Optimize: a classical optimizer (COBYLA, SPSA, gradient descent, ...)
   proposes new angles theta to lower E(theta).
4. Repeat steps 2-3 until the energy stops improving.

Loop (quantum on the left, classical on the right):

    theta --> [ prepare U(theta)|0> ] --> [ measure <H> ] --> E(theta)
      ^                                                          |
      +------------- optimizer updates theta <-------------------+

Things to keep in mind
----------------------
- The ansatz sets the ceiling: if the true ground state is not reachable by
  U(theta), VQE converges to something above E_0.
- Energies are noisy (finite shots, hardware errors), so the optimizer must
  tolerate noise.
- The landscape E(theta) can have local minima and flat regions ("barren
  plateaus"), so the starting point and the ansatz matter.
- VQE is the template for a whole family of variational quantum algorithms
  (VQAs), such as QAOA: swap the cost function and the ansatz, keep the loop.

Execution modes on IBM hardware
-------------------------------
Every energy evaluation is a small Runtime job, and a VQE run needs dozens.
How those jobs are grouped decides how long you wait in the queue:

- job mode (no context): each evaluation is queued on its own. After every
  optimizer step the next job goes to the back of the queue, so a loop of
  50 steps can wait 50 times. Simple, but the slowest option for VQE.
- session: reserves the backend for you for the duration of the loop. Jobs
  inside it start one after another with no new queue wait, and the next one
  can depend on the last result. This is the right fit for an iterative
  algorithm like VQE.
- batch: sends a set of jobs that are all known IN ADVANCE and lets the
  service schedule them together, in parallel where it can. There is no
  exclusive access, so a loop where job k+1 needs the result of job k gains
  nothing. Use it for independent work, such as screening many random
  starting points or scanning the energy landscape.

Rule of thumb: dependent jobs (an optimizer loop) -> session; independent
jobs -> batch; a single one-off job -> neither.

Run with no arguments for the Aer simulator. Use --mode job, batch or
session to run on hardware (default backend ibm_rensselaer).
"""

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
from qiskit_ibm_runtime import Batch, Session
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from scipy.optimize import minimize

OPTIMIZATION_LEVEL = 3


def transpile_problem(
    ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, backend
) -> tuple[QuantumCircuit, SparsePauliOp]:
    """Transpile the ansatz for a backend and remap the Hamiltonian to match.

    Transpiling picks which physical qubits the circuit uses (the "layout")
    and rewrites gates into the backend's native gate set. The Hamiltonian
    acts on the *logical* qubits, so it must be mapped onto the same physical
    qubits, or the Estimator would measure the wrong qubits.

    Parameters
    ----------
    ansatz : QuantumCircuit
        Parameterized circuit. The parameters survive transpilation, so this
        only needs to be done once, not once per optimizer step.
    hamiltonian : SparsePauliOp
        Observable defined on the ansatz's logical qubits.
    backend
        An ``AerSimulator`` or an IBM backend returned by ``connect``.

    Returns
    -------
    tuple[QuantumCircuit, SparsePauliOp]
        The transpiled ansatz and the Hamiltonian expanded to the backend's
        full qubit count.
    """
    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=OPTIMIZATION_LEVEL
    )
    isa_ansatz = pass_manager.run(ansatz)
    isa_hamiltonian = hamiltonian.apply_layout(isa_ansatz.layout)
    print(
        f"Transpiled for {backend.name}: depth {isa_ansatz.depth()}, "
        f"{isa_ansatz.num_nonlocal_gates()} two-qubit gates"
    )
    return isa_ansatz, isa_hamiltonian


def make_estimator(backend, mode: Session | Batch | None = None):
    """Return an Estimator that runs on Aer or on IBM hardware.

    Parameters
    ----------
    backend
        An ``AerSimulator`` or an IBM backend.
    mode : Session or Batch, optional
        An open Runtime ``Session`` or ``Batch`` on ``backend``. With neither,
        every job is queued independently ("job mode"). Ignored for Aer.

    Returns
    -------
    Estimator
        Both kinds are called as ``estimator.run([(circuit, observable,
        parameter_values, precision)])``.
    """
    if isinstance(backend, AerSimulator):
        return AerEstimator()
    return RuntimeEstimator(mode=mode if mode is not None else backend)


def product_ansatz(num_qubits: int) -> QuantumCircuit:
    """Simplest ansatz: one independent RY rotation per qubit, no entanglement.

    Only ``num_qubits`` parameters and depth 1, so every optimizer step is
    cheap and the landscape is easy to search. The price is expressibility:
    a product state cannot represent entanglement, so it stalls above E_0
    whenever the true ground state is entangled.
    """
    theta = ParameterVector("theta", num_qubits)
    circuit = QuantumCircuit(num_qubits, name="product")
    for qubit in range(num_qubits):
        circuit.ry(theta[qubit], qubit)
    return circuit


def entangled_ansatz(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Hardware-efficient ansatz: RY/RZ rotation layers split by CX ladders.

    Has ``2 * num_qubits * (reps + 1)`` parameters. The CX ladder entangles
    neighbouring qubits, so it can reach ground states the product ansatz
    cannot. The price is a bigger search space for the optimizer (more
    iterations to converge) and extra two-qubit gates, which are the noisiest
    operations on hardware.
    """
    theta = ParameterVector("theta", 2 * num_qubits * (reps + 1))
    params = iter(theta)
    circuit = QuantumCircuit(num_qubits, name=f"entangled_reps{reps}")
    for layer in range(reps + 1):
        for qubit in range(num_qubits):
            circuit.ry(next(params), qubit)
            circuit.rz(next(params), qubit)
        if layer < reps:
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
    return circuit


def _chain_terms(num_qubits: int, pauli: str, coeff: float) -> list[tuple[str, float]]:
    """Return ``coeff * pauli_i pauli_(i+1)`` for each neighbouring pair."""
    terms = []
    for i in range(num_qubits - 1):
        label = ["I"] * num_qubits
        label[i] = label[i + 1] = pauli
        # Qiskit labels put qubit 0 on the right, so reverse the string.
        terms.append(("".join(reversed(label)), coeff))
    return terms


def _field_terms(num_qubits: int, pauli: str, coeff: float) -> list[tuple[str, float]]:
    """Return ``coeff * pauli_i`` for every qubit."""
    terms = []
    for i in range(num_qubits):
        label = ["I"] * num_qubits
        label[i] = pauli
        terms.append(("".join(reversed(label)), coeff))
    return terms


def ising_hamiltonian(num_qubits: int) -> SparsePauliOp:
    """Easy case: H = -sum X_i X_(i+1) - 0.5 sum X_i.

    Every term commutes with every other, so the ground state is simply every
    qubit in |+>. That is a *product* state: no entanglement is needed, and
    both ansatzes should reach the exact energy.
    """
    terms = _chain_terms(num_qubits, "X", -1.0) + _field_terms(num_qubits, "X", -0.5)
    return SparsePauliOp.from_list(terms)


def heisenberg_hamiltonian(num_qubits: int) -> SparsePauliOp:
    """Hard case: H = sum (X_i X_(i+1) + Y_i Y_(i+1) + Z_i Z_(i+1)).

    The antiferromagnetic Heisenberg chain. For two qubits the ground state is
    the entangled singlet (|01> - |10>)/sqrt(2) with E_0 = -3. No product state
    gets below -1, so the product ansatz must stall while the entangled one
    can converge.
    """
    terms = []
    for pauli in "XYZ":
        terms += _chain_terms(num_qubits, pauli, 1.0)
    return SparsePauliOp.from_list(terms)


def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """Lowest eigenvalue by exact diagonalization (fine for a few qubits).

    This is the reference VQE is trying to reach. It costs memory that grows
    as 4^n, which is exactly why VQE exists for larger systems.
    """
    return float(np.linalg.eigvalsh(hamiltonian.to_matrix())[0])


def energy(estimator, ansatz, hamiltonian, values, shots: int):
    """Submit one job and return the energy for ``values`` (blocks until done).

    ``values`` may be one parameter vector (returns a float) or an array of
    shape (k, num_parameters) (returns k energies from a single job).
    """
    # The Estimator takes a target standard error, not a shot count. The error
    # on a +/-1 observable after N shots is about 1/sqrt(N).
    precision = 1 / np.sqrt(shots)
    job = estimator.run([(ansatz, hamiltonian, values, precision)])
    return job.result()[0].data.evs


def run_vqe(estimator, ansatz, hamiltonian, x0, shots: int, maxiter: int) -> dict:
    """Minimize E(theta) with COBYLA and record every energy it saw.

    COBYLA needs no gradients, which suits noisy energies: each evaluation is
    one Estimator job, and the next theta depends on the last result.

    Returns
    -------
    dict
        ``energy`` (best found), ``theta``, ``history`` (energy per
        evaluation), and ``seconds`` (wall-clock time of the whole loop).
    """
    history: list[float] = []

    def cost(theta):
        history.append(float(energy(estimator, ansatz, hamiltonian, theta, shots)))
        return history[-1]

    start = time.perf_counter()
    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return {
        "energy": min(history),
        "theta": result.x,
        "history": history,
        "seconds": time.perf_counter() - start,
    }


def screen_starts(estimator, ansatz, hamiltonian, starts, shots: int) -> np.ndarray:
    """Evaluate many independent starting points, one job each, all at once.

    This is the batch-friendly pattern: every job is known up front, so all of
    them are submitted before any result is read and the service can run them
    in parallel. Returns the energies in the same order as ``starts``.
    """
    precision = 1 / np.sqrt(shots)
    jobs = [estimator.run([(ansatz, hamiltonian, x0, precision)]) for x0 in starts]
    return np.array([job.result()[0].data.evs for job in jobs], dtype=float)


def convergence_summary(history: list[float], target: float, tol: float = 0.05) -> str:
    """Describe how far and how fast the loop got toward ``target`` energy."""
    hit = next((i + 1 for i, e in enumerate(history) if e <= target + tol), None)
    reached = (
        f"within {tol} of E_0 after {hit} evaluations"
        if hit
        else (f"never within {tol} of E_0")
    )
    return f"best {min(history):+.4f}, {reached}"


MODES = ["aer", "job", "session", "batch"]
# Colour = ansatz (colour-blind-safe pair); line style = execution mode.
ANSATZ_COLOR = {"product": "#0072B2", "entangled": "#D55E00"}
MODE_STYLE = {"aer": "-", "job": ":", "session": "--", "batch": "-."}


def save_results(outdir: Path, args, target: float, records: dict) -> Path:
    """Write one run's results to JSON so runs can be plotted together later."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"vqa_{args.hamiltonian}_{args.qubits}q_{args.mode}.json"
    payload = {
        "hamiltonian": args.hamiltonian,
        "qubits": args.qubits,
        "mode": args.mode,
        "backend": "aer_simulator" if args.mode == "aer" else args.backend,
        "shots": args.shots,
        "exact_energy": target,
        "ansatzes": records,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {path}")
    return path


def plot_results(outdir: Path, show: bool = False) -> int:
    """Compare every saved run: convergence, wall-clock time and batch screens.

    Reads all ``vqa_*.json`` files in ``outdir``, so simulator and hardware
    runs launched separately (even in parallel) appear on the same figure.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = [json.loads(f.read_text()) for f in sorted(outdir.glob("vqa_*.json"))]
    if not runs:
        print(f"No vqa_*.json files in {outdir}; run the script first.")
        return 1

    hamiltonians = sorted({r["hamiltonian"] for r in runs})
    screens = [r for r in runs if r["mode"] == "batch"]
    fig = plt.figure(
        figsize=(6 * max(len(hamiltonians), 2), 9), constrained_layout=True
    )
    grid = fig.add_gridspec(2, max(len(hamiltonians), 2))

    # Top row: energy per evaluation, one panel per Hamiltonian.
    for col, ham in enumerate(hamiltonians):
        ax = fig.add_subplot(grid[0, col])
        group = [r for r in runs if r["hamiltonian"] == ham and r["mode"] != "batch"]
        ax.axhline(
            group[0]["exact_energy"] if group else runs[0]["exact_energy"],
            color="black",
            lw=1,
            label="exact $E_0$",
        )
        for r in group:
            for name, a in r["ansatzes"].items():
                ax.plot(
                    range(1, len(a["history"]) + 1),
                    a["history"],
                    color=ANSATZ_COLOR[name],
                    ls=MODE_STYLE[r["mode"]],
                    lw=1.5,
                )
        ax.set(
            title=f"{ham} chain: convergence",
            xlabel="energy evaluation (one job each)",
            ylabel="energy",
        )
        ax.legend(
            handles=[Line2D([], [], color="black", lw=1, label="exact $E_0$")]
            + [
                Line2D([], [], color=c, label=f"{n} ansatz")
                for n, c in ANSATZ_COLOR.items()
            ]
            + [
                Line2D([], [], color="gray", ls=MODE_STYLE[m], label=m)
                for m in MODES
                if m != "batch" and any(r["mode"] == m for r in group)
            ],
            fontsize=8,
        )

    # Bottom left: wall-clock time of the whole loop, grouped by mode.
    ax = fig.add_subplot(grid[1, 0])
    timed = [r for r in runs if r["mode"] != "batch"]
    labels = [f"{r['hamiltonian']}\n{r['mode']}" for r in timed]
    width = 0.38
    for k, name in enumerate(ANSATZ_COLOR):
        xs = np.arange(len(timed)) + (k - 0.5) * width
        ax.bar(
            xs,
            [r["ansatzes"][name]["seconds"] for r in timed],
            width,
            color=ANSATZ_COLOR[name],
            label=f"{name} ansatz",
        )
    ax.set_xticks(np.arange(len(timed)), labels, fontsize=8)
    ax.set(title="Wall-clock time of the optimizer loop", ylabel="seconds")
    ax.legend(fontsize=8)

    # Bottom right: best energy per ansatz, with batch-screened starts as dots.
    ax = fig.add_subplot(grid[1, 1])
    entries = [(r, n) for r in runs for n in ANSATZ_COLOR if r["mode"] != "batch"]
    ax.set_title("Final energy vs exact (gap above $E_0$)")
    labels = []
    for i, (r, name) in enumerate(entries):
        a = r["ansatzes"][name]
        ax.bar(i, min(a["history"]) - r["exact_energy"], color=ANSATZ_COLOR[name])
        labels.append(f"{r['hamiltonian'][:4]}\n{r['mode']}\n{name[:4]}")
    for r in screens:
        for name, a in r["ansatzes"].items():
            ax.scatter(
                [len(entries) - 0.5] * len(a["energies"]),
                np.array(a["energies"]) - r["exact_energy"],
                color=ANSATZ_COLOR[name],
                marker="x",
                zorder=3,
            )
    if screens:
        labels.append("batch\nstarts")
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=7)
    ax.set_ylabel("energy $-$ $E_0$ (lower is better)")

    out = outdir / "vqa_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    if show:
        plt.show()
    return 0


def main() -> int:
    parser = make_parser("Simple VQE: compare two ansatzes on a small spin chain.")
    parser.add_argument(
        "--hamiltonian",
        choices=["ising", "heisenberg"],
        default="heisenberg",
        help="ising: product ground state; heisenberg: entangled (default).",
    )
    parser.add_argument("--qubits", type=int, default=2, help="Chain length.")
    parser.add_argument(
        "--mode",
        choices=["aer", "job", "batch", "session"],
        default="aer",
        help="Where/how to run (default: aer, the noiseless simulator).",
    )
    parser.add_argument("--maxiter", type=int, default=150, help="COBYLA iterations.")
    parser.add_argument(
        "--starts", type=int, default=4, help="Random starts to screen in batch mode."
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--outdir", type=Path, default=Path("vqa_results"), help="Results folder."
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
    if args.qubits < 2:
        parser.error("need at least 2 qubits")

    hamiltonian = (
        ising_hamiltonian(args.qubits)
        if args.hamiltonian == "ising"
        else heisenberg_hamiltonian(args.qubits)
    )
    target = exact_ground_energy(hamiltonian)
    print(f"{args.hamiltonian} chain, {args.qubits} qubits, exact E_0 = {target:+.4f}")

    ansatzes = {
        "product": product_ansatz(args.qubits),
        "entangled": entangled_ansatz(args.qubits),
    }
    rng = np.random.default_rng(args.seed)
    records: dict[str, dict] = {}

    session = batch = None
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
        elif args.mode == "batch":
            batch = Batch(backend=backend)

    try:
        estimator = make_estimator(backend, session or batch)
        for name, ansatz in ansatzes.items():
            isa_ansatz, isa_hamiltonian = transpile_problem(
                ansatz, hamiltonian, backend
            )
            x0 = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
            print(f"\n--- {name} ansatz: {ansatz.num_parameters} parameters ---")

            if args.mode == "batch":
                # Independent jobs only: screen random starts, no optimizer loop.
                starts = rng.uniform(0, 2 * np.pi, (args.starts, ansatz.num_parameters))
                t0 = time.perf_counter()
                energies = screen_starts(
                    estimator, isa_ansatz, isa_hamiltonian, starts, args.shots
                )
                print(
                    f"{args.starts} independent jobs in "
                    f"{time.perf_counter() - t0:.1f} s; energies "
                    f"{np.round(energies, 3).tolist()}"
                )
                print(f"best start {energies.min():+.4f} (E_0 {target:+.4f})")
                records[name] = {
                    "parameters": ansatz.num_parameters,
                    "energies": energies.tolist(),
                    "seconds": time.perf_counter() - t0,
                }
                continue

            out = run_vqe(
                estimator, isa_ansatz, isa_hamiltonian, x0, args.shots, args.maxiter
            )
            print(
                f"{len(out['history'])} jobs in {out['seconds']:.1f} s; "
                + convergence_summary(out["history"], target)
            )
            records[name] = {
                "parameters": ansatz.num_parameters,
                "history": out["history"],
                "seconds": out["seconds"],
            }
    finally:
        for context in (session, batch):
            if context is not None:
                context.close()
    save_results(args.outdir, args, target, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
