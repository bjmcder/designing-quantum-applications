"""Shared helpers for the algorithm scripts in this folder.

This file is not an algorithm. It holds the code every script needs in order
to run its circuits, so each algorithm file can stay focused on the physics.

Each algorithm script does two things:

1. Builds one or more named circuits.
2. Writes a small ``report`` function that prints what a run of those
   circuits means (for example, "the hidden string is 101").

``run_experiments`` then runs every circuit on the noiseless Aer simulator
and, unless you skip it, on IBM hardware. For each backend it does this:

1. Transpile every circuit for that backend (``build_pubs``).
2. Wrap each transpiled circuit in a PUB, one PUB per circuit.
3. Submit ALL the PUBs together as a single job. One job means one wait in
   the hardware queue, however many circuits you have.
4. Read the result: entry ``i`` belongs to circuit ``i`` (``counts_by_name``).

The hardware steps are the same ones taught in
``00-Setup-Start-Here/04_test_quantum.py``: transpile, submit, save the job ID,
wait, and read the counts. If a hardware run is interrupted, recover the job
with ``00-Setup-Start-Here/05_retrieve_job.py``.

A note on reading results: Qiskit prints bitstrings with qubit 0 on the
*right*. For example, the counts key ``"101"`` means qubit 2 = 1, qubit 1 = 0,
qubit 0 = 1.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

# The folder name "00-Setup-Start-Here" can't be imported as a package (it
# contains hyphens), so add it to the import path and import the file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00-Setup-Start-Here"))
from _job_state import (
    DEFAULT_POLL_INTERVAL,
    extract_counts,
    save_last_job,
    wait_for_job,
)

logging.basicConfig(
    level=logging.WARNING, format="[%(levelname)-8s] %(name)s: %(message)s"
)
# Suppress harmless Qiskit warnings about instance discovery
logging.getLogger("qiskit_runtime_service").setLevel(logging.ERROR)

DEFAULT_SHOTS = 4096
DEFAULT_BACKEND = "ibm_rensselaer"
PREFERRED_INSTANCE = "MANE-4960-dedicated"

# A report function is called as report(where, circuit_name, counts), where
# `where` is "aer" or the hardware backend's name.
ReportFn = Callable[[str, str, dict], None]


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser with the options every script shares.

    Parameters
    ----------
    description : str
        Text shown at the top of ``--help``.

    Returns
    -------
    argparse.ArgumentParser
        A parser with --shots, --backend, --name, --skip-hardware, and
        --poll-interval. Each script adds its own algorithm options.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Shots (repeated runs) per circuit (default: {DEFAULT_SHOTS}).",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"IBM backend to run on (default: {DEFAULT_BACKEND}).",
    )
    parser.add_argument(
        "--name", default="default", help="Saved account name (default: 'default')."
    )
    parser.add_argument(
        "--skip-hardware", action="store_true", help="Only run on the Aer simulator."
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between hardware status checks (default: {DEFAULT_POLL_INTERVAL}).",
    )
    return parser


def build_pubs(
    circuits: dict[str, QuantumCircuit], backend, show_depth: bool = False
) -> list[tuple]:
    """Transpile every circuit for a backend and wrap each one in a PUB.

    A PUB (Primitive Unified Bloc) is a tuple that starts with a circuit. Ours
    have no parameters, so each tuple holds only the circuit. A single job can
    carry many PUBs, so we make one per circuit and send them all together.

    Parameters
    ----------
    circuits : dict[str, QuantumCircuit]
        Circuits to run. Each must end with measurements.
    backend
        What to transpile for: ``AerSimulator()`` or an IBM backend. Real
        hardware only understands its own basis gates and qubit layout, so
        this step converts each circuit into an ISA (Instruction Set
        Architecture) circuit. The simulator accepts almost any gate, so
        little changes for Aer, but the call is the same.
    show_depth : bool, optional
        Print the circuit depth before and after transpiling.

    Returns
    -------
    list[tuple]
        One PUB per circuit, in the same order as ``circuits``.
    """
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    pubs = []
    for name, circuit in circuits.items():
        isa_circuit = pass_manager.run(circuit)
        if show_depth:
            print(
                f"  {name}: depth {circuit.depth()} -> {isa_circuit.depth()}, "
                f"{isa_circuit.num_nonlocal_gates()} two-qubit gates"
            )
        pubs.append((isa_circuit,))
    return pubs


def counts_by_name(circuits: dict[str, QuantumCircuit], result) -> dict[str, dict]:
    """Match each PUB result back to the circuit it came from.

    Parameters
    ----------
    circuits : dict[str, QuantumCircuit]
        The same dictionary passed to ``build_pubs``.
    result
        The job's result. ``result[i]`` belongs to the i-th PUB, which was
        built from the i-th circuit.

    Returns
    -------
    dict[str, dict]
        Maps each circuit name to its counts (bitstring -> number of shots).
    """
    return {
        name: extract_counts(pub_result)[1]
        for name, pub_result in zip(circuits, result)
    }


def run_on_aer(circuits: dict[str, QuantumCircuit], shots: int) -> dict[str, dict]:
    """Run all circuits in one job on the noiseless Aer simulator.

    Parameters
    ----------
    circuits : dict[str, QuantumCircuit]
        Circuits to simulate, keyed by name. Each must end with measurements.
    shots : int
        Number of times to run each circuit.

    Returns
    -------
    dict[str, dict]
        Maps each circuit name to its counts.
    """
    pubs = build_pubs(circuits, AerSimulator())
    job = AerSampler().run(pubs, shots=shots)
    return counts_by_name(circuits, job.result())


def connect(backend_name: str, account_name: str):
    """Connect to an IBM backend and warn if not on the preferred instance.

    Parameters
    ----------
    backend_name : str
        Name of the backend, for example "ibm_rensselaer".
    account_name : str
        Name of the saved IBM Quantum account.

    Returns
    -------
    backend or None
        The backend, or None if the connection failed.
    """
    try:
        service = QiskitRuntimeService(name=account_name)
        backend = service.backend(backend_name)
    except Exception as exc:  # noqa: BLE001 - report connection failure
        print(f"Could not reach backend '{backend_name}': {exc}")
        print(
            "Check that 00-Setup-Start-Here/02_save_token.py has been run, and "
            "run 03_check_token.py to see which backends you have access to. "
            "Pass a different backend with: --backend <name>"
        )
        return None

    print(f"Connected to {backend.name} ({backend.num_qubits} qubits)")

    # Using a different instance only means a longer wait in the queue, so
    # warn but don't stop. The active account stores an instance ID (a CRN),
    # so look up its friendly name before comparing.
    active_crn = service.active_account()["instance"]
    active_instance = next(
        (inst["name"] for inst in service.instances() if inst["crn"] == active_crn),
        active_crn,
    )
    if active_instance != PREFERRED_INSTANCE:
        print(
            f"** Note **  You're using instance '{active_instance}', not the "
            f"preferred instance '{PREFERRED_INSTANCE}'. Your job may wait "
            "longer in the queue than necessary."
        )
    return backend


def run_on_hardware(
    circuits: dict[str, QuantumCircuit],
    shots: int,
    backend,
    account_name: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> dict[str, dict] | None:
    """Run all circuits in ONE job on IBM hardware and wait for the result.

    Parameters
    ----------
    circuits : dict[str, QuantumCircuit]
        Circuits to run, keyed by name. Each must end with measurements.
    shots : int
        Number of times to run each circuit.
    backend
        The backend returned by ``connect``.
    account_name : str
        Saved account name, stored with the job ID for later retrieval.
    poll_interval : int, optional
        Seconds between status checks while waiting.

    Returns
    -------
    dict[str, dict] or None
        Maps each circuit name to its counts, or None if you stopped waiting,
        the job failed, or the connection dropped. After a dropped connection
        the job keeps running on IBM's servers and can be fetched with
        00-Setup-Start-Here/05_retrieve_job.py.
    """
    # Step 1 and 2: transpile every circuit and wrap each in a PUB.
    pubs = build_pubs(circuits, backend, show_depth=True)

    # Step 3: one job carries every PUB, so we wait in the queue only once.
    job = RuntimeSampler(mode=backend).run(pubs, shots=shots)
    # Save the job ID *before* waiting. The queue can be long, and the job
    # keeps running on IBM's servers even if this script loses its connection.
    save_last_job(job.job_id(), backend.name, account_name)
    print(
        f"Submitted ONE job ({job.job_id()}) carrying {len(pubs)} PUB(s) "
        f"on {backend.name}."
    )
    print(
        "If interrupted, retrieve it later with: "
        "uv run ../00-Setup-Start-Here/05_retrieve_job.py"
    )

    try:
        result = wait_for_job(job, backend, poll_interval=poll_interval)
    except KeyboardInterrupt:
        print(
            f"\nStopped waiting. Job {job.job_id()} is still running on IBM's "
            "servers; fetch it with: uv run "
            f"../00-Setup-Start-Here/05_retrieve_job.py {job.job_id()}"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        # Either the job itself failed, or we lost contact with IBM.
        try:
            status = str(job.status())
        except Exception:  # noqa: BLE001
            status = None
        if status in ("ERROR", "CANCELLED"):
            print(f"\nJob {job.job_id()} ended with status {status}: {exc}")
        else:
            print(
                f"\nLost connection: {exc}\nJob {job.job_id()} is probably "
                "still running on IBM's servers; fetch it with: uv run "
                f"../00-Setup-Start-Here/05_retrieve_job.py {job.job_id()}"
            )
        return None

    # Step 4: match each PUB's result back to its circuit.
    return counts_by_name(circuits, result)


def run_experiments(circuits: dict[str, QuantumCircuit], args, report: ReportFn) -> int:
    """Run every circuit on Aer, then (unless skipped) on IBM hardware.

    Each backend gets one job that carries all the circuits (one PUB each).

    Parameters
    ----------
    circuits : dict[str, QuantumCircuit]
        Circuits to run, keyed by a short name used in the report.
    args : argparse.Namespace
        Parsed arguments from a parser built with ``make_parser``.
    report : ReportFn
        Called once per circuit and backend to print what the counts mean.

    Returns
    -------
    int
        Exit code: always 0 (a failed hardware run is reported, not fatal).
    """
    first = next(iter(circuits.values()))
    print(f"Example circuit ({first.name}):")
    print(first.draw(output="text"))

    print(f"\n--- Aer simulator (noiseless), {args.shots} shots ---")
    for name, counts in run_on_aer(circuits, args.shots).items():
        report("aer", name, counts)

    if args.skip_hardware:
        return 0

    print(f"\n--- IBM hardware '{args.backend}', {args.shots} shots ---")
    backend = connect(args.backend, args.name)
    if backend is None:
        return 0
    start = time.perf_counter()
    all_counts = run_on_hardware(
        circuits, args.shots, backend, args.name, args.poll_interval
    )
    if all_counts is None:
        return 0
    for name, counts in all_counts.items():
        report(backend.name, name, counts)
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    return 0
