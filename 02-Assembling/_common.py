"""Shared helpers for the algorithm scripts in this folder.

This file is not an algorithm. It holds the code every script needs in order
to run its circuits, so each algorithm file can stay focused on the physics.

Each algorithm script does two things:

1. Builds one or more named circuits.
2. Writes a small ``report`` function that prints what a run of those
   circuits means (for example, "the hidden string is 101").

``run_experiments`` then runs every circuit on the noiseless Aer simulator
and, unless you skip it, on IBM hardware. The hardware steps are the same
ones taught in ``00-Setup-Start-Here/04_test_quantum.py``: transpile, submit,
save the job ID, wait, and read the counts. If a hardware run is interrupted,
recover the job with ``00-Setup-Start-Here/05_retrieve_job.py``.

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


def run_on_aer(circuit: QuantumCircuit, shots: int) -> dict:
    """Run a circuit on the noiseless Aer simulator.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to simulate. It must end with measurements.
    shots : int
        Number of times to run the circuit.

    Returns
    -------
    dict
        Maps each measured bitstring to how many shots produced it.
    """
    # A PUB (Primitive Unified Bloc) is a tuple holding the circuit; ours has
    # no parameters, so the tuple has just one entry.
    # Transpile for Aer the same way we do for hardware. The simulator accepts
    # almost any gate, so little changes, but this lets us use high-level gates
    # such as QFTGate without decomposing them by hand.
    isa_circuit = generate_preset_pass_manager(
        backend=AerSimulator(), optimization_level=1
    ).run(circuit)
    job = AerSampler().run([(isa_circuit,)], shots=shots)
    _, counts = extract_counts(job.result()[0])
    return counts


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
    circuit: QuantumCircuit,
    shots: int,
    backend,
    account_name: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> dict | None:
    """Transpile a circuit, run it on IBM hardware, and wait for the result.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to run. It must end with measurements.
    shots : int
        Number of times to run the circuit.
    backend
        The backend returned by ``connect``.
    account_name : str
        Saved account name, stored with the job ID for later retrieval.
    poll_interval : int, optional
        Seconds between status checks while waiting.

    Returns
    -------
    dict or None
        Maps each measured bitstring to its count, or None if the wait was
        interrupted or the connection dropped. In that case the job keeps
        running on IBM's servers and can be fetched with 00-Setup-Start-Here/05_retrieve_job.py.
    """
    # Real hardware only understands its own basis gates and qubit layout, so
    # convert the circuit to an ISA (Instruction Set Architecture) circuit.
    # (run_on_aer does the same for Aer, but the simulator accepts almost any
    # gate, so little changes there.)
    isa_circuit = generate_preset_pass_manager(
        backend=backend, optimization_level=1
    ).run(circuit)
    print(
        f"Transpiled: depth {circuit.depth()} -> {isa_circuit.depth()}, "
        f"{isa_circuit.num_nonlocal_gates()} two-qubit gates"
    )

    job = RuntimeSampler(mode=backend).run([(isa_circuit,)], shots=shots)
    # Save the job ID *before* waiting. The queue can be long, and the job
    # keeps running on IBM's servers even if this script loses its connection.
    save_last_job(job.job_id(), backend.name, account_name)
    print(f"Submitted job {job.job_id()} on {backend.name}.")
    print("If interrupted, retrieve it later with: uv run ../00-Setup-Start-Here/05_retrieve_job.py")

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

    _, counts = extract_counts(result[0])
    return counts


def run_experiments(circuits: dict[str, QuantumCircuit], args, report: ReportFn) -> int:
    """Run every circuit on Aer, then (unless skipped) on IBM hardware.

    Parameters
    ----------
    circuits : dict[str, QuantumCircuit]
        Circuits to run, keyed by a short name used in the report.
    args : argparse.Namespace
        Parsed arguments from a parser built with ``make_parser``.
    report : ReportFn
        Called once per finished run to print what the counts mean.

    Returns
    -------
    int
        Exit code: always 0 (a failed hardware run is reported, not fatal).
    """
    first = next(iter(circuits.values()))
    print(f"Example circuit ({first.name}):")
    print(first.draw(output="text"))

    print(f"\n--- Aer simulator (noiseless), {args.shots} shots ---")
    for name, circuit in circuits.items():
        report("aer", name, run_on_aer(circuit, args.shots))

    if args.skip_hardware:
        return 0

    print(f"\n--- IBM hardware '{args.backend}', {args.shots} shots ---")
    backend = connect(args.backend, args.name)
    if backend is None:
        return 0
    for name, circuit in circuits.items():
        start = time.perf_counter()
        counts = run_on_hardware(
            circuit, args.shots, backend, args.name, args.poll_interval
        )
        if counts is None:
            break
        report(backend.name, name, counts)
        print(f"Elapsed: {time.perf_counter() - start:.1f}s\n")
    return 0
