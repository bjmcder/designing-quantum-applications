#!/usr/bin/env python3
"""Build a Bell state and run it on a simulator, then on real IBM hardware.

A guided walkthrough of the modern Qiskit Runtime workflow: named
registers, a named circuit, PUBs, samplers, job submission/retrieval, and
post-processing raw shot data into counts. No plotting here.

The hardware queue can be long, so while waiting this script prints a
status/queue update every few seconds (see --poll-interval). The job ID is
also saved locally as soon as it's submitted -- if this script loses its
connection (or you close your laptop) while waiting, the job keeps running
on IBM's servers regardless; fetch its result later with
05_retrieve_job.py instead of resubmitting.

Run with no arguments. Pass --skip-hardware to run only the simulator step.
"""

import argparse
import sys

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

from _job_state import DEFAULT_POLL_INTERVAL, extract_counts, save_last_job, wait_for_job

DEFAULT_SHOTS = 10000
DEFAULT_BACKEND = "ibm_rensselaer"


def build_bell_circuit() -> QuantumCircuit:
    # Step 1: named registers. Naming them makes the circuit diagram and
    # the post-processing results (pub_result.data.<name>) self-explanatory
    # instead of falling back to generic names like "q" and "c0".
    qubits = QuantumRegister(2, name="q")
    bits = ClassicalRegister(2, name="meas")

    # Step 2: a named circuit, built from those registers.
    circuit = QuantumCircuit(qubits, bits, name="bell_state")

    # Step 3: the Bell state itself. H puts qubit 0 into superposition, then
    # CX entangles it with qubit 1.
    circuit.h(qubits[0])
    circuit.cx(qubits[0], qubits[1])
    circuit.measure(qubits, bits)

    return circuit


def run_on_aer(circuit: QuantumCircuit, shots: int) -> dict:
    print(f"\n--- Running on a perfect (noiseless) Aer simulator, {shots} shots ---")

    # Step 4: a PUB (Primitive Unified Bloc) is what you hand to a sampler.
    # A PUB is a tuple of (circuit, [parameter values], [shots]) -- here
    # just the circuit, since the Bell circuit has no parameters.
    pub = (circuit,)

    # Step 5: create the sampler. qiskit_aer.primitives.SamplerV2 samples
    # locally against an ideal, noiseless simulator by default -- no real
    # hardware noise, so counts should split ~50/50 between "00" and "11".
    sampler = AerSampler()

    # Step 6: submit the job, then retrieve its result.
    job = sampler.run([pub], shots=shots)
    print(f"Submitted job {job.job_id()}")
    result = job.result()

    # Step 7: post-process. pub_result.data.<classical register name> is a
    # BitArray of raw shot outcomes; get_counts() collapses it into a
    # bitstring -> count dictionary.
    pub_result = result[0]
    _, counts = extract_counts(pub_result)

    print(f"Counts: {counts}")
    return counts


def run_on_hardware(
    circuit: QuantumCircuit,
    shots: int,
    backend_name: str,
    account_name: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> dict | None:
    print(f"\n--- Running on IBM hardware backend '{backend_name}', {shots} shots ---")

    try:
        service = QiskitRuntimeService(name=account_name)
        backend = service.backend(backend_name)
    except Exception as exc:  # noqa: BLE001 - report any connection failure to the user
        print(f"Could not reach backend '{backend_name}': {exc}")
        print(
            "Check that 02_save_token.py has been run, and run "
            "03_check_token.py to see which instances (and backends) you "
            "actually have access to. If the class backend has a different "
            f"name, pass it with: --backend <name>"
        )
        return None

    print(f"Connected to {backend.name} ({backend.num_qubits} qubits)")

    # Extra step, only needed for real hardware: convert to an ISA
    # (Instruction Set Architecture) circuit -- transpiled into this
    # backend's actual basis gates and qubit connectivity. The simulator
    # above skipped this because it supports every gate directly and has
    # no connectivity restrictions.
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pass_manager.run(circuit)

    # Steps 4-5 again, this time targeting real hardware: same PUB shape,
    # but the sampler now runs in "backend mode" against the IBM backend.
    pub = (isa_circuit,)
    sampler = RuntimeSampler(mode=backend)

    # Step 6: submit, then retrieve. Save the job ID to disk *before*
    # waiting. The queue can be long, and the job keeps running on IBM's
    # servers even if this script's connection drops while it waits.
    job = sampler.run([pub], shots=shots)
    save_last_job(job.job_id(), backend.name, account_name)
    print(f"Submitted job {job.job_id()} on {backend.name}.")
    print(
        "If this script is interrupted, retrieve the result later with:\n"
        "  uv run 05_retrieve_job.py"
    )
    print(f"Waiting for it to run (checking every {poll_interval}s)...")

    try:
        result = wait_for_job(job, backend, poll_interval=poll_interval)
    except (KeyboardInterrupt, Exception) as exc:
        cause = "Cancelled." if isinstance(exc, KeyboardInterrupt) else f"Lost connection: {exc}"
        print(
            f"\n{cause} The job (id: {job.job_id()}) is still running on IBM's "
            "servers -- it doesn't need this script to stay connected.\n"
            "To pick it back up:\n"
            "  1. Reconnect (if you lost network/power), then open a terminal here.\n"
            "  2. Run:\n"
            "       uv run 05_retrieve_job.py\n"
            f"     (or explicitly:  uv run 05_retrieve_job.py {job.job_id()})\n"
            "  3. It will reconnect, show the job's current status, and wait "
            "for it to finish if it hasn't already."
        )
        return None

    # Step 7: post-process, exactly as with the Aer result above.
    pub_result = result[0]
    _, counts = extract_counts(pub_result)

    print(f"Counts: {counts}")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Bell state and run it on Aer, then on real IBM hardware.",
    )
    parser.add_argument(
        "--shots", type=int, default=DEFAULT_SHOTS, help=f"Shots per run (default: {DEFAULT_SHOTS})."
    )
    parser.add_argument(
        "--backend", default=DEFAULT_BACKEND, help=f"IBM backend to run on (default: {DEFAULT_BACKEND})."
    )
    parser.add_argument(
        "--name", default="default", help="Name of the saved account to use (default: 'default')."
    )
    parser.add_argument(
        "--skip-hardware", action="store_true", help="Only run on the Aer simulator."
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between status/queue checks while waiting on hardware "
        f"(default: {DEFAULT_POLL_INTERVAL}).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    circuit = build_bell_circuit()
    print("Circuit:")
    print(circuit.draw(output="text"))

    aer_counts = run_on_aer(circuit, args.shots)

    hardware_counts = None
    if not args.skip_hardware:
        hardware_counts = run_on_hardware(
            circuit, args.shots, args.backend, args.name, poll_interval=args.poll_interval
        )

    print("\n=== Summary ===")
    print(f"Aer (ideal) counts:       {aer_counts}")
    if hardware_counts is not None:
        print(f"{args.backend} counts: {hardware_counts}")
    elif not args.skip_hardware:
        print(f"{args.backend}: run failed or was unreachable (see above).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
