#!/usr/bin/env python3
"""Retrieve results for a previously submitted IBM Quantum job.

Use this if 04_test_quantum.py lost its connection while waiting in the
hardware queue -- the job keeps running on IBM's servers regardless of
whether anything is connected to watch it, and can be fetched again here
using its job ID.

With no arguments, retrieves the most recently submitted job (saved
locally by 04_test_quantum.py). Pass a job ID directly to retrieve any
other job you have access to.
"""

import argparse
import sys

from qiskit_ibm_runtime import QiskitRuntimeService

from _job_state import DEFAULT_POLL_INTERVAL, extract_counts, load_last_job, wait_for_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve results for a previously submitted IBM Quantum job.",
    )
    parser.add_argument(
        "job_id",
        nargs="?",
        help="Job ID to retrieve. If omitted, uses the most recently "
        "submitted job from 04_test_quantum.py.",
    )
    parser.add_argument(
        "--name",
        help="Name of the saved account to use. Defaults to whichever "
        "account submitted the job, if known, otherwise 'default'.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between status/queue checks while waiting "
        f"(default: {DEFAULT_POLL_INTERVAL}).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    job_id = args.job_id
    account_name = args.name

    if job_id is None:
        last_job = load_last_job()
        if last_job is None:
            parser.error(
                "no job ID given, and no locally saved job was found. "
                "Run 04_test_quantum.py first, or pass a job ID directly."
            )
        job_id = last_job["job_id"]
        account_name = account_name or last_job.get("account_name")
        print(
            f"Using most recently submitted job: {job_id} "
            f"(submitted {last_job.get('submitted_at', 'at an unknown time')})"
        )

    account_name = account_name or "default"

    try:
        service = QiskitRuntimeService(name=account_name)
        job = service.job(job_id)
    except Exception as exc:  # noqa: BLE001 - report any connection failure to the user
        print(f"Could not retrieve job '{job_id}': {exc}")
        return 1

    backend = job.backend()
    print(f"Job {job_id} on {backend.name if backend else '?'}: status = {job.status()}")

    if not job.in_final_state():
        print(f"Job is still queued or running (checking every {args.poll_interval}s)...")

    try:
        result = wait_for_job(job, backend, poll_interval=args.poll_interval)
    except (KeyboardInterrupt, Exception) as exc:
        cause = "Cancelled." if isinstance(exc, KeyboardInterrupt) else f"Lost connection: {exc}"
        print(
            f"\n{cause} The job is still running on IBM's servers -- it "
            "doesn't need this script to stay connected. Try again later "
            f"with:\n  uv run 05_retrieve_job.py {job_id}"
        )
        return 1

    pub_result = result[0]
    register_name, counts = extract_counts(pub_result)

    print(f"Counts ({register_name}): {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
