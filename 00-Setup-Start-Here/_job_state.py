"""Shared helpers for saving/loading the last submitted hardware job.

Not a numbered teaching script -- used by 04_test_quantum.py and
05_retrieve_job.py so both agree on where job state lives and how a PUB
result gets turned into counts.
"""

import json
import time
from datetime import datetime
from pathlib import Path

STATE_FILE = Path(__file__).parent / ".last_job.json"

DEFAULT_POLL_INTERVAL = 20


def save_last_job(job_id: str, backend_name: str, account_name: str) -> None:
    STATE_FILE.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "backend": backend_name,
                "account_name": account_name,
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )


def load_last_job() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None


def extract_counts(pub_result) -> tuple[str, dict]:
    """Pull counts out of a PUB result without assuming a register name."""
    register_name, bit_array = next(iter(pub_result.data.items()))
    return register_name, bit_array.get_counts()


def wait_for_job(job, backend=None, poll_interval: int = DEFAULT_POLL_INTERVAL):
    """Poll a job until it finishes, printing status/queue updates as it goes.

    IBM's Runtime API doesn't expose *this job's* exact position in line
    (that was a older-API feature that's gone now) -- the closest available
    signal is how many jobs are pending on the backend overall, which is
    what gets printed here. Returns the finished job's result. Propagates
    KeyboardInterrupt or a connection failure so the caller can print
    restart instructions with the job's ID.
    """
    last_status = None
    while not job.in_final_state():
        status = job.status()
        pending = None
        if backend is not None:
            try:
                pending = backend.status().pending_jobs
            except Exception:
                pending = None

        if pending is not None:
            print(f"  status: {status} -- {pending} job(s) pending on {backend.name}")
        elif status != last_status:
            print(f"  status: {status}")
        last_status = status

        time.sleep(poll_interval)

    return job.result()
