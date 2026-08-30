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
    """Save the last submitted hardware job to a local JSON file.

    Parameters
    ----------
    job_id : str
        The IBM Quantum job ID.
    backend_name : str
        Name of the backend the job was submitted to.
    account_name : str
        Name of the saved account used for submission.

    Returns
    -------
    None
    """
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
    """Load the last submitted hardware job from the local state file.

    Returns
    -------
    dict or None
        Dictionary containing job_id, backend, account_name, and submitted_at,
        or None if no saved job exists or the file is corrupted.
    """
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None


def extract_counts(pub_result) -> tuple[str, dict]:
    """Extract measurement counts from a PUB result.

    Retrieves counts without assuming a specific classical register name,
    making the function flexible across different circuit definitions.

    Parameters
    ----------
    pub_result
        A Qiskit PUB (Primitive Unified Bloc) result object.

    Returns
    -------
    tuple[str, dict]
        A tuple of (register_name: str, counts: dict) where counts maps
        bitstrings to their measurement frequencies.
    """
    register_name, bit_array = next(iter(pub_result.data.items()))
    return register_name, bit_array.get_counts()


def wait_for_job(job, backend=None, poll_interval: int = DEFAULT_POLL_INTERVAL):
    """Poll a job until it reaches a final state, printing status updates.

    IBM's Runtime API doesn't expose this job's exact position in queue
    (older API feature that's gone now) -- the closest available signal is
    how many jobs are pending on the backend overall, which is printed here.
    Propagates KeyboardInterrupt or connection failures so the caller can
    implement recovery instructions with the job ID.

    Parameters
    ----------
    job
        The submitted Qiskit Runtime job object.
    backend : optional
        The backend object (for retrieving pending job count). If None,
        status is printed without queue information.
    poll_interval : int, optional
        Seconds to wait between status checks (default: DEFAULT_POLL_INTERVAL).

    Returns
    -------
    Result
        The finished job's result object.

    Raises
    ------
    KeyboardInterrupt
        If the user interrupts while waiting.
    Exception
        If a connection error occurs while checking job status.
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

    status = job.status()
    print(f"  status: {status}")
    return job.result()
