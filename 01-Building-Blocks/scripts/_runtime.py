"""Plumbing shared by the building-block example scripts.

This file is not a lesson. It only answers one question for the scripts:
"where do I run?" Each script asks for a list of *targets* (the Aer
simulator, plus IBM hardware unless you pass --skip-hardware) and then runs
the same steps against each one.

Everything that teaches a Qiskit concept -- building the circuit,
transpiling it, writing the observable, assembling the PUB, running the job,
and reading the result -- stays in the scripts themselves so you can see it.

A note on reading results: Qiskit prints bitstrings with qubit 0 on the
*right*. The counts key "01" means qubit 1 = 0 and qubit 0 = 1.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

# "00-Setup-Start-Here" has hyphens, so it can't be imported as a package.
# Put it on the import path and import the helper file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00-Setup-Start-Here"))
from _job_state import DEFAULT_POLL_INTERVAL, save_last_job, wait_for_job

logging.basicConfig(
    level=logging.WARNING, format="[%(levelname)-8s] %(name)s: %(message)s"
)
logging.getLogger("qiskit_runtime_service").setLevel(logging.ERROR)

DEFAULT_SHOTS = 4096
DEFAULT_PRECISION = 0.02
DEFAULT_BACKEND = "ibm_rensselaer"
PREFERRED_INSTANCE = "MANE-4960-dedicated"


@dataclass
class Target:
    """A place to run circuits: the Aer simulator or an IBM backend.

    Attributes
    ----------
    label : str
        Short name used when printing results ("aer" or the backend name).
    backend
        What you transpile *for*. Pass it to ``generate_preset_pass_manager``.
    sampler
        A SamplerV2 that runs on this target.
    estimator
        An EstimatorV2 that runs on this target.
    hardware : bool
        True for real IBM hardware, False for the simulator.
    account_name : str
        Saved IBM account, stored with the job ID so it can be found later.
    poll_interval : int
        Seconds between status checks while waiting for hardware.
    """

    label: str
    backend: object
    sampler: object
    estimator: object
    hardware: bool = False
    account_name: str = ""
    poll_interval: int = DEFAULT_POLL_INTERVAL

    def wait(self, job):
        """Return a job's result, waiting if the job is on real hardware.

        Parameters
        ----------
        job
            The job returned by ``sampler.run`` or ``estimator.run``.

        Returns
        -------
        PrimitiveResult or None
            One entry per PUB you submitted, in order. None if you stopped
            waiting or the job failed. Callers should skip any later steps
            that depend on it.
        """
        if not self.hardware:
            return job.result()

        # Save the ID *before* waiting: the queue can be long and the job
        # keeps running on IBM's servers even if this script disconnects.
        # (05_retrieve_job.py only knows how to print Sampler counts, so use
        # the IBM Quantum dashboard to look at an Estimator job.)
        save_last_job(job.job_id(), self.backend.name, self.account_name)
        print(f"  submitted job {job.job_id()} on {self.label}; waiting...")
        try:
            return wait_for_job(job, self.backend, poll_interval=self.poll_interval)
        except KeyboardInterrupt:
            print(
                f"\nStopped waiting. Job {job.job_id()} is still running on "
                "IBM's servers."
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
                    "still running on IBM's servers."
                )
            return None


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create the command-line options every script shares.

    Parameters
    ----------
    description : str
        Text shown at the top of ``--help`` (usually the script docstring).

    Returns
    -------
    argparse.ArgumentParser
        A parser with --shots, --precision, --backend, --name,
        --skip-hardware and --poll-interval.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Shots per circuit for Sampler jobs (default: {DEFAULT_SHOTS}).",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=DEFAULT_PRECISION,
        help=(
            "Target standard error for Estimator jobs (default: "
            f"{DEFAULT_PRECISION}). On IBM hardware, smaller means more shots. "
            "On Aer there are no shots: it returns the exact value plus random "
            "noise of this size."
        ),
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


def _connect(backend_name: str, account_name: str):
    """Connect to an IBM backend, or return None (with a hint) on failure."""
    try:
        service = QiskitRuntimeService(name=account_name)
        backend = service.backend(backend_name)
    except Exception as exc:  # noqa: BLE001 - report connection failure
        print(f"Could not reach backend '{backend_name}': {exc}")
        print(
            "Check that 00-Setup-Start-Here/02_save_token.py has been run, and "
            "run 03_check_token.py to see which backends you can use. "
            "Pass a different backend with: --backend <name>"
        )
        return None

    print(f"Connected to {backend.name} ({backend.num_qubits} qubits)")

    # A different instance only means a longer queue, so warn but continue.
    # The active account stores a CRN (an ID), so look up its friendly name.
    active_crn = service.active_account()["instance"]
    active_instance = next(
        (i["name"] for i in service.instances() if i["crn"] == active_crn),
        active_crn,
    )
    if active_instance != PREFERRED_INSTANCE:
        print(
            f"** Note **  You're using instance '{active_instance}', not "
            f"'{PREFERRED_INSTANCE}'. Your job may wait longer in the queue."
        )
    return service, backend


def get_targets(args) -> list[Target]:
    """Build the list of places to run: Aer first, then hardware if wanted.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from a parser built with ``make_parser``.

    Returns
    -------
    list[Target]
        Always starts with the noiseless Aer simulator. IBM hardware is
        appended unless --skip-hardware was given or the connection failed.
    """
    # AerSimulator is a "backend" like any other, so we can transpile for it
    # with exactly the same call we use for hardware. The simulator accepts
    # nearly any gate, so the transpiled circuit barely changes.
    targets = [
        Target(
            label="aer",
            backend=AerSimulator(),
            sampler=AerSampler(),
            estimator=AerEstimator(),
        )
    ]
    if args.skip_hardware:
        return targets

    connection = _connect(args.backend, args.name)
    if connection is not None:
        _, backend = connection
        targets.append(
            Target(
                label=backend.name,
                backend=backend,
                # mode=backend: each .run() call is its own independent job.
                sampler=RuntimeSampler(mode=backend),
                estimator=RuntimeEstimator(mode=backend),
                hardware=True,
                account_name=args.name,
                poll_interval=args.poll_interval,
            )
        )
    return targets


def banner(text: str) -> None:
    """Print a section heading."""
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")
