"""Execution: run the calibration, the sweep, and the twirled versions.

All circuits go out as one job (on hardware the queue wait is paid per job):

    1. calibration: |0> and |1>, 20000 shots each, no twirl
    2. calibration with twirl: |0> and |1>, split over 100 randomizations
       (many, because every sweep point is divided by the same result)
    3. sweep: every RY angle, ``shots`` shots, no twirl
    4. sweep with twirl: every RY angle, ``shots`` shots split over R
       randomizations

A randomization is one circuit with one coin toss: the twirl X is either
present or absent, and all ``shots // R`` shots of the circuit share the
toss. After the run, the answer of every randomization with an X is flipped
back: the counts of 0 and 1 are exchanged. Only two compiled circuits exist per state
(with and without the X), and the coin flips pick between them.

Called from main.py. With no backend name it runs on the Aer simulator with
the read-out error of ops.py and nothing else, so the corrections can be
checked against the built-in values. With a backend name it runs on IBM
hardware, and the device supplies the error. On hardware one more job runs:
the Estimator with IBM's own measurement mitigation, the same idea as the
twirl above (see ``run_estimator``).
"""

import time

import numpy as np
from encoding import (
    calibration_circuit,
    preparation_circuit,
    sweep_circuit,
    transpile_for,
)
from ops import SIM_READOUT, angle_grid, observable, simulator_truth
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.primitives import SamplerV2 as AerSampler

CAL_SHOTS = 20000  # shots of each calibration state
CAL_RANDOMIZATIONS = 100  # twirl randomizations of each calibration state


def get_primitives(backend_name: str | None):
    """Create the backend and the Sampler.

    Args:
        backend_name: IBM backend name, or None for the Aer simulator.

    Returns:
        (backend, sampler). On the simulator the backend carries the
        read-out error.
    """
    if backend_name is None:
        noise = NoiseModel()
        p01, p10 = SIM_READOUT
        noise.add_all_qubit_readout_error(
            ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        )
        backend = AerSimulator(noise_model=noise)
        return backend, AerSampler.from_backend(backend)
    # Only hardware runs need qiskit_ibm_runtime and a saved IBM account.
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    backend = QiskitRuntimeService().backend(backend_name)
    return backend, SamplerV2(mode=backend)


def unflipped_counts(counts: dict, flip: bool) -> tuple[int, int]:
    """Number of 0s and 1s, in the bit labels of the ideal circuit.

    If the twirl X was in the circuit, the bit that was read is the opposite
    of the bit before the X, so the two counts are exchanged.

    Args:
        counts: Counts from the Sampler, such as ``{"0": 990, "1": 10}``.
        flip: Whether the circuit contained the twirl X.

    Returns:
        (n0, n1).
    """
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    return (n1, n0) if flip else (n0, n1)


def run_estimator(backend, thetas, qubit: int | None, shots: int) -> dict:
    """IBM's own mitigation: the Estimator with measurement twirling on.

    Hardware only, and not run by the notebook. The Estimator measures Z
    itself. The two options below ask for the twirl of the measurement and
    for the rescaling by a learned factor.

    Args:
        backend: Hardware backend.
        thetas: RY angles in radians.
        qubit: Physical qubit to run on.
        shots: Sets the target precision, 1 / sqrt(shots).

    Returns:
        The expectation values and standard errors of the Estimator.
    """
    from qiskit_ibm_runtime import EstimatorV2

    estimator = EstimatorV2(mode=backend)
    estimator.options.resilience_level = 0  # start with everything off
    estimator.options.resilience.measure_mitigation = True
    estimator.options.twirling.enable_measure = True
    estimator.options.dynamical_decoupling.enable = False
    obs = observable()
    pubs = []
    for theta in thetas:
        isa = transpile_for(preparation_circuit(float(theta)), backend, qubit)
        pubs.append((isa, obs.apply_layout(isa.layout), None, shots**-0.5))
    result = estimator.run(pubs).result()
    return {
        "evs": [float(r.data.evs) for r in result],
        "stds": [float(r.data.stds) for r in result],
    }


def run(
    points: int = 13,
    shots: int = 4000,
    randomizations: int = 20,
    backend_name: str | None = None,
    qubit: int | None = None,
    seed: int | None = None,
) -> dict:
    """Measure the calibration states and the RY sweep, with and without twirl.

    Args:
        points: Number of RY angles from 0 to pi.
        shots: Shots per angle (split over the randomizations when twirled).
        randomizations: Number R of randomizations of each twirled run.
        backend_name: IBM backend name, or None for the Aer simulator.
        qubit: Physical qubit to measure on hardware (ignored on the
            simulator).
        seed: Seed of the coin flips, or None for a different set each run.

    Returns:
        The settings and the counts. Each count is a pair (n0, n1) in the
        bit labels of the ideal circuit. ``twirled`` entries hold one pair per
        randomization.
    """
    backend, sampler = get_primitives(backend_name)
    simulated = backend_name is None
    layout_qubit = None if simulated else qubit
    rng = np.random.default_rng(seed)
    thetas = angle_grid(points)
    per_run = shots // randomizations
    cal_per_run = CAL_SHOTS // CAL_RANDOMIZATIONS

    def compile_(circuit):
        return transpile_for(circuit, backend, layout_qubit)

    # Two compiled circuits per state: without and with the twirl X.
    cal = {
        s: [compile_(calibration_circuit(s, f)) for f in (False, True)] for s in (0, 1)
    }
    sweep = [
        [compile_(sweep_circuit(float(t), f)) for f in (False, True)] for t in thetas
    ]

    pubs, plan = [], []  # plan: (key, flip) for every pub, in order

    def add(key, circuit, n_shots, flip=False):
        pubs.append((circuit, None, n_shots))
        plan.append((key, flip))

    for s in (0, 1):
        add(("cal", s), cal[s][0], CAL_SHOTS)
        for flip in rng.integers(0, 2, CAL_RANDOMIZATIONS).astype(bool):
            add(("cal_twirl", s), cal[s][int(flip)], cal_per_run, bool(flip))
    for j, pair in enumerate(sweep):
        add(("raw", j), pair[0], shots)
        for flip in rng.integers(0, 2, randomizations).astype(bool):
            add(("twirl", j), pair[int(flip)], per_run, bool(flip))

    t0 = time.perf_counter()
    job = sampler.run(pubs)
    print(f"  {len(pubs)} circuits in one job", end="", flush=True)
    result = job.result()
    seconds = time.perf_counter() - t0
    print(f" ({seconds:.0f} s)")

    found: dict = {}
    for pub_result, (key, flip) in zip(result, plan):
        n = unflipped_counts(pub_result.data.c.get_counts(), flip)
        found.setdefault(key, []).append(n)

    data = {
        "backend": backend.name,
        "qubit": qubit,
        "shots": shots,
        "randomizations": randomizations,
        "cal_shots": CAL_SHOTS,
        "seconds": seconds,
        "truth": simulator_truth() if simulated else None,
        "angles": [float(t) for t in thetas],
        "calibration": {str(s): found[("cal", s)][0] for s in (0, 1)},
        "calibration_twirled": {str(s): found[("cal_twirl", s)] for s in (0, 1)},
        "raw": [found[("raw", j)][0] for j in range(points)],
        "twirled": [found[("twirl", j)] for j in range(points)],
        "estimator": None,
    }
    if not simulated:
        data["estimator"] = run_estimator(backend, thetas, qubit, shots)
    return data


def randomization_study(
    counts=(1, 2, 4, 8, 16, 32, 64),
    repeats: int = 100,
    total_shots: int = 4000,
    theta: float = np.pi / 2,
    seed: int | None = None,
) -> dict:
    """Spread of the twirled <Z> as the number of randomizations grows.

    For each R the same measurement is repeated ``repeats`` times with a
    fresh set of coin flips, always with the same total number of shots. The
    standard deviation across the repeats is the error bar that one run
    would have. Simulator only. The estimate is divided by the built-in
    attenuation, so the study shows the effect of R alone.

    Args:
        counts: Numbers of randomizations R to compare.
        repeats: Repeats for each R.
        total_shots: Shots of one measurement, split over the R randomizations.
        theta: RY angle. The default has a true <Z> of 0.
        seed: Seed of the coin flips and the sampling, or None.

    Returns:
        The R values, the estimates of every repeat, and the true <Z>.
    """
    backend, sampler = get_primitives(None)
    rng = np.random.default_rng(seed)
    pair = [transpile_for(sweep_circuit(theta, f), backend) for f in (False, True)]
    a = simulator_truth()["a"]

    estimates = {}
    for count in counts:
        flips = rng.integers(0, 2, (repeats, count)).astype(bool)
        pubs = [(pair[int(f)], None, total_shots // count) for f in flips.ravel()]
        result = sampler.run(pubs).result()
        z = np.empty(flips.size)
        for k, (pub_result, f) in enumerate(zip(result, flips.ravel())):
            n0, n1 = unflipped_counts(pub_result.data.c.get_counts(), bool(f))
            z[k] = (n0 - n1) / (n0 + n1)
        estimates[count] = (z.reshape(repeats, count).mean(axis=1) / a).tolist()
    return {
        "counts": list(counts),
        "estimates": estimates,
        "total_shots": total_shots,
        "ideal": float(np.cos(theta)),
    }
