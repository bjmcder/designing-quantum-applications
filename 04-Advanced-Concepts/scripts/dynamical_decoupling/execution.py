"""Execution: run the idle circuits and collect <X> for each pulse sequence.

For every sequence and every idle time t:

    1. Build the circuit (encoding.py) and compile it for the backend.
    2. The Estimator runs it and returns the expectation value of X.

All circuits go out as one job, because on hardware the queue wait is paid
per job.

Called from main.py. With no backend name it runs on the Aer simulator with
the noise of ops.py: T1 and T2 relaxation during every delay and an optional
depolarizing error on every pulse. The frequency noise is added as an
ensemble: the same circuit runs for many noise realizations, each with its own
offsets, and the results are averaged.

With a backend name it runs on IBM hardware and the device supplies the
noise. Two kinds of runs are possible there:

- Pulses written into the circuit as X and Y gates, with the dynamical
  decoupling of the runtime switched off. This is the same experiment as on
  the simulator.
- The runtime's own dynamical decoupling (``runtime_dd=True``): the circuit
  holds only the delay, and the Estimator option
  ``dynamical_decoupling.enable`` makes the service insert the pulses of a
  chosen ``sequence_type`` ("XpXm" or "XY4") into the idle time.
"""

import time

import numpy as np
from encoding import build_circuit, snap, timing, transpile_for
from ops import (
    ARMS,
    DEFAULT_T_MAX,
    PULSE_ERROR,
    SIM_T1,
    SIM_T2,
    delay_grid,
    noise_offsets,
    observable,
    pulse_train,
)
from qiskit.circuit.delay import Delay
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.noise.passes import RelaxationNoisePass
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

RUNTIME_SEQUENCES = {"runtime_xpxm": "XpXm", "runtime_xy4": "XY4"}


def get_primitives(
    backend_name: str | None,
    pulse_error: float = 0.0,
    sequence_type: str | None = None,
):
    """Create the backend and the Estimator.

    Args:
        backend_name: IBM backend name, or None for the Aer simulator.
        pulse_error: Depolarizing probability of every X and Y pulse
            (simulator only).
        sequence_type: ``"XpXm"`` or ``"XY4"`` to switch on the dynamical
            decoupling of the runtime (hardware only), or None to switch it
            off.

    Returns:
        (backend, estimator). On the simulator the Estimator carries the
        pulse error.
    """
    if backend_name is None:
        noise = NoiseModel()
        if pulse_error > 0:
            noise.add_all_qubit_quantum_error(
                depolarizing_error(pulse_error, 1), ["x", "y"]
            )
        options = {"backend_options": {"noise_model": noise}}
        return AerSimulator(), AerEstimator(options=options)
    # Only hardware runs need qiskit_ibm_runtime and a saved IBM account.
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

    backend = QiskitRuntimeService().backend(backend_name)
    estimator = EstimatorV2(mode=backend)
    # Raw measurements: no error mitigation.
    estimator.options.resilience_level = 0
    if sequence_type is None:
        estimator.options.dynamical_decoupling.enable = False
    else:
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = sequence_type
    return backend, estimator


def add_relaxation(circuit, dt: float):
    """Make every delay of a compiled circuit relax like the simulated qubit.

    Args:
        circuit: Compiled circuit with Delay instructions in dt.
        dt: Sample time in seconds.

    Returns:
        The circuit with a thermal-relaxation error on each delay.
    """
    relax = RelaxationNoisePass(
        t1s=[SIM_T1 * 1e-6], t2s=[SIM_T2 * 1e-6], dt=dt, op_types=Delay
    )
    return PassManager([relax]).run(circuit)


def make_pub(
    kind: str,
    total: float,
    backend,
    dt: float,
    granularity: int,
    simulated: bool,
    qubit: int | None,
    precision: float,
    count: int | None = None,
):
    """One circuit with its observable, ready for the Estimator.

    On the simulator the pub binds one row of noise offsets per ensemble
    member, and the precision is scaled so that the average over members has
    the requested standard error.

    Args:
        kind: Sequence, see ``ops.pulse_train``. ``"none"`` also serves the
            runtime dynamical decoupling, which adds the pulses itself.
        total: Idle time in microseconds.
        backend: Backend to compile for.
        dt: Sample time in seconds.
        granularity: Allowed delays are a multiple of this many dt.
        simulated: True on the Aer simulator.
        qubit: Physical qubit on hardware.
        precision: Target standard error of the expectation value.
        count: Number of pulses, see ``ops.pulse_train``.

    Returns:
        A tuple (circuit, observable, parameter values, precision).
    """
    pulses = pulse_train(kind, total, count)
    circuit = build_circuit(total, pulses, dt, granularity, noise=simulated)
    obs = observable()
    isa = transpile_for(circuit, backend, None if simulated else qubit)
    if not simulated:
        return (isa, obs.apply_layout(isa.layout), None, precision)
    isa = add_relaxation(isa, dt)
    blocks = circuit.num_parameters
    if blocks == 0:
        return (isa, obs, None, precision)
    offsets = noise_offsets()[:, :blocks]
    return (isa, obs, offsets, precision * np.sqrt(len(offsets)))


def read_results(result, count: int) -> tuple[list, list]:
    """Expectation values and standard errors of the first ``count`` pubs.

    The simulator gives an array over noise members, which is averaged;
    hardware gives a number.
    """
    evs, stds = [], []
    for k in range(count):
        res = result[k].data
        evs.append(float(np.mean(res.evs)))
        stds.append(float(np.sqrt(np.mean(np.square(res.stds)) / np.size(res.evs))))
    return evs, stds


def run(
    arms=ARMS,
    t_max: float = DEFAULT_T_MAX,
    points: int = 31,
    backend_name: str | None = None,
    qubit: int | None = None,
    precision: float = 0.02,
    pulse_error: float = PULSE_ERROR,
    runtime_dd: bool = False,
) -> dict:
    """Measure <X> against idle time for each pulse sequence.

    Args:
        arms: Sequences to run, from ``ops.ARMS``.
        t_max: Longest idle time in microseconds.
        points: Idle times per sequence.
        backend_name: IBM backend name, or None for the Aer simulator.
        qubit: Physical qubit to measure on hardware (ignored on the
            simulator).
        precision: Target standard error of each expectation value.
        pulse_error: Depolarizing probability of one pulse (simulator only).
        runtime_dd: On hardware, also run the runtime's dynamical decoupling
            with XpXm and with XY4 (ignored on the simulator).

    Returns:
        The settings and, for each sequence, the idle times actually run,
        the expectation values, and their standard errors.
    """
    simulated = backend_name is None
    backend, estimator = get_primitives(backend_name, pulse_error)
    dt, granularity = timing(backend)
    times = snap(delay_grid(t_max, points), dt, granularity)

    jobs = [(estimator, list(arms), "none")]
    if runtime_dd and not simulated:
        for name, sequence in RUNTIME_SEQUENCES.items():
            _, runtime_estimator = get_primitives(backend_name, 0.0, sequence)
            jobs.append((runtime_estimator, [name], "none"))

    t0 = time.perf_counter()
    curves = {}
    for job_estimator, names, _ in jobs:
        pubs = []
        for name in names:
            kind = "none" if name in RUNTIME_SEQUENCES else name
            for total in times:
                pubs.append(
                    make_pub(
                        kind,
                        total,
                        backend,
                        dt,
                        granularity,
                        simulated,
                        qubit,
                        precision,
                    )
                )
        print(f"  {len(pubs)} circuits in one job", end="", flush=True)
        result = job_estimator.run(pubs).result()
        print(" done")
        evs, stds = read_results(result, len(pubs))
        for i, name in enumerate(names):
            part = slice(i * len(times), (i + 1) * len(times))
            curves[name] = {
                "times_us": [float(t) for t in times],
                "evs": evs[part],
                "stds": stds[part],
            }
    seconds = time.perf_counter() - t0

    return {
        "backend": backend.name,
        "qubit": qubit,
        "points": points,
        "precision": precision,
        "pulse_error": pulse_error if simulated else None,
        "seconds": seconds,
        "simulated": simulated,
        "curves": curves,
    }


def run_sweep(
    counts=(0, 1, 2, 4, 8, 16, 32, 64),
    pulse_errors=(0.0, 0.005, 0.02),
    total: float = 50.0,
    backend_name: str | None = None,
    qubit: int | None = None,
    precision: float = 0.02,
) -> dict:
    """Measure <X> at one idle time for an increasing number of pulses.

    The pulses are a train of equal X pulses (CPMG). A count of 0 is no
    pulses. On the simulator the run repeats for each pulse error. Hardware
    has its own pulse error, so it runs once.

    Args:
        counts: Numbers of pulses.
        pulse_errors: Depolarizing probabilities of one pulse (simulator
            only).
        total: Idle time in microseconds.
        backend_name: IBM backend name, or None for the Aer simulator.
        qubit: Physical qubit to measure on hardware.
        precision: Target standard error of each expectation value.

    Returns:
        The settings and, for each pulse error, the expectation values and
        standard errors at each count.
    """
    simulated = backend_name is None
    errors = tuple(pulse_errors) if simulated else (None,)
    curves = {}
    for p in errors:
        backend, estimator = get_primitives(backend_name, p or 0.0)
        dt, granularity = timing(backend)
        [snapped] = snap(np.array([total]), dt, granularity)
        pubs = []
        for n in counts:
            kind = "cpmg" if n else "none"
            pubs.append(
                make_pub(
                    kind,
                    snapped,
                    backend,
                    dt,
                    granularity,
                    simulated,
                    qubit,
                    precision,
                    count=n or None,
                )
            )
        print(f"  {len(pubs)} circuits in one job", end="", flush=True)
        result = estimator.run(pubs).result()
        print(" done")
        evs, stds = read_results(result, len(pubs))
        curves[str(p)] = {"evs": evs, "stds": stds}
    return {
        "backend": "aer_simulator" if simulated else backend_name,
        "total_us": float(snapped),
        "counts": list(counts),
        "simulated": simulated,
        "curves": curves,
    }
