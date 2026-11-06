"""Execution: run the delay circuits and collect <O>(t) for each experiment.

For every experiment and every delay t:

    1. Build the circuit and compile it for the backend.
    2. The Estimator runs it and returns the expectation value of the
       observable from ops.py.

All circuits of all experiments go out as one job, because on hardware the
queue wait is paid per job.

Called from main.py. With no backend name it runs on the Aer simulator with
the noise of ops.py: T1 and T2 relaxation during every delay, plus a
read-out error. The slow noise that makes T2* shorter than T2 is added as an
ensemble: the same circuit runs for many frequency offsets and the results
are averaged. With a backend name it runs on IBM hardware, and the device
supplies all of this.
"""

import time

import numpy as np
from encoding import build_circuit, snap, timing, transpile_for
from ops import (
    DEFAULT_T_MAX,
    EXPERIMENTS,
    SIM_READOUT,
    SIM_T1,
    SIM_T2,
    delay_grid,
    observable,
    simulator_truth,
    slow_shifts,
)
from qiskit.circuit import Parameter
from qiskit.circuit.delay import Delay
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.noise.passes import RelaxationNoisePass
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


def get_primitives(backend_name: str | None):
    """Create the backend and the Estimator.

    Args:
        backend_name: IBM backend name, or None for the Aer simulator.

    Returns:
        (backend, estimator). On the simulator the backend carries the
        read-out error.
    """
    if backend_name is None:
        noise = NoiseModel()
        p01, p10 = SIM_READOUT
        noise.add_all_qubit_readout_error(
            ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        )
        backend = AerSimulator(noise_model=noise)
        return backend, AerEstimator()
    # Only hardware runs need qiskit_ibm_runtime and a saved IBM account.
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

    backend = QiskitRuntimeService().backend(backend_name)
    estimator = EstimatorV2(mode=backend)
    # Raw measurements: no error mitigation, and no dynamical decoupling,
    # which would insert pulses into the waits and stretch every time.
    estimator.options.resilience_level = 0
    estimator.options.dynamical_decoupling.enable = False
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


def run(
    experiments=EXPERIMENTS,
    t_max: dict | None = None,
    points: int = 31,
    backend_name: str | None = None,
    qubit: int | None = None,
    precision: float = 0.02,
) -> dict:
    """Measure the decay curve of each experiment.

    Args:
        experiments: Experiments to run, from ``ops.EXPERIMENTS``.
        t_max: Longest delay of each experiment in microseconds. Missing
            entries use ``ops.DEFAULT_T_MAX``.
        points: Delays per experiment.
        backend_name: IBM backend name, or None for the Aer simulator.
        qubit: Physical qubit to measure on hardware (ignored on the
            simulator).
        precision: Target standard error of each expectation value.

    Returns:
        The settings and, for each experiment, the delays actually run, the
        expectation values, and their standard errors.
    """
    limits = {**DEFAULT_T_MAX, **(t_max or {})}
    backend, estimator = get_primitives(backend_name)
    simulated = backend_name is None
    dt, granularity = timing(backend)
    shifts = slow_shifts()
    shift = Parameter("shift")

    pubs, layout = [], []
    for name in experiments:
        parts = 2 if name == "t2" else 1
        waits, times = snap(delay_grid(limits[name], points), dt, granularity, parts)
        obs = observable(name)
        for wait in waits:
            circuit = build_circuit(name, int(wait), dt, shift if simulated else None)
            isa = transpile_for(circuit, backend, None if simulated else qubit)
            if simulated:
                isa = add_relaxation(isa, dt)
                if name == "t1":  # slow noise along Z does not matter for T1
                    pubs.append((isa, obs, None, precision))
                else:
                    # One value per frequency offset; the results are averaged.
                    scaled = precision * np.sqrt(len(shifts))
                    pubs.append((isa, obs, shifts[:, None], scaled))
            else:
                pubs.append((isa, obs.apply_layout(isa.layout), None, precision))
        layout.append((name, times))

    t0 = time.perf_counter()
    job = estimator.run(pubs)
    print(f"  {len(pubs)} circuits in one job", end="", flush=True)
    result = job.result()
    seconds = time.perf_counter() - t0
    print(f" ({seconds:.0f} s)")

    data = {
        "backend": backend.name,
        "qubit": qubit,
        "points": points,
        "precision": precision,
        "seconds": seconds,
        "truth": simulator_truth() if simulated else None,
        "experiments": {},
    }
    # The Aer Estimator reads the exact expectation value and skips the
    # read-out error, so apply it here. A read-out that turns 0 into 1 with
    # probability p01 and 1 into 0 with probability p10 maps a true value v
    # to (1 - p01 - p10) v + (p10 - p01).
    p01, p10 = SIM_READOUT
    scale, shift_ro = (1 - p01 - p10, p10 - p01) if simulated else (1.0, 0.0)
    k = 0
    for name, times in layout:
        evs, stds = [], []
        for _ in times:
            res = result[k].data
            k += 1
            # The simulator gives an array over offsets; hardware a number.
            ev = float(np.mean(res.evs))
            std = float(np.sqrt(np.mean(np.square(res.stds)) / np.size(res.evs)))
            evs.append(scale * ev + shift_ro)
            stds.append(scale * std)
        data["experiments"][name] = {
            "times_us": [float(t) for t in times],
            "evs": evs,
            "stds": stds,
        }
    return data
