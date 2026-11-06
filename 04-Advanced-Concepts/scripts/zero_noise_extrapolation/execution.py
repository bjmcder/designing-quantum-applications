"""Execution: measure <ZZ> of the test circuit at several noise factors.

For every circuit length N and every noise factor lambda:

    1. Build the circuit of encoding.py and fold it to the noise factor.
    2. The Estimator runs it and returns <ZZ>.

The whole measurement can be repeated, to see how much the answer changes from
run to run. All circuits go out as one job, because on hardware the queue
wait is paid per job.

Called from main.py. With no backend name it runs on the Aer simulator with a
depolarizing error on every CX gate (ops.SIM_CX_ERROR), and the folding is
done here, with encoding.fold_gates. With a backend name it runs on IBM
hardware. The Runtime Estimator folds the gates and extrapolates for us when
we switch on ``options.resilience.zne_mitigation``. It returns <ZZ> at every
noise factor, so the analysis fits the same data in both cases. The hardware
path has not been tested.
"""

import time

import numpy as np
from encoding import build_circuit, fold_gates, transpile_for
from ops import (
    FACTORS,
    METHODS,
    N_CX,
    RUNTIME_NAMES,
    SIM_CX_ERROR,
    exact_value,
    ideal_value,
    observable,
)
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


def get_primitives(backend_name: str | None, factors=FACTORS):
    """Create the backend and the Estimator.

    Args:
        backend_name: IBM backend name, or None for the Aer simulator.
        factors: Noise factors the Runtime Estimator should use (hardware
            only; on the simulator we fold the circuits ourselves).

    Returns:
        (backend, estimator). On the simulator the backend is None and the
        Estimator carries the depolarizing error of every CX gate.
    """
    if backend_name is None:
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(SIM_CX_ERROR, 2), ["cx"])
        return None, AerEstimator(options={"backend_options": {"noise_model": noise}})
    # Only hardware runs need qiskit_ibm_runtime and a saved IBM account.
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

    backend = QiskitRuntimeService().backend(backend_name)
    estimator = EstimatorV2(mode=backend)
    zne = estimator.options.resilience
    zne.zne_mitigation = True
    zne.zne.noise_factors = tuple(float(f) for f in factors)
    zne.zne.extrapolator = tuple(RUNTIME_NAMES[m] for m in METHODS)
    zne.zne.amplifier = "gate_folding"
    # Twirling turns coherent gate errors into the depolarizing kind that
    # folding amplifies cleanly.
    estimator.options.twirling.enable_gates = True
    return backend, estimator


def run(
    n_cx=(N_CX,),
    factors=FACTORS,
    repeats: int = 1,
    precision: float = 0.02,
    backend_name: str | None = None,
    qubits: list[int] | None = None,
) -> dict:
    """Measure <ZZ> at every noise factor for every circuit length.

    Args:
        n_cx: Numbers of CX gates (even numbers), one circuit length each.
        factors: Noise factors, from 1 up.
        repeats: How many times to repeat the whole measurement (simulator
            only; hardware runs once).
        precision: Target standard error of each expectation value. On
            hardware it sets the shots, about 1 / precision^2. The simulator
            adds Gaussian noise of this size to the exact value.
        backend_name: IBM backend name, or None for the Aer simulator.
        qubits: The two physical qubits to run on (hardware only).

    Returns:
        Dict with ``backend``, ``n_cx``, ``factors``, ``precision``,
        ``evs`` (nested list: circuit length, repeat, noise
        factor), ``ideal``, ``exact`` (simulator only: the exact noisy value
        for each circuit length and noise factor), ``runtime`` (hardware
        only: the zero-noise value from each Runtime extrapolator), and
        ``seconds``.
    """
    simulated = backend_name is None
    factors = [float(f) for f in factors]
    backend, estimator = get_primitives(backend_name, factors)
    obs = observable()
    n_cx = [int(n) for n in n_cx]
    repeats = repeats if simulated else 1

    pubs = []
    for n in n_cx:
        base = build_circuit(n)
        if simulated:
            for _ in range(repeats):
                for f in factors:
                    pubs.append((fold_gates(base, f), obs, None, precision))
        else:
            isa = transpile_for(base, backend, qubits)
            pubs.append((isa, obs.apply_layout(isa.layout), None, precision))

    t0 = time.perf_counter()
    job = estimator.run(pubs)
    print(f"  {len(pubs)} circuits in one job", end="", flush=True)
    result = job.result()
    seconds = time.perf_counter() - t0
    print(f" ({seconds:.0f} s)")

    evs, runtime = [], []
    for k, n in enumerate(n_cx):
        if simulated:
            block = result[
                k * repeats * len(factors) : (k + 1) * repeats * len(factors)
            ]
            values = np.array([float(r.data.evs) for r in block])
            evs.append(values.reshape(repeats, len(factors)).tolist())
        else:
            data = result[k].data
            evs.append([np.asarray(data.evs_noise_factors, dtype=float).tolist()])
            # Row i is extrapolator i; the first column is noise factor 0.
            zero = np.asarray(data.evs_extrapolated, dtype=float)[..., 0]
            runtime.append(dict(zip(METHODS, zero.tolist())))

    return {
        "backend": backend_name or "aer",
        "qubits": qubits,
        "n_cx": n_cx,
        "factors": factors,
        "precision": precision,
        "evs": evs,
        "ideal": ideal_value(),
        "exact": [[exact_value(n, f) for f in factors] for n in n_cx]
        if simulated
        else None,
        "runtime": runtime or None,
        "seconds": seconds,
    }
