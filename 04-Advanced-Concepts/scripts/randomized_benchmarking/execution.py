"""Execution: run the random sequences and collect survival probabilities.

For every length m, several random sequences are drawn. Each is a separate
circuit, because the average over sequences is what the theory describes.
All circuits go out as one job, since on hardware the queue wait is paid per
job.

With no backend name the circuits run on the Aer simulator with a
depolarizing error on every H and S gate and a read-out error. The correct
decay rate follows from those numbers (``ops.simulator_truth``), so the fit
can be checked. With a backend name the circuits run on IBM hardware and the
device supplies the noise.
"""

import time

import numpy as np
from encoding import build_circuit, gate_count, transpile_for
from ops import (
    LENGTHS,
    SIM_GATE_ERROR,
    SIM_READOUT,
    random_sequence,
    simulator_truth,
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def get_backend(backend_name: str | None, readout=SIM_READOUT):
    """Create the backend and the sampler.

    Args:
        backend_name: IBM backend name, or None for the Aer simulator.
        readout: (P(read 1 | state 0), P(read 0 | state 1)) of the simulator.

    Returns:
        (backend, sampler).
    """
    if backend_name is None:
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(
            depolarizing_error(SIM_GATE_ERROR, 1), ["h", "s"]
        )
        p01, p10 = readout
        noise.add_all_qubit_readout_error(
            ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        )
        backend = AerSimulator(noise_model=noise)
        return backend, None
    # Only hardware runs need qiskit_ibm_runtime and a saved IBM account.
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    backend = QiskitRuntimeService().backend(backend_name)
    sampler = SamplerV2(mode=backend)
    # Raw gates only: no dynamical decoupling and no twirling.
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    return backend, sampler


def run(
    lengths=LENGTHS,
    sequences: int = 20,
    shots: int = 1000,
    backend_name: str | None = None,
    qubit: int | None = None,
    seed: int = 1,
    readout=SIM_READOUT,
) -> dict:
    """Measure the survival probability of random sequences.

    Args:
        lengths: Numbers of Cliffords per sequence.
        sequences: Random sequences per length.
        shots: Shots per sequence.
        backend_name: IBM backend name, or None for the Aer simulator.
        qubit: Physical qubit on hardware.
        seed: Seed of the random sequences.
        readout: Read-out error of the simulator.

    Returns:
        Dict with ``backend``, ``qubit``, ``lengths``, ``shots``,
        ``survival`` (a list per length, one value per sequence),
        ``gates_per_clifford``, ``seconds``, and ``truth`` (simulator only).
    """
    rng = np.random.default_rng(seed)
    backend, sampler = get_backend(backend_name, readout)

    circuits, owners, counts = [], [], []
    for m in lengths:
        for _ in range(sequences):
            circuit = transpile_for(
                build_circuit(random_sequence(int(m), rng)), backend, qubit
            )
            circuits.append(circuit)
            owners.append(int(m))
            counts.append(gate_count(circuit))

    start = time.time()
    if sampler is None:
        result = backend.run(circuits, shots=shots).result()
        zeros = [result.get_counts(i).get("0", 0) for i in range(len(circuits))]
    else:
        job = sampler.run(circuits, shots=shots)
        result = job.result()
        zeros = [r.data.meas.get_counts().get("0", 0) for r in result]
    seconds = time.time() - start

    survival = {int(m): [] for m in lengths}
    for m, z in zip(owners, zeros):
        survival[m].append(z / shots)

    # Each sequence has m + 1 Cliffords (the inverse is one more).
    per_clifford = float(np.mean([g / (m + 1) for g, m in zip(counts, owners)]))
    return {
        "backend": backend_name or "aer",
        "qubit": qubit,
        "lengths": [int(m) for m in lengths],
        "shots": shots,
        "survival": survival,
        "gates_per_clifford": per_clifford,
        "seconds": seconds,
        "truth": simulator_truth() if backend_name is None else None,
    }
