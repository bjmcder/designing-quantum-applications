"""Execution: run the layer circuits and collect survival probabilities.

For every layer, every length m, and every random draw, one circuit. All
circuits of all layers go out as one job, because on hardware the queue wait
is paid per job.

With no backend name the circuits run on the Aer simulator. Every CX gets a
depolarizing error that depends on its link (ops.SIM_LINK_ERROR), and
read-out has an error. The decay rate of each link follows from those
numbers (``ops.simulator_truth``), so the fits can be checked. With a backend
name the circuits run on IBM hardware, and the device supplies the noise.
"""

import time

import numpy as np
from encoding import build_layer_circuit, cx_count, transpile_for
from ops import LENGTHS, SIM_READOUT, link_layers, sim_link_error, simulator_truth
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def get_backend(backend_name: str | None, n_qubits: int):
    """Create the backend and the sampler.

    Args:
        backend_name: IBM backend name, or None for the Aer simulator.
        n_qubits: Number of qubits in the chain.

    Returns:
        (backend, sampler). The sampler is None on the simulator.
    """
    if backend_name is None:
        noise = NoiseModel()
        for i in range(n_qubits - 1):
            error = depolarizing_error(sim_link_error(i), 2)
            noise.add_quantum_error(error, "cx", [i, i + 1])
            noise.add_quantum_error(error, "cx", [i + 1, i])
        p01, p10 = SIM_READOUT
        noise.add_all_qubit_readout_error(
            ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        )
        return AerSimulator(noise_model=noise), None
    # Only hardware runs need qiskit_ibm_runtime and a saved IBM account.
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    backend = QiskitRuntimeService().backend(backend_name)
    sampler = SamplerV2(mode=backend)
    # Raw gates only: no dynamical decoupling and no twirling.
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    return backend, sampler


def pair_survival(counts: dict, shots: int, pair: tuple[int, int]) -> float:
    """Fraction of shots in which both qubits of a pair read 0.

    Args:
        counts: Bit strings and their counts. The rightmost bit is qubit 0.
        shots: Total number of shots.
        pair: The two qubit numbers.

    Returns:
        The survival probability of the pair.
    """
    zeros = 0
    for bits, n in counts.items():
        reversed_bits = bits.replace(" ", "")[::-1]
        if all(reversed_bits[q] == "0" for q in pair):
            zeros += n
    return zeros / shots


def run(
    n_qubits: int = 4,
    lengths=LENGTHS,
    sequences: int = 20,
    shots: int = 500,
    backend_name: str | None = None,
    chain: list[int] | None = None,
    seed: int = 1,
    simultaneous: bool = True,
) -> dict:
    """Measure the survival probability of every link of a chain.

    Args:
        n_qubits: Number of qubits in the chain.
        lengths: Numbers of two-qubit Cliffords per sequence.
        sequences: Random sequences per length and link.
        shots: Shots per circuit.
        backend_name: IBM backend name, or None for the Aer simulator.
        chain: Physical qubits of the chain, in order (hardware only).
        seed: Seed of the random sequences.
        simultaneous: If True, the links of a layer run together. If False,
            each link runs alone, which leaves out the interference.

    Returns:
        Dict with ``backend``, ``chain``, ``n_qubits``, ``lengths``,
        ``shots``, ``layers``, ``survival`` ({"0-1": {m: [values]}}),
        ``cx_per_clifford``, ``seconds``, and ``truth`` (simulator only).
    """
    rng = np.random.default_rng(seed)
    backend, sampler = get_backend(backend_name, n_qubits)

    layers = link_layers(n_qubits)
    if not simultaneous:
        layers = [[link] for layer in layers for link in layer]

    circuits, owners = [], []  # owner: (layer links, m)
    for layer in layers:
        for m in lengths:
            for _ in range(sequences):
                qc = build_layer_circuit(layer, n_qubits, int(m), rng)
                circuits.append(transpile_for(qc, backend, chain))
                owners.append((layer, int(m)))

    start = time.time()
    if sampler is None:
        result = backend.run(circuits, shots=shots).result()
        all_counts = [result.get_counts(i) for i in range(len(circuits))]
    else:
        result = sampler.run(circuits, shots=shots).result()
        all_counts = [r.data.meas.get_counts() for r in result]
    seconds = time.time() - start

    survival = {
        f"{a}-{b}": {int(m): [] for m in lengths} for layer in layers for a, b in layer
    }
    cx, cliffords = 0, 0
    for circuit, (layer, m), counts in zip(circuits, owners, all_counts):
        for a, b in layer:
            survival[f"{a}-{b}"][m].append(pair_survival(counts, shots, (a, b)))
        cx += cx_count(circuit)
        cliffords += (m + 1) * len(layer)

    return {
        "backend": backend_name or "aer",
        "chain": chain,
        "n_qubits": n_qubits,
        "lengths": [int(m) for m in lengths],
        "shots": shots,
        "layers": [[list(link) for link in layer] for layer in layers],
        "survival": survival,
        "cx_per_clifford": cx / cliffords,
        "seconds": seconds,
        "truth": simulator_truth(n_qubits) if backend_name is None else None,
    }
