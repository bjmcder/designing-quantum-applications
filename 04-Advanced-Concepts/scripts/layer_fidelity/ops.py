"""Ops: layers of two-qubit gates, the RB model for a pair, and layer fidelity.

The problem: randomized benchmarking (see the randomized_benchmarking
example) measures the fidelity of the gates on one qubit. A useful circuit
has many qubits, runs gates on several at once, and its two-qubit gates are
about ten times worse than its one-qubit gates. We want one number that says
how well a whole line of qubits works when it is used this way.

Layer fidelity does that for a chain of N qubits joined by N - 1 links:

    1. Split the links into layers of gates that can run at the same time.
       Two links that share a qubit cannot, so a chain needs two layers:
       the even links (0-1, 2-3, ...) and the odd links (1-2, 3-4, ...).
    2. For each layer, run randomized benchmarking on every link of the layer
       at the same time, each with its own random two-qubit Cliffords. Running
       them together includes the crosstalk between neighbors.
    3. Fit the decay of each link. Convert its decay rate to a process
       fidelity.
    4. Multiply the N - 1 process fidelities. The product is the layer
       fidelity LF.

For a pair of qubits the survival probability (both read 0) follows

    F(m) = A p^m + B

as in the one-qubit case, but the floor is B = 1/4 instead of 1/2, because
a pair has 4 outcomes and a random guess hits 00 one time in four. The
dimension of the pair is d = 4. The decay rate p gives the process fidelity

    PF = p + (1 - p) / d^2,

which is the probability that a gate does its job under the same picture as
before: with probability p the state is kept, and with probability 1 - p it
is replaced by a random one.

The layer fidelity is the product of the link fidelities,

    LF = PF_1 PF_2 ... PF_(N-1).

It falls as the chain grows, so we also quote the error per layered gate,

    EPLG = (4/5) (1 - LF^(1 / (N - 1)))

the error of an average link. The factor 4/5 = d/(d + 1) turns a process
error into the error of an average gate, the number a single gate is quoted
with.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, random_clifford

LENGTHS = (1, 2, 4, 8, 16, 24, 32, 48, 64)  # two-qubit Cliffords per sequence
DIM = 4  # dimension of a pair of qubits

# The simulated chain: a depolarizing error on every CX of each link, in the
# order of the links 0-1, 1-2, 2-3. The pattern repeats for longer chains.
# Read-out flips 0 to 1 with probability SIM_READOUT[0], 1 to 0 with [1].
SIM_LINK_ERROR = (0.010, 0.025, 0.015)
SIM_READOUT = (0.02, 0.04)


def link_layers(n_qubits: int) -> list[list[tuple[int, int]]]:
    """Split the links of a chain into two layers of non-overlapping links.

    Args:
        n_qubits: Number of qubits in the chain, at least 2.

    Returns:
        [even links, odd links], each a list of (a, a + 1). The odd layer is
        empty for a chain of 2.
    """
    links = [(i, i + 1) for i in range(n_qubits - 1)]
    return [links[0::2], links[1::2]]


def sim_link_error(link: int) -> float:
    """Depolarizing error of the CX on link number ``link`` (0 is qubits 0-1)."""
    return SIM_LINK_ERROR[link % len(SIM_LINK_ERROR)]


def pair_sequence(m: int, rng: np.random.Generator) -> QuantumCircuit:
    """Random sequence of m two-qubit Cliffords followed by their inverse.

    Args:
        m: Number of random Cliffords.
        rng: Random number generator.

    Returns:
        Two-qubit circuit that is the identity if there is no noise.
    """
    qc = QuantumCircuit(2)
    total = Clifford(QuantumCircuit(2))  # the identity
    for _ in range(m):
        cliff = random_clifford(2, seed=int(rng.integers(2**31)))
        qc.compose(cliff.to_circuit(), inplace=True)
        qc.barrier()
        total = total.compose(cliff)  # apply total, then cliff
    qc.compose(total.adjoint().to_circuit(), inplace=True)
    return qc


def decay(m, amp, p, offset):
    """RB model: F(m) = A p^m + B.

    Args:
        m: Sequence length (number of Cliffords), scalar or array.
        amp: A, the size of the decaying part.
        p: Decay rate per Clifford, between 0 and 1.
        offset: B, the value the curve approaches for long sequences.

    Returns:
        The survival probability.
    """
    return amp * p ** np.asarray(m, dtype=float) + offset


def initial_guess(m: np.ndarray, y: np.ndarray) -> list[float]:
    """Starting values for the fit: [A, p, B], read off the data."""
    offset = float(y[-1])
    amp = float(y[0] - offset)
    p = float(np.clip(1 - 1 / m[-1], 0.5, 0.9999))
    return [amp, p, offset]


def process_fidelity(p: float) -> float:
    """PF = p + (1 - p) / d^2 for a pair of qubits."""
    return p + (1 - p) / DIM**2


def error_per_clifford(p: float) -> float:
    """Error of an average two-qubit Clifford, r = (1 - p)(d - 1)/d."""
    return (1 - p) * (DIM - 1) / DIM


def layer_fidelity(fidelities) -> float:
    """LF, the product of the process fidelities of the links.

    Args:
        fidelities: Process fidelity of each link.

    Returns:
        The product.
    """
    return float(np.prod(fidelities))


def eplg(lf: float, n_links: int) -> float:
    """Error per layered gate.

    The geometric mean of the link fidelities is LF^(1 / n_links). One minus
    it is the process error of an average link. Multiplying by d / (d + 1)
    = 4/5 turns a process error into the error of an average gate.

    Args:
        lf: Layer fidelity.
        n_links: Number of links in the chain.

    Returns:
        The error per layered gate.
    """
    return DIM / (DIM + 1) * (1 - lf ** (1 / n_links))


def simulator_truth(n_qubits: int, samples: int = 300, seed: int = 0) -> dict:
    """The decay rates that the simulator's noise gives.

    A depolarizing error e on a CX multiplies the signal by (1 - e), and it
    does so for every one of the 15 non-trivial errors of a pair alike. The
    noise therefore passes through the Cliffords unchanged, and a Clifford
    with k CX gates multiplies the signal by (1 - e)^k. Here p is the
    average of (1 - e)^k over random Cliffords. We estimate it from
    ``samples`` random Cliffords compiled the way execution.py compiles them.

    Args:
        n_qubits: Number of qubits in the chain.
        samples: Number of random Cliffords to compile.
        seed: Seed of the random Cliffords.

    Returns:
        Dict with, per link name like ``"0-1"``: ``p`` and ``pf``, plus
        ``cx`` (average CX gates per Clifford), ``lf``, and ``eplg``.
    """
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(samples):
        cliff = random_clifford(2, seed=int(rng.integers(2**31)))
        compiled = transpile(
            cliff.to_circuit(), basis_gates=["cx", "sx", "rz"], optimization_level=0
        )
        counts.append(compiled.count_ops().get("cx", 0))
    counts = np.array(counts)

    out = {}
    for i in range(n_qubits - 1):
        p = float(np.mean((1 - sim_link_error(i)) ** counts))
        out[f"{i}-{i + 1}"] = {"p": p, "pf": process_fidelity(p)}
    lf = layer_fidelity([v["pf"] for v in out.values()])
    out["cx"] = float(counts.mean())
    out["lf"] = lf
    out["eplg"] = eplg(lf, n_qubits - 1)
    return out
