"""Ops: turn T1 and T2 into Kraus operators.

The problem: the benchmarking example measured T1 and T2 of a qubit. To
predict what a circuit does on that qubit, we need a noise model: a rule that
says what one idle step of duration dt does to the state. This file builds
that rule from the two times.

A state of one qubit is described by its density matrix, a 2x2 complex matrix

    rho = [[ p0,  c  ],
           [ c*,  p1 ]]

The diagonal holds the probabilities of reading 0 and 1. The off-diagonal c
is the coherence, the phase information that interference needs. A pure
state |psi> has rho = |psi><psi|.

Noise acts on rho through Kraus operators K_0, K_1, ...:

    rho  ->  sum_i  K_i  rho  K_i^dagger

with the condition  sum_i K_i^dagger K_i = I,  which keeps the total
probability at 1. Each K_i is one way the environment can act on the qubit,
and the sum averages over them.

Two processes make up the noise, and each has a short list of operators.

Amplitude damping (T1). The qubit falls from |1> to |0> with probability g:

    K_0 = [[1, 0], [0, sqrt(1-g)]]       nothing fell
    K_1 = [[0, sqrt(g)], [0, 0]]         |1> became |0>

Phase damping (the part of T2 that T1 does not explain). The environment
learns whether the qubit is in |1> and scrambles the phase, with probability l:

    K_0 = [[1, 0], [0, sqrt(1-l)]]       nothing happened
    K_1 = [[0, 0], [0, sqrt(l)]]         the phase was scrambled

Times are in microseconds.
"""

import numpy as np

T1 = 100.0  # us, the value built into the benchmarking simulator
T2 = 70.0  # us, ditto. Replace both with the times you measured.
STEP = 1.0  # us, the duration of one idle step


def dephasing_time(t1: float, t2: float) -> float:
    """Time T_phi of the pure dephasing that T1 does not explain.

    Amplitude damping alone makes the coherence decay as exp(-t / 2 T1).
    The measured decay is exp(-t / T2). The rates add:

        1 / T2 = 1 / (2 T1) + 1 / T_phi

    Args:
        t1: Relaxation time T1.
        t2: Dephasing time T2. It cannot exceed 2 * t1.

    Returns:
        T_phi, in the same unit.

    Raises:
        ValueError: If t2 > 2 * t1, which no physical qubit can do.
    """
    if t2 > 2 * t1:
        raise ValueError("T2 cannot be larger than 2 * T1.")
    rate = 1 / t2 - 1 / (2 * t1)
    return np.inf if rate == 0 else 1 / rate


def amplitude_damping(t: float, t1: float) -> list[np.ndarray]:
    """Kraus operators for a wait of t, with the qubit decaying at rate 1/T1.

    Args:
        t: Duration of the wait.
        t1: Relaxation time.

    Returns:
        [K_0, K_1].
    """
    g = 1 - np.exp(-t / t1)  # probability of falling during t
    k0 = np.array([[1, 0], [0, np.sqrt(1 - g)]])
    k1 = np.array([[0, np.sqrt(g)], [0, 0]])
    return [k0, k1]


def phase_damping(t: float, t1: float, t2: float) -> list[np.ndarray]:
    """Kraus operators for the pure dephasing during a wait of t.

    The coherence must shrink by exp(-t / T_phi). K_0 multiplies it by
    sqrt(1 - l), so l = 1 - exp(-2 t / T_phi).

    Args:
        t: Duration of the wait.
        t1: Relaxation time.
        t2: Dephasing time.

    Returns:
        [K_0, K_1].
    """
    l = 1 - np.exp(-2 * t / dephasing_time(t1, t2))  # noqa: E741
    k0 = np.array([[1, 0], [0, np.sqrt(1 - l)]])
    k1 = np.array([[0, 0], [0, np.sqrt(l)]])
    return [k0, k1]


def relaxation_kraus(t: float, t1: float = T1, t2: float = T2) -> list[np.ndarray]:
    """Kraus operators for T1 and T2 together: damping, then dephasing.

    Doing one channel and then the other means every operator of the second is
    multiplied with every operator of the first, so 2 x 2 = 4 operators.

    Args:
        t: Duration of the wait.
        t1: Relaxation time.
        t2: Dephasing time.

    Returns:
        List of four 2x2 matrices.
    """
    first = amplitude_damping(t, t1)
    second = phase_damping(t, t1, t2)
    return [b @ a for a in first for b in second]


def check_complete(kraus: list[np.ndarray]) -> float:
    """How far the operators are from sum K^dagger K = I.

    Args:
        kraus: List of 2x2 Kraus operators.

    Returns:
        The largest entry of |sum K^dagger K - I|. It should be near 0.
    """
    total = sum(k.conj().T @ k for k in kraus)
    return float(np.abs(total - np.eye(2)).max())


def apply_channel(kraus: list[np.ndarray], rho: np.ndarray) -> np.ndarray:
    """Apply the channel: rho -> sum K rho K^dagger.

    Args:
        kraus: List of 2x2 Kraus operators.
        rho: 2x2 density matrix.

    Returns:
        The new density matrix.
    """
    return sum(k @ rho @ k.conj().T for k in kraus)


def expected(name: str, t, t1: float = T1, t2: float = T2):
    """The decay curves fitted in the benchmarking example.

    These are formulas, not simulations, so they are what the noise model
    should reproduce.

    t1   prepare |1>, measure Z:   <Z>(t) = 1 - 2 exp(-t / T1)
    t2   prepare |+>, measure X:   <X>(t) = exp(-t / T2)

    Args:
        name: "t1" or "t2".
        t: Wait time or array of wait times.
        t1: Relaxation time.
        t2: Dephasing time.

    Returns:
        Expectation value at t.
    """
    t = np.asarray(t, dtype=float)
    if name == "t1":
        return 1 - 2 * np.exp(-t / t1)
    return np.exp(-t / t2)
