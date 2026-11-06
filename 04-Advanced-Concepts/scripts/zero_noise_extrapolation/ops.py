"""Ops: the observable, the ideal answer, and the extrapolation models for ZNE.

The problem: noise pulls an expectation value <O> toward zero, and the longer
the circuit, the harder it pulls. Zero-noise extrapolation (ZNE) measures <O>
at several noise levels and reads off the value at zero noise. We call the
noise level the noise factor: 1 is the device as it is, 3 is a circuit run so
that its noise is three times larger, and so on.

The test circuit is built so that the ideal answer is known without any
simulation. Qubit 0 is rotated by an angle THETA about the Y axis, which gives
<Z0> = cos(THETA). Then N CX gates act on the pair of qubits. A CX gate is its
own inverse, so an even number of them does nothing, and the ideal value of

    <ZZ> = <Z0 Z1> = cos(THETA)

does not depend on N. Everything that makes the device value differ from it
is noise, and N sets how much noise there is.

The simulator puts a depolarizing error on every CX: with probability
SIM_CX_ERROR the two-qubit state is replaced by the fully mixed state. That
shrinks <ZZ> by a factor (1 - SIM_CX_ERROR) per CX gate, so with a noise
factor of lambda (lambda times as many CX gates) the exact value is

    <ZZ>(lambda) = (1 - SIM_CX_ERROR)^(N lambda) cos(THETA).

We know the exact noisy value, so we can check every step.

Models of <O>(lambda), fitted to the data and evaluated at lambda = 0:

linear       c0 + c1 lambda
quadratic    c0 + c1 lambda + c2 lambda^2. With three noise factors this is
             Richardson extrapolation: the one parabola through all three points.
exponential  a exp(-b lambda). Full depolarizing noise ends at <ZZ> = 0, so
             the curve has no offset.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

THETA = np.pi / 3  # rotation of qubit 0, so the ideal <ZZ> is cos(60 deg) = 0.5
SIM_CX_ERROR = 0.01  # depolarizing probability per CX gate on the simulator
N_CX = 20  # CX gates in the default circuit (even, so they cancel)
FACTORS = (1, 3, 5)  # noise factors measured by default

METHODS = ("linear", "quadratic", "exponential")

NAMES = {
    "linear": "Linear",
    "quadratic": "Quadratic (Richardson)",
    "exponential": "Exponential",
}

# The names the IBM Runtime Estimator uses for the same extrapolators.
RUNTIME_NAMES = {
    "linear": "linear",
    "quadratic": "polynomial_degree_2",
    "exponential": "exponential",
}


def observable() -> SparsePauliOp:
    """The operator the circuit measures: Z on both qubits.

    Returns:
        The two-qubit Pauli operator ZZ.
    """
    return SparsePauliOp("ZZ")


def ideal_value(theta: float = THETA) -> float:
    """The noiseless <ZZ> of the test circuit.

    Args:
        theta: Rotation angle of qubit 0 in radians.

    Returns:
        cos(theta). The CX gates cancel in pairs and do not change it.
    """
    return float(np.cos(theta))


def folds_needed(n_gates: int, factor: float) -> int:
    """How many gates must be folded to reach a noise factor.

    Folding one gate adds two gates, so it raises the gate count from n to
    n + 2. Reaching a factor lambda takes (lambda - 1) n / 2 folds.

    Args:
        n_gates: Number of gates that carry noise.
        factor: Requested noise factor, at least 1.

    Returns:
        The number of folds, rounded to a whole gate.
    """
    return round((factor - 1) * n_gates / 2)


def actual_factor(n_gates: int, factor: float) -> float:
    """The noise factor that folding really reaches.

    Args:
        n_gates: Number of gates that carry noise.
        factor: Requested noise factor.

    Returns:
        (n + 2 folds) / n. It equals ``factor`` unless rounding to a whole
        gate changed it.
    """
    return (n_gates + 2 * folds_needed(n_gates, factor)) / n_gates


def exact_value(
    n_cx: int,
    factor: float = 1.0,
    error: float = SIM_CX_ERROR,
    theta: float = THETA,
) -> float:
    """The exact noisy <ZZ> of the simulator.

    Args:
        n_cx: CX gates in the circuit before any folding.
        factor: Noise factor (0 gives the ideal value).
        error: Depolarizing probability of one CX gate.
        theta: Rotation angle of qubit 0 in radians.

    Returns:
        (1 - error)^(n_cx factor) cos(theta).
    """
    return float((1 - error) ** (n_cx * factor) * np.cos(theta))


def simulator_truth(n_cx: int, factors) -> dict:
    """What the simulator was built with, for checking the analysis.

    Args:
        n_cx: CX gates in the circuit before any folding.
        factors: Noise factors that were run.

    Returns:
        The ideal value and the exact value at each noise factor.
    """
    return {
        "ideal": ideal_value(),
        "exact": [exact_value(n_cx, f) for f in factors],
    }


# ------------------------------------------------------------------ models


def linear(x, c0, c1):
    """Straight line: ``c0 + c1 x``. The value at x = 0 is ``c0``."""
    return c0 + c1 * x


def quadratic(x, c0, c1, c2):
    """Parabola: ``c0 + c1 x + c2 x^2``. The value at x = 0 is ``c0``."""
    return c0 + c1 * x + c2 * x**2


def exponential(x, a, b):
    """Decay without offset: ``a exp(-b x)``. The value at x = 0 is ``a``."""
    return a * np.exp(-b * x)


MODELS = {"linear": linear, "quadratic": quadratic, "exponential": exponential}


def initial_guess(method: str, factors: np.ndarray, values: np.ndarray) -> list:
    """Starting parameters for the fit, read off the data.

    The line and the parabola start from the straight line through the first
    and last point. The exponential starts from the rate that joins them.

    Args:
        method: One of ``METHODS``.
        factors: Noise factors.
        values: Measured expectation values.

    Returns:
        Starting values in the order of the model's parameters.
    """
    x0, x1 = factors[0], factors[-1]
    y0, y1 = values[0], values[-1]
    slope = (y1 - y0) / (x1 - x0)
    if method == "linear":
        return [y0 - slope * x0, slope]
    if method == "quadratic":
        return [y0 - slope * x0, slope, 0.0]
    ratio = y1 / y0 if y0 * y1 > 0 else 0.5
    rate = max(-np.log(ratio) / (x1 - x0), 0.01)
    return [y0 * np.exp(rate * x0), rate]
