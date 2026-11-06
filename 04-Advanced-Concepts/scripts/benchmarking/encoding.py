"""Encoding: the delay circuits of the three experiments.

Each circuit prepares a state, waits, and leaves the qubit ready for the
observable in ops.py (nothing is measured here, the Estimator does that).

t1        X                         Z
t2star    H, wait, RZ(2 pi f t)     X
t2        H, wait t/2, X, wait t/2  X

H puts the qubit in |+>. The wait is a Delay instruction: the qubit sits idle
while the clock runs. RZ(2 pi f t) is a rotation of the phase by the angle
that a detuning f would have collected in the wait. We apply it in software
so that the Ramsey fringes oscillate at a frequency we know. The X in the
middle of the echo exchanges |0> and |1>, so the phase collected in the first
half is undone in the second half.

Delays are counted in dt, the sample time of the control electronics, and
have to be a multiple of a granularity. ``snap`` rounds a requested delay to
the nearest allowed one, and the analysis uses the delay that was actually run.
"""

import numpy as np
from ops import DETUNING
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

DEFAULT_DT = 2.2222e-10  # s, used when the backend does not report one
DEFAULT_GRANULARITY = 16  # delays are a multiple of this many dt


def timing(backend) -> tuple[float, int]:
    """Sample time and delay granularity of a backend.

    Args:
        backend: Backend the circuits will run on.

    Returns:
        (dt in seconds, granularity in dt).
    """
    dt = getattr(backend, "dt", None) or DEFAULT_DT
    constraints = getattr(backend.target, "timing_constraints", None)
    granularity = constraints().granularity if constraints else 1
    return dt, max(granularity, DEFAULT_GRANULARITY)


def snap(
    times_us: np.ndarray, dt: float, granularity: int, parts: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Round delays to what the hardware can run.

    Args:
        times_us: Requested total delays in microseconds.
        dt: Sample time in seconds.
        granularity: Allowed delays are a multiple of this many dt.
        parts: Number of equal waits the delay is split into (2 for an echo).

    Returns:
        (length of one wait in dt, total delay actually run in microseconds).
    """
    unit = granularity * dt * 1e6  # us
    waits = np.rint(np.asarray(times_us) / parts / unit).astype(int) * granularity
    return waits, waits * parts * dt * 1e6


def build_circuit(
    name: str, wait: int, dt: float, shift: Parameter | None = None
) -> QuantumCircuit:
    """The delay circuit of one experiment at one delay.

    Args:
        name: Experiment, one of ``ops.EXPERIMENTS``.
        wait: Length of one wait, in dt.
        dt: Sample time in seconds.
        shift: Optional frequency offset in radians per microsecond, applied
            as a phase rotation during the wait. The simulator uses it to
            stand in for slow noise. Hardware has its own, so pass None.

    Returns:
        A one-qubit circuit.
    """
    circuit = QuantumCircuit(1, name=name)
    wait_us = wait * dt * 1e6
    if name == "t1":
        circuit.x(0)
        circuit.delay(wait, 0, unit="dt")
    elif name == "t2star":
        circuit.h(0)
        circuit.delay(wait, 0, unit="dt")
        phase = 2 * np.pi * DETUNING * wait_us
        circuit.rz(phase if shift is None else phase + shift * wait_us, 0)
    else:
        circuit.h(0)
        circuit.delay(wait, 0, unit="dt")
        if shift is not None:
            circuit.rz(shift * wait_us, 0)
        circuit.x(0)
        circuit.delay(wait, 0, unit="dt")
        if shift is not None:
            circuit.rz(shift * wait_us, 0)
    return circuit


def transpile_for(circuit: QuantumCircuit, backend, qubit: int | None = None):
    """Compile a delay circuit without changing it.

    Optimization is off. The optimizer would see that X, wait, X does
    nothing and delete the echo, or merge the delay away.

    Args:
        circuit: A circuit from ``build_circuit``.
        backend: Backend to compile for.
        qubit: Physical qubit to run on, or None to let the compiler choose.

    Returns:
        The compiled circuit.
    """
    layout = None if qubit is None else [qubit]
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=0, initial_layout=layout
    )
    return pm.run(circuit)
