"""Encoding: the idle circuits, with and without pulses.

Each circuit prepares |+> with H, idles for a total time t, and leaves the
qubit ready for the observable in ops.py (nothing is measured here, the
Estimator does that).

none     H, wait t
echo     H, wait t/2, X, wait t/2
xy4      H, wait, X, wait, Y, wait, X, wait, Y, wait (pulses at the centers
         of equal slices of t, see ops.pulse_train)

The wait is a Delay instruction: the qubit sits idle while the clock runs.
The pulses are ordinary X and Y gates. On the simulator, the frequency noise
is added as a phase rotation RZ(offset * duration) on every piece of the wait.
Because the pulses are real gates in the circuit, the cancellation is not put
in by hand. It happens because X RZ(a) X = RZ(-a): a pulse reverses the sign of
the phase that comes after it, relative to the phase that came before.

Delays are counted in dt, the sample time of the control electronics, and
have to be a multiple of a granularity. The pieces of the wait are rounded to
the nearest allowed length, and the analysis uses the idle time that was
actually run.
"""

import numpy as np
from ops import segments
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
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


def snap(times_us: np.ndarray, dt: float, granularity: int) -> np.ndarray:
    """Round idle times to what the hardware can run.

    Args:
        times_us: Requested idle times in microseconds.
        dt: Sample time in seconds.
        granularity: Allowed delays are a multiple of this many dt.

    Returns:
        The idle times that can be run, in microseconds.
    """
    unit = granularity * dt * 1e6  # us
    return np.rint(np.asarray(times_us) / unit) * unit


def build_circuit(
    total: float,
    pulses: list,
    dt: float,
    granularity: int,
    noise: bool = False,
) -> QuantumCircuit:
    """The idle circuit for one idle time and one set of pulses.

    Args:
        total: Idle time in microseconds.
        pulses: Output of ``ops.pulse_train``, empty for no pulses.
        dt: Sample time in seconds.
        granularity: Allowed delays are a multiple of this many dt.
        noise: True to add the frequency noise of the simulator as phase
            rotations. They use one Parameter per block of fast noise, named
            ``shift[0]``, ``shift[1]``, and so on, in radians per microsecond.
            Hardware has its own noise, so pass False.

    Returns:
        A one-qubit circuit.
    """
    unit = granularity * dt * 1e6  # us
    circuit = QuantumCircuit(1, name=f"idle_{total:.1f}us")
    circuit.h(0)
    pieces = segments(total, pulses)
    shifts = None
    if noise and pieces:
        shifts = ParameterVector("shift", pieces[-1][2] + 1)
    for start, stop, block, _, axis in pieces:
        length = round(stop / unit) - round(start / unit)
        if length > 0:
            circuit.delay(length * granularity, 0, unit="dt")
            if shifts is not None:
                circuit.rz(shifts[block] * (length * unit), 0)
        if axis == "x":
            circuit.x(0)
        elif axis == "y":
            circuit.y(0)
    return circuit


def transpile_for(circuit: QuantumCircuit, backend, qubit: int | None = None):
    """Compile an idle circuit without changing it.

    Optimization is off. The optimizer would see that X, wait, X does
    nothing and delete the pulses, or merge the delays away.

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
