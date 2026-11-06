"""Encoding: the circuits of the calibration, the sweep, and the twirl.

Every circuit has one qubit and one classical bit.

calibration   X (only for the state |1>), then measure.
sweep         RY(theta), then measure. The true <Z> is cos(theta).
twirl         The same circuit with an extra X gate just before the
              measurement. The X flips the bit. The classical answer is
              flipped back in execution.py, so the ideal answer does not
              change.

A barrier sits between the state preparation, the twirl X, and the
measurement. The optimizer is also off at compile time. Without these, the
compiler may merge the X into the state preparation and the twirl would
vanish before it reached the device.
"""

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def _measure(circuit: QuantumCircuit, flip: bool) -> QuantumCircuit:
    """Add the optional twirl X and the measurement to a prepared circuit."""
    circuit.barrier()
    if flip:
        circuit.x(0)
        circuit.barrier()
    circuit.measure(0, 0)
    return circuit


def calibration_circuit(state: int, flip: bool = False) -> QuantumCircuit:
    """Prepare |0> or |1> and measure.

    Args:
        state: 0 or 1, the bit to prepare.
        flip: If True, add the twirl X before the measurement.

    Returns:
        A one-qubit circuit with one classical bit.
    """
    if state not in (0, 1):
        raise ValueError("state must be 0 or 1")
    circuit = QuantumCircuit(1, 1, name=f"cal{state}{'_twirl' if flip else ''}")
    if state == 1:
        circuit.x(0)
    return _measure(circuit, flip)


def sweep_circuit(theta: float, flip: bool = False) -> QuantumCircuit:
    """Prepare RY(theta)|0> and measure.

    Args:
        theta: Rotation angle in radians.
        flip: If True, add the twirl X before the measurement.

    Returns:
        A one-qubit circuit with one classical bit.
    """
    circuit = QuantumCircuit(1, 1, name=f"ry{'_twirl' if flip else ''}")
    circuit.ry(theta, 0)
    return _measure(circuit, flip)


def preparation_circuit(theta: float) -> QuantumCircuit:
    """RY(theta)|0> without a measurement, for the Estimator.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A one-qubit circuit.
    """
    circuit = QuantumCircuit(1, name="ry")
    circuit.ry(theta, 0)
    return circuit


def transpile_for(circuit: QuantumCircuit, backend, qubit: int | None = None):
    """Compile a circuit without changing it.

    Optimization is off, so the twirl X stays where we put it.

    Args:
        circuit: A circuit from this module.
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
