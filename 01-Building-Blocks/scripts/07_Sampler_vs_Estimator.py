#!/usr/bin/env python3
"""Observables and change of basis: the Sampler versus the Estimator.

A quantum computer only ever returns bitstrings from Z-basis measurements.
Yet we often want an *expectation value* <O> of an observable O. There are
two ways to get one:

Sampler (do it yourself)
    Run circuits that end in measurements, get counts, and compute the
    expectation value from the counts. For O = Z, with n0 zeros and n1 ones,
    <Z> = (n0 - n1) / N. To measure X you must first change basis: since
    X = H Z H, apply H before measuring and then treat the result as Z.
    An observable made of several non-commuting terms (here Z + X) needs one
    circuit per term, and you add up the results.

Estimator (let Qiskit do it)
    Give it a circuit WITHOUT measurements and the observable. It picks the
    basis changes, runs the circuits, combines the terms, and returns the
    expectation value together with its standard error. (On Aer there are no
    shots: it returns the exact value plus random noise of size --precision,
    and "std" simply repeats that precision.)

    Hardware caveat: by default IBM's Estimator applies some error mitigation
    (typically readout-error mitigation) on the server, and the Sampler applies
    none. So on hardware the Estimator can look more accurate than raw Sampler
    counts for that reason, not just because of how it measures.

The experiment: the state Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>,
which has <Z> = cos(theta) and <X> = sin(theta). The observable is
O = Z + X, so <O> = cos(theta) + sin(theta). We sweep theta in a single PUB
using a circuit *parameter*.

The steps
---------
    1. Circuit    a parameterized circuit (theta is a placeholder).
    2. Transpile  once for the backend; the parameter survives.
    3. Observable O = Z + X as a SparsePauliOp (Estimator only), mapped to the
                  transpiled circuit's layout.
    4. PUB        Sampler:   (circuit, theta_values)
                  Estimator: (circuit, observable, theta_values)
    5. Execute    Sampler job (two PUBs), Estimator job (one PUB).
    6. Read       counts -> expectation values by hand, versus evs and stds.

Run with no arguments to use Aer and IBM hardware. Pass --skip-hardware to
run only the simulator.
"""

import sys

import numpy as np
from _runtime import banner, get_targets, make_parser
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

THETA = Parameter("theta")
# A 2D array of shape (num_points, num_parameters): one row per theta value.
THETA_VALUES = np.linspace(0, np.pi, 7).reshape(-1, 1)


def build_state() -> QuantumCircuit:
    """Return the parameterized state circuit Ry(theta)|0> with no measurement."""
    qubits = QuantumRegister(1, name="q")
    circuit = QuantumCircuit(qubits, name="ry_state")
    circuit.ry(THETA, qubits[0])
    return circuit


def with_measurement(state: QuantumCircuit, basis: str) -> QuantumCircuit:
    """Return ``state`` followed by a Z measurement, after an optional H.

    Parameters
    ----------
    state : QuantumCircuit
        The circuit without measurements.
    basis : str
        "Z" to measure Z directly. "X" to apply H first (change of basis),
        which turns the Z measurement into an X measurement.

    Returns
    -------
    QuantumCircuit
        The circuit ready for a Sampler.
    """
    circuit = state.copy(name=f"measure_{basis}")
    bits = ClassicalRegister(1, name="meas")
    circuit.add_register(bits)
    if basis == "X":
        circuit.h(0)
    circuit.measure(0, bits)
    return circuit


def expectation_from_counts(counts: dict[str, int]) -> float:
    """Turn counts of '0' and '1' into <Z> = (n0 - n1) / N."""
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    return (n0 - n1) / (n0 + n1)


def main() -> int:
    """Estimate <Z + X> over a sweep of theta, both ways."""
    args = make_parser(__doc__).parse_args()

    # ---- STEP 1: circuit ---------------------------------------------------
    state = build_state()
    banner("STEP 1 - Circuit (theta is a parameter, not yet a number)")
    print(state.draw(output="text"))

    # ---- STEP 3: observable (does not depend on the target) ----------------
    # "Z + X" as Pauli strings. This is a sum of two terms that do not commute,
    # so no single measurement basis gives both: that is why a Sampler needs
    # two circuits for it.
    observable = SparsePauliOp(["Z", "X"], coeffs=[1.0, 1.0])
    print("\nSTEP 3 - Observable O =", observable.to_list())

    exact = np.cos(THETA_VALUES[:, 0]) + np.sin(THETA_VALUES[:, 0])

    for target in get_targets(args):
        banner(f"Running on {target.label}")

        # ---- STEP 2: transpile -------------------------------------------
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )

        # ================= Sampler: two circuits, by hand =================
        isa_z = pass_manager.run(with_measurement(state, "Z"))
        isa_x = pass_manager.run(with_measurement(state, "X"))
        # STEP 4: PUB = (circuit, parameter values). All 7 theta values travel
        # in one PUB, so the sweep is 7 circuits from 1 PUB.
        sampler_pubs = [(isa_z, THETA_VALUES), (isa_x, THETA_VALUES)]
        # STEP 5
        result = target.wait(target.sampler.run(sampler_pubs, shots=args.shots))
        if result is None:
            continue  # don't submit the Estimator job if this one failed
        # STEP 6: data.meas is a BitArray with one entry per theta value;
        # get_counts(i) picks the i-th.
        z_counts, x_counts = (r.data.meas for r in result)
        by_hand = np.array(
            [
                expectation_from_counts(z_counts.get_counts(i))
                + expectation_from_counts(x_counts.get_counts(i))
                for i in range(len(THETA_VALUES))
            ]
        )

        # ================= Estimator: one circuit, one observable =========
        isa_state = pass_manager.run(state)  # no measurements!
        # The transpiler may place the qubit on a different physical qubit.
        # Rewrite the observable to match.
        isa_observable = observable.apply_layout(isa_state.layout)
        # STEP 4: PUB = (circuit, observable, parameter values).
        estimator_pubs = [(isa_state, isa_observable, THETA_VALUES)]
        # STEP 5
        result = target.wait(
            target.estimator.run(estimator_pubs, precision=args.precision)
        )
        evs = stds = None
        if result is not None:
            # STEP 6: one number and one standard error per theta value.
            evs = np.asarray(result[0].data.evs)
            stds = np.asarray(result[0].data.stds)

        # ---- compare -----------------------------------------------------
        print("\n  theta/pi   exact   Sampler (by hand)   Estimator (ev +/- std)")
        for i, theta in enumerate(THETA_VALUES[:, 0]):
            s = f"{by_hand[i]:+.3f}" if by_hand is not None else "  n/a"
            e = (
                f"{evs[i]:+.3f} +/- {stds[i]:.3f}"
                if evs is not None
                else "n/a"
            )
            print(f"  {theta / np.pi:7.3f}  {exact[i]:+.3f}   {s:>17}   {e}")

    print(
        "\nTakeaway: both recover cos(theta) + sin(theta). The Sampler needed us to "
        "add an H and combine counts;\nthe Estimator did that bookkeeping and also "
        "reported the uncertainty."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
