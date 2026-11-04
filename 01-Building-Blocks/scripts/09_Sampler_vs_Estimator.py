#!/usr/bin/env python3
"""Notebook 09, the experiment: <Z + X> two ways.

This script runs the experiment from notebooks/09_Sampler_vs_Estimator.ipynb
from the command line. The notebook defines every term used here and explains
each step; this file is the same code without the explanations.

We sweep theta over 7 values from 0 to pi for the state Ry(theta)|0> and
measure the observable O = Z + X, whose exact expectation value is
cos(theta) + sin(theta): first with the Sampler, combining counts by hand,
then with the Estimator. A second Estimator PUB measures the penalty
Hamiltonian (I + Z1 Z0)/2 of Notebook 08 in the state |++>, made by an H on
each qubit; its exact expectation value is 0.5.

The recipe
----------
    Encoding   Ry(theta)|0>, with theta a circuit parameter; and |++>.
    Operator   Z + X and the penalty Hamiltonian, as SparsePauliOps.
    Execution  part A: two Sampler circuits, one measuring Z and one with an H
               before the measurement (the X basis); part B: the Estimator,
               on the unmeasured circuits with each observable laid out to
               match.
    Analysis   each measured value with its standard error, next to the exact
               value. Most points should lie within twice their standard error.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

THETA = Parameter("theta")
# Shape (num_points, num_parameters): one row per theta value.
THETA_VALUES = np.linspace(0, np.pi, 7).reshape(-1, 1)


def with_measurement(state, basis):
    circuit = state.copy(name=f"measure_{basis}")
    bits = ClassicalRegister(1, name="meas")
    circuit.add_register(bits)
    if basis == "X":
        circuit.h(0)
    circuit.measure(0, bits)
    return circuit


def expectation_from_counts(counts):
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    return (n0 - n1) / (n0 + n1)


def sampler_std(z_counts, x_counts):
    variance = 0.0
    for counts in (z_counts, x_counts):
        variance += (1 - expectation_from_counts(counts) ** 2) / sum(counts.values())
    return float(np.sqrt(variance))


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    qubits = QuantumRegister(1, name="q")
    state = QuantumCircuit(qubits, name="ry_state")
    state.ry(THETA, qubits[0])
    print(state.draw("text"))
    plus_plus = QuantumCircuit(2, name="plus_plus")
    plus_plus.h([0, 1])
    print(plus_plus.draw("text"))

    banner("Operator")
    observable = SparsePauliOp(["Z", "X"], coeffs=[1.0, 1.0])
    print([(p, float(c.real)) for p, c in observable.to_list()])
    exact = np.cos(THETA_VALUES[:, 0]) + np.sin(THETA_VALUES[:, 0])

    penalty = SparsePauliOp.from_list([("II", 0.5), ("ZZ", 0.5)])
    exact_penalty = Statevector(plus_plus).expectation_value(penalty).real
    print("exact <penalty> on |++>:", round(exact_penalty, 3))

    banner("Execution, part A: the Sampler, by hand")
    targets = get_targets(args)
    print(with_measurement(state, "X").draw("text"))
    sampler_results = {}
    for target in targets:
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        isa_z = pass_manager.run(with_measurement(state, "Z"))
        isa_x = pass_manager.run(with_measurement(state, "X"))
        sampler_pubs = [(isa_z, THETA_VALUES), (isa_x, THETA_VALUES)]
        # wait() returns None if a hardware job failed or you stopped waiting.
        sampler_results[target.label] = target.wait(
            target.sampler.run(sampler_pubs, shots=args.shots)
        )
        print(f"{target.label}: ran {len(sampler_pubs)} Sampler PUBs")

    banner("Execution, part B: the Estimator")
    estimator_results = {}
    for target in targets:
        if sampler_results[target.label] is None:
            continue  # the Sampler job failed; do not submit another
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        isa_state = pass_manager.run(state)
        isa_plus_plus = pass_manager.run(plus_plus)
        estimator_pubs = [
            (isa_state, observable.apply_layout(isa_state.layout), THETA_VALUES),
            (isa_plus_plus, penalty.apply_layout(isa_plus_plus.layout)),
        ]
        estimator_results[target.label] = target.wait(
            target.estimator.run(estimator_pubs, precision=args.precision)
        )
        print(f"{target.label}: ran {len(estimator_pubs)} Estimator PUBs")

    banner("Analysis")
    by_hand, by_hand_std, estimates = {}, {}, {}
    for label, result in sampler_results.items():
        if result is None:
            continue
        z_bits, x_bits = (r.data.meas for r in result)
        points = range(len(THETA_VALUES))
        z_counts = [z_bits.get_counts(i) for i in points]
        x_counts = [x_bits.get_counts(i) for i in points]
        by_hand[label] = np.array(
            [
                expectation_from_counts(zc) + expectation_from_counts(xc)
                for zc, xc in zip(z_counts, x_counts)
            ]
        )
        by_hand_std[label] = np.array(
            [sampler_std(zc, xc) for zc, xc in zip(z_counts, x_counts)]
        )

    for label, result in estimator_results.items():
        if result is None:
            continue
        estimates[label] = (
            np.asarray(result[0].data.evs),
            np.asarray(result[0].data.stds),
        )
        penalty_ev = float(result[1].data.evs)
        penalty_std = float(result[1].data.stds)
        print(
            f"{label}: <penalty> on |++> = {penalty_ev:.3f} +/- {penalty_std:.3f}"
            "   (exact 0.500)"
        )

    for label in by_hand:
        print(f"\n{label}: <Z + X>, each measured value +/- its standard error")
        print(
            f"  {'theta/pi':>8s}  {'exact':>6s}  {'Sampler':>15s}  {'Estimator':>15s}"
        )
        for i, theta in enumerate(THETA_VALUES[:, 0]):
            s = f"{by_hand[label][i]:+.3f} +/- {by_hand_std[label][i]:.3f}"
            e = (
                f"{estimates[label][0][i]:+.3f} +/- {estimates[label][1][i]:.3f}"
                if label in estimates
                else "n/a"
            )
            print(f"  {theta / np.pi:8.3f}  {exact[i]:+6.3f}  {s:>15}  {e:>15}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
