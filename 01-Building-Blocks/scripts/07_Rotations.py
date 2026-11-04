#!/usr/bin/env python3
"""Notebook 07, the experiment: R_y versus R_x.

This script runs the experiment from notebooks/07_Rotations.ipynb from the
command line. The notebook defines every term used here and explains each
step; this file is the same code without the explanations.

R_y(theta)|0> and R_x(theta)|0> have the same Z-basis probability,
P(1) = sin^2(theta/2), so a Z-basis measurement cannot tell them apart. In
the X basis they separate: P(0) = (1 + sin(theta))/2 for R_y and 1/2 for R_x
at every theta. The experiment sweeps theta over 9 angles from 0 to 2 pi for
both gates and measures in both bases.

The recipe
----------
    Encoding   every qubit starts in |0>; nothing else is needed.
    Operator   R_y(theta) or R_x(theta), then a Z-basis or an X-basis
               measurement (H first): four circuits, each with the single
               parameter theta.
    Execution  transpile each circuit once, with theta still a placeholder;
               each PUB is (circuit, THETA_VALUES), so four PUBs carry all 36
               runs in one Sampler job.
    Analysis   a table of measured probabilities next to the predictions, for
               each angle.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

THETA = Parameter("theta")
# One row per point of the sweep, one column per parameter.
THETA_VALUES = np.linspace(0, 2 * np.pi, 9).reshape(-1, 1)


def rotation_circuit(gate, basis):
    qubits = QuantumRegister(1, name="q")
    bits = ClassicalRegister(1, name="meas")
    circuit = QuantumCircuit(qubits, bits, name=f"{gate}_{basis}")
    getattr(circuit, gate)(THETA, qubits[0])
    if basis == "X":
        circuit.h(qubits[0])  # X-basis measurement, Notebook 06
    circuit.measure(qubits, bits)
    return circuit


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    print("Every qubit starts in |0>, which is the starting state here.")

    banner("Operator")
    names = [("ry", "Z"), ("ry", "X"), ("rx", "Z"), ("rx", "X")]
    circuits = {name: rotation_circuit(*name) for name in names}
    print(circuits[("ry", "X")].draw("text"))

    banner("Execution")
    targets = get_targets(args)
    results = {}
    for target in targets:
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        pubs = [(pass_manager.run(c), THETA_VALUES) for c in circuits.values()]
        job = target.sampler.run(pubs, shots=args.shots)
        # None if a hardware job failed or you stopped waiting.
        results[target.label] = target.wait(job)
        print(f"{target.label}: ran {len(pubs)} PUBs of {len(THETA_VALUES)} angles")

    banner("Analysis")
    thetas = THETA_VALUES[:, 0]
    predicted = {
        ("ry", "Z"): np.sin(thetas / 2) ** 2,  # P(1)
        ("rx", "Z"): np.sin(thetas / 2) ** 2,  # P(1)
        ("ry", "X"): (1 + np.sin(thetas)) / 2,  # P(0)
        ("rx", "X"): np.full_like(thetas, 0.5),  # P(0)
    }
    outcome = {"Z": "1", "X": "0"}

    for label, result in results.items():
        if result is None:
            continue
        measured = {}
        for name, pub_result in zip(circuits, result):
            bits = pub_result.data.meas
            hits = [
                bits.get_counts(i).get(outcome[name[1]], 0) for i in range(len(thetas))
            ]
            measured[name] = np.array(hits) / args.shots
        print(f"{label}, {args.shots} shots per angle")
        print(
            "  theta/pi   Ry: P(1) Z / pred   P(0) X / pred"
            "     Rx: P(1) Z / pred   P(0) X / pred"
        )
        for i, t in enumerate(thetas):
            row = [f"{measured[n][i]:.3f} / {predicted[n][i]:.3f}" for n in names]
            ry_part = f"{row[0]}     {row[1]}"
            rx_part = f"{row[2]}     {row[3]}"
            print(f"  {t / np.pi:7.3f}     {ry_part}         {rx_part}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
