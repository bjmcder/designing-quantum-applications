#!/usr/bin/env python3
"""Notebook 04, the experiment: same counts, different states.

This script runs the experiment from notebooks/04_Amplitudes_and_Phase.ipynb
from the command line. The notebook defines every term used here and explains
each step; this file is the same code without the explanations.

Four states, |0>, |+>, |-> and |+i>, are measured twice each: once directly
in the Z basis, and once after one more H. Measured directly, |+>, |-> and
|+i> all give about 50/50, as the Born rule predicts. After one more H, |+>
returns 0 on every shot and |-> returns 1 on every shot, which rules out the
picture of a qubit as a coin that is secretly 0 or 1. |+i> then H still gives
50/50.

The recipe
----------
    Encoding   |+> = H|0>, |-> = Z H|0>, |+i> = S H|0>; in a circuit the
               first gate applied is written first, so Z H|0> is h then z.
               Statevector gives the exact amplitudes and P(0).
    Operator   nothing (measure directly), or a barrier and one more H. Both
               end in a Z-basis measurement.
    Execution  transpile for each target, one PUB per circuit, one Sampler job.
    Analysis   the measured fraction of 0s against the exact P(0), with the
               scatter sqrt(p(1 - p)/N) for p = 0.5.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

np.set_printoptions(precision=3, suppress=True)

PREPARE = {"|0>": [], "|+>": ["h"], "|->": ["h", "z"], "|+i>": ["h", "s"]}


def build(name, extra_h):
    qubits = QuantumRegister(1, name="q")
    bits = ClassicalRegister(1, name="meas")
    circuit = QuantumCircuit(qubits, bits, name=f"{name}{' then H' if extra_h else ''}")
    for gate in PREPARE[name]:
        getattr(circuit, gate)(qubits[0])
    if extra_h:
        # The barrier stops the transpiler from canceling H H to nothing.
        circuit.barrier()
        circuit.h(qubits[0])
    circuit.measure(qubits, bits)
    return circuit


def show(z):
    """Format one amplitude compactly: 0.707, -0.707i, 0.5+0.5i."""
    re, im = round(z.real, 3) + 0, round(z.imag, 3) + 0  # + 0 turns -0.0 into 0.0
    if im == 0:
        return f"{re:g}"
    if re == 0:
        return f"{im:g}i"
    return f"{re:g}{im:+g}i"


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    circuits = {}
    for name in PREPARE:
        circuits[name] = build(name, extra_h=False)
        circuits[f"{name} then H"] = build(name, extra_h=True)
    print(circuits["|-> then H"].draw("text"))

    ideal_p0 = {}
    print("Exact amplitudes (of |0>, |1>) and P(0):")
    for label, circuit in circuits.items():
        state = Statevector(circuit.remove_final_measurements(inplace=False))
        ideal_p0[label] = state.probabilities()[0]
        amplitudes = "(" + ", ".join(show(z) for z in state.data) + ")"
        print(f"  {label:12s} {amplitudes:20s}  P(0) = {ideal_p0[label]:.3f}")

    banner("Operator")
    print("Nothing (measure directly), or one more H; both end in a Z measurement.")

    banner("Execution")
    targets = get_targets(args)
    results = {}
    for target in targets:
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        pubs = [(pass_manager.run(c),) for c in circuits.values()]
        job = target.sampler.run(pubs, shots=args.shots)
        # wait() returns None if a hardware job failed or you stopped waiting.
        results[target.label] = target.wait(job)
        print(f"{target.label}: ran {len(pubs)} PUBs")

    banner("Analysis")
    for label, result in results.items():
        if result is None:
            continue
        print(f"{label}, {args.shots} shots:")
        for name, pub_result in zip(circuits, result):
            counts = pub_result.data.meas.get_counts()
            p0 = counts.get("0", 0) / args.shots
            print(
                f"  {name:12s} measured P(0) = {p0:.3f}   exact = {ideal_p0[name]:.3f}"
            )
    scatter = np.sqrt(0.5 * 0.5 / args.shots)
    print(
        f"\nScatter of a fraction p = 0.5 from {args.shots} shots: {scatter:.3f}."
        f" Most runs land within {2 * scatter:.3f} of 0.500."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
