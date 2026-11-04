#!/usr/bin/env python3
"""Notebook 06, the experiment: one gate, two gates, and the Y basis.

This script runs the experiment from notebooks/06_Interference_H_vs_SX.ipynb
from the command line. The notebook defines every term used here and explains
each step; this file is the same code without the explanations.

H and SX (sqrt(X)) each take |0> to a state that gives 0 or 1 with probability
1/2 in the Z basis, so one Z measurement cannot tell them apart. Applying each
gate twice can: H H = I returns 0 on every shot, and SX SX = X returns 1 on
every shot. A Y-basis measurement also separates them after one gate:
SX|0> is |-i> up to a global phase and returns 1 on every shot, while H|0> =
|+> stays near 50/50.

The recipe
----------
    Encoding   every circuit starts from |0>; the state is made by H or SX,
               once or twice.
    Operator   six circuits: H, SX, H H and SX SX measured in the Z basis, and
               H and SX measured in the Y basis (S-dagger, then H, then
               measure). Barriers keep the transpiler from canceling H H or
               merging SX SX into X.
    Execution  transpile for each target, one PUB per circuit, one Sampler job.
    Analysis   counts and P(0) for each circuit next to the exact P(0) from
               Statevector.

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


def build_circuit(name, gates, basis="Z"):
    qubits = QuantumRegister(1, name="q")
    bits = ClassicalRegister(1, name="meas")
    circuit = QuantumCircuit(qubits, bits, name=name)
    for k, gate in enumerate(gates):
        if k > 0:
            circuit.barrier()
        getattr(circuit, gate)(qubits[0])
    if basis == "Y":
        circuit.barrier()
        circuit.sdg(qubits[0])  # change of basis: S-dagger, then H
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
    print("Every circuit starts from |0>; H or SX, once or twice, makes the state.")

    banner("Operator")
    circuits = {
        "H": build_circuit("H", ["h"]),
        "SX": build_circuit("SX", ["sx"]),
        "H H": build_circuit("H H", ["h", "h"]),
        "SX SX": build_circuit("SX SX", ["sx", "sx"]),
        "H, Y basis": build_circuit("H, Y basis", ["h"], basis="Y"),
        "SX, Y basis": build_circuit("SX, Y basis", ["sx"], basis="Y"),
    }
    print(circuits["SX, Y basis"].draw("text"))

    print("Exact amplitudes and P(0) from Statevector:")
    ideal_p0 = {}
    for name, circuit in circuits.items():
        state = Statevector(circuit.remove_final_measurements(inplace=False))
        ideal_p0[name] = state.probabilities()[0]
        amplitudes = "(" + ", ".join(show(z) for z in state.data) + ")"
        print(f"  {name:12s} amplitudes {amplitudes:20s}  P(0) = {ideal_p0[name]:.3f}")

    banner("Execution")
    targets = get_targets(args)
    results = {}
    for target in targets:
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        isa_circuits = {k: pass_manager.run(c) for k, c in circuits.items()}
        print(target.label)
        for name, isa in isa_circuits.items():
            print(f"  {name:12s} {dict(isa.count_ops())}")
        pubs = [(isa,) for isa in isa_circuits.values()]
        job = target.sampler.run(pubs, shots=args.shots)
        # None if a hardware job failed or you stopped waiting.
        results[target.label] = target.wait(job)

    banner("Analysis")
    for label, result in results.items():
        if result is None:
            continue
        print(f"{label}, {args.shots} shots:")
        for name, pub_result in zip(circuits, result):
            counts = pub_result.data.meas.get_counts()
            p0 = counts.get("0", 0) / args.shots
            exact = ideal_p0[name]
            counts_text = str(dict(sorted(counts.items())))
            probabilities = f"P(0) = {p0:.3f}   exact = {exact:.3f}"
            print(f"  {name:12s} counts {counts_text:22s}  {probabilities}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
