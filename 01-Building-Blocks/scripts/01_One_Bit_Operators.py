#!/usr/bin/env python3
"""Notebook 01, the experiment: I, X, X X and Z on each basis state.

This script runs the experiment from notebooks/01_One_Bit_Operators.ipynb
from the command line. The notebook defines every term used here and explains
each step; this file is the same code without the explanations.

We apply four operators, I, X, X X and Z, to each of the basis states |0> and
|1>, measure, and compare the counts with the predictions: I, X X and Z leave
the bit's value unchanged, and X flips it. The operators are built from
creation and annihilation, a_dag = |1><0| and a = |0><1|:

    n = a_dag a,   n_bar = a a_dag,   I = n + n_bar,
    X = a + a_dag,   Z = n_bar - n.

The recipe
----------
    Encoding   start in |0>, or apply NOT (X) first to start in |1>.
    Operator   the gate sequences id, x, x x and z; Qiskit's Operator class
               checks each circuit's matrix against the one derived from a
               and a_dag.
    Execution  transpile for each target, one PUB per circuit, one Sampler job.
    Analysis   the fraction of shots on the predicted bit, which should be
               1.000 on the simulator. The Z rows match the I rows.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

np.set_printoptions(precision=3, suppress=True)

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])

a = np.outer(ket0, ket1)  # |0><1|
a_dag = np.outer(ket1, ket0)  # |1><0|
n = a_dag @ a
n_bar = a @ a_dag
I = np.eye(2, dtype=int)
X = a + a_dag
Z = n_bar - n

GATES = {"I": ["id"], "X": ["x"], "X X": ["x", "x"], "Z": ["z"]}
EXPECTED = {"I": I, "X": X, "X X": X @ X, "Z": Z}


def build_circuit(start_bit, gate_names, measure=True):
    """Prepare |start_bit>, apply the gates in order, and measure."""
    qubits = QuantumRegister(1, name="q")
    bits = ClassicalRegister(1, name="meas")
    circuit = QuantumCircuit(qubits, bits)
    if start_bit == 1:
        circuit.x(qubits[0])
    for gate in gate_names:
        getattr(circuit, gate)(qubits[0])
    if measure:
        circuit.measure(qubits, bits)
    return circuit


def predicted_bit(start_bit, label):
    """Return the bit that I, X X and Z keep and that X flips."""
    return 1 - start_bit if label == "X" else start_bit


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    print("Start in |0>, or apply NOT first to start in |1>.")

    banner("Operator")
    print("X =\n", X)
    print("Z =\n", Z)
    for label, gate_names in GATES.items():
        # Operator stores complex numbers (they appear from Notebook 04 on).
        # Here every imaginary part is 0, so .real keeps the real matrix.
        matrix = Operator(build_circuit(0, gate_names, measure=False)).data.real
        matches = np.allclose(matrix, EXPECTED[label])
        print(f"{label:4s} circuit matrix matches derived matrix: {matches}")

    circuits = {}
    for start_bit in (0, 1):
        for label, gate_names in GATES.items():
            circuits[(start_bit, label)] = build_circuit(start_bit, gate_names)
    print(circuits[(1, "X X")].draw("text"))
    for start_bit, label in circuits:
        prediction = predicted_bit(start_bit, label)
        print(f"start |{start_bit}>, apply {label:4s} -> predicted bit {prediction}")

    banner("Execution")
    targets = get_targets(args)
    results = {}
    for target in targets:
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        isa_circuits = {key: pass_manager.run(c) for key, c in circuits.items()}
        print(target.label)
        for (start_bit, label), isa in isa_circuits.items():
            print(f"  start |{start_bit}>, {label:4s}: {dict(isa.count_ops())}")
        pubs = [(isa,) for isa in isa_circuits.values()]
        job = target.sampler.run(pubs, shots=args.shots)
        # wait() returns None if a hardware job failed or you stopped waiting.
        results[target.label] = target.wait(job)

    banner("Analysis")
    for label, result in results.items():
        if result is None:
            continue
        print(f"{label}, {args.shots} shots:")
        for (start_bit, gate_label), pub_result in zip(circuits, result):
            counts = pub_result.data.meas.get_counts()
            expected = str(predicted_bit(start_bit, gate_label))
            fraction = counts.get(expected, 0) / args.shots
            print(
                f"  start |{start_bit}>, {gate_label:4s}: counts {counts!s:14s}"
                f"  fraction on {expected} = {fraction:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
