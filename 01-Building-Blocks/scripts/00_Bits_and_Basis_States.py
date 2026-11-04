#!/usr/bin/env python3
"""Notebook 00, the experiment: prepare three bitstrings and read them back.

This script runs the experiment from notebooks/00_Bits_and_Basis_States.ipynb
from the command line. The notebook defines every term used here and explains
each step; this file is the same code without the explanations.

We prepare |101>, |011> and |110> and check that the Sampler returns each one.
The last two are not palindromes, so getting them back confirms the bit order:
Qiskit writes bit 0 on the RIGHT, so bit k is character bitstring[-1 - k].

The recipe
----------
    Encoding   put qubit k in |1> with a NOT (X) gate when bit k is 1.
    Operator   none: prepare, then measure every qubit.
    Execution  transpile for each target, one PUB per circuit, one Sampler job.
    Analysis   the fraction of shots on the prepared bitstring, which should
               be 1.000 on the simulator.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

BITSTRINGS = ["101", "011", "110"]

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])


def prepare(bitstring):
    """Return a circuit that prepares |bitstring> and measures every qubit."""
    num_bits = len(bitstring)
    qubits = QuantumRegister(num_bits, name="q")
    bits = ClassicalRegister(num_bits, name="meas")
    circuit = QuantumCircuit(qubits, bits, name=bitstring)
    for k in range(num_bits):
        if bitstring[-1 - k] == "1":
            circuit.x(qubits[k])
    circuit.measure(qubits, bits)
    return circuit


def basis_vector(bitstring):
    """Return the vector for |bitstring> as a tensor product of |0> and |1>."""
    vector = np.array([1])
    for character in bitstring:  # leftmost character is the leftmost factor
        vector = np.kron(vector, ket1 if character == "1" else ket0)
    return vector


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    circuits = {b: prepare(b) for b in BITSTRINGS}
    print(circuits["011"].draw("text"))
    for b in BITSTRINGS:
        slot = int(np.argmax(basis_vector(b)))
        print(f"|{b}> -> 1 in slot {slot}  (binary {b} = {int(b, 2)})")

    banner("Operator")
    print("None: prepare the bitstring, then measure every qubit.")

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
        for b, pub_result in zip(BITSTRINGS, result):
            counts = pub_result.data.meas.get_counts()
            on_target = counts.get(b, 0) / args.shots
            print(
                f"  prepared {b}: counts {counts}   fraction on {b} = {on_target:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
