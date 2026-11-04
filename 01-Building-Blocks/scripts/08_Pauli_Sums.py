#!/usr/bin/env python3
"""Notebook 08, the experiment: the cheapest valid choice.

This script runs the experiment from notebooks/08_Pauli_Sums.ipynb from the
command line. The notebook defines every term used here and explains each
step; this file is the same code without the explanations.

Choose exactly one of three options, numbered 0, 1 and 2, with costs
w = (3, 1, 2). Store the choice one-hot: bit k is 1 when option k is chosen.
The cost of a bitstring is the cost of what it chooses plus a penalty, with
weight lambda = 4, for not choosing exactly one:

    C(x) = sum_k w_k x_k + lambda * (sum_k x_k - 1)^2.

We build its Hamiltonian as a sum of Pauli strings, find the ground state
exactly (|010>, energy 1), and check on the quantum computer that the
ground-state bitstring has the lowest cost of all the bitstrings it returns.

The recipe
----------
    Encoding   one-hot over three bits; prepare the ground state with X gates,
               or every bitstring at once with an H on each qubit.
    Operator   n_k = (I - Z_k)/2 substituted into C(x), as a SparsePauliOp.
    Execution  transpile for each target, one PUB per circuit, one Sampler job.
    Analysis   the cost of each measured bitstring, read from the diagonal of
               the Hamiltonian: 1 for every "ground" shot, and an average near
               7.0 for the "all" circuit.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

NUM_BITS = 3
WEIGHTS = [3, 1, 2]
LAMBDA = 4


def number_op(k, num_qubits=NUM_BITS):
    """n_k = (I - Z_k)/2 as a SparsePauliOp; label character -1-k is qubit k."""
    label = ["I"] * num_qubits
    label[-1 - k] = "Z"
    return SparsePauliOp.from_list([("I" * num_qubits, 0.5), ("".join(label), -0.5)])


def cost(bitstring):
    """C(x) computed directly from the bits, to check the Hamiltonian's diagonal."""
    bits = [int(bitstring[-1 - k]) for k in range(NUM_BITS)]
    return sum(w * x for w, x in zip(WEIGHTS, bits)) + LAMBDA * (sum(bits) - 1) ** 2


def ground_circuit(bitstring):
    qubits = QuantumRegister(NUM_BITS, name="q")
    bits = ClassicalRegister(NUM_BITS, name="meas")
    circuit = QuantumCircuit(qubits, bits, name="ground")
    for k in range(NUM_BITS):
        if bitstring[-1 - k] == "1":
            circuit.x(qubits[k])
    circuit.measure(qubits, bits)
    return circuit


def all_bitstrings_circuit():
    qubits = QuantumRegister(NUM_BITS, name="q")
    bits = ClassicalRegister(NUM_BITS, name="meas")
    circuit = QuantumCircuit(qubits, bits, name="all")
    circuit.h(qubits)  # every bitstring with probability 1/8
    circuit.measure(qubits, bits)
    return circuit


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    print("One-hot over three bits: the valid bitstrings are 001, 010 and 100.")

    banner("Operator")
    identity = SparsePauliOp("I" * NUM_BITS)
    count = sum((number_op(k) for k in range(NUM_BITS)), 0 * identity)
    weights_part = sum(WEIGHTS[k] * number_op(k) for k in range(NUM_BITS))
    penalty_part = LAMBDA * (count - identity).compose(count - identity)
    cost_op = (weights_part + penalty_part).simplify()
    print("Pauli string  coefficient")
    for label, coeff in cost_op.to_list():
        print(f"    {label}         {coeff.real:+5.1f}")

    diagonal = np.real(np.diag(cost_op.to_matrix()))
    cost_of = {
        format(i, f"0{NUM_BITS}b"): float(diagonal[i]) for i in range(2**NUM_BITS)
    }
    print()
    print("bitstring  diagonal  C(x)")
    for b, value in cost_of.items():
        print(f"   {b}     {value:5.1f}   {cost(b):5.1f}")

    energies, vectors = np.linalg.eigh(cost_op.to_matrix())
    ground = format(int(np.argmax(abs(vectors[:, 0]))), f"0{NUM_BITS}b")
    print("ground-state energy:", round(energies[0].real, 3), "  ground state:", ground)

    circuits = {"ground": ground_circuit(ground), "all": all_bitstrings_circuit()}
    print(circuits["all"].draw("text"))

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
    print("Expected: every 'ground' shot costs 1.0; the 'all' average is near")
    print(f"the mean of the diagonal, {diagonal.sum():.0f}/8 = {diagonal.mean():.1f}.")
    for label, result in results.items():
        if result is None:
            continue
        print(f"{label}, {args.shots} shots")
        for name, pub_result in zip(circuits, result):
            counts = pub_result.data.meas.get_counts()
            average = sum(cost_of[b] * c for b, c in counts.items()) / args.shots
            cheapest = min(counts, key=cost_of.get)
            print(
                f"  {name}: average cost over shots = {average:.3f}"
                f"   cheapest bitstring seen = {cheapest}"
            )
            if name == "all":
                for b in sorted(counts, key=cost_of.get):
                    print(f"      {b}  cost {cost_of[b]:5.1f}  counts {counts[b]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
