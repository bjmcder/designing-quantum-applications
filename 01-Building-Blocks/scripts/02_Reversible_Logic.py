#!/usr/bin/env python3
"""Notebook 02, the experiment: truth tables on a quantum computer.

This script runs the experiment from notebooks/02_Reversible_Logic.ipynb
from the command line. The notebook defines every term used here and explains
each step; this file is the same code without the explanations.

We run CX, SWAP, CCX and the oracle U_f for f(x1, x0) = x1 AND NOT x0 on
every input bitstring and compare the measured truth tables with the ideal
ones. Each gate is first built from a, a_dag, n and n_bar and checked against
Qiskit's gate:

    CX   = n_bar_0 + n_0 X_1
    SWAP = n_bar_0 n_bar_1 + n_0 n_1 + a_dag_0 a_1 + a_0 a_dag_1
    CCX  = I - n_0 n_1 + n_0 n_1 X_2

The recipe
----------
    Encoding   prepare each input bitstring with NOT gates.
    Operator   the gate as a circuit, and the classical rule it should follow;
               also the number of CX gates each operation costs.
    Execution  transpile for each target; all the circuits go into one job.
    Analysis   for each gate, the truth table: input, most frequent output,
               ideal output, and the fraction of shots on the ideal output,
               which should be 1.000 on the simulator.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

np.set_printoptions(precision=3, suppress=True)

ket0, ket1 = np.array([1, 0]), np.array([0, 1])
a = np.outer(ket0, ket1)  # |0><1|
a_dag = a.T  # |1><0|
n, n_bar = a_dag @ a, a @ a_dag
I = np.eye(2, dtype=int)
X = a + a_dag

GATE_NAMES = ["CX", "SWAP", "CCX", "oracle"]


def on_bit(op, k, num_bits):
    """Matrix of a one-bit operator on bit k of a register.

    Bit 0 is the rightmost factor of the Kronecker product.
    """
    factors = [op if j == k else I for j in reversed(range(num_bits))]
    matrix = np.array([[1]])
    for f in factors:
        matrix = np.kron(matrix, f)
    return matrix


def derived_matrices():
    """Return CX, SWAP and CCX built from a, a_dag, n and n_bar."""
    CX = np.kron(I, n_bar) + np.kron(X, n)  # n_bar_0 + n_0 X_1
    SWAP = (
        np.kron(n_bar, n_bar)
        + np.kron(n, n)
        + np.kron(a_dag, a)  # a_1^dagger a_0: bit 1 is the left factor
        + np.kron(a, a_dag)  # a_1 a_0^dagger = a_0^dagger a_1
    )
    n0, n1, X2 = on_bit(n, 0, 3), on_bit(n, 1, 3), on_bit(X, 2, 3)
    CCX = np.eye(8, dtype=int) - n0 @ n1 + n0 @ n1 @ X2
    return {"CX": CX, "SWAP": SWAP, "CCX": CCX}


def oracle_circuit():
    """U_f for f(x1, x0) = x1 AND NOT x0; input on qubits 0, 1, ancilla on qubit 2."""
    circuit = QuantumCircuit(3, name="U_f")
    circuit.x(0)
    circuit.ccx(0, 1, 2)
    circuit.x(0)
    return circuit


def f(x):
    x0, x1 = x & 1, (x >> 1) & 1
    return x1 & (1 - x0)


def prepare(bitstring):
    """Circuit that prepares |bitstring> (bit 0 is the rightmost character)."""
    num_bits = len(bitstring)
    circuit = QuantumCircuit(num_bits)
    for k in range(num_bits):
        if bitstring[-1 - k] == "1":
            circuit.x(k)
    return circuit


def gate_circuit(name):
    circuit = QuantumCircuit(3 if name in ("CCX", "oracle") else 2)
    if name == "CX":
        circuit.cx(0, 1)
    elif name == "SWAP":
        circuit.swap(0, 1)
    elif name == "CCX":
        circuit.ccx(0, 1, 2)
    else:
        circuit.compose(oracle_circuit(), inplace=True)
    return circuit


def ideal_output(name, x):
    x0, x1 = x & 1, (x >> 1) & 1
    if name == "CX":
        return x ^ (x0 << 1)  # bit 1 becomes x1 XOR x0
    if name == "SWAP":
        return (x0 << 1) | x1
    if name == "CCX":
        return x ^ ((x0 & x1) << 2)  # bit 2 becomes x2 XOR (x0 AND x1)
    return x ^ (f(x & 0b11) << 2)  # oracle


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    print("Each input bitstring is prepared with NOT gates, for example 010:")
    print(prepare("010").draw("text"))

    banner("Operator")
    for name, matrix in derived_matrices().items():
        matches = np.allclose(Operator(gate_circuit(name)).data, matrix)
        print(f"{name:4s} derived from a and a_dag matches Qiskit's gate: {matches}")
    U_f = Operator(oracle_circuit()).data.real
    print("U_f @ U_f == I:", np.allclose(U_f @ U_f, np.eye(8)))

    circuits, expected = {}, {}
    for name in GATE_NAMES:
        num_bits = gate_circuit(name).num_qubits
        for x in range(2**num_bits):
            bitstring = format(x, f"0{num_bits}b")
            circuit = prepare(bitstring).compose(gate_circuit(name))
            # Adds a classical register named "meas".
            circuit.measure_all()
            circuits[(name, bitstring)] = circuit
            expected[(name, bitstring)] = format(ideal_output(name, x), f"0{num_bits}b")
    print(len(circuits), "circuits")
    print(circuits[("oracle", "010")].draw("text"))

    reference = generate_preset_pass_manager(
        basis_gates=["cx", "rz", "sx", "x"], optimization_level=1
    )
    for name in GATE_NAMES:
        ops = dict(reference.run(gate_circuit(name)).count_ops())
        print(f"{name:6s} uses {ops.get('cx', 0)} CX gates   (all gates: {ops})")

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
        print(f"{target.label}: ran {len(pubs)} PUBs in one job")

    banner("Analysis")
    for label, result in results.items():
        if result is None:
            continue
        print(f"{label}, {args.shots} shots per input")
        for name in GATE_NAMES:
            print(f"  {name}")
            for key, pub_result in zip(circuits, result):
                if key[0] != name:
                    continue
                counts = pub_result.data.meas.get_counts()
                measured = max(counts, key=counts.get)
                fraction = counts.get(expected[key], 0) / args.shots
                print(
                    f"    in {key[1]} -> measured {measured}, ideal {expected[key]},"
                    f" fraction {fraction:.3f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
