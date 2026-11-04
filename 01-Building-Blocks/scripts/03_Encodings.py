#!/usr/bin/env python3
"""Notebook 03, the experiment: one neighbor step in each encoding.

This script runs the experiment from notebooks/03_Encodings.ipynb from the
command line. The notebook defines every term used here and explains each
step; this file is the same code without the explanations.

The values m = 0, ..., 7 are stored three ways: binary and Gray code on 3
qubits, and one-hot (unary) on 8 qubits. The neighbor step E_m exchanges the
codes of m and m + 1 and leaves every other code alone. For each encoding and
each m from 0 to 6 we prepare the code of m, apply E_m, measure, and decode.
Every shot should decode to m + 1. The CX cost of E_m differs by encoding:
6 to 10 in binary, 6 in Gray, 3 in one-hot.

The recipe
----------
    Encoding   prepare the code of m with NOT gates.
    Operator   the neighbor step E_m, then measure every qubit; the table of
               CX gates each E_m needs, and a check that it exchanges the
               two codes.
    Execution  transpile for each target; all the circuits go into one job.
    Analysis   decode each shot as correct (m + 1), wrong (another value) or
               invalid (no value; only possible for one-hot), and report the
               average one-hot penalty (w - 1)^2. On the simulator every shot
               is correct.

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

N = 8


def binary(m, num_bits=3):
    return format(m, f"0{num_bits}b")


def gray(m, num_bits=3):
    return format(m ^ (m >> 1), f"0{num_bits}b")


def decode_gray(code):
    bits = [int(c) for c in code]  # bits[0] is the top bit
    b = [bits[0]]
    for g in bits[1:]:
        b.append(b[-1] ^ g)  # b_k = b_(k+1) XOR g_k
    return int("".join(map(str, b)), 2)


def unary(m, num_bits=N):
    return format(1 << m, f"0{num_bits}b")


def decode_unary(code):
    """Return the value, or None if the bitstring is invalid."""
    if code.count("1") != 1:
        return None
    return len(code) - 1 - code.index("1")  # bit 0 is the rightmost character


def one_hot_penalty(code):
    return (code.count("1") - 1) ** 2


ENCODINGS = {"binary": binary, "Gray": gray, "one-hot": unary}
DECODERS = {"binary": lambda c: int(c, 2), "Gray": decode_gray, "one-hot": decode_unary}


def neighbor_step(encoding, m):
    """Circuit that exchanges the codes of m and m + 1.

    Every other code is left alone.
    """
    code_m, code_next = ENCODINGS[encoding](m), ENCODINGS[encoding](m + 1)
    num_bits = len(code_m)
    circuit = QuantumCircuit(num_bits, name=f"E_{m}")
    if encoding == "one-hot":
        circuit.swap(m, m + 1)
        return circuit

    def bit(code, k):
        return code[-1 - k]

    differing = [k for k in range(num_bits) if bit(code_m, k) != bit(code_next, k)]
    pivot, others = differing[0], differing[1:]
    # Make the two codes differ only in the pivot bit.
    for k in others:
        circuit.cx(pivot, k)
    # The CX gates leave the code whose pivot bit is 0 unchanged,
    # so its other bits are the values the CCX controls test for.
    reference = code_m if bit(code_m, pivot) == "0" else code_next
    controls = [k for k in range(num_bits) if k != pivot]
    zeros = [k for k in controls if bit(reference, k) == "0"]
    for k in zeros:  # a control that must be 0 is sandwiched by X
        circuit.x(k)
    circuit.ccx(controls[0], controls[1], pivot)
    for k in zeros:
        circuit.x(k)
    for k in reversed(others):  # undo the CX gates
        circuit.cx(pivot, k)
    return circuit


def is_exchange(circuit, code_a, code_b, check_all):
    """True if the circuit swaps the two codes.

    With check_all, every other bitstring must also be left unchanged.
    """
    U = Operator(circuit).data.real
    i, j = int(code_a, 2), int(code_b, 2)
    P = np.eye(U.shape[0])
    P[[i, j]] = P[[j, i]]
    if check_all:
        return np.allclose(U, P)
    return np.allclose(U[:, [i, j]], P[:, [i, j]])


def prepare(code):
    circuit = QuantumCircuit(len(code))
    for k in range(len(code)):
        if code[-1 - k] == "1":
            circuit.x(k)
    return circuit


def print_cx_costs():
    """Print the CX gates each neighbor step needs; '!' marks a failed check."""
    reference = generate_preset_pass_manager(
        basis_gates=["cx", "rz", "sx", "x"], optimization_level=1
    )
    print("CX gates needed for the neighbor step m -> m + 1 (and back):")
    print(f"  {'m':8s}", "  ".join(f"{m:>2}" for m in range(N - 1)))
    for encoding, encode in ENCODINGS.items():
        row = []
        for m in range(N - 1):
            step = neighbor_step(encoding, m)
            # For one-hot only the valid codes matter;
            # SWAP also moves some invalid patterns.
            check_all = encoding != "one-hot"
            ok = is_exchange(step, encode(m), encode(m + 1), check_all=check_all)
            cx = dict(reference.run(step).count_ops()).get("cx", 0)
            row.append(f"{cx:>2}{'' if ok else '!'}")
        print(f"  {encoding:8s}", "  ".join(row), f"  ({len(encode(0))} qubits)")


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    for m in range(N):
        print(f"  m = {m}: binary {binary(m)}, Gray {gray(m)}, one-hot {unary(m)}")

    banner("Operator")
    print_cx_costs()
    print("E_3 in binary:")
    print(neighbor_step("binary", 3).draw("text"))

    circuits, expected = {}, {}
    for encoding, encode in ENCODINGS.items():
        for m in range(N - 1):
            circuit = prepare(encode(m)).compose(neighbor_step(encoding, m))
            circuit.measure_all()
            circuits[(encoding, m)] = circuit
            expected[(encoding, m)] = m + 1
    print(len(circuits), "circuits")

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
        print(f"{label}, {args.shots} shots per circuit")
        for encoding in ENCODINGS:
            correct = wrong = invalid = 0
            penalty_sum = 0
            for key, pub_result in zip(circuits, result):
                if key[0] != encoding:
                    continue
                for code, count in pub_result.data.meas.get_counts().items():
                    value = DECODERS[encoding](code)
                    if value is None:
                        invalid += count
                    elif value == expected[key]:
                        correct += count
                    else:
                        wrong += count
                    penalty_sum += one_hot_penalty(code) * count
            total = correct + wrong + invalid
            line = (
                f"  {encoding:8s} correct {correct / total:.3f},"
                f" wrong {wrong / total:.3f}, invalid {invalid / total:.3f}"
            )
            if encoding == "one-hot":
                line += f", average penalty {penalty_sum / total:.3f}"
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
