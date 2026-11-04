#!/usr/bin/env python3
"""Notebook 05, the experiment: three start-and-goal pairs.

This script runs the experiment from
notebooks/05_Rotation_From_Outer_Product.ipynb from the command line. The
notebook defines every term used here and explains each step; this file is the
same code without the explanations.

For a start state |psi> and a goal state |phi>, the recipe gate

    U = |phi><psi| + |phi_perp><psi_perp|

takes |psi> to |phi>. We test it on three cases: |0> -> |+>, |+> -> |-i>, and
a generic pair with Bloch angles (theta, phi) = (1.0, 0.5) for the start and
(2.2, -1.3) for the goal. Each case runs V_psi, then U, then V_phi^dagger,
then a Z-basis measurement, so a correct U gives 0 on every shot. The control
circuit leaves out U; it gives 0 with probability equal to the overlap
|<phi|psi>|^2.

The recipe
----------
    Encoding   V_psi prepares the start state |psi> from |0>.
    Operator   U, then V_phi^dagger, which maps the goal back to |0>; the
               control circuit leaves out U.
    Execution  transpile for each target, one PUB per circuit, one Sampler job.
    Analysis   P(0) for each circuit next to the ideal value: 1 with U, the
               overlap without.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

np.set_printoptions(precision=3, suppress=True)

plus = np.array([1, 1]) / np.sqrt(2)
minus_i = np.array([1, -1j]) / np.sqrt(2)


def orthogonal(state):
    """(alpha, beta) -> (-conj(beta), conj(alpha))"""
    return np.array([-np.conj(state[1]), np.conj(state[0])])


def rotation_matrix(start, goal):
    """U = |goal><start| + |goal_perp><start_perp|"""
    goal_perp, start_perp = orthogonal(goal), orthogonal(start)
    return np.outer(goal, start.conj()) + np.outer(goal_perp, start_perp.conj())


def prepare_matrix(state):
    """V_s = |s><0| + |s_perp><1|, which maps |0> to |s>."""
    return np.outer(state, [1, 0]) + np.outer(orthogonal(state), [0, 1])


def bloch_state(theta, phi):
    """cos(theta/2)|0> + e^(i phi) sin(theta/2)|1>"""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def build_circuit(start, goal, rotate):
    qubits = QuantumRegister(1, name="q")
    bits = ClassicalRegister(1, name="meas")
    circuit = QuantumCircuit(qubits, bits, name="with_U" if rotate else "no_U")
    circuit.append(UnitaryGate(prepare_matrix(start), label="V_psi"), qubits)
    circuit.barrier()
    if rotate:
        circuit.append(UnitaryGate(rotation_matrix(start, goal), label="U"), qubits)
        circuit.barrier()
    undo_goal = prepare_matrix(goal).conj().T
    circuit.append(UnitaryGate(undo_goal, label="V_phi^dag"), qubits)
    circuit.measure(qubits, bits)
    return circuit


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    cases = {
        "|0> -> |+>": (bloch_state(0, 0), plus),
        "|+> -> |-i>": (plus, minus_i),
        "generic": (bloch_state(1.0, 0.5), bloch_state(2.2, -1.3)),
    }
    for case, (start, goal) in cases.items():
        u = rotation_matrix(start, goal)
        print(
            f"{case:12s} unitary: {np.allclose(u.conj().T @ u, np.eye(2))}"
            f"   U|psi> == |phi>: {np.allclose(u @ start, goal)}"
            f"   overlap |<phi|psi>|^2 = {abs(np.vdot(goal, start)) ** 2:.3f}"
        )

    banner("Operator")
    circuits, expected_p0 = {}, {}
    for case, (start, goal) in cases.items():
        circuits[f"{case} with U"] = build_circuit(start, goal, rotate=True)
        circuits[f"{case} no U"] = build_circuit(start, goal, rotate=False)
        expected_p0[f"{case} with U"] = 1.0
        expected_p0[f"{case} no U"] = abs(np.vdot(goal, start)) ** 2

    # Statevector gives the exact probability of 0 for each circuit.
    for name, circuit in circuits.items():
        state = Statevector(circuit.remove_final_measurements(inplace=False))
        assert np.isclose(state.probabilities()[0], expected_p0[name])
    print("Exact P(0) of every circuit matches the prediction.")
    print(circuits["|0> -> |+> with U"].draw("text"))

    banner("Execution")
    targets = get_targets(args)
    results = {}
    for target in targets:
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        isa_circuits = {k: pass_manager.run(c) for k, c in circuits.items()}
        print(
            target.label,
            "-> 'generic with U' became:",
            dict(isa_circuits["generic with U"].count_ops()),
        )
        pubs = [(isa,) for isa in isa_circuits.values()]
        job = target.sampler.run(pubs, shots=args.shots)
        # None if a hardware job failed or you stopped waiting.
        results[target.label] = target.wait(job)

    banner("Analysis")
    for label, result in results.items():
        if result is None:
            continue
        print(f"P(0) from {args.shots} shots on {label}:")
        for name, pub_result in zip(circuits, result):
            counts = pub_result.data.meas.get_counts()
            p0 = counts.get("0", 0) / args.shots
            print(f"  {name:22s} P(0) = {p0:.3f}   ideal = {expected_p0[name]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
