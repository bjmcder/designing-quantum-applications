#!/usr/bin/env python3
"""Notebook 11, the experiment: hopping past an empty or an occupied orbital.

This script runs the experiment from notebooks/11_Fermions.ipynb from the
command line. The notebook defines every term used here and explains each
step; this file is the same code without the explanations.

We share one particle equally between orbitals 0 and 2, with orbital 1 either
empty or occupied, and measure two energies:

    h_qubit   = (X0 X2 + Y0 Y2) / 2           (the qubit hopping energy)
    h_fermion = (X0 Z1 X2 + Y0 Z1 Y2) / 2     (the fermion hopping energy)

Expected: <h_qubit> = +1 in both cases. <h_fermion> = +1 when orbital 1 is
empty and -1 when it is occupied, because the Z string of the Jordan-Wigner
transformation multiplies the state by (-1)^x1.

The recipe
----------
    Encoding   a NOT on qubit 1 when orbital 1 is occupied; then H on qubit 0,
               CX from qubit 0 to qubit 2, and a NOT on qubit 2, which give
               (|1 x1 0> + |0 x1 1>)/sqrt(2).
    Operator   the two energies as SparsePauliOps. Qiskit labels put qubit 0
               on the right, so X0 X2 is "XIX" and X0 Z1 X2 is "XZX".
    Execution  one Estimator job, one PUB per state carrying both observables.
    Analysis   each energy with its standard error, next to its exact value.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser


def shared_particle(orbital_1_occupied):
    """One particle shared equally between orbitals 0 and 2.

    Orbital 1 is occupied when orbital_1_occupied is True, empty otherwise.
    """
    qubits = QuantumRegister(3, name="q")
    circuit = QuantumCircuit(qubits, name=f"x1={int(orbital_1_occupied)}")
    if orbital_1_occupied:
        circuit.x(qubits[1])
    circuit.h(qubits[0])
    circuit.cx(qubits[0], qubits[2])
    circuit.x(qubits[2])
    return circuit


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    states = {
        "orbital 1 empty": shared_particle(False),
        "orbital 1 occupied": shared_particle(True),
    }
    for name, circuit in states.items():
        probabilities = Statevector(circuit).probabilities_dict()
        rounded = {str(b): round(float(q), 3) for b, q in probabilities.items()}
        print(f"{name:19s} bitstrings with probability: {rounded}")
    print(states["orbital 1 occupied"].draw("text"))

    banner("Operator")
    h_qubit = SparsePauliOp(["XIX", "YIY"], coeffs=[0.5, 0.5])
    h_fermion = SparsePauliOp(["XZX", "YZY"], coeffs=[0.5, 0.5])
    observables = [h_qubit, h_fermion]
    print("exact values:")
    for name, circuit in states.items():
        ideal = Statevector(circuit)
        values = [ideal.expectation_value(o).real for o in observables]
        print(
            f"  {name:19s} <h_qubit> = {values[0]:+.3f}"
            f"   <h_fermion> = {values[1]:+.3f}"
        )

    banner("Execution")
    targets = get_targets(args)
    results = {}
    for target in targets:
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        pubs = []
        for circuit in states.values():
            isa = pass_manager.run(circuit)
            pubs.append((isa, [obs.apply_layout(isa.layout) for obs in observables]))
        job = target.estimator.run(pubs, precision=args.precision)
        # wait() returns None if a hardware job failed or you stopped waiting.
        results[target.label] = target.wait(job)
        print(f"{target.label}: ran {len(pubs)} Estimator PUBs")

    banner("Analysis")
    for label, result in results.items():
        if result is None:
            continue
        print(f"Estimator on {label} (precision {args.precision})")
        print(
            f"  {'state':19s} {'<h_qubit>':>17s} {'exact':>6s}"
            f"   {'<h_fermion>':>17s} {'exact':>6s}"
        )
        for (name, circuit), pub_result in zip(states.items(), result):
            ev = np.asarray(pub_result.data.evs)
            std = np.asarray(pub_result.data.stds)
            ideal = Statevector(circuit)
            exact = [ideal.expectation_value(o).real for o in observables]
            print(
                f"  {name:19s} {ev[0]:+7.3f} +/- {std[0]:.3f} {exact[0]:+6.0f}"
                f"   {ev[1]:+7.3f} +/- {std[1]:.3f} {exact[1]:+6.0f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
