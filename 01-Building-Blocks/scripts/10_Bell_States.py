#!/usr/bin/env python3
"""Notebook 10, the experiment: Bell states, product states and a mixture.

This script runs the experiment from notebooks/10_Bell_States.ipynb from the
command line. The notebook defines every term used here and explains each
step; this file is the same code without the explanations.

We prepare the four Bell states and four product states (|00>, |11>, |++>
and |0+>), look at their counts in the Z and X bases with the Sampler,
measure the three correlators <ZZ>, <XX> and <YY> with the Estimator, and add
the coin-flip mixture of |00> and |11> as a control. The entanglement witness
S = |<ZZ>| + |<XX>| + |<YY>| is 3 for every Bell state and at most 1 for
every product state and for the mixture.

Bitstrings read q1 q0, with qubit 0 on the RIGHT: the Bell input "01" means
qubit 0 starts as 1 and qubit 1 as 0.

The recipe
----------
    Encoding   the Bell states from X gates, then H on qubit 0 and CX; the
               product states from X and H gates. The mixture combines the
               results of |00> and |11>, half weight each.
    Operator   for the Sampler, a Z-basis measurement, or H on both qubits
               followed by one (the X basis); for the Estimator, the three
               correlators.
    Execution  one Sampler job with two PUBs per state; one Estimator job with
               one PUB per state that carries all three correlators.
    Analysis   the counts in both bases, then the correlators and S with its
               standard error. A state is declared entangled when S is above
               1 by more than three standard errors.

Run with no arguments to use only the Aer simulator. Add --hardware to also
run on ibm_rensselaer (see the README for the IBM token you need).
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from _runtime import banner, get_targets, make_parser

BELL_INPUT = {"Phi+": "00", "Phi-": "01", "Psi+": "10", "Psi-": "11"}


def build_state(name):
    """Return a two-qubit circuit (no measurement) that prepares the named state."""
    qubits = QuantumRegister(2, name="q")
    circuit = QuantumCircuit(qubits, name=name)
    if name in BELL_INPUT:
        input_bits = BELL_INPUT[name]
        for k in range(2):
            if input_bits[-1 - k] == "1":
                circuit.x(qubits[k])
        circuit.barrier()
        circuit.h(qubits[0])  # the only non-classical gate
        circuit.cx(qubits[0], qubits[1])  # classical reversible XOR
    elif name == "|11>":
        circuit.x(qubits)
    elif name == "|++>":
        circuit.h(qubits)
    elif name == "|0+>":
        circuit.h(qubits[0])
    elif name != "|00>":
        raise ValueError(f"Unknown state {name!r}")
    return circuit


def measured(state, basis):
    """Return a copy of `state` that measures both qubits in the Z or X basis."""
    circuit = state.copy(name=f"{state.name}_{basis}")
    bits = ClassicalRegister(2, name="meas")
    circuit.add_register(bits)
    circuit.barrier()
    if basis == "X":
        circuit.h([0, 1])  # change of basis: X basis -> Z basis
    circuit.measure(circuit.qubits, bits)
    return circuit


def add_counts(first, second):
    total = dict(first)
    for key, value in second.items():
        total[key] = total.get(key, 0) + value
    return dict(sorted(total.items()))


def report(name, ev, std, ideal_s):
    s = np.abs(ev).sum()
    s_err = np.sqrt(np.sum(std**2))
    verdict = "entangled" if s > 1 + 3 * s_err else "no entanglement shown"
    print(
        f"  {name:8s} {ev[0]:+7.3f} {ev[1]:+7.3f} {ev[2]:+7.3f} "
        f"{s:6.2f} +/- {s_err:.2f} {ideal_s:8.1f}   {verdict}"
    )


def main():
    args = make_parser(__doc__).parse_args()

    banner("Encoding")
    names = list(BELL_INPUT) + ["|00>", "|11>", "|++>", "|0+>"]
    states = {n: build_state(n) for n in names}
    print("amplitudes of |00>, |01>, |10>, |11> (all real here):")
    for name in BELL_INPUT:
        amplitudes = Statevector(states[name]).data.real
        print(f"  {name:5s} " + "  ".join(f"{v:+.3f}" for v in amplitudes))
    print(states["Psi+"].draw("text"))

    banner("Operator")
    observables = [SparsePauliOp(p) for p in ("ZZ", "XX", "YY")]
    print(measured(states["Phi+"], "X").draw("text"))
    print(f"{'state':6s}   <ZZ>   <XX>   <YY>   (ideal)")
    for name, circuit in states.items():
        ideal = Statevector(circuit)
        values = [ideal.expectation_value(o).real for o in observables]
        print(f"{name:6s} " + " ".join(f"{v:+6.2f}" for v in values))

    banner("Execution")
    targets = get_targets(args)
    sampler_results = {}
    for target in targets:
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        sampler_circuits = [
            pass_manager.run(measured(c, basis))
            for c in states.values()
            for basis in ("Z", "X")
        ]
        sampler_pubs = [(isa,) for isa in sampler_circuits]
        print(f"{target.label}: {len(sampler_pubs)} Sampler PUBs in one job")
        # wait() returns None if a hardware job failed or you stopped waiting.
        sampler_results[target.label] = target.wait(
            target.sampler.run(sampler_pubs, shots=args.shots)
        )

    estimator_results = {}
    for target in targets:
        if sampler_results[target.label] is None:
            continue  # the Sampler job failed; do not submit another
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        estimator_pubs = []
        for circuit in states.values():
            isa = pass_manager.run(circuit)
            isa_observables = [obs.apply_layout(isa.layout) for obs in observables]
            estimator_pubs.append((isa, isa_observables))
        print(f"{target.label}: {len(estimator_pubs)} Estimator PUBs in one job")
        estimator_results[target.label] = target.wait(
            target.estimator.run(estimator_pubs, precision=args.precision)
        )

    banner("Analysis")
    for label, result in sampler_results.items():
        if result is None:
            continue
        print(f"Sampler counts on {label} ({args.shots} shots per state)")
        counts = {}
        for i, name in enumerate(states):
            counts[name] = {
                "Z": result[2 * i].data.meas.get_counts(),
                "X": result[2 * i + 1].data.meas.get_counts(),
            }
        counts["mixture"] = {
            b: add_counts(counts["|00>"][b], counts["|11>"][b]) for b in ("Z", "X")
        }
        for name, by_basis in counts.items():
            print(f"  {name:8s} Z: {dict(sorted(by_basis['Z'].items()))}")
            print(f"  {'':8s} X: {dict(sorted(by_basis['X'].items()))}")

    for label, result in estimator_results.items():
        if result is None:
            continue
        print(f"\nEstimator on {label} (precision {args.precision})")
        print(
            f"  {'state':8s} {'<ZZ>':>7s} {'<XX>':>7s} {'<YY>':>7s} "
            f"{'S +/- std':>15s} {'ideal S':>8s}   verdict"
        )
        measured_values = {}
        for (name, circuit), pub_result in zip(states.items(), result):
            ev = np.asarray(pub_result.data.evs)
            std = np.asarray(pub_result.data.stds)
            measured_values[name] = (ev, std)
            ideal = Statevector(circuit)
            ideal_s = sum(abs(ideal.expectation_value(o).real) for o in observables)
            report(name, ev, std, ideal_s)
        (ev_a, std_a) = measured_values["|00>"]
        (ev_b, std_b) = measured_values["|11>"]
        report("mixture", (ev_a + ev_b) / 2, np.sqrt(std_a**2 + std_b**2) / 2, 1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
