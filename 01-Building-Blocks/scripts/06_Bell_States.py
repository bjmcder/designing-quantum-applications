#!/usr/bin/env python3
"""Entanglement: the four Bell states versus unentangled (product) states.

The four Bell states are the simplest maximally entangled two-qubit states:

    |Phi+> = (|00> + |11>) / sqrt(2)        |Phi-> = (|00> - |11>) / sqrt(2)
    |Psi+> = (|01> + |10>) / sqrt(2)        |Psi-> = (|01> - |10>) / sqrt(2)

One circuit makes all four. Start from the basis state |a b>, then apply H on
qubit 0 and CX (control 0, target 1):

    |00> -> Phi+     |10> -> Phi-     |01> -> Psi+     |11> -> Psi-
    (the X gates that set a, b come first)

For comparison we also build three *unentangled* states, which can always be
written as (one-qubit state) x (one-qubit state): |00>, |++> and |+0>.

Telling them apart takes more than one measurement basis
--------------------------------------------------------
* Z basis (plain measurement). Phi+ and Phi- give the SAME counts (only 00 and
  11). Both are perfectly correlated, but the sign between the terms is a
  phase, invisible to a Z measurement.
* X basis. To measure in the X basis we do a *change of basis*: apply H to
  each qubit, then measure in Z (because X = H Z H). Now Phi+ and Phi- differ.

Correlators with an Estimator
-----------------------------
Instead of counting bitstrings ourselves, an Estimator returns expectation
values of observables directly. We ask for three correlators, each in [-1, 1]:

               ZZ      XX      YY
    Phi+       +1      +1      -1
    Phi-       +1      -1      +1
    Psi+       -1      +1      +1
    Psi-       -1      -1      -1
    |++>        0      +1       0       (product states are much weaker)

Entanglement witness: S = |<ZZ>| + |<XX>| + |<YY>|. For any product state
S <= 1 (by the Cauchy-Schwarz inequality). Bell states have S = 3. Seeing
S > 1 proves the two qubits are entangled.

The steps
---------
    1. Circuit    build each two-qubit state.
    2. Transpile  map onto a pair of connected qubits on the backend.
    3. Observable ZZ, XX, YY as SparsePauliOp (Estimator part only). The
                  observable must follow the circuit's qubit layout.
    4. PUB        Sampler: (circuit,)   Estimator: (circuit, observables)
    5. Execute    one Sampler job, one Estimator job.
    6. Read       counts for the Sampler, evs and stds for the Estimator.

Run with no arguments to use Aer and IBM hardware. Pass --skip-hardware to
run only the simulator.
"""

import sys

import numpy as np
from _runtime import banner, get_targets, make_parser
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

BELL_BITS = {"Phi+": (0, 0), "Phi-": (1, 0), "Psi+": (0, 1), "Psi-": (1, 1)}


def build_state(name: str) -> QuantumCircuit:
    """Build one of the named two-qubit states (no measurement).

    Parameters
    ----------
    name : str
        "Phi+", "Phi-", "Psi+", "Psi-" (entangled), or "|00>", "|++>",
        "|+0>" (unentangled).

    Returns
    -------
    QuantumCircuit
        A two-qubit circuit that prepares the state from |00>.
    """
    qubits = QuantumRegister(2, name="q")
    circuit = QuantumCircuit(qubits, name=name)
    if name in BELL_BITS:
        a, b = BELL_BITS[name]
        if a:
            circuit.x(qubits[0])
        if b:
            circuit.x(qubits[1])
        circuit.barrier()
        circuit.h(qubits[0])  # superposition on qubit 0 ...
        circuit.cx(qubits[0], qubits[1])  # ... shared with qubit 1: entanglement
    elif name == "|++>":
        circuit.h(qubits[0])
        circuit.h(qubits[1])
    elif name == "|+0>":
        circuit.h(qubits[0])
    elif name != "|00>":
        raise ValueError(f"Unknown state {name!r}")
    return circuit


def measured(state: QuantumCircuit, basis: str) -> QuantumCircuit:
    """Return a copy of ``state`` that measures both qubits in X or Z.

    Parameters
    ----------
    state : QuantumCircuit
        A circuit without measurements.
    basis : str
        "Z" measures directly. "X" first applies H to each qubit (the change
        of basis), then measures.

    Returns
    -------
    QuantumCircuit
        The state circuit followed by the basis change and measurements.
    """
    circuit = state.copy(name=f"{state.name}_{basis}")
    bits = ClassicalRegister(2, name="meas")
    circuit.add_register(bits)
    circuit.barrier()
    if basis == "X":
        circuit.h([0, 1])
    circuit.measure(circuit.qubits, bits)
    return circuit


def main() -> int:
    """Compare Bell and product states with a Sampler and an Estimator."""
    args = make_parser(__doc__).parse_args()

    # ---- STEP 1: circuits --------------------------------------------------
    names = list(BELL_BITS) + ["|00>", "|++>", "|+0>"]
    states = {n: build_state(n) for n in names}
    banner("STEP 1 - Circuit for Phi+ (the X-basis version)")
    print(measured(states["Phi+"], "X").draw(output="text"))

    # ---- STEP 3 (built early, it does not depend on the target) -----------
    # Qiskit labels read right-to-left, qubit 0 last. These are symmetric
    # (same Pauli on both qubits), so the order does not matter here.
    observables = [SparsePauliOp(p) for p in ("ZZ", "XX", "YY")]

    for target in get_targets(args):
        banner(f"Running on {target.label}")

        # ---- STEP 2: transpile -------------------------------------------
        # On hardware the pass manager picks two physically connected qubits
        # and rewrites H and CX into native gates. isa.layout records which
        # physical qubits were chosen.
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )

        # ================= Sampler: counts in Z and X bases ================
        sampler_circuits = {
            (n, basis): pass_manager.run(measured(c, basis))
            for n, c in states.items()
            for basis in ("Z", "X")
        }
        # STEP 4: PUB per circuit. STEP 5: one job carries all of them.
        sampler_pubs = [(isa,) for isa in sampler_circuits.values()]
        print(f"Sampler: {len(sampler_pubs)} PUBs in one job")
        result = target.wait(target.sampler.run(sampler_pubs, shots=args.shots))
        if result is not None:
            # STEP 6: counts keyed by bitstring "q1 q0".
            print(f"\nSampler counts ({args.shots} shots) -- Z basis | X basis")
            for i, (name, _) in enumerate(states.items()):
                z = result[2 * i].data.meas.get_counts()
                x = result[2 * i + 1].data.meas.get_counts()
                print(f"  {name:5s} Z: {dict(sorted(z.items()))}")
                print(f"  {'':5s} X: {dict(sorted(x.items()))}")

        # ================= Estimator: correlators ==========================
        # STEP 2 again: transpile the *unmeasured* circuits. The Estimator
        # adds whatever basis changes and measurements it needs by itself.
        estimator_pubs = []
        for name, circuit in states.items():
            isa = pass_manager.run(circuit)
            # STEP 3: the transpiler may have moved our qubits onto other
            # physical qubits. apply_layout rewrites each observable to act
            # on those same physical qubits.
            isa_observables = [obs.apply_layout(isa.layout) for obs in observables]
            # STEP 4: Estimator PUB = (circuit, observables). A list of three
            # observables gives three expectation values from one PUB.
            estimator_pubs.append((isa, isa_observables))
        print(f"\nEstimator: {len(estimator_pubs)} PUBs in one job")
        # STEP 5: precision is the target standard error on each value.
        result = target.wait(
            target.estimator.run(estimator_pubs, precision=args.precision)
        )
        if result is None:
            continue

        # STEP 6: evs are the expectation values, stds their standard errors.
        print(f"\nEstimator (precision {args.precision})")
        print(f"  {'state':5s}   <ZZ>    <XX>    <YY>      S    ideal S")
        for (name, circuit), pub_result in zip(states.items(), result):
            ev = np.asarray(pub_result.data.evs)
            ideal = Statevector(circuit)
            ideal_s = sum(abs(ideal.expectation_value(o).real) for o in observables)
            s = np.abs(ev).sum()
            # Shot noise can push a product state slightly above 1, so only
            # claim entanglement when S clears 1 by three standard errors.
            s_err = np.sqrt(np.sum(np.asarray(pub_result.data.stds) ** 2))
            verdict = "entangled" if s > 1 + 3 * s_err else "no entanglement shown"
            print(
                f"  {name:5s} {ev[0]:+7.3f} {ev[1]:+7.3f} {ev[2]:+7.3f}"
                f" {s:6.2f}    {ideal_s:4.1f}   {verdict}"
            )

    print(
        "\nTakeaway: Phi+ and Phi- look identical in the Z basis but differ in X; "
        "S clearly above 1 certifies\nentanglement (product states can never exceed 1)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
