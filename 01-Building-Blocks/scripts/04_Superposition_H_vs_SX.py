#!/usr/bin/env python3
"""Superposition: the Hadamard gate versus the square-root-of-NOT gate.

Two different gates make a "50/50" superposition from |0>:

    H  |0> = (|0> + |1>) / sqrt(2)          = |+>
    SX |0> = ((1+i)|0> + (1-i)|1>) / 2      = |-i>   (up to a global phase)

(SX is Qiskit's name for sqrt(X); "square root of NOT" is the same thing.)

Measuring in the Z basis cannot tell them apart: both give 0 or 1 with
probability 1/2. The difference is in the *phase* between the two amplitudes,
and phase only shows up through interference. Apply each gate twice:

    H  H  = I   ->  back to |0>, so we always measure 0
    SX SX = X   ->  flipped to |1>, so we always measure 1

So one application of each gate looks identical, and two applications give
opposite, deterministic answers. That is superposition plus interference.

The steps (every script in this folder follows the same recipe)
---------------------------------------------------------------
    1. Circuit    build the abstract circuit.
    2. Transpile  rewrite it into the gates and qubits the backend supports
                  (an "ISA circuit"). Aer is transpiled for too, for uniformity.
    3. Observable not needed here: a Sampler just measures bitstrings.
    4. PUB        bundle what to run: here just (circuit,).
    5. Execute    hand the PUBs to a Sampler; get a job; wait for the result.
    6. Read       turn the raw shots into counts and compare to theory.

Run with no arguments to use Aer and IBM hardware. Pass --skip-hardware to
run only the simulator.
"""

import sys

import numpy as np
from _runtime import banner, get_targets, make_parser
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RXGate, SXGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

np.set_printoptions(precision=3, suppress=True)


def build_circuits() -> dict[str, QuantumCircuit]:
    """Build the four circuits we compare, each ending in a Z measurement.

    Returns
    -------
    dict[str, QuantumCircuit]
        Keys are "H", "SX", "H H" and "SX SX".
    """
    gates = {"H": ["h"], "SX": ["sx"], "H H": ["h", "h"], "SX SX": ["sx", "sx"]}
    circuits = {}
    for name, sequence in gates.items():
        qubits = QuantumRegister(1, name="q")
        bits = ClassicalRegister(1, name="meas")
        circuit = QuantumCircuit(qubits, bits, name=name)
        for k, gate in enumerate(sequence):
            if k > 0:
                # A barrier stops the transpiler from "helping" by canceling
                # H H to nothing (or merging SX SX into X). We want the gates
                # to actually run.
                circuit.barrier()
            getattr(circuit, gate)(qubits[0])
        circuit.measure(qubits, bits)
        circuits[name] = circuit
    return circuits


def show_the_math(circuits: dict[str, QuantumCircuit]) -> None:
    """Print the gate matrices and the ideal output states (no hardware)."""
    banner("The math (exact, computed on your laptop)")
    h = Operator(circuits["H"].remove_final_measurements(inplace=False)).data
    sx = Operator(circuits["SX"].remove_final_measurements(inplace=False)).data
    print("H =\n", h)
    print("SX =\n", sx)
    print("H @ H  =\n", h @ h, "  (identity)")
    print("SX @ SX =\n", sx @ sx, "  (Pauli X: a NOT gate)")

    # SX is a quarter turn about the x axis of the Bloch sphere, up to an
    # unobservable global phase. H is a half turn about the (x+z) axis.
    same = Operator(SXGate()).equiv(Operator(RXGate(np.pi / 2)))
    print(f"\nSX equals Rx(pi/2) up to a global phase: {same}")

    print("\nIdeal states after each circuit (amplitudes of |0>, |1>):")
    for name, circuit in circuits.items():
        state = Statevector(circuit.remove_final_measurements(inplace=False))
        print(f"  {name:6s} {state.data}   P(0) = {state.probabilities()[0]:.3f}")


def main() -> int:
    """Run the four circuits on Aer and (optionally) IBM hardware."""
    args = make_parser(__doc__).parse_args()

    # ---- STEP 1: circuits --------------------------------------------------
    circuits = build_circuits()
    show_the_math(circuits)
    banner("STEP 1 - Circuit (the 'SX SX' example)")
    print(circuits["SX SX"].draw(output="text"))

    for target in get_targets(args):
        banner(f"Running on {target.label}")

        # ---- STEP 2: transpile -------------------------------------------
        # The pass manager knows the backend's native gates and layout. On
        # hardware, H is rewritten as rz-sx-rz (the chip's native gates) and
        # a physical qubit is chosen. optimization_level=1 is a light touch.
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        isa_circuits = {n: pass_manager.run(c) for n, c in circuits.items()}
        print("STEP 2 - Transpiled gate counts:")
        for name, isa in isa_circuits.items():
            print(f"  {name:6s} {dict(isa.count_ops())}")

        # ---- STEP 3: observable ------------------------------------------
        # Not used: a Sampler has no observable. It just returns bitstrings.

        # ---- STEP 4: PUBs ------------------------------------------------
        # A PUB (Primitive Unified Bloc) is a tuple: (circuit, ...). Ours have
        # no parameters, so each tuple holds only the circuit. One job can
        # carry many PUBs.
        pubs = [(isa,) for isa in isa_circuits.values()]
        print(f"STEP 4 - Built {len(pubs)} PUBs")

        # ---- STEP 5: execute ---------------------------------------------
        job = target.sampler.run(pubs, shots=args.shots)
        result = target.wait(job)
        if result is None:
            continue

        # ---- STEP 6: read the results ------------------------------------
        # result[i] belongs to pubs[i]. `.data.meas` is the classical
        # register we named "meas"; get_counts() turns the raw shots into
        # {bitstring: count}.
        print(f"STEP 6 - Counts from {args.shots} shots:")
        for name, pub_result in zip(circuits, result):
            counts = pub_result.data.meas.get_counts()
            p0 = counts.get("0", 0) / args.shots
            print(f"  {name:6s} {counts}   P(0) = {p0:.3f}")

    print(
        "\nTakeaway: 'H' and 'SX' both give ~50/50, but 'H H' returns to 0 and "
        "'SX SX' flips to 1.\nThe gates differ in phase, which a single Z "
        "measurement cannot see."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
