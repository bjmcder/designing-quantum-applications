#!/usr/bin/env python3
"""Build a rotation gate from a starting state and a goal state.

The goal: given any single-qubit state |psi> (start) and |phi> (goal), find
a gate U with  U|psi> = |phi>.

The outer-product recipe
------------------------
The outer product |phi><psi| is a matrix that sends |psi> to |phi> and
everything orthogonal to |psi> to zero. That is not unitary by itself, so we
add the same thing for the orthogonal pair:

    U = |phi><psi|  +  |phi_perp><psi_perp|

For a state |s> = (a, b), the orthogonal state is |s_perp> = (-conj(b), conj(a)).
Then U is unitary, U|psi> = |phi>, and U|psi_perp> = |phi_perp>: it rotates
the whole Bloch sphere so that psi lands on phi.

U is not unique: following it with any extra rotation about the psi axis still
maps psi to phi. The phase we give |phi_perp> decides which one we get, and
this recipe is just one valid choice.

A handy way to see it: let  P_s = |s><0| + |s_perp><1|  (a gate that prepares
|s> from |0>). Then  U = P_phi P_psi^dagger: "undo psi, then prepare phi".

How we check it on a quantum computer
-------------------------------------
We cannot read a state directly, so we uncompute instead:

    |0> --P_psi--  U  --P_phi^dagger-- measure

P_psi makes psi, U turns it into phi, and P_phi^dagger sends phi back to |0>.
If U is right we measure 0 every time. As a control, we also skip U: then we
measure 0 only with probability |<phi|psi>|^2, which is less than 1.

The steps
---------
    1. Circuit    build it from the matrices above (UnitaryGate).
    2. Transpile  rewrite the arbitrary 2x2 unitary into native gates.
    3. Observable not needed: a Sampler just measures bitstrings.
    4. PUB        (circuit,)
    5. Execute    Sampler job.
    6. Read       P(0) should be 1.0 with U and |<phi|psi>|^2 without.

Run with no arguments to use Aer and IBM hardware. Pass --skip-hardware to
run only the simulator.
"""

import sys

import numpy as np
from _runtime import banner, get_targets, make_parser
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

np.set_printoptions(precision=3, suppress=True)


def bloch_state(theta: float, phi: float) -> np.ndarray:
    """Return the state cos(theta/2)|0> + e^(i phi) sin(theta/2)|1>.

    Parameters
    ----------
    theta : float
        Polar angle on the Bloch sphere (0 is |0>, pi is |1>).
    phi : float
        Azimuthal angle (0 is |+>, pi/2 is |+i> when theta = pi/2).

    Returns
    -------
    np.ndarray
        The two amplitudes, shape (2,).
    """
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def orthogonal(state: np.ndarray) -> np.ndarray:
    """Return the state orthogonal to ``state``: (a, b) -> (-conj(b), conj(a))."""
    return np.array([-np.conj(state[1]), np.conj(state[0])])


def prepare_matrix(state: np.ndarray) -> np.ndarray:
    """Return P_s = |s><0| + |s_perp><1|, the unitary that maps |0> to |s>."""
    return np.outer(state, [1, 0]) + np.outer(orthogonal(state), [0, 1])


def rotation_matrix(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Return U = |goal><start| + |goal_perp><start_perp|.

    Parameters
    ----------
    start : np.ndarray
        The state the gate should act on, shape (2,).
    goal : np.ndarray
        The state the gate should produce, shape (2,).

    Returns
    -------
    np.ndarray
        A 2x2 unitary with U @ start == goal.
    """
    return np.outer(goal, start.conj()) + np.outer(
        orthogonal(goal), orthogonal(start).conj()
    )


def build_circuit(start: np.ndarray, goal: np.ndarray, rotate: bool) -> QuantumCircuit:
    """Build |0> -> start -> (U) -> goal -> back to |0> -> measure.

    Parameters
    ----------
    start, goal : np.ndarray
        The starting and goal states.
    rotate : bool
        If False, skip U (the control experiment).

    Returns
    -------
    QuantumCircuit
        A one-qubit circuit ending in a Z measurement.
    """
    qubits = QuantumRegister(1, name="q")
    bits = ClassicalRegister(1, name="meas")
    circuit = QuantumCircuit(qubits, bits, name="with_U" if rotate else "no_U")

    circuit.append(UnitaryGate(prepare_matrix(start), label="P_psi"), qubits)
    circuit.barrier()  # keep the transpiler from merging the pieces together
    if rotate:
        circuit.append(UnitaryGate(rotation_matrix(start, goal), label="U"), qubits)
        circuit.barrier()
    circuit.append(
        UnitaryGate(prepare_matrix(goal).conj().T, label="P_phi^dag"), qubits
    )
    circuit.measure(qubits, bits)
    return circuit


def main() -> int:
    """Check three start/goal pairs on Aer and (optionally) IBM hardware."""
    args = make_parser(__doc__).parse_args()

    # ---- STEP 1: circuits --------------------------------------------------
    plus = bloch_state(np.pi / 2, 0)
    minus_i = bloch_state(np.pi / 2, -np.pi / 2)
    cases = {
        "|0> -> |+>": (bloch_state(0, 0), plus),
        "|+> -> |-i>": (plus, minus_i),
        "generic": (bloch_state(1.0, 0.5), bloch_state(2.2, -1.3)),
    }

    banner("The math (exact)")
    circuits = {}
    expected_p0 = {}
    for case, (start, goal) in cases.items():
        u = rotation_matrix(start, goal)
        is_unitary = np.allclose(u.conj().T @ u, np.eye(2))
        maps_ok = np.allclose(u @ start, goal)
        overlap = abs(np.vdot(goal, start)) ** 2
        print(f"{case:12s} unitary: {is_unitary}   U|psi> == |phi>: {maps_ok}")
        circuits[f"{case} with U"] = build_circuit(start, goal, rotate=True)
        circuits[f"{case} no U"] = build_circuit(start, goal, rotate=False)
        expected_p0[f"{case} with U"] = 1.0
        expected_p0[f"{case} no U"] = overlap

    banner("STEP 1 - Circuit (first case)")
    print(circuits["|0> -> |+> with U"].draw(output="text"))

    for target in get_targets(args):
        banner(f"Running on {target.label}")

        # ---- STEP 2: transpile -------------------------------------------
        # An arbitrary 2x2 unitary is not a native gate. The transpiler
        # decomposes each one into the backend's basis gates (Aer: u3; IBM
        # chips: rz, sx and x), so one UnitaryGate becomes a few real gates.
        pass_manager = generate_preset_pass_manager(
            backend=target.backend, optimization_level=1
        )
        isa_circuits = {n: pass_manager.run(c) for n, c in circuits.items()}
        first = isa_circuits["generic with U"]
        print(f"STEP 2 - 'generic with U' became: {dict(first.count_ops())}")

        # ---- STEP 3: observable: not used (Sampler) ----------------------

        # ---- STEP 4: PUBs ------------------------------------------------
        pubs = [(isa,) for isa in isa_circuits.values()]

        # ---- STEP 5: execute ---------------------------------------------
        job = target.sampler.run(pubs, shots=args.shots)
        result = target.wait(job)
        if result is None:
            continue

        # ---- STEP 6: read the results ------------------------------------
        print(f"STEP 6 - P(0) from {args.shots} shots:")
        for name, pub_result in zip(circuits, result):
            counts = pub_result.data.meas.get_counts()
            p0 = counts.get("0", 0) / args.shots
            print(f"  {name:22s} P(0) = {p0:.3f}   ideal = {expected_p0[name]:.3f}")

    print(
        "\nTakeaway: with U the qubit always returns to 0 (the rotation worked); "
        "without it, P(0) = |<phi|psi>|^2."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
