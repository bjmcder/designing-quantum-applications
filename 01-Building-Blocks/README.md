# Unit 1: Building blocks

By the end of this unit you will be able to write a small problem in terms of
bits and operators, run it on a simulator on your own computer or on
`ibm_rensselaer` (the IBM quantum computer on the RPI campus), and check the
answer against a hand calculation. The unit starts with ordinary bits and the
logic you already know, and adds the quantum ingredients one at a time.
Units 2 and 3 use exactly these pieces to build algorithms and applications,
and Unit 4 holds optional advanced examples.

## Before you start

The notebooks assume:

- Python, including NumPy arrays and functions;
- linear algebra: vectors, matrices, matrix products, transpose, inverse,
  determinant, trace, eigenvalues and eigenvectors;
- complex numbers: real and imaginary parts, the complex conjugate
  $\bar z$, the modulus $|z|$, and Euler's formula
  $e^{i\varphi} = \cos\varphi + i\sin\varphi$;
- calculus: the power series of $e^x$, $\cos x$ and $\sin x$ (used once, in
  Notebook 07);
- Boolean logic: NOT, AND, OR, XOR, and binary numbers;
- probability: probability, mean, variance, and standard deviation.

No physics is assumed. Every quantum term is defined before it is used, in
the notebook listed in [Where each term is defined](#where-each-term-is-defined).
Gate names that first appear in the transpiler's printouts, such as `u3`,
`rz` and `sx`, get a one-line description where they appear and a full
definition in Notebook 07.

If any item above is rusty, three quick checks:

1. What is the product of the matrix $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ and
   the vector $\begin{pmatrix}1\\0\end{pmatrix}$?
2. What is $|e^{i\pi/4}|$, and what is the complex conjugate of $e^{i\pi/4}$?
3. What is $1 \oplus 1$ (XOR)?

Answers: $\begin{pmatrix}0\\1\end{pmatrix}$; $1$ and $e^{-i\pi/4}$; $0$.

## The recipe

Every notebook in this course, from single bits here to the molecules in
Unit 3, follows the same four steps.

| Step | What you decide | Qiskit pieces |
|---|---|---|
| Encoding | Which qubit states stand for the objects in the problem, and the circuit that prepares the starting state | `QuantumCircuit` and its gates |
| Operator | What is applied to that state, and what is measured | gates; from Notebook 08 on, `SparsePauliOp` |
| Execution | Running the circuit on the simulator or on hardware | transpile, PUB, primitive (`Sampler` or `Estimator`), job |
| Analysis | Turning the raw results into an answer and checking it against theory | counts, expectation values, standard errors |

The Execution and Analysis pieces are introduced in Notebook 00 and reused
unchanged after that. In some problems the Operator step comes before the
Encoding step, because how you store the problem in qubits depends on the
operator. The hydrogen-molecule notebook in Unit 3 is one such case.

## The notebooks

Work through them in order. Each one builds on the ones listed in its last
column.

| # | Notebook | After it you can | Builds on |
|---|---|---|---|
| 00 | [Bits and basis states](notebooks/00_Bits_and_Basis_States.ipynb) | write a bitstring as a vector, set and clear bits with $a^\dagger$ and $a$, and run a circuit | prerequisites |
| 01 | [One-bit operators](notebooks/01_One_Bit_Operators.ipynb) | derive $n$, $\bar n$, $I$, $X$ and $Z$ from $a^\dagger$ and $a$ | 00 |
| 02 | [Reversible logic](notebooks/02_Reversible_Logic.ipynb) | build CX, SWAP and CCX from the same parts and compute any Boolean function reversibly | 01 |
| 03 | [Encodings](notebooks/03_Encodings.ipynb) | store a number or a choice in binary, Gray or one-hot form and compare the costs | 00 to 02 |
| 04 | [Amplitudes and phase](notebooks/04_Amplitudes_and_Phase.ipynb) | compute measurement probabilities from amplitudes and explain what phase changes | 01, 02 |
| 05 | [An operator from two states](notebooks/05_Rotation_From_Outer_Product.ipynb) | build a gate that takes any start state to any goal state | 04 |
| 06 | [Interference: H versus SX](notebooks/06_Interference_H_vs_SX.ipynb) | predict when amplitudes cancel, and measure in the X and Y bases | 04, 05 |
| 07 | [Rotations](notebooks/07_Rotations.ipynb) | write rotation gates as exponentials of Pauli operators and sweep their angles | 04, 06 |
| 08 | [Operators as Pauli sums](notebooks/08_Pauli_Sums.ipynb) | write any operator, or any cost function of bits, as a sum of Pauli strings | 01, 03, 07 |
| 09 | [Expectation values: Sampler versus Estimator](notebooks/09_Sampler_vs_Estimator.ipynb) | measure the average value of an operator, with its uncertainty, two ways | 06 to 08 |
| 10 | [Entanglement: Bell states](notebooks/10_Bell_States.ipynb) | tell entangled states from product states and from classical mixtures | 02, 09 |
| 11 | [Fermions](notebooks/11_Fermions.ipynb) | write operators for electrons or nucleons on qubits, with the correct signs | 01, 02, 04, 08 to 10 |

Notebooks 00 to 03 use only ordinary bits: every state is a bitstring, and
every operator maps bitstrings to bitstrings. Notebook 04 adds the one new
ingredient that makes the rest of the course possible.

## How each notebook is organized

1. **Before you start**: what the notebook assumes, two or three questions to
   test it, and what you will be able to do at the end.
2. **Concepts**: each new idea is motivated, defined, and then used in a
   worked example that ends with a check.
3. **The experiment**: the four steps of the recipe, run on the simulator.
4. **Exercises**: ordered from the same structure with new numbers, to new
   situations, to one that mixes in earlier notebooks. The last exercises
   use `ibm_rensselaer`.
5. **Solutions** to every exercise, and a table of the terms the notebook
   introduced.

## Running the notebooks

Open each notebook from its own folder (`01-Building-Blocks/notebooks`); this
is the default in Jupyter and VS Code. The helper file
`scripts/_runtime.py` decides where circuits run.

By default every notebook runs only on the noiseless Aer simulator, so it
finishes in seconds. To also run on the campus quantum computer, set
`skip_hardware=False` in the `SETTINGS` cell near the top. You need a saved
IBM token (see `00-Setup-Start-Here`). Hardware jobs wait in a queue; the job
ID is saved as soon as a job is submitted, so you can stop waiting and
retrieve the result later with `00-Setup-Start-Here/05_retrieve_job.py`.

## Running the scripts

Each notebook has a script with the same number and name in `scripts/`, for
example `scripts/07_Rotations.py`. A script runs only the notebook's
experiment, with the same four steps and the same code, from the command line
and without the explanations. Run it from the `scripts` folder:

```bash
cd 01-Building-Blocks/scripts
python 07_Rotations.py              # Aer simulator only
python 07_Rotations.py --hardware   # also ibm_rensselaer
python 07_Rotations.py --help       # shots, precision, backend, account name
```

As in the notebooks, the simulator is the default and `--hardware` adds the
campus quantum computer. `--shots` and `--precision` set the same values as
`shots` and `precision` in a notebook's `SETTINGS` cell.

## Notation

| Symbol | Meaning | Defined in |
|---|---|---|
| $\lvert 0\rangle$, $\lvert 1\rangle$ | the two basis states of one bit, the column vectors $(1,0)^T$ and $(0,1)^T$ | 00 |
| $\lvert x_{n-1}\cdots x_1x_0\rangle$ | a bitstring, written with bit 0 on the **right**, the order Qiskit prints | 00 |
| $a^\dagger$, $a$ | creation (sets a bit from 0 to 1) and annihilation (clears a bit from 1 to 0) | 00 |
| $A_k$ | the one-bit operator $A$ acting on bit $k$, with the identity on every other bit | 02 |
| $A_1 \otimes A_0$ | the matrix of a two-bit operator; the left factor acts on bit 1, matching the bit order | 02 |
| $n = a^\dagger a$, $\bar n = a a^\dagger$ | the projectors onto $\lvert 1\rangle$ and onto $\lvert 0\rangle$ | 01 |
| $x \oplus y$ | XOR of two bits | 02 |
| $\alpha$, $\beta$ | the complex amplitudes of $\lvert 0\rangle$ and $\lvert 1\rangle$ in a one-qubit state | 04 |
| $M^\dagger$ | the conjugate transpose of a matrix $M$; for $a$ it gives $a^\dagger$, which is where the name comes from | 04 |
| $\langle\phi\vert\psi\rangle$ | the inner product, with the complex conjugate on the left vector | 04 |
| $R_P(\theta)$ | the rotation $e^{-i\theta P/2}$ about the axis of the Pauli operator $P$ | 07 |
| `"ZI"` | a Qiskit Pauli label; the **rightmost** letter acts on qubit 0, so `"ZI"` is $Z_1$ | 08 |
| $H$, $\mathcal H$, $H_{\text{cost}}$ | bare $H$ is always the Hadamard gate; a general Hamiltonian is $\mathcal H$, and a specific one carries a subscript | 04, 08 |
| $\langle O\rangle$ | the expectation value of the operator $O$ | 09 |
| $c_p^\dagger$, $c_p$ | fermion creation and annihilation operators for orbital $p$ (Unit 3 writes these as $a_p^\dagger$, $a_p$) | 11 |
| $x_p$ | the occupation of orbital $p$, 0 or 1, the same as bit $p$ of a bitstring | 11 |

## Where each term is defined

| Term | Notebook |
|---|---|
| bit, bitstring, register, qubit, basis state, ket | 00 |
| tensor product of states | 00 |
| creation and annihilation operators | 00 |
| circuit, gate, measurement, shot, counts | 00 |
| simulator, backend, transpile, ISA circuit, primitive, Sampler, PUB, job | 00 |
| target, pass manager, readout error | 00 |
| scatter of a measured fraction, and the scatter rule for comparing results | 00 |
| operator, bra, outer product, inner product of real vectors | 01 |
| number operator, projector, completeness | 01 |
| Pauli $X$ and $Z$, Z-basis measurement | 01 |
| tensor product of operators, mixed-product rule, commute, reversible function, permutation matrix | 02 |
| CX (control, target), SWAP, hopping term, CCX (Toffoli) | 02 |
| ancilla, oracle $U_f$, uncomputation | 02 |
| encoding, decoding, binary, Gray, unary (one-hot), neighbor step | 03 |
| Hamming weight, Hamming distance, valid and invalid bitstrings, penalty | 03 |
| amplitude, superposition, normalization, Born rule (one qubit and several) | 04 |
| conjugate transpose, reverse-order rule $(AB)^\dagger = B^\dagger A^\dagger$, complex inner product | 04 |
| unitary operator, orthogonal states, overlap, angle between two states | 04 |
| Pauli $Y$, Hadamard $H$, $\lvert\pm\rangle$, $\lvert\pm i\rangle$, phase gate $P(\varphi)$, $S$, $S^\dagger$ | 04 |
| global phase, relative phase, Bloch sphere, statevector, barrier | 04 |
| orthogonal state $\lvert s_\perp\rangle$, orthonormal, preparation operator $V_s$, phase choice for $\lvert s_\perp\rangle$ | 05 |
| interference, $\sqrt{X}$ (SX), measurement basis, change of basis, X basis, Y basis | 06 |
| matrix exponential, rotation gates $R_x$, $R_y$, $R_z$, $R_{zz}$, parity | 07 |
| circuit parameter, parameter sweep | 07 |
| Pauli string, Pauli decomposition, Hermitian operator | 08 |
| Hamiltonian, ground state, ground-state energy, diagonal Hamiltonian | 08 |
| observable, expectation value, standard error | 09 |
| commuting observables, qubit-wise commuting, anticommute, Estimator, precision | 09 |
| physical qubit, layout, error mitigation, readout-error mitigation | 09 |
| product state, entangled state, Bell states, correlator | 10 |
| Bloch vector, classical mixture, entanglement witness | 10 |
| fermion, orbital, occupation-number state | 11 |
| anticommutator, anticommutation relations, Jordan-Wigner transformation, Z string, sign rule | 11 |
