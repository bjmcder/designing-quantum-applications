# Qubit Encodings for Lattice Protein Folding

Source: H. Linn, I. Lyngfelt, L. García-Álvarez, and G. Johansson, "Resource
analysis of quantum algorithms for coarse-grained protein folding models,"
*Phys. Rev. Research* **6**, 033112 (2024). Section, equation, table, and
appendix numbers below refer to that paper. Code from the authors:
<https://github.com/HannaLinn/resources-qaoa-protein-folding>.

These notes cover only how a protein conformation becomes a bitstring and what
circuit the paper assumes. The Hamiltonian appears only where it forces extra
qubits or rules out bitstrings.

## 1. Common framework

**Chain.** A protein is a sequence of $N$ beads (amino acids) $a_1,\dots,a_N$.
In the HP model each bead is hydrophobic (H) or polar (P).

**Cardinality vector (Eq. 1).** Each bead $a_i$ picks one conformation $r_i$
from a finite set $R_i$ with $c_i = |R_i|$ elements. The paper describes every
model by

$$\mathbf{C} = (c_1, c_2, \dots, c_N).$$

The meaning of "conformation" changes from model to model: a lattice site
(coordinate-based model), a step direction (turn-based model), or a side-chain
rotamer (side-chain model).

**Bitstring.** Each integer $r_i$ is written in some binary code, and the codes
are concatenated: $x = r_1 r_2 \cdots r_N = x_1 x_2 \cdots x_M$ with $x_k \in
\{0,1\}$ (Sec. III). The encoding (unary, binary, or BU-binary, Sec. 2 below)
sets $M$.

**Bit to qubit (Sec. III, before Eq. 7).** Each bit becomes one qubit through

$$x_k \;\to\; \frac{1-\sigma^z_k}{2},$$

so $x_k = 0 \leftrightarrow |0\rangle$ ($\sigma^z = +1$) and $x_k = 1
\leftrightarrow |1\rangle$ ($\sigma^z = -1$). Any polynomial cost in the bits
then becomes a $k$-local Ising Hamiltonian $H_\text{cost} = \sum h_i\sigma^z_i +
\sum J_{ij}\sigma^z_i\sigma^z_j + \sum J_{ijk}\sigma^z_i\sigma^z_j\sigma^z_k +
\cdots$ (Eq. 7). A product of $m$ bits gives Pauli-$Z$ strings on up to $m$
qubits, which is why dense encodings produce high-order terms.

**Feasible set (Sec. IV C).** $S$ is the set of all $2^M$ bitstrings. $F
\subseteq S$ holds the bitstrings that pick exactly one valid conformation per
bead and violate no physical constraint. Every bitstring outside $F$ has to be
penalized in the cost or excluded by the circuit.

> Qiskit note: Qiskit prints count keys with qubit 0 as the **rightmost**
> character. The paper writes bitstrings left to right in index order. Reverse
> the string when decoding.

## 2. Integer-to-qubit encodings (Sec. III A, Tables I–III)

Every encoding below takes one integer $r \in \{0,\dots,c-1\}$, the conformation
index for one bead.

### 2.1 Unary (one-hot, "sparse")

- Uses $c$ bits. Bit number $r$ is 1 and all the others are 0. Table I writes
  position 0 on the **left**: $0 \to 10000$, $1 \to 01000$, …, $4 \to 00001$ for
  $c=5$.
- Qubits: $\sum_i c_i$ (Table II).
- Validity: each block of $c_i$ bits must have Hamming weight exactly 1. The
  bitstring as a whole has Hamming weight $N$, one per bead (the paper calls
  this count $M$ in Sec. III A 1).
- Feasible fraction (one-hot condition only): $|F|/|S| = \prod_i c_i/2^{c_i}$
  (Table III). For $c_i = 2$ this is at most $2^{-N}$.
- Locality: 2. Only one- and two-body terms appear. One-body terms: $\sum_i
  c_i$. Two-body terms: $\binom{\sum_i c_i}{2}$ (Table II, Sec. IV B).

### 2.2 Binary ("dense")

- Uses $n_i = \lceil \log_2 c_i \rceil$ bits holding the ordinary base-2 value
  of $r$, most significant bit on the left: $0 \to 000$, $3 \to 011$, $4
  \to 100$ (Table I).
- Qubits: $\sum_i \lceil \log_2 c_i \rceil$ (Table II).
- Validity: codes with value $\ge c_i$ are invalid. Hamming weight cannot detect
  them (Sec. III A 2). When $c_i$ is a power of two, every code is valid.
- Feasible fraction: $\prod_i c_i / 2^{\lceil\log_2 c_i\rceil}$ (Table III).
  Worst case is $c_i = 2^l + 1$.
- Locality: $\lceil\log_2 \max(c_i)\rceil + \lceil\log_2 \max_2(c_i)\rceil$,
  where $\max_2$ is the second-largest $c_i$ (Table II). A two-bead interaction
  couples all bits of both beads.

### 2.3 Block-unary binary (BU-binary)

- Choose a block size $g$ (the paper uses $g = 3$ throughout). Each bead gets
  $\lceil c_i/g \rceil$ blocks. Each block has $n = \lceil\log_2(g+1)\rceil$
  bits.
- The all-zero block means "inactive." Exactly one block per bead is active, and
  its nonzero binary value (1 to $g$) picks one of $g$ conformations inside that
  block.
- Mapping in Table I ($g=3$, $n=2$, blocks written right to left): $0
  \to 00\,01$, $1 \to 00\,10$, $2 \to 00\,11$, $3 \to 01\,00$, $4 \to 10\,00$.
  In general, block $b = \lfloor r/g \rfloor$ (counted from the right) holds the
  value $(r \bmod g) + 1$ and every other block is 00.
- Qubits: $\sum_i \lceil c_i/g\rceil\,\lceil\log_2(g+1)\rceil$ (Table II).
- Validity: exactly one nonzero block per bead, and that block's value must map
  to $r < c_i$.
- Feasible fraction: $\prod_i c_i / 2^{2\lceil c_i/g\rceil}$ as printed in Table
  III. This is the $g=3$ case of $\prod_i c_i/2^{n\lceil c_i/g\rceil}$.
- Locality: $2\lceil\log_2(g+1)\rceil$ (Table II), set by $g$ and not by $c_i$.
  This lets you match $g$ to the largest native multiqubit gate. $g = 1$
  recovers unary. $g \to \max(c_i)$ approaches binary.

### Worked example: $c = 5$ (Table I)

| $r$ | Unary (5 qubits) | Binary (3 qubits) | BU-binary, $g=3$ (4 qubits) |
|---|---|---|---|
| 0 | 10000 | 000 | 00 01 |
| 1 | 01000 | 001 | 00 10 |
| 2 | 00100 | 010 | 00 11 |
| 3 | 00010 | 011 | 01 00 |
| 4 | 00001 | 100 | 10 00 |
| valid codes | 5 of 32 | 5 of 8 | 5 of 16 |

The invalid codes are, for unary, any weight other than 1 (for example 11000 or
00000). For binary they are 101, 110, and 111. For BU-binary there are 11: 00 00
(no active block), the 9 strings with both blocks active (for example 01 01),
and 11 00, which would mean $r = 5 \ge c$.

## 3. Lattice models

The paper studies square (2D) and simple cubic (3D) lattices. It also compares
against the tetrahedral-lattice turn model of Robert *et al.* (Ref. 13). The
intro notes that early coordinate encodings (Ref. 32) need at least $DN\log_2 N$
qubits, where $D$ is the lattice dimension.

### 3.1 Coordinate-based HP lattice model (Sec. II A 1, Eqs. 2–3, Ref. 41)

**Idea.** Each bead's conformation is the lattice site it occupies.

**Lattice.** An $L_1 \times L_2$ rectangular grid (or an $L_1\times L_2\times
L_3$ cubic grid). The resource study scans every grid with $N \le L_1L_2
\le 1.5N$ sites (or $L_1L_2L_3$ in 3D) (Sec. IV).

**Checkerboard symmetry.** Color sites even or odd by the parity of $x+y(+z)$.
Consecutive beads must sit on opposite colors, so bead $i$ is allowed only on
sites of its own parity. This halves the choices:

$$
\mathbf{C}^\text{coord}_\text{square} = \left(\left\lceil
\tfrac{L_1L_2}{2}\right\rceil, \left\lfloor \tfrac{L_1L_2}{2}\right\rfloor,
\left\lceil \tfrac{L_1L_2}{2}\right\rceil, \dots\right) \quad\text{(Eq. 2)}
$$

$$
\mathbf{C}^\text{coord}_\text{cubic} = \left(\left\lceil
\tfrac{L_1L_2L_3}{2}\right\rceil, \left\lfloor
\tfrac{L_1L_2L_3}{2}\right\rfloor, \dots\right) \quad\text{(Eq. 3)}
$$

The first bead uses the ceiling, which is the larger (even) sublattice when the
site count is odd. After that the entries alternate.

**Bitstring to conformation.** Index the sites of each parity class
$0,1,2,\dots$ in some fixed order, for example row-major. Bead $i$'s code
(unary, binary, or BU-binary) gives an index $r_i$ into its parity list, and
that index is its site. Unary is the "one qubit per (bead, site)" formulation:
qubit $(i,s) = 1$ means bead $i$ is on site $s$.

**Qubits.** Apply Table II to $\mathbf{C}$:
- Unary: $\sum_i c_i \approx N\cdot L_1L_2/2$, which is about $N^2$ because the
  site count grows with $N$ (Sec. IV A).
- Binary: $\sum_i \lceil\log_2 c_i\rceil$, about $N\log N$ (the text writes
  $ND\log N$).
- BU-binary: $\sum_i \lceil c_i/g\rceil\lceil\log_2(g+1)\rceil$.

**Constraints the encoding needs (Sec. II A 1).** No auxiliary qubits are
needed. Three penalty terms remove nonphysical bitstrings:
1. One site per bead. This is the one-hot condition in unary. Binary needs it
   only when $c_i$ is not a power of two.
2. Self-avoidance. No two beads on the same site. Only beads of the same parity
   can collide.
3. Chain connectivity. Beads $i$ and $i+1$ must be on neighboring sites.

The contact energy is evaluated only between adjacent even and odd sites.

**Worked example: $N=4$ on a $2\times2$ square grid.** Even sites (row-major)
are $[(0,0),(1,1)]$ and odd sites are $[(0,1),(1,0)]$. $\mathbf{C} = (2,2,2,2)$.
- Binary uses 1 qubit per bead, 4 qubits total. The bitstring
  $b_0b_1b_2b_3 = 0110$ decodes to bead 0 at $(0,0)$, bead 1 at $(1,0)$, bead 2
  at $(1,1)$, and bead 3 at $(0,1)$, which is a valid U-shaped fold. The
  bitstring $0100$ places beads 0 and 2 both at $(0,0)$, which violates
  self-avoidance.
- Unary uses 2 qubits per bead, 8 total. The same fold is $10\;01\;01\;10$.
- BU-binary with $g=3$ uses one 2-bit block per bead, 8 total. Each bead is $01$
  for $r=0$ and $10$ for $r=1$, so the fold is $01\;10\;10\;01$.
- Brute force finds exactly 4 valid folds ($0011, 0110, 1001, 1100$). That is 4
  of 16 binary strings and 4 of 256 unary strings. Table III predicts a feasible
  fraction of 1 for binary and 1/16 for unary. The difference shows that Table
  III counts only the one-conformation-per-bead condition, so for lattice models
  it is an upper bound (the paper calls it approximate, Sec. IV C).

### 3.2 Turn-based HP lattice model (Sec. II A 2, Eqs. 4–5, App. C)

**Idea.** The conformation is the sequence of step directions ("turns") from
each bead to the next. Bead 0 sits at the origin. Bead $i$'s position is the sum
of the earlier turns.

**Cardinality.** The number of choices equals the lattice coordination number:

$$
\mathbf{C}^\text{turn}_\text{square} = (4,\dots,4)\ \ \text{(Eq. 4)}, \qquad
\mathbf{C}^\text{turn}_\text{cubic} = (6,\dots,6)\ \ \text{(Eq. 5)}.
$$

The tetrahedral lattice (Ref. 13) also has 4 choices per turn.

**Turn qubits per step.** Unary uses $2D$ (4 on square, 6 on cubic, one per
direction $\pm x, \pm y, \pm z$). Binary uses $\lceil\log_2 2D\rceil$ (2 on
square, 3 on cubic). BU-binary uses $\lceil 2D/g\rceil\lceil\log_2(g+1)\rceil$
by the general rule.

**Binary indexing on the cubic lattice (App. C, Eqs. C1, C2, C7).** The appendix
follows Babej *et al.* (Ref. 36), with beads $0,\dots,N-1$ and turns
$j = 0,\dots,N-2$ (turn $j$ goes from bead $j$ to bead $j+1$):

- **Symmetry fixing.**
  - Turn 0 is fixed to $+x$, so bead 1 is at $(1,0,0)$. This costs no qubits.
  - Turn 1 uses a single qubit $q_0$: $q_0 = 1$ means $+x$ and $q_0 = 0$ means
    $+y$. Rotations make every other perpendicular choice equivalent to $+y$,
    and $-x$ would be a backturn. This follows from $x_i = 1 + q_0 + \dots$ in
    Eq. C7 and the first two terms of $H_\text{back}$ in Eq. C6.
- **Turns $j = 2,\dots,N-2$** use three qubits each, $(q_{3j-5}, q_{3j-4},
  q_{3j-3})$. The paper gives only the $\pm x$ indicator functions:

$$
d^j_{+x} = (1-q_{3j-4})\,q_{3j-5}\,q_{3j-3} \quad\text{(C1)} \;\Rightarrow\;
(q_{3j-5},q_{3j-4},q_{3j-3}) = (1,0,1)
$$
$$
d^j_{-x} = (1-q_{3j-3})\,q_{3j-5}\,q_{3j-4} \quad\text{(C2)} \;\Rightarrow\;
(q_{3j-5},q_{3j-4},q_{3j-3}) = (1,1,0)
$$

  The $\pm y$ and $\pm z$ codes come from Ref. 36. To implement, assign four of
  the six remaining 3-bit codes to $\pm y, \pm z$. The last two codes then mean
  no direction, so they are invalid and must be penalized. The paper does not
  write that penalty out.
- **Conformation qubits.** There are $1 + 3(N-3) = 3N-8 = DN-8$, numbered
  $q_0,\dots,q_{3N-9}$. This matches the slack pointer offset $DN-8$ in Eq. C14
  and the general form "$DN - c_1$" in Eq. C25. The paper gives no explicit 2D
  indexing. The direct analog (turn 0 fixed, 1 qubit for turn 1, then 2 qubits
  per turn) would use $2N-5$. That figure is our derivation, not the paper's.
- **Decoding positions (Eq. C7).** $x_0 = 0$ and $x_1 = 1$. For $i \ge 2$,
  $$
  x_i = 1 + q_0 + \sum_{j=2}^{i-1}\left(d^j_{+x} - d^j_{-x}\right), \qquad
  d^j_{+x}-d^j_{-x} = q_{3j-3}q_{3j-5} - q_{3j-4}q_{3j-5}\ \text{(C8)}.
  $$
  The paper says $y_i$ and $z_i$ follow the same pattern. By the same logic
  $y_i = (1-q_0) + \sum_{j\ge2}(d^j_{+y}-d^j_{-y})$ and $z_i =
  \sum_{j\ge 2}(d^j_{+z}-d^j_{-z})$.

**Constraints the encoding needs.** Here the encoding forces extra qubits.

1. **No backturns, $H_\text{back}$ (Eq. C6).** Penalize turn $j$ followed by its
   opposite at turn $j+1$: $d^j_{+x}d^{j+1}_{-x} + d^j_{-x}d^{j+1}_{+x} + (\pm
   y) + (\pm z)$ for $j = 2,\dots,N-3$. Add two boundary terms that combine
   $q_0$ with turn 2. The paper writes these with $\vee$, but they must be read
   as products, for example $q_0\,d^2_{-x}$. This term needs no extra qubits.
2. **No overlap (Eqs. C9–C16).** Squared distance $D_{ij} = (x_i-x_j)^2 +
   (y_i-y_j)^2 + (z_i-z_j)^2$ (C9). Bounds are $0\le D_{ij}\le (i-j)^2$ (C10),
   and the constraint is $D_{ij}\ge 1$ (C11–C12).
   - Only pairs with $j \ge i+4$ and $j - i$ **even** are checked. Lattice
     parity means only same-parity beads can collide, and the case $j = i+2$ is
     already covered by $H_\text{back}$. The factor $(|1+i-j| \bmod 2)$ selects
     these pairs.
   - The inequality becomes an equality with a **slack integer** $\alpha_{ij}$
     stored on $\mu_{ij}$ auxiliary qubits:
     $$
     \alpha_{ij} = \sum_{k=0}^{\mu_{ij}-1} q_{p_{ij}+k}\,2^{\mu_{ij}-1-k}\
     \text{(C15)}, \qquad H_\text{overlap} =
     \lambda\sum_{i=0}^{N-5}\sum_{j=i+4}^{N-1}(|1+j-i|\bmod 2)\,(2^{\mu_{ij}} -
     D_{ij} - \alpha_{ij})^2\ \text{(C16)}.
     $$
     Because $0\le\alpha_{ij}\le 2^{\mu_{ij}}-1$, the penalty vanishes only when
     $D_{ij} = 2^{\mu_{ij}}-\alpha_{ij} \ge 1$.
   - The qubits per pair are $\mu_{ij} = \lceil 2u\rceil$, with $u$ from Table
     IV. For unary $u = j-i$. For binary $u = \log_2(j-i)$, so $\mu_{ij} =
     \lceil 2\log_2(j-i)\rceil$, enough to hold values up to $(j-i)^2$. For
     BU-binary $u = \lceil (j-i)/g\rceil\lceil\log_2(g+1)\rceil$.
   - Totals and layout:
     $$
     M^\text{dist}_\text{aux} =
     \sum_{i=0}^{N-5}\sum_{j=i+4}^{N-1}
     \left\lceil 2u\,(|1+i-j|\bmod 2)\right\rceil
     \ \text{(C13)},\qquad p_{ij} = (DN-8) +
     \sum_{u=0}^{i}\sum_{n=u+4}^{N-1}\mu_{un} - \sum_{m=j}^{N-1}\mu_{im}\
     \text{(C14)}.
     $$
     The slack registers follow the conformation qubits in lexicographic order
     of $(i,j)$. Set $\mu_{ij}=0$ for odd-separation pairs.
3. **Contacts, $H_\text{pair}$ (Eqs. C19–C21).** One auxiliary qubit
   $\omega_{jk}$ marks each pair that can touch, with $\omega_{jk} = 1$ iff
   $D_{jk} = 1$ (C19). Candidate pairs have $k \ge j+3$ and $k - j$ **odd**:
   $$
   M^\text{pair}_\text{aux} = \sum_{j=0}^{N-4}\sum_{k=j+3}^{N-1}(|j-k|\bmod 2)\
   \text{(C20)}, \qquad H_\text{pair} = \sum
   (|j-k|\bmod 2)\,\omega_{jk}P_{jk}(2-D_{jk})\ \text{(C21)}.
   $$
   The paper does not say where the $\omega$ qubits go. Placing them after the
   slack registers is a natural choice.

**Total qubits (Eq. C24).** $M_\text{total} = M_\text{conf} +
M^\text{dist}_\text{aux} + M^\text{pair}_\text{aux}$. Explicitly:

- Binary (C25): $(DN - c_1) +
  \sum_{i=0}^{N-5}\sum_{j=i+4}^{N-1}\lceil 2\log_2(j-i)\rceil(|1+j-i| \bmod 2) +
  \sum_{j=0}^{N-4}\sum_{k=j+3}^{N-1}(|j-k|\bmod 2)$. Asymptotically
  $\tfrac12N^2\log_2N - (\tfrac14 - \tfrac{3}{4\ln2})N^2 + O(N)$ (C26).
- Unary (C27): $2DN + \sum\sum\lceil 2(j-i)\rceil(|1+j-i| \bmod 2) +
  \sum\sum(|j-k|\bmod 2)$. Asymptotically $\tfrac16N^3 + \tfrac14N^2 + O(N)$
  (C28).
- BU-binary (C29): $\lceil\log_2(g+1)\rceil\lceil ND/g\rceil + \sum\sum 2\lceil
  (j-i)/g\rceil\lceil\log_2(g+1)\rceil(|1+j-i|\bmod 2) +
  \sum\sum(|j-k|\bmod 2)$. Asymptotically $\frac{\log_2(1+g)}{6g}N^3 +
  \frac{\log_2(1+g)}{4g}N^2 + O(N)$ (C30).

The auxiliary qubits dominate, so turn-based models scale quadratically or worse
in $N$ even though the turn qubits alone grow linearly (Sec. IV A, Sec. V).

**Worked example: $N = 5$ on the cubic lattice, binary.**
- Conformation qubits: $3N-8 = 7$ ($q_0$ for turn 1, $q_1q_2q_3$ for turn 2,
  $q_4q_5q_6$ for turn 3).
- Bitstring $q_0\dots q_6 = 0\;110\;110$ decodes as turn 0 $=+x$, turn 1 $=+y$,
  turn 2 $=-x$, turn 3 $=-x$. Bead positions are $(0,0,0), (1,0,0), (1,1,0),
  (0,1,0), (-1,1,0)$. Eq. C7 check: $x_4 = 1 + 0 - 1 - 1 = -1$.
- Changing turn 3 to $+x$ ($q_4q_5q_6 = 101$) gives a backturn onto bead 2,
  which $H_\text{back}$ penalizes.
- Overlap slack: the only checked pair is $(0,4)$, with $\mu_{04} =
  \lceil 2\log_2 4\rceil = 4$ qubits at $p_{04} = 7$, i.e., $q_7\dots q_{10}$.
  For the fold above $D_{04} = 2$, so $\alpha_{04} = 16 - 2 = 14 = 1110_2$.
- Contact qubits: pairs $(0,3)$ and $(1,4)$ get $\omega_{03}, \omega_{14}$ at
  $q_{11}, q_{12}$. In the fold above $D_{03} = 1$, so $\omega_{03}=1$, and
  $D_{14}=5$, so $\omega_{14}=0$.
- Total is 13 qubits, matching C24 and C25 with $c_1 = 8$. The unary count by
  C27 is $30 + 8 + 2 = 40$, and in unary each turn would be a 6-bit one-hot
  word, for example $-x \to 010000$ with order $(+x,-x,+y,-y,+z,-z)$.

**Variants named in Sec. II A 2.**
- *Turn ancilla encoding* is the one above. It stores distance and interaction
  information in auxiliary qubits, and its qubit count grows quadratically.
- *Turn circuit encoding* (Refs. 12, 35) tracks the same information with
  multiqubit terms built from nonunitary half adders and XNOR gates. Its qubit
  count is linear, but half adders may add ancillas. The paper does not analyze
  it.
- *Tetrahedral turn model* (Ref. 13) has 4 directions per turn. For $N = 15$ it
  needs 76 qubits (unary) or 53 (binary). The paper reports 404/226 for the
  square turn model and 135/52.5 on average for the coordinate model at
  coordination number 4 (Sec. IV A). Plugging into the printed C25 and C27 does
  not exactly reproduce 404/226 (we get about 279 minus $c_1$ for binary). Treat
  the formulas as the definition and check counts with your own code.

## 4. Side-chain conformation model (Sec. II B, Eq. 6)

This model is off-lattice but uses the same encodings, so it is included for
completeness. The backbone is fixed, and residue $i$ picks one of $c_i$
rotamers. The energy is pairwise, $E(\mathbf r) =
\sum_{i=2}^{N}\sum_{j<i}E_{ij}(r_i,r_j)$ (Eq. 6). The only constraint is one
rotamer per residue, so Table II and Table III are exact for this model, with no
geometry penalties and no auxiliary qubits. The study draws $c_i \sim \mathcal
U(2,100)$ (and $\mathcal U(2,20)$ for the small-$N$ insets).

## 5. Ansatz and circuit structure assumed (Sec. III B, Sec. IV, App. B)

- **What the paper builds.** It builds one QAOA cost layer, $e^{-i\gamma
  H_\text{cost}}$ with a real parameter $\gamma$. All resource numbers count a
  **single** cost layer, which is a lower bound. The paper does not determine
  how many layers are needed or whether the algorithm succeeds (Sec. I).
- **Gate compilation (App. B, Eq. B1, Fig. 10).** Each $m$-body term
  $\exp[-i\tfrac{\alpha}{2}\sigma^z_i\cdots\sigma^z_\ell]$ becomes a CNOT ladder
  that computes parity onto one qubit, then $R_z(\alpha)$, then the reverse
  ladder. That costs $2(m-1)$ CNOTs. One-body terms are single $R_z$ gates. The
  study assumes all-to-all connectivity. Naive routing on limited connectivity
  can add $O(n^3)$ depth, and swap networks reduce that to linear overhead (Sec.
  III B).
- **What the paper does not specify.** It gives no initial state, no mixer, and
  no hardware-efficient layers. It cites QAOA (Ref. 9) and the quantum
  alternating operator ansatz (Ref. 37), which Ref. 12 used to enforce hard
  constraints when folding a 4-bead chain. Implementation choices beyond the
  paper:
  - With the standard $|+\rangle^{\otimes M}$ start and $X$ mixer, every
    bitstring is reachable, so all constraints must be penalty terms.
  - For unary codes, a Hamming-weight-preserving (XY) mixer started from a valid
    one-hot state stays inside the one-hot subspace. That removes constraint 1
    of Sec. 3.1 from the cost. This is the alternating operator ansatz idea of
    Ref. 37 and is not analyzed in this paper.
- **Encoding versus hardware (Sec. V).**
  - Binary has the fewest qubits and the smallest infeasible set but the highest
    locality. It suits hardware with native multiqubit gates, such as trapped
    ions.
  - Unary has the most qubits but only 2-local terms (in the coordinate and
    side-chain models). It suits hardware limited to two-qubit gates, such as
    superconducting qubits.
  - BU-binary sits between the two, tuned by $g$.

## 6. Comparison

| Encoding | Qubits per bead | Locality (coord./side-chain) | Feasible fraction (Table III) | Invalid strings |
|---|---|---|---|---|
| Unary | $c_i$ | 2 | $\prod c_i/2^{c_i}$ | weight $\ne 1$ per block |
| Binary | $\lceil\log_2 c_i\rceil$ | $\lceil\log_2 c_{\max}\rceil+\lceil\log_2 c_{\max,2}\rceil$ | $\prod c_i/2^{\lceil\log_2 c_i\rceil}$ | codes $\ge c_i$ |
| BU-binary | $\lceil c_i/g\rceil\lceil\log_2(g+1)\rceil$ | $2\lceil\log_2(g+1)\rceil$ | $\prod c_i/2^{2\lceil c_i/g\rceil}$ ($g=3$) | not exactly one active block, or value $\ge c_i$ |

| Model | $c_i$ | Conformation qubits | Auxiliary qubits | Geometric constraints |
|---|---|---|---|---|
| Coordinate, square | $\lceil L_1L_2/2\rceil$, $\lfloor L_1L_2/2\rfloor$ alternating | Table II on $\mathbf C$ (unary about $N^2$, binary about $N\log N$) | none | one site per bead, self-avoidance, connectivity |
| Coordinate, cubic | same with $L_1L_2L_3$ | same | none | same |
| Turn, square | 4 | unary $4N$, binary about $2N$ | $M^\text{dist}_\text{aux}+M^\text{pair}_\text{aux}$ | no backturn, overlap (slack), contact flags |
| Turn, cubic | 6 | unary $6N$, binary $3N-8$ | same | same |
| Side-chain | rotamers, $c_i$ | Table II | none | one rotamer per residue |

## 7. Implementation checklist

1. Build $\mathbf C$ for the model, then a per-bead `encode(r, c)` and
   `decode(bits, c)` for each encoding, following the bit orders of Table I.
   Return `None` for invalid codes.
2. Lay out qubits bead by bead in a fixed order. For the turn model, place
   conformation qubits first, then slack registers in $(i,j)$ order (Eq. C14),
   then contact flags.
3. Write the classical decoder from bitstring to coordinates, plus validity
   checks (one-hot, range, self-avoidance, connectivity, no backturn). Use them
   to classify sampled bitstrings.
4. Build costs as polynomials in bits, substitute $x = (1-Z)/2$, and collect
   Pauli-$Z$ strings (for example as a Qiskit `SparsePauliOp`).
5. Use `PauliEvolutionGate` or explicit CNOT ladders for $e^{-i\gamma H}$, and
   confirm the CNOT count is $2(m-1)$ per $m$-body term.
6. Reverse Qiskit count keys before decoding.
