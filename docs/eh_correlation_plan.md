# Electron–hole correlation maps from ORCA SA-CASSCF / NEVPT2 — theory and plan

Goal: for a state-averaged CASSCF (+ NEVPT2 / QD-NEVPT2) run, explain *where the emission
of root I comes from* when its oscillator strength is small, and quantify how much of its
intensity is borrowed from another root.

## 0. Where it should live

**Engine → `dyson-orca-tools`. Figures/report → `orca_analysis` (thin optional hook).**

Why: the transition density matrix (TDM) is the Dyson orbital's sibling. The Dyson orbital is
`⟨Ψ_N-1| a_p |Ψ_N⟩` (one operator); the TDM is `⟨Ψ_I| a†_p a_q |Ψ_J⟩` (two operators). Everything
the TDM needs already exists in `dyson-orca-tools`:

| need | already there |
|---|---|
| determinant CI vectors with signs (`[2u0d]  -0.052…`) | `io/orca_output.parse_orca_output` |
| MO coefficients + AO overlap `S` | `Dyson.get_mo_coeff_array`, `get_s_matrix_ao` (ORCA JSON) |
| apply `a_p`/`a†_p` with the fermionic sign, hash lookup of the result | `Dyson.apply_operator`, `casci_dyson_coefficients` |
| AO → atom map (for fragments) | `Molecule.Atoms[i].Basis` shells in the JSON, or the `OrbitalLabels` |
| optional PySCF for integrals/cubes | the `[cube]` extra |

`orca_analysis` reads only the text output and *deliberately skips* the determinant list
(`csf.py`), so it has none of the machinery. It should stay the "report generator" and, if
`dyson_orca_tools` is importable, add an `ehmap` panel to the results folder.

One-line analogy: the Dyson orbital answers "which orbital did the electron leave *the molecule*
from?"; the TDM answers "which orbital did the electron leave, and which did it land in, *inside*
the molecule?".

## 1. The physics question, and which number answers it

A small oscillator strength `f = (2/3) ΔE |μ_0I|²` can have three different origins. Each one has
its own diagnostic, and they are all derived from the same object, the 1-TDM.

| origin of small f | what you see | diagnostic |
|---|---|---|
| not a one-electron transition (double excitation, "2Ag-like") | the TDM is *small as a whole* | `Ω = ‖γ‖²` ≪ 1 |
| one-electron, but hole and electron sit on different fragments (CT) | TDM is large but *off-diagonal* in fragment space | `Ω_AB` map, CT number |
| one-electron, local, but the dipole cancels (pseudo-symmetry forbidden) | TDM large, diagonal, yet `μ` ≈ 0 | atom-resolved transition dipole `μ_A` (contributions of opposite sign) |

And intensity borrowing has two flavours:

* **electronic mixing** (what QD-NEVPT2 does): the perturbed state is `|Φ_K⟩ = Σ_J U_JK |Ψ_J⟩`, so
  `μ_K = Σ_J U_JK μ_0J`. If the dark CAS root gets a 10 % admixture of a bright one, it inherits
  ~1 % of its `|μ|²`. This is fully computable from what ORCA already prints.
* **vibronic (Herzberg–Teller)**: `μ(Q) = μ⁰ + Σ_n ⟨n|∂H/∂Q|k⟩/(E_k−E_n) · μ_n⁰ · Q`. Same structure
  (mixing coefficient × bright dipole), but the coefficient comes from a nuclear-coordinate
  derivative. Out of scope for a post-processing tool; ORCA's ESD module (`%esd DoHT true`) is the
  route if you need the actual vibronic number.

Analogy: a quiet singer next to a loud one. The audience hears the quiet one only to the extent
the two voices are mixed — by the arrangement (electronic mixing, `U_JK`) or by the room shaking
(vibronic coupling). The e–h map tells you *why* the quiet singer is quiet; the mixing analysis
tells you *whose voice* you actually hear.

For **emission**, do the analysis at the emitting geometry (CASSCF `!Opt` on the emitting `IRoot`),
not at the Franck–Condon one; at the ground-state geometry the map describes absorption.

## 2. Theory

### 2.1 The one-particle transition density matrix

For two CI states sharing one orbital set (SA-CASSCF), spin-summed:

```
γ^{IJ}_{tu} = ⟨Ψ_I | E_tu | Ψ_J⟩ ,   E_tu = Σ_σ a†_{tσ} a_{uσ}
```

`t, u` run over **active** orbitals only. Inactive orbitals are spectators: they are doubly
occupied in every determinant, so `⟨Ψ_I|E_ij|Ψ_J⟩ = 2 δ_ij ⟨Ψ_I|Ψ_J⟩ = 0` for `I ≠ J`. That is why the
TDM is a small `norb × norb` block and the whole thing is cheap.

Read `γ_{tu}` as a *transfer ledger*: "going from J to I, how much amplitude moves an electron
from orbital `u` (hole) to orbital `t` (particle)". Row = where it lands, column = where it left.

* `I = J` gives the 1-RDM; its eigenvalues are ORCA's natural-orbital occupations → free unit test.
* `γ^{IJ} = (γ^{JI})ᵀ` for real wavefunctions → second free test.
* Different spin multiplicity → `γ = 0` (spin-forbidden). Only pair roots within one MULT block.
  For SA over one block with the same `M_S`, `E_tu` conserves `M_S`, nothing else to worry about.

### 2.2 Evaluating it on determinants (Slater–Condon, the hash way)

In an orthonormal orbital basis `⟨D'| a†_p a_q |D⟩` is `±1` **only if** `D' = a†_p a_q D`, else 0.
So instead of an `N_det²` double loop, for every determinant `D` of `Ψ_J` generate all
`a†_p a_q D` (`≤ 4·norb²` of them), look each one up in a dict of `Ψ_I`'s determinants, and
accumulate `c_I(D') · sign · c_J(D)`. Cost `O(N_det · norb²)` — this is exactly the trick
`casci_dyson_coefficients` already uses, applied twice:

1. `a_q`: `sign_q = (−1)^{#occupied spin-orbitals before q}`, clear bit `q`.
2. `a†_p` on the *new* occupation: `sign_p = (−1)^{#occupied before p}`, set bit `p`.
3. `γ[p//2, q//2] += sign_p · sign_q · c_I(D') · c_J(D)`, restricting `p % 2 == q % 2` (spin-summed).

Watch the truncation: ORCA prints only determinants above `TPrintWF`. Report `Σ c²` of what was
parsed for each root and lower `TPrintWF` (1e-5 or smaller) if it is not ≈ 1. A truncated
vector under-estimates `Ω`, and that is exactly the quantity you want to trust.
Sizes: CAS(8,8) singlet ≈ 5k dets, CAS(12,12) at `TPrintWF 1e-5` typically a few 10⁴ — seconds in
pure Python with dict lookups; no need for anything cleverer.

### 2.3 From MO ledger to real space

`D^{IJ} = C_act γ^{IJ} C_actᵀ` (AO basis, `nbas × nbas`, non-symmetric in general).

**Fragment e–h map** (Plasser & Lischka, JCTC 2012; Plasser, Bäppler, Wormit, Dreuw, JCP 2014):

```
Ω_AB = ½ Σ_{μ∈A} Σ_{ν∈B} [ (D S)_{μν} (S D)_{μν} + D_{μν} (S D S)_{μν} ]
```

`Ω_AB` = "hole on fragment A, electron on fragment B". Diagonal = local excitation, off-diagonal
= charge transfer. `Σ_AB Ω_AB = Ω = tr(Dᵀ S D S) = ‖γ‖²_F` — the single-excitation character.
Derived numbers: `CT = (1/Ω) Σ_{A≠B} Ω_AB`; participation ratio `PR = Ω² / Σ_AB Ω_AB²`.
Fragments default to atoms; a JSON/YAML of atom-index groups gives chemical fragments (donor,
acceptor, bridge…). This origin–destination matrix is the classic "electron–hole correlation
plot" (Tretiak & Mukamel), just on fragments instead of atoms.

**NTOs**: SVD `γ = U Σ Vᵀ` (in the orthonormal active MOs — no `S` needed).
Particle NTOs `C_act U`, hole NTOs `C_act V`, weights `σ_k²`, `Σ σ_k² = Ω`, `PR_NTO = Ω²/Σσ_k⁴`.
One dominant `σ` → a clean hole→particle pair; many → collective/multi-configurational.
(Check whether your ORCA 6.1 offers CASSCF NTOs in `%casscf`; if it does, it is a cross-check,
not a replacement — you still need `γ` for `Ω_AB` and the dipole decomposition.)

**Atom-resolved transition dipole** (needs AO dipole integrals `d_{μν} = ⟨μ|r|ν⟩`, PySCF extra):

```
μ_IJ = −Σ_{μν} D_{μν} d_{νμ}          μ_A = −½ Σ_{μ∈A} [ (D d)_{μμ} + (d D)_{μμ} ]
```

`Σ_A μ_A = μ_IJ`, and `|μ_0I|` must match ORCA's `ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC
DIPOLE MOMENTS` table (TX, TY, TZ in a.u.) — the third free unit test. Opposite-sign `μ_A` that
cancel is the "pseudo-symmetry forbidden" signature.

### 2.4 Intensity borrowing by state mixing (QD-NEVPT2)

ORCA prints the eigenvectors `U` of the QD-NEVPT2 effective Hamiltonian in the CASSCF-root basis.
Then, exactly (transition density is bilinear):

```
γ^{0K}_QD = Σ_{I,J} U_I0 U_JK γ^{IJ}     ≈ Σ_J U_JK γ^{0J}   when U_00 ≈ 1
μ_K       = Σ_J U_JK μ_0J
```

Report, per perturbed root K: the CAS roots it is made of (`U_JK²`), each root's vector
contribution `U_JK μ_0J`, and the share of `|μ_K|²` that comes from cross terms. If the emitting
state's `|μ_K|²` is dominated by a term with `J ≠ K`, the emission is borrowed — and the e–h map
of *that* `γ^{0J}` is the picture of where the light comes from. Plain NEVPT2 (no QD) does not
mix roots; ORCA's NEVPT2 spectrum reuses the CASSCF `μ` with corrected energies, so the comparison
"CASSCF f vs QD-NEVPT2 f for the same root" is itself the first borrowing indicator.

## 3. Implementation plan (dyson-orca-tools, branch `eh_correlation`)

Roles, to avoid confusion: `orca_analysis` = our report package (parses `out.out`, renders cubes
through ORCA's `orca_plot` executable). `dyson-orca-tools` = our JSON/determinant package. ORCA's
NTO/NDO `*_nto-donor.gbw` / `*_nto-acceptor.gbw` / `*_ndo-*.gbw` files are only meant for
`orca_plot`; the numbers we need (CI vectors, NTO singular values λ_k) are in `out.out`, and the
MOs + S come from `orca_2json mol.gbw` as in the Dyson workflow.

### 3.0 Git

```bash
git checkout -b eh_correlation spectral_list_roots     # keeps all spectral_list_roots commits
git add docs/eh_correlation_plan.md && git commit -m "docs: e-h correlation plan"
# later, if spectral_list_roots moves:  git rebase spectral_list_roots
```

One step = one commit with its own test; each step is designed and approved before any file is
touched. `Dyson.apply_operator` is **not** refactored; the new class carries its own tiny
`_excite` helper and a test pins the two sign conventions together.

### 3.1 Step 1 — inputs and reference numbers (no code)

On the CAS(14,14) triplet folder (`cas1414_new_t6`):

* `orca_2json mol.gbw` → `mol.json` (MOs, S, atoms/basis).
* Confirm `out.out` has `Spin-Determinant CI Printing` for every root, note the `TPrintWF` used.
* Locate the NTO block per state (λ_k, Σλ_k²) and the `ABSORPTION SPECTRUM` table (TX/TY/TZ, f).
* Record, for the emitting root: Σλ_k² (this is Ω), the largest λ_k², f_CASSCF and f_QD-NEVPT2.

Check: these four numbers already give a first reading (Ω ≪ 1 → multi-excited; one λ dominates →
clean pair; f_QD ≫ f_CAS → borrowing by mixing). Everything below reproduces and explains them.

### 3.2 Step 2 — `tdm.py` + `parse_nto_block` in `io/orca_output.py`

```python
class TransitionDensity:
    """γ[t,u] = <I| E_tu |J> over active MOs; I == J gives the 1-RDM."""
    def __init__(self, ci_i, ci_j, norb): ...
    def gamma(self) -> np.ndarray: ...          # hash-lookup Slater–Condon, §2.2
    def to_ao(self, c_active) -> np.ndarray: ...  # C_act γ C_actᵀ
    @property
    def parsed_norms(self) -> tuple[float, float]  # Σc² of each root (truncation check)
```

`_excite(occ, p, q) -> (sign, new_occ)`: apply `a_q` then `a†_p`, sign `(-1)^(occupied before)` each.
`parse_nto_block(text) -> {root: [λ_k]}` in the style of `csf.py`'s regexes (layout taken from the
real `out.out`).

Checks (tests on pentacene + the cas1414 output):
1. `_excite` sign for the `a_q` half equals `Dyson.apply_operator`'s on a handful of determinants.
2. `γ^{II}` eigenvalues == ORCA natural occupations of root I.
3. `γ^{IJ} == (γ^{JI})ᵀ`.
4. `svd(γ^{0I}).S` == ORCA's λ_k within the truncation error; report `Σλ² (ORCA)` vs `‖γ‖² (ours)`.

### 3.3 Step 3 — `ehmap.py`

* `ao_to_atom(json_state)`: atom index per AO from `Atoms[i].Basis` (2l+1 per shell); assert
  total == `len(S)`.
* `fragments(atom_map, groups=None)`: default one fragment per atom; optional JSON of atom-index
  groups.
* `omega_matrix(D_ao, S, frag)`, `metrics(Ω_AB) -> {Omega, CT, PR}`, `ntos(gamma, c_active)`.
* `plot_ehmap(...)`: heatmap, hole on x, electron on y, common colour scale across roots.

Checks: `Σ_AB Ω_AB == ‖γ‖²_F`; NTO weights == λ_k²; pentacene S0→S1 map concentrated on the
central rings.

### 3.4 Step 4 — CLI `dyson ehmap`

```
dyson ehmap out.out mol.json --root 0 --root 3 [--fragments frags.json] [--dipole] [--qd]
```
→ `ehmap/`: `gamma_r0_r3.npy`, `omega_r0_r3.csv`, `ehmap_r0_r3.png`, `summary.json`.
Check: runs end-to-end on pentacene and on cas1414; `summary.json` reproduces the step-1 numbers.

### 3.5 Step 5 — `dipole.py` (no PySCF: AO dipole integrals from `orca_2json`)

`orca_2json` exports the AO dipole integrals when `mol.json.conf` contains
`"1elPropertyIntegrals": ["dipole"]` (alongside `"1elIntegrals": ["S"]`), so the atom-resolved
transition dipole needs nothing outside the JSON: `transition_dipole(D, d)`,
`atom_contributions(D, d, atom_map)`.
Check: `|μ_0I|` vs ORCA's DX/DY/DZ (1e-3 a.u.); `Σ_A μ_A == μ`.
(`"Densities": ["all"]` would also export the QD-NEVPT2-corrected densities from `mol.densities`,
which is the route to NDOs of the *perturbed* states if ever needed.)

### 3.6 Step 6 — `mixing.py` (QD-NEVPT2 borrowing)

Parse the QD-NEVPT2 eigenvector block → `U`; `μ_K = Σ_J U_JK μ_0J`; table per perturbed root
with `U_JK²`, the vector contributions and the cross-term share of `|μ_K|²`.
Check: `f_K` rebuilt from `μ_K` and the QD energies == ORCA's QD-NEVPT2 f.

### 3.7 Step 7 — `orca_analysis` hook

* Cube panels of `*_nto-donor/acceptor.gbw` and `*_ndo-donor/acceptor.gbw` through the existing
  `orca_plot` → render chain (first one or two orbitals of each file).
* If `dyson_orca_tools` is importable and `mol.json` exists, run `ehmap` for every excited root
  against root 0 and add the maps next to the CSF lists. No hard dependency.

### 3.8 Inputs to request from ORCA

```
%casscf  ...  PrintWF det   TPrintWF 1e-6   DoNTO true   DoNDO true  end
```
plus `orca_2json mol.gbw` with `mol.json.conf` = `{"MOCoefficients": true, "Basisset": true, "1elIntegrals": ["S"], "1elPropertyIntegrals": ["dipole"]}`. For the emitting state, repeat at the CASSCF-optimised geometry of that
root.

## References

* F. Plasser, H. Lischka, *JCTC* **8**, 2777 (2012) — Ω_AB, CT numbers.
* F. Plasser, M. Wormit, A. Dreuw, *JCP* **141**, 024106 (2014) — TDM analysis, NTOs, e–h correlation.
* S. Tretiak, S. Mukamel, *Chem. Rev.* **102**, 3171 (2002) — electron–hole correlation plots.
* G. Herzberg, E. Teller, *Z. Phys. Chem. B* **21**, 410 (1933) — intensity borrowing.
