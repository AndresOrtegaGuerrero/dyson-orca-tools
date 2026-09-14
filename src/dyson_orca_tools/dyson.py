from collections import defaultdict
import numpy as np


class Dyson:
    """Dyson orbital for ONE (initial, final) pair of ORCA CASCI/CASSCF states.

    Determinants are ORCA strings like '[2u0d]'; internally an occupation vector
    interleaved as (α1, β1, α2, β2, ...) so a spin-orbital index p maps to
    orbital p // 2 and spin p % 2.
    """

    SPIN_MAP = {"2": (1, 1), "u": (1, 0), "d": (0, 1), "0": (0, 0)}

    def __init__(self, initial: dict, final: dict, parameters: dict):
        self.initial = initial
        self.final = final
        self.parameters = parameters

        self.MO_coeff_initial = self.get_mo_coeff_array(self.initial)
        self.MO_coeff_final = self.get_mo_coeff_array(self.final)
        self.s_matrix_ao = self.get_s_matrix_ao(self.initial)

        self.CI_initial = self.get_ci_coeff("initial")
        self.CI_final = self.get_ci_coeff("final")

        # <φ_i^initial | φ_j^final>
        self.s_matrix_mo = self.get_s_matrix_mo()
        self.mult_initial = self.initial["Molecule"]["Multiplicity"]
        self.mult_final = self.final["Molecule"]["Multiplicity"]

        self.num_inactive_orbs = sum(
            orb["Occupancy"] == 2.0
            for orb in self.initial["Molecule"]["MolecularOrbitals"]["MOs"]
        )
        self.num_active_orbs = self.parameters["parameters"]["initial"]["norb"]

        self.add_or_remove = (
            self.parameters["parameters"]["initial"]["nelc"]
            - self.parameters["parameters"]["final"]["nelc"]
        )
        self.operator = "annihilate" if self.add_or_remove > 0 else "create"

    # ------------------------------------------------------------------ inputs
    def get_s_matrix_ao(self, state: dict):
        return np.array(state["Molecule"]["S-Matrix"])

    def get_mo_coeff_array(self, state: dict):
        mo_orbs = state["Molecule"]["MolecularOrbitals"]["MOs"]
        return np.column_stack([orb["MOCoefficients"] for orb in mo_orbs])

    def get_s_matrix_mo(self):
        return self.MO_coeff_initial.T @ self.s_matrix_ao @ self.MO_coeff_final

    def get_ci_coeff(self, state: str) -> dict:
        spin_ci = self.parameters["parameters"][state]["spin_ci"]
        return {k.strip("[]"): v for k, v in spin_ci.items()}

    @property
    def active_slice(self) -> slice:
        return slice(
            self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs
        )

    @property
    def active_overlap(self) -> np.ndarray:
        """Active-active block, rows = initial MOs, cols = final MOs."""
        return self.s_matrix_mo[self.active_slice, self.active_slice]

    @property
    def core_overlap(self) -> np.ndarray:
        core = slice(0, self.num_inactive_orbs)
        return self.s_matrix_mo[core, core]

    @property
    def effective_active_overlap(self) -> np.ndarray:
        """Schur complement of the core block: det[[C, X], [Y, A]] = det C · det(A − Y C⁻¹ X),
        so determinants over active strings with this matrix, times det(C) per spin,
        equal the full core+active overlap determinants."""
        core, act = slice(0, self.num_inactive_orbs), self.active_slice
        C, X, Y, A = (
            self.s_matrix_mo[core, core],
            self.s_matrix_mo[core, act],
            self.s_matrix_mo[act, core],
            self.s_matrix_mo[act, act],
        )
        return A - Y @ np.linalg.solve(C, X)

    # ------------------------------------------------------------ determinants
    def ci_vector_to_array(self, vector: str) -> list:
        """'2u0d' -> [1,1, 1,0, 0,0, 0,1]"""
        return [bit for c in vector for bit in self.SPIN_MAP[c]]

    def ci_vector_to_string(self, vector: str) -> str:
        return "".join(str(b) for b in self.ci_vector_to_array(vector))

    def _target_mult(self, occ) -> int:
        return int(occ[0::2].sum() - occ[1::2].sum()) + 1

    @staticmethod
    def _reorder_sign(occ) -> int:
        """Parity of moving every β past the α's of higher orbitals (interleaved -> α-first)."""
        alpha, beta = occ[0::2], occ[1::2]
        alpha_after = np.cumsum(alpha[::-1])[::-1] - alpha  # α occupied at orbitals > q
        return -1 if int(np.dot(beta, alpha_after)) % 2 else 1

    def apply_operator(self, ci_string: str):
        """Yield (sign, p, new_occ) for a_p / a†_p acting on the determinant,
        keeping only results with the final multiplicity (M_S = S component)."""
        occ = np.array(self.ci_vector_to_array(ci_string), dtype=np.int8)
        want = 0 if self.operator == "create" else 1
        for p in np.flatnonzero(occ == want):
            new = occ.copy()
            new[p] = 1 - want
            if self._target_mult(new) != self.mult_final:
                continue
            sign = (
                -1 if occ[:p].sum() % 2 else 1
            )  # (-1)^(occupied spin-orbitals before p)
            yield sign, int(p), new

    # ------------------------------------------------------------ coefficients
    def _spin_orbital_to_ao(self, dyson_coeff: np.ndarray) -> np.ndarray:
        mo_coeff = (
            dyson_coeff[0::2] + dyson_coeff[1::2]
        )  # one channel is 0 by spin choice
        return self.MO_coeff_initial[:, self.active_slice] @ mo_coeff

    def casci_dyson_coefficients(self) -> np.ndarray:
        """Orthonormal MOs: <J| a_p |I> is ±1 only if J == a_p I, so hash the finals."""
        final = {self.ci_vector_to_string(k): c for k, c in self.CI_final.items()}
        dyson_coeff = np.zeros(2 * self.num_active_orbs)
        for sd_i, ci_i in self.CI_initial.items():
            for sign, p, new in self.apply_operator(sd_i):
                ci_f = final.get("".join(map(str, new)))
                if ci_f is not None:
                    dyson_coeff[p] += sign * ci_i * ci_f
        return dyson_coeff

    def dyson_coefficients(self) -> np.ndarray:
        """Non-orthogonal MOs. <J|I'> factorises into an α and a β determinant of
        MO overlaps once both determinants are reordered α-first (sign σ), so with
        A_p[a', b'] = Σ_I c_I sign σ(I') δ(a_p I = (a', b')):

            d_p = Σ_J c_J σ(J) (Sα^T A_p Sβ)[J_α, J_β],   Sα[a', J_α] = det S_mo[occ a', occ J_α]
        """
        norb = self.num_active_orbs

        # final determinants as (alpha string, beta string) indices
        a_index, b_index = {}, {}
        j_a, j_b, c_j = [], [], []
        for sd_f, ci_f in self.CI_final.items():
            occ = np.array(self.ci_vector_to_array(sd_f), dtype=np.int8)
            a, b = tuple(occ[0::2]), tuple(occ[1::2])
            j_a.append(a_index.setdefault(a, len(a_index)))
            j_b.append(b_index.setdefault(b, len(b_index)))
            c_j.append(ci_f * self._reorder_sign(occ))
        j_a, j_b, c_j = np.array(j_a), np.array(j_b), np.array(c_j)

        # operator applied to the initial determinants, grouped by spin orbital p
        ap_index, bp_index = {}, {}
        entries = defaultdict(list)  # p -> [(a', b', coeff)]
        for sd_i, ci_i in self.CI_initial.items():
            for sign, p, new in self.apply_operator(sd_i):
                a, b = tuple(new[0::2]), tuple(new[1::2])
                entries[p].append(
                    (
                        ap_index.setdefault(a, len(ap_index)),
                        bp_index.setdefault(b, len(bp_index)),
                        sign * ci_i * self._reorder_sign(new),
                    )
                )

        S_eff = self.effective_active_overlap
        S_alpha = self._string_overlaps(ap_index, a_index, S_eff)
        S_beta = self._string_overlaps(bp_index, b_index, S_eff)
        core_factor = np.linalg.det(self.core_overlap) ** 2  # α core × β core

        dyson_coeff = np.zeros(2 * norb)
        for p, rows in entries.items():
            A = np.zeros((len(ap_index), len(bp_index)))
            for ia, ib, c in rows:
                A[ia, ib] += c
            M = S_alpha.T @ A @ S_beta  # [J_alpha, J_beta]
            dyson_coeff[p] = core_factor * np.dot(c_j, M[j_a, j_b])
        return dyson_coeff

    def _string_overlaps(
        self, initial_strings: dict, final_strings: dict, S: np.ndarray
    ) -> np.ndarray:
        """det S[occ(initial string), occ(final string)] for every pair; batched."""
        occ_i = [np.flatnonzero(s) for s in initial_strings]
        occ_f = [np.flatnonzero(s) for s in final_strings]
        out = np.zeros((len(occ_i), len(occ_f)))
        for i, oi in enumerate(occ_i):
            blocks = np.stack([S[np.ix_(oi, of)] for of in occ_f])  # (n_f, k, k)
            out[i] = np.linalg.det(blocks) if blocks.shape[1] else 1.0
        return out

    # ------------------------------------------------------------------ public
    def calculation_is_casci(self, atol: float = 1e-8) -> bool:
        sub = self.active_overlap
        return np.allclose(sub, np.eye(sub.shape[0]), atol=atol)

    def dyson_mo_coefficients(self) -> np.ndarray:
        """Dyson orbital expanded in the active MOs of the initial state, ϱ = Σ_p d_p φ_p.
        The MOs are orthonormal, so Σ d_p² = <ϱ|ϱ> and d_p² is the weight of orbital p."""
        coeffs = (
            self.casci_dyson_coefficients()
            if self.calculation_is_casci()
            else self.dyson_coefficients()
        )
        return coeffs[0::2] + coeffs[1::2]  # one spin channel is 0 by the M_S choice

    def dyson_orbital(self, mo_coeff: np.ndarray | None = None) -> np.ndarray:
        """Dyson orbital as AO coefficients in the basis of the initial state."""
        if mo_coeff is None:
            mo_coeff = self.dyson_mo_coefficients()
        return self.MO_coeff_initial[:, self.active_slice] @ mo_coeff

    def active_labels(self) -> list[str]:
        """Active orbitals named relative to the initial state's HOMO (Aufbau filling)."""
        n_occ = -(-self.parameters["parameters"]["initial"]["nelc"] // 2)  # ceil
        labels = []
        for i in range(self.num_active_orbs):
            if i < n_occ:
                k = n_occ - 1 - i
                labels.append("HOMO" if k == 0 else f"HOMO-{k}")
            else:
                k = i - n_occ
                labels.append("LUMO" if k == 0 else f"LUMO+{k}")
        return labels

    def strength(self, dyson_ao: np.ndarray) -> float:
        """Pole strength <ϱ|ϱ> = d·S·d."""
        return float(dyson_ao @ self.s_matrix_ao @ dyson_ao)
