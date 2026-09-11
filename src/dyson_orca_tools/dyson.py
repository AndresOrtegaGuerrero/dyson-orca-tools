from typing import List, Tuple, Dict
from itertools import product
import numpy as np


class Dyson:
    """Dyson orbital for ONE (initial, final) pair of ORCA CASCI/CASSCF states."""

    def __init__(self, initial: dict, final: dict, parameters: dict):
        self.initial = initial
        self.final = final
        self.parameters = parameters

        self.MO_coeff_initial = self.get_mo_coeff_array(self.initial)
        self.MO_coeff_final = self.get_mo_coeff_array(self.final)
        self.s_matrix_ao = self.get_s_matrix_ao(self.initial)

        self.CI_initial = self.get_ci_coeff("initial")
        self.CI_final = self.get_ci_coeff("final")

        # MO-MO overlap
        self.s_matrix_mo = self.get_s_matrix_mo()
        self.mult_initial = self.initial["Molecule"]["Multiplicity"]
        self.mult_final = self.final["Molecule"]["Multiplicity"]

        # Info system
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
        """Active-active block of the initial/final MO overlap."""
        return self.s_matrix_mo[self.active_slice, self.active_slice]

    @property
    def core_overlap(self) -> np.ndarray:
        """Inactive-inactive block; its determinant should be ~1 for CASSCF pairs."""
        core = slice(0, self.num_inactive_orbs)
        return self.s_matrix_mo[core, core]

    # ------------------------------------------------------------ determinants
    def ci_vector_to_array(self, vector: str) -> list:
        """'[2u0d]' -> [1,1, 1,0, 0,0, 0,1] (alpha, beta interleaved)."""
        spin_map = {"2": [1, 1], "u": [1, 0], "d": [0, 1], "0": [0, 0]}
        return [bit for c in vector for bit in spin_map[c]]

    def ci_vector_to_string(self, vector: str) -> str:
        return "".join(str(b) for b in self.ci_vector_to_array(vector))

    def casci_occupation_diff(self, sd_f, sd_i):
        """Spin-orbital index and sign if the two determinants differ by one electron."""
        occ_f = np.array(self.ci_vector_to_array(sd_f), dtype=int)
        occ_i = np.array(self.ci_vector_to_array(sd_i), dtype=int)
        diff = occ_f - occ_i
        if np.count_nonzero(diff) == 1 and abs(np.sum(diff)) == 1:
            idx = np.flatnonzero(diff)[0]
            sign = (-1) ** np.sum(occ_i[:idx])
            return idx, sign
        return None, None

    def _extract_alpha_beta(self, sd_list):
        alpha_set, beta_set = set(), set()
        for sd in sd_list:
            sd = tuple(sd)
            alpha_set.add(sd[::2])
            beta_set.add(sd[1::2])
        return {"alpha": list(alpha_set), "beta": list(beta_set)}

    def alpha_beta_map(self, ci_list):
        return self._extract_alpha_beta(ci_list)

    def alpha_beta_operator_map(self, operator_initial):
        sd_list = [
            np.array(list(vector[2]), dtype=int)
            for value in operator_initial.values()
            for vectors in value.values()
            for vector in vectors
        ]
        return self._extract_alpha_beta(sd_list)

    def generate_overlaps_dict(self, psi_final, operator_psi_initial):
        sub_s_mo = self.active_overlap
        overlaps_set = {
            tuple(i + j)
            for spin in ["alpha", "beta"]
            for i, j in product(psi_final[spin], operator_psi_initial[spin])
        }
        overlaps_dictionary = {}
        for overlap in overlaps_set:
            final = np.array(overlap[: self.num_active_orbs])
            initial = np.array(overlap[self.num_active_orbs :])
            occupied_final = np.flatnonzero(final == 1)
            occupied_initial = np.flatnonzero(initial == 1)
            det = np.linalg.det(sub_s_mo[np.ix_(occupied_final, occupied_initial)])
            overlaps_dictionary["".join(str(b) for b in overlap)] = det
        return overlaps_dictionary

    def generate_sds_initial(
        self, ci_string: str, mult_final: int, mode: str = "create"
    ) -> Dict[str, List[Tuple[int, int, str]]]:
        """All determinants reachable from ci_string by one a/a† that match mult_final."""
        if mode not in ("create", "annihilate"):
            raise ValueError("mode must be 'create' or 'annihilate'")

        occ0 = np.array(self.ci_vector_to_array(ci_string), dtype=int)
        results: List[Tuple[int, int, str]] = []
        str_occ0 = "".join(str(x) for x in occ0)

        for i, val in enumerate(occ0):
            if mode == "create" and val == 0:
                occ1 = occ0.copy()
                occ1[i] = 1
            elif mode == "annihilate" and val == 1:
                occ1 = occ0.copy()
                occ1[i] = 0
            else:
                continue

            new_mult = (occ1[0::2].sum() - occ1[1::2].sum()) + 1
            if new_mult != mult_final:
                continue

            sign = (-1) ** int(occ0[:i].sum())
            results.append((sign, i, "".join(str(x) for x in occ1)))

        return {str_occ0: results}

    def generate_sds_dict(
        self, ci_strings: Dict[str, float], mult_final: int, mode: str = "create"
    ) -> Dict[str, Dict[str, List[Tuple[int, int, str]]]]:
        return {
            ci_str: self.generate_sds_initial(ci_str, mult_final, mode)
            for ci_str in ci_strings
        }

    def _spin_orbital_to_ao(self, dyson_coeff: np.ndarray) -> np.ndarray:
        """Contract spin-orbital coefficients (alpha+beta) with the initial MOs."""
        mo_coeff = (
            dyson_coeff[0::2] + dyson_coeff[1::2]
        )  # one channel is 0 by spin choice
        return self.MO_coeff_initial[:, self.active_slice] @ mo_coeff

    def casci_dyson_coefficients(self) -> np.ndarray:
        """Orthonormal-MO branch: <Ψ_f| a |Ψ_i> reduces to CI products."""
        dyson_coeff = np.zeros(2 * self.num_active_orbs)
        for sd_i, ci_i in self.CI_initial.items():
            for sd_f, ci_f in self.CI_final.items():
                idx, sign = self.casci_occupation_diff(sd_f, sd_i)
                if idx is not None:
                    dyson_coeff[idx] += sign * ci_i * ci_f
        return dyson_coeff

    def dyson_coefficients(self) -> np.ndarray:
        """Non-orthogonal branch (different MO sets): weight by determinant overlaps."""
        dyson_coeff = np.zeros(2 * self.num_active_orbs)

        sds_dict = self.generate_sds_dict(
            self.CI_initial, self.mult_final, mode=self.operator
        )
        operator_psi_initial = self.alpha_beta_operator_map(sds_dict)
        ci_final_list = [self.ci_vector_to_array(key) for key in self.CI_final]
        psi_final = self.alpha_beta_map(ci_final_list)
        overlaps_dict = self.generate_overlaps_dict(psi_final, operator_psi_initial)

        for sd_i, ci_i in self.CI_initial.items():
            string_sd_i = self.ci_vector_to_string(sd_i)
            for sd_f, ci_f in self.CI_final.items():
                string_sd_f = self.ci_vector_to_string(sd_f)
                for sign, idx, new_sd_i in sds_dict[sd_i][string_sd_i]:
                    overlap_alpha = overlaps_dict[string_sd_f[::2] + new_sd_i[::2]]
                    overlap_beta = overlaps_dict[string_sd_f[1::2] + new_sd_i[1::2]]
                    dyson_coeff[idx] += (
                        sign * ci_i * ci_f * overlap_alpha * overlap_beta
                    )

        return dyson_coeff

    def calculation_is_casci(self, atol: float = 1e-8) -> bool:
        """True when initial and final share the active orbitals (MO overlap = 1)."""
        sub = self.active_overlap
        return np.allclose(sub, np.eye(sub.shape[0]), atol=atol)

    def dyson_orbital(self) -> np.ndarray:
        """Dyson orbital as AO coefficients in the basis of the initial state."""
        coeffs = (
            self.casci_dyson_coefficients()
            if self.calculation_is_casci()
            else self.dyson_coefficients()
        )
        return self._spin_orbital_to_ao(coeffs)

    def strength(self, dyson_ao: np.ndarray) -> float:
        """Pole strength <ϱ|ϱ> = d·S·d (valid in a non-orthogonal AO basis)."""
        return float(dyson_ao @ self.s_matrix_ao @ dyson_ao)
