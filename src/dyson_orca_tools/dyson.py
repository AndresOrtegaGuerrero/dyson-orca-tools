from pyscf import gto
from pyscf.tools import cubegen
from .utils import generate_basis_dict, parse_orca_labels_pyscf
from typing import List, Tuple, Dict
from itertools import product
import numpy as np


class Dyson:
    def __init__(self, initial: dict, final: dict, parameters: dict, output_dir: str):
        self.initial = initial
        self.final = final
        self.parameters = parameters
        self.output_dir = output_dir
        self.pyscf_molecule = self.create_pyscf_molecule()
        self.pyscf_ao_labels = self.get_pyscf_ao_labels()
        self.orca_ao_labels = self.get_orca_ao_labels()

        #
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

    def get_s_matrix_ao(self, state: dict):
        """Get the overlap matrix in the AO basis."""
        return np.array(state["Molecule"]["S-Matrix"])

    def get_mo_coeff_array(self, state: dict):
        mo_orbs = state["Molecule"]["MolecularOrbitals"]["MOs"]
        mo_coeff_list = [orb["MOCoefficients"] for orb in mo_orbs]
        return np.column_stack(mo_coeff_list)

    def get_s_matrix_mo(self):
        """Get the overlap matrix in the MO basis."""
        return self.MO_coeff_initial.T @ self.s_matrix_ao @ self.MO_coeff_final

    def get_ci_coeff(self, state: str) -> dict:
        """Get the CI coefficients for the given state."""
        spin_ci = self.parameters["parameters"][state]["spin_ci"]
        return {k.strip("[]"): v for k, v in spin_ci.items()}  # Remove the brackets

    def ci_vector_to_array(self, vector: str) -> list:
        """Convert CI dict with keys like '[2200]', '[udud]' into lists of 0/1 for alpha/beta occupation."""
        spin_map = {"2": [1, 1], "u": [1, 0], "d": [0, 1], "0": [0, 0]}

        return [bit for c in vector for bit in spin_map[c]]

    def ci_vector_to_string(self, vector: str) -> list:
        """Convert CI dict with keys like '[2200]', '[udud]' into lists of 0/1 for alpha/beta occupation."""
        spin_map = {"2": [1, 1], "u": [1, 0], "d": [0, 1], "0": [0, 0]}
        return "".join(str(bit) for c in vector for bit in spin_map[c])

    def casci_occupation_diff(self, sd_f, sd_i):
        """Compute the difference between two Slater determinants convert to 0 1 occupation."""
        occ_f = np.array(list(self.ci_vector_to_array(sd_f))).astype(int)
        occ_i = np.array(list(self.ci_vector_to_array(sd_i))).astype(int)
        diff = occ_f - occ_i
        if np.count_nonzero(diff) == 1:
            if np.sum(diff) == 1:
                # Electron added in final state
                idx = np.where(diff == 1)[0][0]
                sign = (-1) ** np.sum(occ_i[:idx])
                return idx, sign
            elif np.sum(diff) == -1:
                # Electron removed in final state
                idx = np.where(diff == -1)[0][0]
                sign = (-1) ** np.sum(occ_i[:idx])
                return idx, sign

        return None, None

    def _extract_alpha_beta(self, sd_list):
        alpha_set = set()
        beta_set = set()

        for sd in sd_list:
            sd = tuple(sd)
            alpha_set.add(sd[::2])
            beta_set.add(sd[1::2])

        return {"alpha": list(alpha_set), "beta": list(beta_set)}

    def alpha_beta_map(self, ci_list):
        """Create a map of alpha and beta Slater determinants for Psi_final"""
        return self._extract_alpha_beta(ci_list)

    def alpha_beta_operator_map(self, operator_initial):
        """Create a map of alpha and beta Slater determinants from operator-initial structure."""
        sd_list = [
            np.array(list(vector[2]), dtype=int)
            for value in operator_initial.values()
            for vectors in value.values()
            for vector in vectors
        ]
        return self._extract_alpha_beta(sd_list)

    def generate_overlaps_dict(self, psi_final, operator_psi_initial):
        sub_s_mo = self.s_matrix_mo[
            self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs,
            self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs,
        ]

        overlaps_set = {
            tuple(i + j)
            for spin in ["alpha", "beta"]
            for i, j in product(psi_final[spin], operator_psi_initial[spin])
        }
        overlaps_list = list(overlaps_set)
        # Determinants
        overlaps_dictionary = {}
        for overlap in overlaps_list:
            final = np.array(overlap[: self.num_active_orbs])
            initial = np.array(overlap[self.num_active_orbs :])
            occupied_final = np.where(final == 1)[0]
            occupied_initial = np.where(initial == 1)[0]
            Slater = sub_s_mo[np.ix_(occupied_final, occupied_initial)]
            det = np.linalg.det(Slater)
            string = "".join(str(b) for b in overlap)
            overlaps_dictionary[string] = det
        return overlaps_dictionary

    def dyson_coefficients(self):
        dyson_coeff = np.zeros(2 * self.parameters["parameters"]["initial"]["norb"])  # noqa: F841

        # Step 1: Generate the Slater determinants for create or annihilate in Slater determinants of Psi_Initial

        sds_dict = self.generate_sds_dict(
            self.CI_initial, self.mult_final, mode=self.operator
        )
        # Step 2: Create dictionary of unique Slater determinants coming from a|a_dager | Psi_initial >
        # in alpha/beta space
        operator_psi_initial = self.alpha_beta_operator_map(sds_dict)
        # Step 3: Create a dictionary of unique Slater determinants of Psi_final in alpha/beta space
        ci_final_list = [self.ci_vector_to_array(key) for key in self.CI_final.keys()]
        psi_final = self.alpha_beta_map(ci_final_list)

        # Step 4: Generate the overlaps dictionary
        overlaps_dict = self.generate_overlaps_dict(psi_final, operator_psi_initial)  # noqa: F841

        # print("Overlaps:", overlaps_dict)
        # Step 5: Compute dyson orbital
        for sd_i, ci_i in self.CI_initial.items():
            for sd_f, ci_f in self.CI_final.items():
                coeff = ci_i * ci_f
                string_sd_f = self.ci_vector_to_string(sd_f.strip("[]"))
                string_sd_i = self.ci_vector_to_string(sd_i.strip("[]"))
                operator = sds_dict[sd_i][string_sd_i]
                for op_sd_i in operator:
                    sign = op_sd_i[0]
                    idx = op_sd_i[1]
                    new_sd_i = op_sd_i[2]
                    overlap_alpha = overlaps_dict.get(string_sd_f[::2] + new_sd_i[::2])
                    overlap_beta = overlaps_dict.get(string_sd_f[1::2] + new_sd_i[1::2])
                    dyson_coeff[idx] += sign * coeff * overlap_alpha * overlap_beta

        # print("Dyson coefficients:", dyson_coeff)

        dyson_ao = np.zeros(self.s_matrix_ao.shape[0])
        for i in range(self.num_active_orbs):
            ao_coeff = dyson_coeff[2 * i] + dyson_coeff[2 * i + 1]  # alpha + beta
            # note: one of the spin channels gives 0 due to choice of multiplicity
            mo_index = self.num_inactive_orbs + i
            dyson_ao += ao_coeff * self.MO_coeff_initial[:, mo_index]

        return dyson_ao

    def casci_dyson_coefficients(self):
        dyson_coeff = np.zeros(2 * self.parameters["parameters"]["initial"]["norb"])
        for sd_i, ci_i in self.CI_initial.items():
            for sd_f, ci_f in self.CI_final.items():
                idx, sign = self.casci_occupation_diff(sd_f, sd_i)
                if idx is not None:
                    dyson_coeff[idx] += sign * ci_i * ci_f

        print("Dyson coefficients (CASCI):", dyson_coeff)
        print("Square sum of coefficients:", np.sum(dyson_coeff**2))
        dyson_ao = np.zeros(self.s_matrix_ao.shape[0])
        for i in range(self.num_active_orbs):
            coeff = dyson_coeff[2 * i] + dyson_coeff[2 * i + 1]  # alpha + beta
            mo_index = self.num_inactive_orbs + i

            dyson_ao += coeff * self.MO_coeff_initial[:, mo_index]

        return dyson_ao

    def generate_sds_initial(
        self, ci_string: str, mult_final: int, mode: str = "create"
    ) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        For a given initial CI string, generate all determinants
        by adding or removing one electron in the active space,

        Returns a dict: { ci_string: [ (sign, spin_idx, new_occ_str), … ] }.
        """
        if mode not in ("create", "annihilate"):
            raise ValueError("mode must be 'create' or 'annihilate'")

        occ0 = np.array(self.ci_vector_to_array(ci_string), dtype=int)
        n_spin = len(occ0)
        results: List[Tuple[int, int, str]] = []

        str_occ0 = "".join(str(x) for x in occ0)

        for i in range(n_spin):
            val = occ0[i]
            if mode == "create" and val == 0:
                occ1 = occ0.copy()
                occ1[i] = 1
            elif mode == "annihilate" and val == 1:
                occ1 = occ0.copy()
                occ1[i] = 0
            else:
                continue

            # Allow one given multiplicity
            alpha = occ1[0::2].sum()
            beta = occ1[1::2].sum()
            new_mult = (alpha - beta) + 1
            if new_mult != mult_final:
                continue

            # SD sign
            sign = (-1) ** int(occ0[:i].sum())

            new_key = "".join(str(x) for x in occ1)
            results.append((sign, i, new_key))

        return {str_occ0: results}

    def generate_sds_dict(
        self, ci_strings: Dict[str, float], mult_final: int, mode: str = "create"
    ) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        For a dict of initial CI strings, generate all connected determinants
        by adding or removing one electron in the active spin-orbitals.

        Returns a dict: { ci_string: [ (sign, spin_idx, new_occ_str), … ] }.
        """
        results: Dict[str, List[Tuple[int, int, str]]] = {}

        for ci_str in ci_strings:
            subdict = self.generate_sds_initial(ci_str, mult_final, mode)
            results[ci_str] = subdict
        return results

    def calculation_is_casci(self):
        """Return True if the calculation is CASCI (i.e., MO overlap is identity)."""
        sub_s_mo = self.s_matrix_mo[
            self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs,
            self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs,
        ]

        return np.allclose(sub_s_mo, np.eye(sub_s_mo.shape[0]), atol=1e-8)

    def dyson_orbital(self):
        """Compute the Dyson orbital."""

        calc_type = "CASCI" if self.calculation_is_casci() else "CASSCF"
        print(f"Calculation type: {calc_type}")

        dyson_ao = (
            self.casci_dyson_coefficients()
            if calc_type == "CASCI"
            else self.dyson_coefficients()
        )

        # temporary to avoid for CASSCF ( will be removed once the next steps are done)

        filename = f"{self.output_dir}/dyson_orbital.cube"
        self.cubefile_from_moeff(dyson_ao, filename)

    def create_pyscf_molecule(self):
        """Convert the initial wavefunction data to PySCF molecule object."""

        # Create the PySCF molecule object
        mol = gto.Mole()
        mol.atom = [
            (atom["ElementLabel"], tuple(atom["Coords"]))
            for atom in self.initial["Molecule"]["Atoms"]
        ]
        mol.unit = "Angstrom"
        mol.basis = generate_basis_dict(self.initial["Molecule"]["Atoms"])
        mol.charge = self.initial["Molecule"]["Charge"]
        mol.spin = self.initial["Molecule"]["Multiplicity"]-1
        mol.build()

        return mol

    def get_pyscf_ao_labels(self):
        """Get the AO labels from the PySCF molecule object."""
        return ["".join(label.split()) for label in self.pyscf_molecule.ao_labels()]

    def get_orca_ao_labels(self):
        """Get the AO labels from the wavefunction data in a format compatible with PySCF."""
        return [
            "".join(parse_orca_labels_pyscf(label))
            for label in self.initial["Molecule"]["MolecularOrbitals"]["OrbitalLabels"]
        ]

    def cubefile_from_moeff(self, moeff: list, filename: str):
        """Generate a cube file from the MO coefficients."""

        mo_pyscf_coeff = []

        # Reorder the MO coefficients to match the PySCF AO labels
        for idx in self.pyscf_ao_labels:
            mo_pyscf_coeff.append(moeff[self.orca_ao_labels.index(idx)])
        cubegen.orbital(self.pyscf_molecule, filename, mo_pyscf_coeff, margin=14.0)
