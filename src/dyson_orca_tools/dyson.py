from pyscf import gto
from pyscf.tools import cubegen
from .utils import generate_basis_dict, parse_orca_labels_pyscf
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

    def get_s_matrix_ao(self, state: dict):
        """Get the overlap matrix in the AO basis."""
        return np.array(state["Molecule"]["S-Matrix"])

    def get_mo_coeff_array(self, state: dict):
        mo_orbs = state["Molecule"]["MolecularOrbitals"]["MOs"]
        mo_coeff_list = [orb["MOCoefficients"] for orb in mo_orbs]
        return np.column_stack(mo_coeff_list)

    def get_s_matrix_mo(self):
        # Determine left and right based on charge
        # charge_initial = self.initial["Molecule"]["Charge"]
        # charge_final = self.final["Molecule"]["Charge"]

        # C_left, C_right = (
        #    (self.MO_coeff_final, self.MO_coeff_initial)
        #    if charge_initial > charge_final
        #    else (self.MO_coeff_initial, self.MO_coeff_final)
        # )

        # return C_left.T @ self.s_matrix_ao @ C_right

        return self.MO_coeff_initial.T @ self.s_matrix_ao @ self.MO_coeff_final

    def get_ci_coeff(self, state: str) -> dict:
        """Get the CI coefficients for the given state."""
        spin_ci = self.parameters["parameters"][state]["spin_ci"]
        return {k.strip("[]"): v for k, v in spin_ci.items()}  # Remove the brackets

    def ci_vector_to_array(self, vector: str) -> list:
        """Convert CI dict with keys like '[2200]', '[udud]' into lists of 0/1 for alpha/beta occupation."""
        spin_map = {"2": [1, 1], "u": [1, 0], "d": [0, 1], "0": [0, 0]}

        return [bit for c in vector for bit in spin_map[c]]

    def occupation_diff(self, sd_f, sd_i):
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

    def dyson_coefficients(self):
        dyson_coeff = np.zeros(2 * self.parameters["parameters"]["initial"]["norb"])
        for sd_i, ci_i in self.CI_initial.items():
            for sd_f, ci_f in self.CI_final.items():
                idx, sign = self.occupation_diff(sd_f, sd_i)
                if idx is not None:
                    dyson_coeff[idx] += sign * ci_i * ci_f
        print(dyson_coeff)
        dyson_ao = np.zeros(self.s_matrix_ao.shape[0])
        for i in range(self.num_active_orbs):
            coeff = dyson_coeff[2 * i] + dyson_coeff[2 * i + 1]  # alpha + beta
            mo_index = self.num_inactive_orbs + i

            mo_coeff = (
                self.MO_coeff_final if self.add_or_remove < 0 else self.MO_coeff_initial
            )
            dyson_ao += coeff * mo_coeff[:, mo_index]

        return dyson_ao

    def dyson_orbital(self):
        """Compute the Dyson orbital."""

        dyson_ao = self.dyson_coefficients()

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
        cubegen.orbital(self.pyscf_molecule, filename, mo_pyscf_coeff)
