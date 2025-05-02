from pyscf import gto
from pyscf.tools import cubegen
from .utils import generate_basis_dict, parse_orca_labels_pyscf
import numpy as np


class Dyson:
    def __init__(self, initial: dict, final: dict, parameters: dict):
        self.initial = initial
        self.final = final
        self.parameters = parameters
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

    def get_s_matrix_ao(self, state: dict):
        """Get the overlap matrix in the AO basis."""
        return np.column_stack(state["Molecule"]["S-Matrix"])

    def get_mo_coeff_array(self, state: dict):
        mo_orbs = state["Molecule"]["MolecularOrbitals"]["MOs"]
        mo_coeff_list = [orb["MOCoefficients"] for orb in mo_orbs]
        return np.column_stack(mo_coeff_list)

    def get_s_matrix_mo(self):
        # Determine left and right based on charge
        charge_initial = self.initial["Molecule"]["Charge"]
        charge_final = self.final["Molecule"]["Charge"]

        C_left, C_right = (
            (self.MO_coeff_final, self.MO_coeff_initial)
            if charge_initial > charge_final
            else (self.MO_coeff_initial, self.MO_coeff_final)
        )

        return C_left.T @ self.s_matrix_ao @ C_right

    def get_ci_coeff(self, state: str) -> dict:
        """Get the CI coefficients for the given state."""
        spin_ci = self.parameters["parameters"][state]["spin_ci"]
        return {k.strip("[]"): v for k, v in spin_ci.items()}  # Remove the brackets

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

        cubegen.density(self.pyscf_molecule, filename, mo_pyscf_coeff)
