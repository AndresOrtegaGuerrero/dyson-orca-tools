from pyscf import gto
from pyscf.tools import cubegen
from utils import generate_basis_dict, parse_orca_labels_pyscf


class Dyson:
    def __init__(self, initial: dict, final: dict, parameters: dict):
        self.initial = initial
        self.final = final
        self.parameters = parameters
        self.pyscf_molecule = self.create_pyscf_molecule()
        self.pyscf_ao_labels = self.get_pyscf_ao_labels()
        self.orca_ao_labels = self.get_orca_ao_labels()

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
