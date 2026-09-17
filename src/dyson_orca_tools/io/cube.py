"""Optional cube output through PySCF (`pip install dyson-orca-tools[cube]`)."""

try:
    from pyscf import gto
    from pyscf.tools import cubegen
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Cube output needs PySCF: pip install 'dyson-orca-tools[cube]'"
    ) from exc


def _basis_string(atom_label, shells):
    lines = []
    for shell in shells:
        lines.append(f"{atom_label}    {shell['Shell']}")
        for coeff, exp in zip(shell["Coefficients"], shell["Exponents"]):
            lines.append(f"      {exp: .10E}          {coeff: .8E}")
    return "\n".join(lines) + "\n"


def basis_dict(atoms):
    return {
        atom["ElementLabel"]: gto.basis.parse(
            _basis_string(atom["ElementLabel"], atom["Basis"])
        )
        for atom in atoms
    }


def orca_label_to_pyscf(orca_label):
    """'C 1pz' -> 'C2pz': ORCA counts shells per l, PySCF uses principal n."""
    element, orbital = orca_label.split()[0], orca_label.split()[-1]
    rename = {"dz2": "dz^2", "dx2y2": "dx2-y2", "f0": "f+0"}
    shift = {"p": 1, "d": 2, "f": 3}.get(orbital[1], 0)
    n = int(orbital[0]) + shift
    tail = rename.get(orbital[1:], orbital[1:])
    return f"{element}{n}{tail}"


def pyscf_molecule(state: dict):
    mol = gto.Mole()
    mol.atom = [
        (a["ElementLabel"], tuple(a["Coords"])) for a in state["Molecule"]["Atoms"]
    ]
    mol.unit = "Angstrom"
    mol.basis = basis_dict(state["Molecule"]["Atoms"])
    mol.charge = state["Molecule"]["Charge"]
    mol.spin = state["Molecule"]["Multiplicity"] - 1
    mol.build()
    return mol


def write_cube(state: dict, coeff_ao, filename: str, margin: float = 14.0):
    """Write AO coefficients (ORCA ordering) as a cube file on the state's geometry."""
    mol = pyscf_molecule(state)
    orca_labels = [
        orca_label_to_pyscf(lbl)
        for lbl in state["Molecule"]["MolecularOrbitals"]["OrbitalLabels"]
    ]
    pyscf_labels = ["".join(lbl.split()) for lbl in mol.ao_labels()]
    reordered = [coeff_ao[orca_labels.index(lbl)] for lbl in pyscf_labels]
    cubegen.orbital(mol, filename, reordered, margin=margin)
