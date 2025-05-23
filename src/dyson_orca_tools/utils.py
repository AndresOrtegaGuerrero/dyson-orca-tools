import json
import typer

from pathlib import Path
from pyscf import gto


def generate_basis_string(atom_label, atom_basis_data):
    basis_str = ""
    for shell in atom_basis_data:
        shell_type = shell["Shell"]
        coefficients = shell["Coefficients"]
        exponents = shell["Exponents"]

        basis_str += f"{atom_label}    {shell_type}\n"

        for coeff, exp in zip(coefficients, exponents):
            formatted_exp = f"{exp: .10E}".replace("E", "E")
            formatted_coeff = f"{coeff: .8E}".replace("E", "E")
            basis_str += f"      {formatted_exp}          {formatted_coeff}\n"

    return basis_str


def generate_basis_dict(atoms_data):
    atom_basis_dict = {}

    for atom in atoms_data:
        atom_label = atom["ElementLabel"]
        basis_data = atom["Basis"]

        if atom_label not in atom_basis_dict:
            atom_basis_dict[atom_label] = ""
        string_basis = generate_basis_string(atom_label, basis_data)
        atom_basis_dict[atom_label] = gto.basis.parse(string_basis)

    return atom_basis_dict


def parse_orca_labels_pyscf(orca_label):
    split_str = orca_label.split()
    orbital = split_str[-1]

    orb_formating = {"dz2": "dz^2", "dx2y2": "dx2-y2", "f0": "f+0"}

    if orbital[1] == "p":
        shell = str(int(orbital[0]) + 1) + orbital[1:]
        orbital = shell

    if orbital[1] == "d":
        d_orbital = orb_formating.get(orbital[1:], orbital[1:])
        shell = str(int(orbital[0]) + 2) + d_orbital
        orbital = shell

    if orbital[1] == "f":
        f_orbital = orb_formating.get(orbital[1:], orbital[1:])
        shell = str(int(orbital[0]) + 3) + f_orbital
        orbital = shell

    return [split_str[0], orbital]


def validate_json_file(path: Path, label: str):
    """Validate the JSON file and returns as a dictionary."""
    if not path.exists():
        typer.secho(
            f"❌ Error: {label} file '{path}' does not exist.", fg=typer.colors.RED
        )
        raise typer.Exit(1)
    if not path.is_file():
        typer.secho(
            f"❌ Error: {label} file '{path}' is not a file.", fg=typer.colors.RED
        )
        raise typer.Exit(1)
    if path.suffix != ".json":
        typer.secho(
            f"❌ Error: {label} file '{path}' is not a JSON file.", fg=typer.colors.RED
        )
        raise typer.Exit(1)
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        typer.secho(
            f"❌ Error: {label} file '{path}' is not a valid JSON file. {e}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)


# Pending, we need a validation that the data required is present in the JSON file


def sannity_check(neutral_wfn_data: dict, charged_wfn_data: dict):
    """Perform sanity checks on the JSON data."""
    # Check that charges are correct
    neutral_charge = neutral_wfn_data["Molecule"]["Charge"]
    charged_charge = charged_wfn_data["Molecule"]["Charge"]

    if abs(neutral_charge - charged_charge) != 1:
        typer.secho(
            "❌ Error: The charges of the neutral and charged states are not correct.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    # Check if the number of atoms is the same
    neutral_atoms = len(neutral_wfn_data["Molecule"]["Atoms"])
    charged_atoms = len(charged_wfn_data["Molecule"]["Atoms"])

    if neutral_atoms != charged_atoms:
        typer.secho(
            "❌ Error: The number of atoms in the neutral and charged states is different.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    # Check if the multiplicities are correct

    multiplicity_neutral = neutral_wfn_data["Molecule"]["Multiplicity"]
    multiplicity_charged = charged_wfn_data["Molecule"]["Multiplicity"]

    if abs(multiplicity_neutral - multiplicity_charged) != 1:
        typer.secho(
            "❌ Error: The multiplicities of the neutral and charged states are not correct.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    # Check if the coordinates and basis sets are the same and in order

    for idx in range(neutral_atoms):
        neutral_atom = neutral_wfn_data["Molecule"]["Atoms"][idx]
        charged_atom = charged_wfn_data["Molecule"]["Atoms"][idx]

        if neutral_atom["ElementLabel"] != charged_atom["ElementLabel"]:
            typer.secho(
                f"❌ Error: The elements of the atoms at index {idx} are different.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        if neutral_atom["Coords"] != charged_atom["Coords"]:
            typer.secho(
                f"❌ Error: The coordinates of the atoms at index {idx} are different.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        if neutral_atom["Basis"] != charged_atom["Basis"]:
            typer.secho(
                f"❌ Error: The basis sets of the atoms at index {idx} are different.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    # Check if the MolecularOrbitals are in the files

    if (
        "MolecularOrbitals" not in neutral_wfn_data["Molecule"]
        or "MolecularOrbitals" not in charged_wfn_data["Molecule"]
    ):
        typer.secho(
            "❌ Error: The MolecularOrbitals are missing in one of the files.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    # Check if S-Matrix is in the files

    if (
        "S-Matrix" not in neutral_wfn_data["Molecule"]
        or "S-Matrix" not in charged_wfn_data["Molecule"]
    ):
        typer.secho(
            "❌ Error: The S-Matrix is missing in one of the files.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
