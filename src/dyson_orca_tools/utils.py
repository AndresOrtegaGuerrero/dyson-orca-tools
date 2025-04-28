import json
import typer

from pathlib import Path


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
    # Check that the charges are not the same
    neutral_charge = neutral_wfn_data["Molecule"]["Charge"]
    charged_charge = charged_wfn_data["Molecule"]["Charge"]

    if neutral_charge == charged_charge:
        typer.secho(
            "❌ Error: The charges of the neutral and charged states are the same.",
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

    # Check if S-Matrix in in the files

    if (
        "S-Matrix" not in neutral_wfn_data["Molecule"]
        or "S-Matrix" not in charged_wfn_data["Molecule"]
    ):
        typer.secho(
            "❌ Error: The S-Matrix is missing in one of the files.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
