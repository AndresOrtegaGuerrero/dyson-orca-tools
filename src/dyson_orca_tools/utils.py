import json
import typer

from pathlib import Path


def validate_determinants(det_csf_list, params, state_name):
    """Validate spin determinants/CSFs against expected orbitals and electrons."""
    for det in det_csf_list:
        string = det.strip("[]")
        len_string = len(string)

        if len_string != params["norb"]:
            typer.secho(
                f"❌ Error: The determinant/CSF '{det}' in {state_name} state has {len_string} orbitals "
                f"but expected {params['norb']}.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        num_elec = 2 * string.count("2") + string.count("u") + string.count("d")

        if num_elec != params["nelec"]:
            typer.secho(
                f"❌ Error: The determinant/CSF '{det}' in {state_name} state has {num_elec} electrons "
                f"but expected {params['nelec']}.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)


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


def parameters_sannity_check(params: dict):
    """Perform sanity checks on the parameters."""
    if "parameters" not in params:
        typer.secho(
            "❌ Error: The 'parameters' key is missing in the parameters file.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    states_required = ["initial", "final"]
    missing_states = [
        state for state in states_required if state not in params["parameters"]
    ]

    if missing_states:
        typer.secho(
            f"❌ Error: Missing required states: {', '.join(missing_states)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    initial_params = params["parameters"]["initial"]
    final_params = params["parameters"]["final"]

    required_settings = ["nelec", "norb", "spin_ci"]

    missing_initial = [
        setting for setting in required_settings if setting not in initial_params
    ]
    if missing_initial:
        typer.secho(
            f"❌ Error: Initial state is missing required settings: {', '.join(missing_initial)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    missing_final = [
        setting for setting in required_settings if setting not in final_params
    ]
    if missing_final:
        typer.secho(
            f"❌ Error: Final state is missing required settings: {', '.join(missing_final)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    if abs(initial_params["nelec"] - final_params["nelec"]) != 1:
        typer.secho(
            "❌ Error: The difference in number of electrons between initial and final states must be 1.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    if initial_params["norb"] != final_params["norb"]:
        typer.secho(
            "❌ Error: The number of orbitals must be the same for initial and final states.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    # check spin_ci return a dictionary
    initial_spin_ci = initial_params["spin_ci"]
    final_spin_ci = final_params["spin_ci"]

    if not initial_spin_ci or not final_spin_ci:
        typer.secho(
            "❌ Error: The 'spin_ci' in both states must not be empty.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    det_csf_initial = initial_spin_ci.keys()
    det_csf_final = final_spin_ci.keys()

    validate_determinants(det_csf_initial, initial_params, "initial")
    validate_determinants(det_csf_final, final_params, "final")
