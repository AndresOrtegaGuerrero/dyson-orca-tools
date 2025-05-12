import typer
from pathlib import Path


from ..utils import validate_json_file, sannity_check
from ..dyson import Dyson

app = typer.Typer(help="Compute Dyson orbitals from ORCA CASCI/CASSCF JSON outputs.")


@app.command()
def compute_dyson_orbital(
    initial_wfn: Path = typer.Option(
        ...,
        "-i",
        "--initial-wfn",
        help="JSON file of the neutral state wavefunction.",
    ),
    final_wfn: Path = typer.Option(
        ...,
        "-f",
        "--final-wfn",
        help="JSON file of the charged state wavefunction.",
    ),
    parameters: Path = typer.Option(
        None,
        "-p",
        "--parameters",
        help="JSON file with the parameters for the calculation containing the Spin CI coefficients.",
    ),
    output_dir: Path = typer.Option(
        ".",
        "-o",
        "--output-dir",
        help="Directory to save the Dyson orbital files.",
    ),
):
    """Compute Dyson orbitals from ORCA CASCI/CASSCF JSON outputs."""
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the JSON files
    initial_wfn_data = validate_json_file(initial_wfn, "Initial")
    final_wfn_data = validate_json_file(final_wfn, "Final")
    parameters_data = validate_json_file(parameters, "Parameters")
    typer.secho("✅ Input files are valid JSONs.", fg=typer.colors.GREEN)

    # Perform sanity checks on the JSON data
    sannity_check(initial_wfn_data, final_wfn_data)
    typer.secho("✅ Input files passed sanity checks.", fg=typer.colors.GREEN)

    typer.secho("🔄 Computing Dyson orbitals...", fg=typer.colors.BLUE)

    # Our algorithm to compute the Dyson orbitals goes here

    dyson = Dyson(initial_wfn_data, final_wfn_data, parameters_data, output_dir)  # noqa: F841
    dyson.dyson_orbital()

    typer.secho("🚀 Dyson orbital computed successfully!", fg=typer.colors.CYAN)


if __name__ == "__main__":
    app()
