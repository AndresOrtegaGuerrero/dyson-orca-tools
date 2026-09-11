import typer
import numpy as np
from pathlib import Path

from ..utils import validate_json_file, sannity_check, parameters_sannity_check
from ..dyson import Dyson

app = typer.Typer(help="Compute Dyson orbitals from ORCA CASCI/CASSCF JSON outputs.")


def _write_cube_or_warn(state: dict, coeff_ao, filename: Path):
    try:
        from ..io.cube import write_cube
    except ImportError as exc:
        typer.secho(f"⚠️  {exc}", fg=typer.colors.YELLOW)
        return
    write_cube(state, coeff_ao, str(filename))
    typer.secho(f"   cube written: {filename}", fg=typer.colors.GREEN)


@app.command()
def compute_dyson_orbital(
    initial_wfn: Path = typer.Option(
        ..., "-i", "--initial-wfn", help="JSON of the neutral state."
    ),
    final_wfn: Path = typer.Option(
        ..., "-f", "--final-wfn", help="JSON of the charged state."
    ),
    parameters: Path = typer.Option(
        None, "-p", "--parameters", help="JSON with the spin CI coefficients."
    ),
    output_dir: Path = typer.Option(
        ".", "-o", "--output-dir", help="Directory for the outputs."
    ),
    cube: bool = typer.Option(
        True, "--cube/--no-cube", help="Write a cube file (needs the [cube] extra)."
    ),
):
    """Compute the Dyson orbital between one neutral and one charged state."""
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_wfn_data = validate_json_file(initial_wfn, "Initial")
    final_wfn_data = validate_json_file(final_wfn, "Final")
    parameters_data = validate_json_file(parameters, "Parameters")
    typer.secho("✅ Input files are valid JSONs.", fg=typer.colors.GREEN)

    sannity_check(initial_wfn_data, final_wfn_data)
    typer.secho("✅ Input files passed sanity checks.", fg=typer.colors.GREEN)

    parameters_sannity_check(parameters_data)
    typer.secho("✅ Parameters file passed sanity checks.", fg=typer.colors.GREEN)

    typer.secho("🔄 Computing Dyson orbital...", fg=typer.colors.BLUE)
    dyson = Dyson(initial_wfn_data, final_wfn_data, parameters_data)
    calc_type = "CASCI" if dyson.calculation_is_casci() else "CASSCF"
    typer.secho(
        f"   orbital sets: {calc_type} (same MOs)"
        if calc_type == "CASCI"
        else f"   orbital sets: {calc_type} (non-orthogonal MOs, core det = "
        f"{np.linalg.det(dyson.core_overlap):.6f})"
    )

    dyson_ao = dyson.dyson_orbital()
    strength = dyson.strength(dyson_ao)
    typer.secho(f"   strength <ϱ|ϱ> = {strength:.6f}", fg=typer.colors.CYAN)

    np.savetxt(
        output_dir / "dyson_orbital_ao.txt",
        dyson_ao,
        header="Dyson orbital, AO coefficients (ORCA order)",
    )
    if cube:
        _write_cube_or_warn(
            initial_wfn_data, dyson_ao, output_dir / "dyson_orbital.cube"
        )

    typer.secho("🚀 Dyson orbital computed successfully!", fg=typer.colors.CYAN)


if __name__ == "__main__":
    app()
