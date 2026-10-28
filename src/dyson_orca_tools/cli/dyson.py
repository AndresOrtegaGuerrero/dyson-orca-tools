"""dyson: Dyson orbital between one inition and one final state"""

from pathlib import Path

import numpy as np
import typer

from ..dyson import Dyson
from ..utils import parameters_sanity_check, sanity_check, validate_json_file
from .console import cli_errors, ok, write_cube_or_warn
from .options import CUBE_MARGIN, CUBE_POINTS, CUBE_SPACING, parse_cube_grid

NAME = "dyson"


def run(
    initial_wfn: Path = typer.Option(
        ..., "-i", "--initial-wfn", help="JSON of the initial state."
    ),
    final_wfn: Path = typer.Option(
        ..., "-f", "--final-wfn", help="JSON of the final (N±1) state."
    ),
    parameters: Path = typer.Option(
        ..., "-p", "--parameters", help="JSON with the spin CI coefficients."
    ),
    output_dir: Path = typer.Option(
        ".", "-o", "--output-dir", help="Directory for the outputs."
    ),
    cube: bool = typer.Option(
        True, "--cube/--no-cube", help="Write a cube file (needs the [cube] extra)."
    ),
    cube_points: str = CUBE_POINTS,
    cube_spacing: float = CUBE_SPACING,
    cube_margin: float = CUBE_MARGIN,
):
    """Dyson orbital between one initial and one final state."""
    grid = parse_cube_grid(cube_points, cube_spacing, cube_margin) if cube else None
    output_dir.mkdir(parents=True, exist_ok=True)

    with cli_errors():
        initial_wfn_data = validate_json_file(initial_wfn, "Initial")
        final_wfn_data = validate_json_file(final_wfn, "Final")
        parameters_data = validate_json_file(parameters, "Parameters")
        ok("Input files are valid JSONs.")
        sanity_check(initial_wfn_data, final_wfn_data)
        parameters_sanity_check(parameters_data)
    ok("Inputs passed sanity checks.")

    typer.secho("🔄 Computing Dyson orbital...", fg=typer.colors.BLUE)
    dyson = Dyson(initial_wfn_data, final_wfn_data, parameters_data)
    if dyson.calculation_is_casci():
        typer.echo("   orbital sets: CASCI (same MOs)")
    else:
        typer.echo(
            f"   orbital sets: CASSCF (non-orthogonal MOs, core det = {np.linalg.det(dyson.core_overlap):.6f})"
        )

    dyson_ao = dyson.dyson_orbital()
    typer.secho(
        f"   strength <ϱ|ϱ> = {dyson.strength(dyson_ao):.6f}", fg=typer.colors.CYAN
    )

    np.savetxt(
        output_dir / "dyson_orbital_ao.txt",
        dyson_ao,
        header="Dyson orbital, AO coefficients (ORCA order)",
    )
    if cube and write_cube_or_warn(
        initial_wfn_data, dyson_ao, output_dir / "dyson_orbital.cube", grid
    ):
        ok(f"cube written: {output_dir / 'dyson_orbital.cube'}")
    typer.secho("🚀 Dyson orbital computed successfully!", fg=typer.colors.CYAN)
