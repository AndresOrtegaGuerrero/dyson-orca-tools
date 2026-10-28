"""Shared options for CLI commands."""

import typer
from .console import fail

# cube grid (lengths in Bohr as in PySCF)
CUBE_POINTS = typer.Option(
    "80", "--cube-points", help="Cube grid points: N or NX,NY,NZ."
)
CUBE_SPACING = typer.Option(
    None,
    "--cube-spacing",
    help="Approximate cube grid spacing (Bohr); overrides --cube-points.",
)
CUBE_MARGIN = typer.Option(
    14.0, "--cube-margin", help="Cube box padding around the molecule (Bohr)."
)


def parse_cube_grid(points: str, spacing: float | None, margin: float) -> dict:
    """CLI grid options. keyword arguments for write_cube."""
    try:
        n = [int(x) for x in points.split(",")]
    except ValueError:
        fail(f"--cube-points: '{points}' is not N or NX,NY,NZ")
    if len(n) == 1:
        n = n * 3
    if len(n) != 3 or min(n) < 2:
        fail(f"--cube-points: '{points}' needs 1 or 3 integers >= 2")
    if spacing is not None and spacing <= 0:
        fail("--cube-spacing must be positive")
    if margin < 0:
        fail("--cube-margin must be >= 0")
    nx, ny, nz = n
    return {"nx": nx, "ny": ny, "nz": nz, "resolution": spacing, "margin": margin}
