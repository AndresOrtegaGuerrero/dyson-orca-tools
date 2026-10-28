"""Console prints with color and exits."""

from contextlib import contextmanager
from pathlib import Path
import typer


def ok(msg):
    typer.secho(f"✅ {msg}", fg=typer.colors.GREEN)


def warn(msg):
    typer.secho(f"⚠️  {msg}", fg=typer.colors.YELLOW)


def fail(msg):
    typer.secho(f"❌ {msg}", fg=typer.colors.RED)
    raise typer.Exit(1)


@contextmanager
def cli_errors(*types):
    """Context manager to catch exceptions and print Red error messages to the console."""
    try:
        yield
    except types or (ValueError, FileNotFoundError) as exc:
        fail(str(exc))


def write_cube_or_warn(state, coeff_ao, filename, grid):
    try:
        from ..io.cube import write_cube
    except ImportError as exc:
        warn(str(exc))
        return False
    write_cube(state, coeff_ao, str(filename), **grid)
    return True


def plot_or_warn(
    out_dir: Path,
    title: str | None = None,
    vertical: bool = False,
    legend_outside: bool = False,
    label_threshold: float = 0.0,
    label_energy: bool = False,
):
    try:
        from ..io.plot import read_outputs, plot_spectrum
    except ImportError as exc:
        warn(str(exc))
        return
    peaks, omega, rho, eta = read_outputs(out_dir)
    path = plot_spectrum(
        peaks,
        omega,
        rho,
        out_dir / "spectral_function.png",
        eta=eta,
        title=title,
        label_threshold=label_threshold,
        label_energy=label_energy,
        orientation="vertical" if vertical else "horizontal",
        legend="outside" if legend_outside else "inside",
    )
    ok(f"plot written: {path} (+ .pdf)")
