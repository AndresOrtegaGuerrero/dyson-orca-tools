"""Multireference spectral function"""

from pathlib import Path
import typer
from .options import CUBE_POINTS, CUBE_SPACING, CUBE_MARGIN, parse_cube_grid
from ..io.params import load_parameters
from ..workflows.spectrum import SpectrumOptions, run_spectrum
from .console import cli_errors, ok, warn, plot_or_warn, write_cube_or_warn
from ..utils import validate_json_file, sanity_check
from ..io.tables import write_peaks_csv, write_composition_csv
import numpy as np


NAME = "spectrum"


def run(
    initial_wfn: Path = typer.Option(
        ..., "-i", "--initial-wfn", help="JSON of the initial (reference) state."
    ),
    parameters: Path = typer.Option(
        ...,
        "-p",
        "--parameters",
        help="Parameters v2: initial + list of final runs with roots.",
    ),
    output_dir: Path = typer.Option(
        ".", "-o", "--output-dir", help="Directory for the outputs."
    ),
    eta: float = typer.Option(0.05, "--eta", help="Lorentzian broadening (eV)."),
    omega_min: float = typer.Option(
        None, "--omega-min", help="Grid start (eV); default: lowest peak - 1."
    ),
    omega_max: float = typer.Option(
        None, "--omega-max", help="Grid end (eV); default: highest peak + 1."
    ),
    npts: int = typer.Option(2000, "--npts", help="Grid points."),
    shift: float = typer.Option(
        0.0,
        "--shift",
        help="Rigid shift added to all peak energies (eV), e.g. to align E_F.",
    ),
    shift_removal: float = typer.Option(
        0.0, "--shift-removal", help="Extra shift (eV) for removal peaks only."
    ),
    shift_addition: float = typer.Option(
        0.0, "--shift-addition", help="Extra shift (eV) for addition peaks only."
    ),
    align_fermi: bool = typer.Option(
        False,
        "--align-fermi",
        help="Put ω = 0 at the midgap of the ground-state reference, so ω > 0 is electron addition.",
    ),
    cube: bool = typer.Option(
        False,
        "--cube/--no-cube",
        help="Write one cube per peak (needs the [cube] extra).",
    ),
    cube_points: str = CUBE_POINTS,
    cube_spacing: float = CUBE_SPACING,
    cube_margin: float = CUBE_MARGIN,
    plot: bool = typer.Option(
        False,
        "--plot/--no-plot",
        help="Write spectral_function.png/.pdf (needs the [plot] extra).",
    ),
    vertical: bool = typer.Option(
        False, "--vertical", help="Plot with the energy on the y axis."
    ),
    legend_outside: bool = typer.Option(
        False, "--legend-outside", help="Place the legend outside the axes."
    ),
    label_threshold: float = typer.Option(
        0.0,
        "--label-threshold",
        help="Label only peaks with strength >= this (curves keep all roots).",
    ),
    label_energy: bool = typer.Option(
        False,
        "--label-energy",
        help="Append ω (eV, relative to the chosen zero) to each peak label.",
    ),
    common_reference: bool = typer.Option(
        False,
        "--common-reference",
        help="Measure every spectrum from the lowest reference energy "
        "instead of from its own reference state.",
    ),
    min_strength: float = typer.Option(
        1e-8,
        "--min-strength",
        help="Drop peaks below this strength (e.g. spin-forbidden) before labelling.",
    ),
):
    """Multireference spectral function ρ(ω) = η Σ_j |ϱ_j|² / ((ω − E_j)² + η²)."""
    grid = parse_cube_grid(cube_points, cube_spacing, cube_margin) if cube else None
    style = dict(
        vertical=vertical,
        legend_outside=legend_outside,
        label_threshold=label_threshold,
        label_energy=label_energy,
    )
    opts = SpectrumOptions(
        eta=eta,
        omega_min=omega_min,
        omega_max=omega_max,
        npts=npts,
        shift=shift,
        shift_removal=shift_removal,
        shift_addition=shift_addition,
        align_fermi=align_fermi,
        common_reference=common_reference,
        min_strength=min_strength,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with cli_errors():
        initial = validate_json_file(initial_wfn, "Initial")
        params = load_parameters(parameters)
        for run_ in params.runs:
            sanity_check(initial, run_.data)
    ok(
        f"Loaded {len(params.initials)} reference state(s), {len(params.runs)} final run(s), "
        f"{sum(len(r.roots) for r in params.runs)} root(s)."
    )

    typer.secho("🔄 Computing Dyson orbitals...", fg=typer.colors.BLUE)
    with cli_errors():
        result = run_spectrum(initial, params, opts)
    if result.n_dropped:
        typer.echo(
            f"   dropped {result.n_dropped} peak(s) with strength < {min_strength}"
        )
    if align_fermi:
        ok(f"Midgap of '{result.ground}': μ = {result.mu:.3f} eV; ω → ω − μ.")

    for lab in result.empty:
        warn(f"reference {lab}: no peaks left")
    for leaf in result.leaves:
        out = output_dir / leaf.subdir
        out.mkdir(parents=True, exist_ok=True)
        typer.secho(
            f"\nreference {leaf.initial}, ion multiplicity {leaf.mult}: "
            f"E_ref − E_0 = {result.de_ref[leaf.initial]:.3f} eV"
            + (" (common zero)" if common_reference else ""),
            fg=typer.colors.BLUE,
        )
        _print_table(leaf.peaks)
        _write_leaf(leaf, out, initial, result.zero_note, eta, grid)
        if plot:
            plot_or_warn(out, **style)
    if len(result.leaves) > 1:
        write_peaks_csv(output_dir / "dyson_peaks.csv", result.peaks, with_initial=True)
        ok(f"all references: {output_dir / 'dyson_peaks.csv'}")


def _print_table(peaks):
    typer.echo(
        f"   {'label':>6} {'mult':>4} {'omega/eV':>10} {'strength':>9} {'Σc²':>6}  branch  composition (d_p²/strength)"
    )
    for p in peaks:
        comp = "  ".join(f"{lab} {w:.2f}" for lab, w in p.composition())
        typer.echo(
            f"   {p.label:>6} {p.mult:>4} {p.omega:>10.3f} {p.strength:>9.4f} {p.ci_norm:>6.3f}  "
            f"{'CASCI' if p.casci else 'CASSCF':6s}  {comp}"
        )


def _write_leaf(leaf, out_dir: Path, initial: dict, zero_note: str, eta: float, grid):
    peaks = leaf.peaks
    write_peaks_csv(out_dir / "dyson_peaks.csv", peaks)
    write_composition_csv(out_dir / "dyson_composition.csv", peaks)
    np.savetxt(
        out_dir / "spectral_function.dat",
        np.column_stack([leaf.omega, leaf.rho]),
        header=f"omega_eV  rho   (eta = {eta} eV, {zero_note})",
    )
    np.savetxt(
        out_dir / "dyson_orbitals_ao.txt",
        np.column_stack([p.coeff_ao for p in peaks]),
        header="columns = peaks "
        + " ".join(p.label for p in peaks)
        + " ; rows = AOs (ORCA order)",
    )
    if grid is not None:
        for p in peaks:
            name = out_dir / f"dyson_{p.side}{p.label.split(',')[1]}_m{p.mult}.cube"
            if not write_cube_or_warn(initial, p.coeff_ao, name, grid):
                break
        else:
            ok(f"{len(peaks)} cube files written.")
    ok(
        f"Written to {out_dir}: dyson_peaks.csv, dyson_composition.csv, "
        "spectral_function.dat, dyson_orbitals_ao.txt"
    )
