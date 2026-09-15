import csv
import json
import os
import typer
import numpy as np
from pathlib import Path
from typing import List

from ..utils import validate_json_file, sannity_check, parameters_sannity_check
from ..dyson import Dyson
from ..io.params import load_parameters, parse_parameters
from ..io.orca_output import build_parameters, parse_orca_output
from ..spectral import build_peaks, spectral_function, omega_grid

from ..tdm import TransitionDensity
from ..dipole import ao_dipole, transition_dipole, fragment_contributions
from ..ehmap import (
    ao_to_atom,
    fragment_of_ao,
    omega_matrix,
    metrics,
    plot_ehmap,
    active_mo_coefficients,
    load_fragments,
    orbital_fragment_populations,
    active_occupations,
)

app = typer.Typer(
    help="Dyson orbitals and multireference spectral functions from ORCA CASCI/CASSCF JSON."
)


def _ok(msg):
    typer.secho(f"✅ {msg}", fg=typer.colors.GREEN)


def _fail(msg):
    typer.secho(f"❌ {msg}", fg=typer.colors.RED)
    raise typer.Exit(1)


def _write_cube_or_warn(state: dict, coeff_ao, filename: Path):
    try:
        from ..io.cube import write_cube
    except ImportError as exc:
        typer.secho(f"⚠️  {exc}", fg=typer.colors.YELLOW)
        return False
    write_cube(state, coeff_ao, str(filename))
    return True


@app.command("dyson")
def compute_dyson_orbital(
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
):
    """Dyson orbital between one initial and one final state."""
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_wfn_data = validate_json_file(initial_wfn, "Initial")
    final_wfn_data = validate_json_file(final_wfn, "Final")
    parameters_data = validate_json_file(parameters, "Parameters")
    _ok("Input files are valid JSONs.")
    sannity_check(initial_wfn_data, final_wfn_data)
    parameters_sannity_check(parameters_data)
    _ok("Inputs passed sanity checks.")

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
    if cube and _write_cube_or_warn(
        initial_wfn_data, dyson_ao, output_dir / "dyson_orbital.cube"
    ):
        _ok(f"cube written: {output_dir / 'dyson_orbital.cube'}")
    typer.secho("🚀 Dyson orbital computed successfully!", fg=typer.colors.CYAN)


@app.command("spectrum")
def compute_spectrum(
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
    cube: bool = typer.Option(
        False,
        "--cube/--no-cube",
        help="Write one cube per peak (needs the [cube] extra).",
    ),
    plot: bool = typer.Option(
        False,
        "--plot/--no-plot",
        help="Write spectral_function.png/.pdf (needs the [plot] extra).",
    ),
    vertical: bool = typer.Option(
        False, "--vertical", help="Plot with the energy on the y axis."
    ),
):
    """Multireference spectral function ρ(ω) = η Σ_j |ϱ_j|² / ((ω − E_j)² + η²)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_wfn_data = validate_json_file(initial_wfn, "Initial")
    try:
        params = load_parameters(parameters)
    except (ValueError, FileNotFoundError) as exc:
        _fail(f"Parameters: {exc}")
    for i, run in enumerate(params.runs):
        sannity_check(initial_wfn_data, run.data)
    _ok(
        f"Loaded {len(params.runs)} final run(s), {sum(len(r.roots) for r in params.runs)} root(s)."
    )

    typer.secho("🔄 Computing Dyson orbitals...", fg=typer.colors.BLUE)
    try:
        peaks = build_peaks(initial_wfn_data, params)
    except ValueError as exc:
        _fail(str(exc))
    for p in peaks:
        p.omega += shift

    typer.echo(
        f"   {'label':>6} {'mult':>4} {'omega/eV':>10} {'strength':>9} {'Σc²':>6}  branch  composition (d_p²/strength)"
    )
    for p in peaks:
        comp = "  ".join(f"{lab} {w:.2f}" for lab, w in p.composition())
        typer.echo(
            f"   {p.label:>6} {p.mult:>4} {p.omega:>10.3f} {p.strength:>9.4f} {p.ci_norm:>6.3f}  "
            f"{'CASCI' if p.casci else 'CASSCF':6s}  {comp}"
        )

    with open(output_dir / "dyson_peaks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "side", "mult", "omega_eV", "strength", "branch"])
        for p in peaks:
            w.writerow(
                [
                    p.label,
                    p.side,
                    p.mult,
                    f"{p.omega:.6f}",
                    f"{p.strength:.6f}",
                    "CASCI" if p.casci else "CASSCF",
                ]
            )

    with open(output_dir / "dyson_composition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "strength", "ci_norm", "leading_det", *peaks[0].mo_labels])
        for p in peaks:  # d_p² per initial active MO; the row sums to the strength
            w.writerow(
                [
                    p.label,
                    f"{p.strength:.6f}",
                    f"{p.ci_norm:.6f}",
                    p.leading_det,
                    *(f"{d * d:.6f}" for d in p.coeff_mo),
                ]
            )

    grid = omega_grid(peaks, npts=npts)
    lo = grid[0] if omega_min is None else omega_min
    hi = grid[-1] if omega_max is None else omega_max
    omega = np.linspace(lo, hi, npts)
    rho = spectral_function(peaks, omega, eta=eta)
    np.savetxt(
        output_dir / "spectral_function.dat",
        np.column_stack([omega, rho]),
        header=f"omega_eV  rho   (eta = {eta} eV, shift = {shift} eV)",
    )

    coeffs = np.column_stack([p.coeff_ao for p in peaks])
    np.savetxt(
        output_dir / "dyson_orbitals_ao.txt",
        coeffs,
        header="columns = peaks "
        + " ".join(p.label for p in peaks)
        + " ; rows = AOs (ORCA order)",
    )

    if cube:
        for p in peaks:
            name = output_dir / f"dyson_{p.side}{p.label.split(',')[1]}_m{p.mult}.cube"
            if not _write_cube_or_warn(initial_wfn_data, p.coeff_ao, name):
                break
        else:
            _ok(f"{len(peaks)} cube files written.")

    if plot:
        _plot_or_warn(output_dir, vertical=vertical)
    _ok(
        f"Written to {output_dir}: dyson_peaks.csv, dyson_composition.csv, spectral_function.dat, dyson_orbitals_ao.txt"
    )


@app.command("prepare")
def prepare_parameters(
    initial_out: Path = typer.Option(
        ...,
        "-i",
        "--initial-out",
        help="ORCA output of the initial state (PrintWF det).",
    ),
    final: List[str] = typer.Option(
        ...,
        "-f",
        "--final",
        help="ORCA output of one N±1 calculation, repeatable. "
        "Its orca_2json file is <name>.json next to it, or give 'run.out:file.json'.",
    ),
    output: Path = typer.Option(
        "params.json", "-o", "--output", help="Parameters file to write."
    ),
    initial_root: int = typer.Option(
        0, "--initial-root", help="Root of the initial output to use as reference."
    ),
):
    """Build the v2 parameters file from ORCA outputs (CI vectors, energies, active space)."""
    pairs = []
    for item in final:
        out_file, _, json_file = item.partition(":")
        out_file = Path(out_file)
        pairs.append(
            (out_file, Path(json_file) if json_file else out_file.with_suffix(".json"))
        )
    try:
        raw = build_parameters(initial_out, pairs, initial_root)
        params = parse_parameters(raw)  # validates before writing
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))

    for run in raw["parameters"]["final"]:  # paths relative to the parameters file
        run["file"] = os.path.relpath(
            Path(run["file"]).resolve(), output.resolve().parent
        )
    with open(output, "w") as f:
        json.dump(raw, f, indent=2)
    ini = params.initial
    _ok(
        f"initial: CAS({ini.nelc},{ini.norb}) mult {ini.mult}, E = {ini.energy} Eh, {len(ini.spin_ci)} determinants"
    )
    for run in params.runs:
        norm = [sum(c * c for c in r.spin_ci.values()) for r in run.roots]
        typer.echo(
            f"   final {run.file}: mult {run.mult}, {len(run.roots)} root(s), "
            f"Σc² = {', '.join(f'{n:.3f}' for n in norm)}"
        )
    _ok(f"written {output}")


def _plot_or_warn(out_dir: Path, title: str | None = None, vertical: bool = False):
    try:
        from ..io.plot import read_outputs, plot_spectrum
    except ImportError as exc:
        typer.secho(f"⚠️  {exc}", fg=typer.colors.YELLOW)
        return
    peaks, omega, rho, eta = read_outputs(out_dir)
    path = plot_spectrum(
        peaks,
        omega,
        rho,
        out_dir / "spectral_function.png",
        eta=eta,
        title=title,
        orientation="vertical" if vertical else "horizontal",
    )
    _ok(f"plot written: {path} (+ .pdf)")


@app.command("plot")
def plot_outputs(
    output_dir: Path = typer.Option(
        ".",
        "-o",
        "--output-dir",
        help="Folder with dyson_peaks.csv and spectral_function.dat.",
    ),
    title: str = typer.Option(
        None, "--title", help="Optional title, e.g. 'pentacene CASCI(12,12)'."
    ),
    vertical: bool = typer.Option(
        False, "--vertical", help="Energy on the y axis (as in the paper's figures)."
    ),
):
    """Re-plot a spectrum from the files written by `spectrum` (no recomputation)."""
    _plot_or_warn(output_dir, title, vertical)


@app.command("ehmap")
def electron_hole_map(
    orca_out: Path = typer.Option(
        ..., "-o", "--orca-out", help="ORCA output with PrintWF det."
    ),
    wfn: Path = typer.Option(
        ..., "-j", "--json", help="orca_2json file of the same gbw."
    ),
    pair: List[str] = typer.Option(
        ["0:1"], "--pair", help="Root pair I:J, repeatable."
    ),
    all_pairs: bool = typer.Option(
        False, "--all-pairs", help="Ω for every root pair (table only)."
    ),
    fragments: Path = typer.Option(
        None, "-f", "--fragments", help="JSON {name: [atom indices]}."
    ),
    orbitals: bool = typer.Option(
        False,
        "--orbitals",
        help="Print/write the fragment population of each active MO.",
    ),
    dipole: bool = typer.Option(
        False,
        "--dipole",
        help="Transition dipole and its fragment split (needs dipole integrals in the JSON).",
    ),
    output_dir: Path = typer.Option("ehmap", "-d", "--output-dir"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
):
    """Electron–hole correlation map Ω_AB and NTO weights between CASSCF roots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cas = parse_orca_output(orca_out)
    state = validate_json_file(wfn, "Wavefunction")
    ci = {r.index: r.spin_ci for r in cas.roots}
    c_act = active_mo_coefficients(state, cas.norb)
    s = np.array(state["Molecule"]["S-Matrix"])
    labels, groups = load_fragments(fragments, state)
    frag = fragment_of_ao(ao_to_atom(state), groups)

    dip = ao_dipole(state) if dipole else None

    pairs = [tuple(map(int, p.split(":"))) for p in pair]
    if all_pairs:
        pairs = [(i, j) for i in ci for j in ci if i < j]

    summary, omegas = {}, {}
    for i, j in pairs:
        td = TransitionDensity(ci[i], ci[j], cas.norb)
        g = td.gamma()
        d_ao = td.to_ao(c_act, g)  # rows = particle, cols = hole
        om = omega_matrix(d_ao.T, s, frag)  # .T → rows = hole, cols = electron
        m = metrics(om)
        m["nto_weights"] = td.ntos(g)[0][:5].tolist()
        m["parsed_norms"] = list(td.parsed_norms)
        if dip is not None:
            mu = transition_dipole(d_ao, dip)
            m["mu"] = mu.tolist()
            _write_dipole_csv(
                output_dir / f"dipole_{i}_{j}.csv",
                fragment_contributions(d_ao, dip, frag),
                mu,
                labels,
            )
        summary[f"{i}:{j}"], omegas[(i, j)] = m, om
        np.save(output_dir / f"gamma_{i}_{j}.npy", g)
        _write_omega_csv(output_dir / f"omega_{i}_{j}.csv", om, labels)
        typer.echo(
            f"   {i}→{j}: Ω = {m['Omega']:.4f}  CT = {m['CT']:.2f}  PR = {m['PR']:.1f}"
        )
        if dip is not None:
            typer.echo(
                f"        μ = ({mu[0]:+.5f}, {mu[1]:+.5f}, {mu[2]:+.5f}) a.u.  |μ| = {np.linalg.norm(mu):.5f}"
            )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if all_pairs:
        _write_pair_table(output_dir / "omega_pairs.csv", summary, sorted(ci))
    if plot and not all_pairs:
        vmax = max(om.max() for om in omegas.values())  # one colour scale for all maps
        for (i, j), om in omegas.items():
            plot_ehmap(
                om,
                labels,
                str(output_dir / f"ehmap_{i}_{j}.png"),
                title=f"roots {i}→{j}  Ω={summary[f'{i}:{j}']['Omega']:.3f}",
                vmax=vmax,
            )
    if orbitals:
        occ = active_occupations(state, cas.norb)
        pop = orbital_fragment_populations(c_act, s, frag)
        _write_orbital_table(output_dir / "active_orbitals.csv", occ, pop, labels)
        typer.echo(
            f"   {'act':>4} {'occ':>6} " + "".join(f"{label:>11}" for label in labels)
        )
        for i, (o, row) in enumerate(zip(occ, pop), start=1):
            typer.echo(f"   {i:>4} {o:6.3f} " + "".join(f"{x:11.2f}" for x in row))

    _ok(f"written {output_dir}")


def _write_omega_csv(path, om, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hole\\electron", *labels])
        for lab, row in zip(labels, om):
            w.writerow([lab, *(f"{x:.6e}" for x in row)])


def _write_pair_table(path, summary, roots):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["I\\J", *roots])
        for i in roots:
            w.writerow(
                [
                    i,
                    *(
                        f"{summary[f'{min(i, j)}:{max(i, j)}']['Omega']:.4e}"
                        if i != j
                        else "-"
                        for j in roots
                    ),
                ]
            )


def _write_dipole_csv(path, per_frag, mu, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fragment", "mu_x", "mu_y", "mu_z"])
        for lab, row in zip(labels, per_frag):
            w.writerow([lab, *(f"{x:+.6f}" for x in row)])
        w.writerow(["total", *(f"{x:+.6f}" for x in mu)])


def _write_orbital_table(path, occ, pop, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["active_mo", "occupation", *labels])
        for i, (o, row) in enumerate(zip(occ, pop), start=1):
            w.writerow([i, f"{o:.4f}", *(f"{x:.4f}" for x in row)])


if __name__ == "__main__":
    app()
