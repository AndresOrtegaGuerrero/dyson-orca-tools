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
from ..io.orca_output import (
    build_parameters,
    parse_orca_output,
    detect_pt2,
    parse_pt2_energies,
    parse_qdnevpt2,
)
from ..mixing import mixing_matrix, overlap_matrix, composition
from ..excited import nto_info, ndo_info, write_orbital_cubes
from ..spectral import build_peaks, spectral_function, omega_grid

from ..tdm import TransitionDensity
from ..dipole import (
    ao_dipole,
    transition_dipole,
    fragment_contributions,
    fragment_charges,
    centre_origin,
)
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


# cube grid, shared by every command that writes cubes (lengths in Bohr, as in PySCF)
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


def _cube_grid(points: str, spacing: float | None, margin: float) -> dict:
    """CLI grid options -> keyword arguments of ``write_cube``."""
    try:
        n = [int(x) for x in points.split(",")]
    except ValueError:
        _fail(f"--cube-points: '{points}' is not N or NX,NY,NZ")
    if len(n) == 1:
        n = n * 3
    if len(n) != 3 or min(n) < 2:
        _fail(f"--cube-points: '{points}' needs 1 or 3 integers >= 2")
    if spacing is not None and spacing <= 0:
        _fail("--cube-spacing must be positive")
    if margin < 0:
        _fail("--cube-margin must be >= 0")
    nx, ny, nz = n
    return {"nx": nx, "ny": ny, "nz": nz, "resolution": spacing, "margin": margin}


def _write_cube_or_warn(state: dict, coeff_ao, filename: Path, grid: dict):
    try:
        from ..io.cube import write_cube
    except ImportError as exc:
        typer.secho(f"⚠️  {exc}", fg=typer.colors.YELLOW)
        return False
    write_cube(state, coeff_ao, str(filename), **grid)
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
    cube_points: str = CUBE_POINTS,
    cube_spacing: float = CUBE_SPACING,
    cube_margin: float = CUBE_MARGIN,
):
    """Dyson orbital between one initial and one final state."""
    grid = _cube_grid(cube_points, cube_spacing, cube_margin) if cube else None
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
        initial_wfn_data, dyson_ao, output_dir / "dyson_orbital.cube", grid
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
        False, "--label-energy", help="Append ω − E_0 (eV) to each peak label."
    ),
):
    """Multireference spectral function ρ(ω) = η Σ_j |ϱ_j|² / ((ω − E_j)² + η²)."""
    grid = _cube_grid(cube_points, cube_spacing, cube_margin) if cube else None
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
            if not _write_cube_or_warn(initial_wfn_data, p.coeff_ao, name, grid):
                break
        else:
            _ok(f"{len(peaks)} cube files written.")

    if plot:
        _plot_or_warn(
            output_dir,
            vertical=vertical,
            legend_outside=legend_outside,
            label_threshold=label_threshold,
            label_energy=label_energy,
        )
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


def _plot_or_warn(
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
        label_threshold=label_threshold,
        label_energy=label_energy,
        orientation="vertical" if vertical else "horizontal",
        legend="outside" if legend_outside else "inside",
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
    legend_outside: bool = typer.Option(
        False, "--legend-outside", help="Place the legend outside the axes."
    ),
    label_threshold: float = typer.Option(
        0.0,
        "--label-threshold",
        help="Label only peaks with strength >= this (curves keep all roots).",
    ),
    label_energy: bool = typer.Option(
        False, "--label-energy", help="Append ω − E_0 (eV) to each peak label."
    ),
):
    """Re-plot a spectrum from the files written by `spectrum` (no recomputation)."""
    _plot_or_warn(
        output_dir, title, vertical, legend_outside, label_threshold, label_energy
    )


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
    mult: int = typer.Option(
        None,
        "--mult",
        help="Multiplicity block to analyse (default: first block in the output).",
    ),
    qd: bool = typer.Option(
        False,
        "--qd",
        help="Use the QD-NEVPT2 rotated states instead of the CASSCF roots.",
    ),
    mixing: bool = typer.Option(
        False, "--mixing", help="Write the QD-NEVPT2 mixing table (U²) and composition."
    ),
    cubes: int = typer.Option(
        0,
        "--cubes",
        help="Cubes for the top N NTO pairs and N NDOs per pair (needs the [cube] extra).",
    ),
    cube_points: str = CUBE_POINTS,
    cube_spacing: float = CUBE_SPACING,
    cube_margin: float = CUBE_MARGIN,
    dipole: bool = typer.Option(
        False,
        "--dipole",
        help="Transition dipole and its fragment split (needs dipole integrals in the JSON).",
    ),
    output_dir: Path = typer.Option("ehmap", "-d", "--output-dir"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
):
    """Electron–hole correlation map Ω_AB and NTO/NDO analysis between CASSCF or QD-NEVPT2 roots.

    Per pair I→J the line reads:
      Ω        single-excitation character, Σ_σ ‖γ^σ‖² in the spin-orbital basis (Plasser 2014, Eq. 47; 1 for CIS)
      CT       charge-transfer fraction: share of Ω with hole and electron on different fragments
      PR_frag  fragments involved (participation ratio over fragments)
      PR_NTO   NTO pairs needed to describe the transition (Eq. 59)
      λ        NTO weights σ² (Plasser's λ_i, Σλ = Ω); σ itself is ORCA's printed NTO "n"
      p        promotion number from the density difference: electrons actually moved (≈1 single, ≈2 double)
      μ        transition dipole (a.u.) from the spin-traced γ, with --dipole
    Ω = 0 with p ≈ 1 = one orbital changed but as a double substitution of spin orbitals (excitation +
    spin recoupling): no one-electron transition density, dipole-dark from the reference root.
    """
    grid = _cube_grid(cube_points, cube_spacing, cube_margin) if cubes else None
    output_dir.mkdir(parents=True, exist_ok=True)
    cas = parse_orca_output(orca_out)
    text = Path(orca_out).read_text()
    pt2 = detect_pt2(text)
    if mult is None:
        mult = cas.mults[0]
        if len(cas.mults) > 1:
            typer.echo(
                f"   several multiplicity blocks {cas.mults}: using --mult {mult}"
            )
    elif mult not in cas.mults:
        _fail(f"no multiplicity {mult} block in {orca_out} (found {cas.mults})")
    typer.echo(f"   block mult {mult}, PT2 in output: {pt2 or 'none'}")
    typer.echo(
        "   Ω single-excitation character | CT charge-transfer fraction | PR_frag fragments involved | "
        "PR_NTO NTO pairs involved | λ NTO weights (σ²) | p electrons moved (density) | μ transition dipole"
    )

    state = validate_json_file(wfn, "Wavefunction")
    ci = {r.index: r.spin_ci for r in cas.roots if r.mult == mult}
    cas_energy = {r.index: r.energy for r in cas.roots if r.mult == mult}
    # diagonal NEVPT2 energies belong to the CASSCF roots; QD energies to the QD states (--qd)
    energy_pt2 = (
        {r: e for (m, r), e in parse_pt2_energies(text, "NEVPT2").items() if m == mult}
        if pt2
        else {}
    )
    if qd or mixing:
        if pt2 != "QD-NEVPT2":
            _fail(
                f"--qd/--mixing need a QD-NEVPT2 run; this output has {pt2 or 'no PT2'} "
                "(plain NEVPT2 does not mix roots — analyse the CASSCF roots)"
            )
        qdres = parse_qdnevpt2(orca_out)[mult]
    if qd:
        ci = {r.index: r.spin_ci for r in qdres.roots}
        energy_pt2 = qdres.energies
    tag = f"m{mult}_" + ("qd_" if qd else "")
    c_act = active_mo_coefficients(state, cas.norb)
    s = np.array(state["Molecule"]["S-Matrix"])
    labels, groups = load_fragments(fragments, state)
    frag = fragment_of_ao(ao_to_atom(state), groups)

    dip = centre_origin(ao_dipole(state), s, state) if dipole else None

    pairs = [tuple(map(int, p.split(":"))) for p in pair]
    if all_pairs:
        pairs = [(i, j) for i in ci for j in ci if i < j]

    summary, omegas, rdm = {}, {}, {}
    for i, j in pairs:
        td = TransitionDensity(ci[i], ci[j], cas.norb)
        g_spin = td.gamma_spin()
        g = g_spin.sum(axis=0)  # spin-traced γ: dipole, NTOs
        d_ao = td.to_ao(c_act, g)  # rows = particle, cols = hole
        # Ω_AB in the spin-orbital convention: α map + β map (Plasser 2014, Eq. 51); Σ_AB = Ω
        om = sum(
            omega_matrix(td.to_ao(c_act, gs).T, s, frag) for gs in g_spin
        )  # rows = hole
        m = metrics(om)
        m["Omega_spin_traced"] = float(
            (g * g).sum()
        )  # ‖γ^α+γ^β‖², = 2Ω for singlet–singlet
        nto = nto_info(g, c_act, omega=m["Omega"])
        m["nto_sigma"] = nto["sigma"][:5].tolist()  # ORCA's n
        m["nto_lambda"] = nto["lam"][:5].tolist()  # σ² (Plasser's λ)
        m["PR_NTO"] = nto["pr_nto"]
        m["parsed_norms"] = list(td.parsed_norms)
        for r in (i, j):  # normalised 1-RDMs, cached per root
            if r not in rdm:
                t_r = TransitionDensity(ci[r], ci[r], cas.norb)
                rdm[r] = t_r.gamma() / t_r.parsed_norms[0]
        ndo = ndo_info(rdm[j], rdm[i], c_act)
        m["ndo_kappa"] = ndo["kappa"][:6].tolist()
        m["promotion"] = ndo["p"]
        m["dE_casscf_eV"] = (cas_energy[j] - cas_energy[i]) * 27.2114
        if energy_pt2:
            m["dE_pt2_eV"] = (energy_pt2[j] - energy_pt2[i]) * 27.2114
        if dip is not None:
            mu = transition_dipole(d_ao, dip)
            m["mu"] = mu.tolist()
            _write_dipole_csv(
                output_dir / f"{tag}dipole_{i}_{j}.csv",
                fragment_contributions(d_ao, dip, frag),
                fragment_charges(d_ao, s, frag),
                mu,
                labels,
            )
        summary[f"{i}:{j}"], omegas[(i, j)] = m, om
        np.save(output_dir / f"{tag}gamma_{i}_{j}.npy", g)
        _write_omega_csv(output_dir / f"{tag}omega_{i}_{j}.csv", om, labels)
        typer.echo(
            f"   {i}→{j}: Ω = {m['Omega']:.3f}  CT = {m['CT']:.2f}  PR_frag = {m['PR_frag']:.1f}  "
            f"PR_NTO = {m['PR_NTO']:.1f}  λ = {', '.join(f'{x:.3f}' for x in m['nto_lambda'][:3])}  "
            f"p = {m['promotion']:.2f}"
        )
        if dip is not None:
            typer.echo(
                f"        μ = ({mu[0]:+.5f}, {mu[1]:+.5f}, {mu[2]:+.5f}) a.u.  |μ| = {np.linalg.norm(mu):.5f}"
            )
        if cubes:
            k = min(cubes, cas.norb)
            roles = ("hole", "particle")
            has_nto = m["Omega"] > 1e-6  # σ = 0 → NTO cubes would be noise
            names = [
                f"{tag}nto_r{i}_r{j}_pair{p}_{role}_sig{nto['sigma'][p]:.3f}"
                for p in range(k)
                for role in roles
                if has_nto
            ] + [
                f"{tag}ndo_r{i}_r{j}_orb{p}_{'det' if ndo['kappa'][p] < 0 else 'att'}_kap{ndo['kappa'][p]:+.3f}"
                for p in range(k)
            ]
            vecs = np.column_stack(
                [nto[role][:, p] for p in range(k) for role in roles if has_nto]
                + [ndo["coeff"][:, :k]]
            )
            try:
                write_orbital_cubes(state, vecs, names, output_dir / "cubes", **grid)
                typer.echo(f"        {len(names)} cubes -> {output_dir / 'cubes'}")
            except ImportError as exc:
                typer.secho(f"⚠️  {exc}", fg=typer.colors.YELLOW)

    with open(output_dir / f"{tag}summary.json", "w") as f:
        json.dump(
            {"mult": mult, "pt2": pt2, "qd_states": qd, "pairs": summary}, f, indent=2
        )
    if all_pairs:
        _write_pair_table(output_dir / f"{tag}omega_pairs.csv", summary, sorted(ci))
    if mixing:
        e, u = mixing_matrix(qdres.heff)
        u_ov = overlap_matrix(
            {r.index: r.spin_ci for r in cas.roots if r.mult == mult},
            {r.index: r.spin_ci for r in qdres.roots},
        )
        _write_mixing_csv(output_dir / f"m{mult}_qd_mixing.csv", u, e, qdres.energies)
        for K, comp in enumerate(composition(u)):
            typer.echo(
                f"   QD root {K} ({(e[K] - e[0]) * 27.2114:.3f} eV) = "
                + " + ".join(f"{w:.2f}·CAS{J}" for J, w in comp)
            )
        if np.abs(np.abs(u) - np.abs(u_ov)).max() > 0.1:
            typer.secho(
                "⚠️  mixing from H_eff and from the printed vectors differ by >0.1 (truncation?)",
                fg=typer.colors.YELLOW,
            )
    if plot and not all_pairs:
        vmax = max(om.max() for om in omegas.values())  # one colour scale for all maps
        for (i, j), om in omegas.items():
            plot_ehmap(
                om,
                labels,
                str(output_dir / f"{tag}ehmap_{i}_{j}.png"),
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


def _write_mixing_csv(path, u, e, e_orca):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "QD_root",
                "E_eigh_Eh",
                "E_orca_Eh",
                *(f"U2_CAS{J}" for J in range(u.shape[0])),
            ]
        )
        for K in range(u.shape[1]):
            w.writerow(
                [
                    K,
                    f"{e[K]:.6f}",
                    f"{e_orca[K]:.6f}",
                    *(f"{x:.4f}" for x in u[:, K] ** 2),
                ]
            )


def _write_dipole_csv(path, per_frag, q_frag, mu, labels):
    """Origin = molecular centre; a fragment's μ_A shifts by q_A·ΔR under an origin change."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fragment", "mu_x", "mu_y", "mu_z", "transition_charge"])
        for lab, row, q in zip(labels, per_frag, q_frag):
            w.writerow([lab, *(f"{x:+.6f}" for x in row), f"{q:+.4f}"])
        w.writerow(["total", *(f"{x:+.6f}" for x in mu), f"{q_frag.sum():+.4f}"])


def _write_orbital_table(path, occ, pop, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["active_mo", "occupation", *labels])
        for i, (o, row) in enumerate(zip(occ, pop), start=1):
            w.writerow([i, f"{o:.4f}", *(f"{x:.4f}" for x in row)])


if __name__ == "__main__":
    app()
