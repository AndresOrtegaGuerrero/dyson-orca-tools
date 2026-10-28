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
from ..spectral import (
    build_peaks,
    spectral_function,
    omega_grid,
    relabel,
    midgap,
    shift_peaks,
    HARTREE_EV,
)

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

from ..io.tables import (
    write_peaks_csv,
    write_composition_csv,
    write_omega_csv,
    write_pair_csv,
    write_mixing_csv,
    write_dipole_csv,
    write_orbital_csv,
)

from .console import ok, fail, warn, write_cube_or_warn, plot_or_warn
from .options import CUBE_POINTS, CUBE_SPACING, CUBE_MARGIN, parse_cube_grid

app = typer.Typer(
    help="Dyson orbitals and multireference spectral functions from ORCA CASCI/CASSCF JSON."
)


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
    grid = parse_cube_grid(cube_points, cube_spacing, cube_margin) if cube else None
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_wfn_data = validate_json_file(initial_wfn, "Initial")
    final_wfn_data = validate_json_file(final_wfn, "Final")
    parameters_data = validate_json_file(parameters, "Parameters")
    ok("Input files are valid JSONs.")
    sannity_check(initial_wfn_data, final_wfn_data)
    parameters_sannity_check(parameters_data)
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
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_wfn_data = validate_json_file(initial_wfn, "Initial")
    try:
        params = load_parameters(parameters)
    except (ValueError, FileNotFoundError) as exc:
        fail(f"Parameters: {exc}")
    for run in params.runs:
        sannity_check(initial_wfn_data, run.data)
    ok(
        f"Loaded {len(params.initials)} reference state(s), {len(params.runs)} final run(s), "
        f"{sum(len(r.roots) for r in params.runs)} root(s)."
    )

    typer.secho("🔄 Computing Dyson orbitals...", fg=typer.colors.BLUE)
    try:
        peaks = build_peaks(initial_wfn_data, params, min_strength=min_strength)
    except ValueError as exc:
        fail(str(exc))
    n_all = sum(len(r.roots) for r in params.runs) * len(params.initials)
    if len(peaks) < n_all:
        typer.echo(
            f"   dropped {n_all - len(peaks)} peak(s) with strength < {min_strength}"
        )

    # energy_zero: each reference state has its own zero, unless --common-reference
    e0 = min(s.energy for s in params.initials)
    de_ref = {s.label: (s.energy - e0) * HARTREE_EV for s in params.initials}
    ground = min(params.initials, key=lambda s: s.energy).label

    if common_reference:
        for p in peaks:
            p.omega += de_ref[p.initial]

    mu = 0.0
    if align_fermi:
        try:
            mu = midgap(peaks, ground)
        except ValueError as exc:
            fail(f"Cannot align Fermi level: {exc}")
        ok(f"Midgap of '{ground}': μ = {mu:.3f} eV; ω → ω − μ.")
    shift_peaks(
        peaks,
        mu=mu,
        shift=shift,
        shift_removal=shift_removal,
        shift_addition=shift_addition,
    )
    labels = [s.label for s in params.initials]
    opts = dict(
        eta=eta,
        omega_min=omega_min,
        omega_max=omega_max,
        npts=npts,
        zero_note=(
            f"mu = {mu:.4f} eV, shift = {shift} eV, shift_removal = {shift_removal} eV, "
            f"shift_addition = {shift_addition} eV, common_reference = {common_reference}"
        ),
        plot=plot,
        vertical=vertical,
        legend_outside=legend_outside,
        label_threshold=label_threshold,
        label_energy=label_energy,
    )

    # One spectrum per (reference, ion multiplicity): peaks of different ion
    # multiplicities are never summed (their relative weights need the spin
    # factors, not implemented). Folders: <ref>/ or <ref>/m<mult>/ when several.
    leaves = 0
    for lab in labels:
        mine = [p for p in peaks if p.initial == lab]
        if not mine:
            typer.secho(f"\nreference {lab}: no peaks left", fg=typer.colors.YELLOW)
            continue
        ref_dir = output_dir if len(labels) == 1 else output_dir / lab
        mults = sorted({p.mult for p in mine})
        for m in mults:
            out = ref_dir if len(mults) == 1 else ref_dir / f"m{m}"
            out.mkdir(parents=True, exist_ok=True)
            typer.secho(
                f"\nreference {lab}, ion multiplicity {m}: E_ref − E_0 = {de_ref[lab]:.3f} eV"
                + (" (common zero)" if common_reference else ""),
                fg=typer.colors.BLUE,
            )
            subset = relabel(
                [p for p in mine if p.mult == m]
            )  # in place: table below matches
            _write_spectrum(subset, out, initial_wfn_data, grid=grid, **opts)
            leaves += 1

    if leaves > 1:  # one table over everything; labels as in the leaf folders
        write_peaks_csv(output_dir / "dyson_peaks.csv", peaks, with_initial=True)
        ok(f"all references: {output_dir / 'dyson_peaks.csv'}")


def _write_spectrum(
    peaks,
    out_dir: Path,
    initial_wfn_data: dict,
    *,
    eta,
    omega_min,
    omega_max,
    npts,
    zero_note,
    grid,
    plot,
    vertical,
    legend_outside,
    label_threshold,
    label_energy,
):
    """Table, CSVs, ρ(ω), AO coefficients, cubes and plot for one reference state."""
    typer.echo(
        f"   {'label':>6} {'mult':>4} {'omega/eV':>10} {'strength':>9} {'Σc²':>6}  branch  composition (d_p²/strength)"
    )
    for p in peaks:
        comp = "  ".join(f"{lab} {w:.2f}" for lab, w in p.composition())
        typer.echo(
            f"   {p.label:>6} {p.mult:>4} {p.omega:>10.3f} {p.strength:>9.4f} {p.ci_norm:>6.3f}  "
            f"{'CASCI' if p.casci else 'CASSCF':6s}  {comp}"
        )
    write_peaks_csv(out_dir / "dyson_peaks.csv", peaks)
    write_composition_csv(out_dir / "dyson_composition.csv", peaks)

    omegas = omega_grid(peaks, npts=npts)
    lo = omegas[0] if omega_min is None else omega_min
    hi = omegas[-1] if omega_max is None else omega_max
    omega = np.linspace(lo, hi, npts)
    rho = spectral_function(peaks, omega, eta=eta)
    np.savetxt(
        out_dir / "spectral_function.dat",
        np.column_stack([omega, rho]),
        header=f"omega_eV  rho   (eta = {eta} eV, {zero_note})",
    )

    coeffs = np.column_stack([p.coeff_ao for p in peaks])
    np.savetxt(
        out_dir / "dyson_orbitals_ao.txt",
        coeffs,
        header="columns = peaks "
        + " ".join(p.label for p in peaks)
        + " ; rows = AOs (ORCA order)",
    )

    if grid is not None:
        for p in peaks:
            name = out_dir / f"dyson_{p.side}{p.label.split(',')[1]}_m{p.mult}.cube"
            if not write_cube_or_warn(initial_wfn_data, p.coeff_ao, name, grid):
                break
        else:
            ok(f"{len(peaks)} cube files written.")

    if plot:
        plot_or_warn(
            out_dir,
            vertical=vertical,
            legend_outside=legend_outside,
            label_threshold=label_threshold,
            label_energy=label_energy,
        )
    ok(
        f"Written to {out_dir}: dyson_peaks.csv, dyson_composition.csv, "
        "spectral_function.dat, dyson_orbitals_ao.txt"
    )


def _parse_state_spec(spec: str) -> tuple[int, int]:
    """'3:0' -> (3, 0)."""
    try:
        mult, root = spec.split(":")
        return int(mult), int(root)
    except ValueError:
        raise ValueError(f"--initial-state '{spec}': expected MULT:ROOT, e.g. 1:0")


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
    initial_state: List[str] = typer.Option(
        None,
        "--initial-state",
        help="Reference state as MULT:ROOT, repeatable (e.g. 1:0 3:0). "
        "Default: the lowest-energy root of the output.",
    ),
    all_initial_roots: bool = typer.Option(
        False, "--all-initial-roots", help="Every root of every block as a reference."
    ),
):
    """Build the parameters file (v3) from ORCA outputs (CI vectors, energies, active space)."""
    pairs = []
    for item in final:
        out_file, _, json_file = item.partition(":")
        out_file = Path(out_file)
        pairs.append(
            (out_file, Path(json_file) if json_file else out_file.with_suffix(".json"))
        )
    try:
        cas = parse_orca_output(initial_out)
        if all_initial_roots:
            states = [(r.mult, r.index) for r in cas.roots]
        else:
            states = [_parse_state_spec(s) for s in initial_state or []]

        raw = build_parameters(cas, pairs, states)
        params = parse_parameters(raw)  # validates before writing
    except (ValueError, FileNotFoundError) as exc:
        fail(str(exc))

    for run in raw["parameters"]["final"]:  # paths relative to the parameters file
        run["file"] = os.path.relpath(
            Path(run["file"]).resolve(), output.resolve().parent
        )
    with open(output, "w") as f:
        json.dump(raw, f, indent=2)
    for ini in params.initials:
        ok(
            f"initial {ini.label}: CAS({ini.nelc},{ini.norb}) mult {ini.mult}, "
            f"E = {ini.energy} Eh, {len(ini.spin_ci)} determinants"
        )
    for run in params.runs:
        norm = [sum(c * c for c in r.spin_ci.values()) for r in run.roots]
        typer.echo(
            f"   final {run.file}: mult {run.mult}, {len(run.roots)} root(s), "
            f"Σc² = {', '.join(f'{n:.3f}' for n in norm)}"
        )
    ok(f"written {output}")


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
        False,
        "--label-energy",
        help="Append ω (eV, relative to the chosen zero) to each peak label.",
    ),
):
    """Re-plot a spectrum from the files written by `spectrum` (no recomputation)."""
    plot_or_warn(
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
    grid = parse_cube_grid(cube_points, cube_spacing, cube_margin) if cubes else None
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
        fail(f"no multiplicity {mult} block in {orca_out} (found {cas.mults})")
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
            fail(
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
            write_dipole_csv(
                output_dir / f"{tag}dipole_{i}_{j}.csv",
                fragment_contributions(d_ao, dip, frag),
                fragment_charges(d_ao, s, frag),
                mu,
                labels,
            )
        summary[f"{i}:{j}"], omegas[(i, j)] = m, om
        np.save(output_dir / f"{tag}gamma_{i}_{j}.npy", g)
        write_omega_csv(output_dir / f"{tag}omega_{i}_{j}.csv", om, labels)
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
                warn(str(exc))

    with open(output_dir / f"{tag}summary.json", "w") as f:
        json.dump(
            {"mult": mult, "pt2": pt2, "qd_states": qd, "pairs": summary}, f, indent=2
        )
    if all_pairs:
        write_pair_csv(output_dir / f"{tag}omega_pairs.csv", summary, sorted(ci))
    if mixing:
        e, u = mixing_matrix(qdres.heff)
        u_ov = overlap_matrix(
            {r.index: r.spin_ci for r in cas.roots if r.mult == mult},
            {r.index: r.spin_ci for r in qdres.roots},
        )
        write_mixing_csv(output_dir / f"m{mult}_qd_mixing.csv", u, e, qdres.energies)
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
        write_orbital_csv(output_dir / "active_orbitals.csv", occ, pop, labels)
        typer.echo(
            f"   {'act':>4} {'occ':>6} " + "".join(f"{label:>11}" for label in labels)
        )
        for i, (o, row) in enumerate(zip(occ, pop), start=1):
            typer.echo(f"   {i:>4} {o:6.3f} " + "".join(f"{x:11.2f}" for x in row))

    ok(f"written {output_dir}")


if __name__ == "__main__":
    app()
