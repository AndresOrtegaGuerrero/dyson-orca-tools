import json
import typer
import numpy as np
from pathlib import Path
from typing import List

from ..utils import validate_json_file
from ..io.orca_output import (
    parse_orca_output,
    detect_pt2,
    parse_pt2_energies,
    parse_qdnevpt2,
)
from ..mixing import mixing_matrix, overlap_matrix, composition
from ..excited import nto_info, ndo_info, write_orbital_cubes

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
    write_omega_csv,
    write_pair_csv,
    write_mixing_csv,
    write_dipole_csv,
    write_orbital_csv,
)

from .console import ok, fail, warn
from .options import CUBE_POINTS, CUBE_SPACING, CUBE_MARGIN, parse_cube_grid
from . import dyson, prepare, plot, spectrum

app = typer.Typer(
    help="Dyson orbitals and multireference spectral functions from ORCA CASCI/CASSCF JSON."
)

for module in (dyson, prepare, plot, spectrum):
    app.command(module.NAME)(module.run)


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
