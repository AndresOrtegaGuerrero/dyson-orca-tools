"""ehmap command: states -> per-pair transition-density analysis -> extras. electron hole map"""

import typer
import json
import numpy as np
from pathlib import Path
from typing import List
from .options import CUBE_POINTS, CUBE_SPACING, CUBE_MARGIN, parse_cube_grid
from ..workflows.ehmap import (
    EhmapOptions,
    PairAnalysis,
    MixingAnalysis,
    run_ehmap,
    cube_set,
)
from .console import cli_errors, ok, warn
from ..utils import validate_json_file
from ..constants import HARTREE_EV
from ..io.tables import (
    write_omega_csv,
    write_dipole_csv,
    write_pair_csv,
    write_orbital_csv,
    write_mixing_csv,
)
from ..excited import write_orbital_cubes
from ..ehmap import load_fragments, plot_ehmap


NAME = "ehmap"


def run(
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
    opts = EhmapOptions(
        mult=mult, qd=qd, mixing=mixing, dipole=dipole, orbitals=orbitals, ncubes=cubes
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_list = None if all_pairs else [tuple(map(int, p.split(":"))) for p in pair]

    with cli_errors():
        state = validate_json_file(wfn, "Wavefunction")
        labels, groups = load_fragments(fragments, state)
        result = run_ehmap(orca_out, state, labels, groups, pair_list, opts)
    sys = result.system
    if sys.mult_note:
        typer.echo(f"   {sys.mult_note}")
    typer.echo(f"   block mult {sys.mult}, PT2 in output: {sys.pt2 or 'none'}")
    typer.echo(
        "   Ω single-excitation character | CT charge-transfer fraction | PR_frag fragments involved | "
        "PR_NTO NTO pairs involved | λ NTO weights (σ²) | p electrons moved (density) | μ transition dipole"
    )

    summary = {}
    for pa in result.pairs:
        summary[f"{pa.i}:{pa.j}"] = pa.metrics
        np.save(output_dir / f"{sys.tag}gamma_{pa.i}_{pa.j}.npy", pa.gamma)
        write_omega_csv(
            output_dir / f"{sys.tag}omega_{pa.i}_{pa.j}.csv", pa.omega, labels
        )
        if pa.mu is not None:
            write_dipole_csv(
                output_dir / f"{sys.tag}dipole_{pa.i}_{pa.j}.csv",
                *pa.dipole_fragments,
                pa.mu,
                labels,
            )
        _print_pair(pa)
        if cubes:
            names, vecs = cube_set(pa, min(cubes, sys.cas.norb), sys.tag)
            try:
                write_orbital_cubes(state, vecs, names, output_dir / "cubes", **grid)
                typer.echo(f"        {len(names)} cubes -> {output_dir / 'cubes'}")
            except ImportError as exc:
                warn(str(exc))

    with open(output_dir / f"{sys.tag}summary.json", "w") as f:
        json.dump(
            {"mult": sys.mult, "pt2": sys.pt2, "qd_states": qd, "pairs": summary},
            f,
            indent=2,
        )
    if all_pairs:
        write_pair_csv(
            output_dir / f"{sys.tag}omega_pairs.csv", summary, sorted(sys.ci)
        )
    if result.mixing:
        _report_mixing(result.mixing, output_dir, sys.mult)
    if plot and not all_pairs:
        vmax = max(pa.omega.max() for pa in result.pairs)
        for pa in result.pairs:
            plot_ehmap(
                pa.omega,
                labels,
                str(output_dir / f"{sys.tag}ehmap_{pa.i}_{pa.j}.png"),
                title=f"roots {pa.i}→{pa.j}  Ω={pa.metrics['Omega']:.3f}",
                vmax=vmax,
            )
    if result.orbitals:
        occ, pop = result.orbitals
        write_orbital_csv(output_dir / "active_orbitals.csv", occ, pop, labels)
        _print_orbitals(occ, pop, labels)
    ok(f"written {output_dir}")


def _print_pair(pa: PairAnalysis):
    m = pa.metrics
    typer.echo(
        f"   {pa.i}→{pa.j}: Ω = {m['Omega']:.3f}  CT = {m['CT']:.2f}  PR_frag = {m['PR_frag']:.1f}  "
        f"PR_NTO = {m['PR_NTO']:.1f}  λ = {', '.join(f'{x:.3f}' for x in m['nto_lambda'][:3])}  "
        f"p = {m['promotion']:.2f}"
    )
    if pa.mu is not None:
        mu = pa.mu
        typer.echo(
            f"        μ = ({mu[0]:+.5f}, {mu[1]:+.5f}, {mu[2]:+.5f}) a.u.  |μ| = {np.linalg.norm(mu):.5f}"
        )


def _report_mixing(mx: MixingAnalysis, output_dir: Path, mult: int):
    write_mixing_csv(
        output_dir / f"m{mult}_qd_mixing.csv", mx.u, mx.e, mx.energies_orca
    )
    for K, comp in enumerate(mx.composition):
        typer.echo(
            f"   QD root {K} ({(mx.e[K] - mx.e[0]) * HARTREE_EV:.3f} eV) = "
            + " + ".join(f"{w:.2f}·CAS{J}" for J, w in comp)
        )
    if mx.inconsistent:
        warn(
            "mixing from H_eff and from the printed vectors differ by >0.1 (truncation?)"
        )


def _print_orbitals(occ, pop, labels):
    typer.echo(
        f"   {'act':>4} {'occ':>6} " + "".join(f"{label:>11}" for label in labels)
    )
    for i, (o, row) in enumerate(zip(occ, pop), start=1):
        typer.echo(f"   {i:>4} {o:6.3f} " + "".join(f"{x:11.2f}" for x in row))
