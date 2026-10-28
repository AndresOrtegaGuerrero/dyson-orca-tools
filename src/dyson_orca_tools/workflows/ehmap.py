"""ehmap command: states -> per-pair transition-density analysis -> extras."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import HARTREE_EV
from ..dipole import (
    ao_dipole,
    centre_origin,
    fragment_charges,
    fragment_contributions,
    transition_dipole,
)
from ..ehmap import (
    active_mo_coefficients,
    active_occupations,
    ao_to_atom,
    fragment_of_ao,
    metrics,
    omega_matrix,
    orbital_fragment_populations,
)
from ..excited import ndo_info, nto_info
from ..io.orca_output import (
    detect_pt2,
    parse_orca_output,
    parse_pt2_energies,
    parse_qdnevpt2,
)
from ..mixing import composition, mixing_matrix, overlap_matrix
from ..tdm import TransitionDensity


@dataclass
class EhmapOptions:
    mult: int | None = None
    qd: bool = False
    mixing: bool = False
    dipole: bool = False
    orbitals: bool = False
    ncubes: int = 0


@dataclass
class EhmapSystem:
    cas: object  # parse_orca_output result
    mult: int
    pt2: str | None  # "QD-NEVPT2" or None, as found in the output
    tag: str  # "m1_" or "m1_qd_"
    ci: dict  # root -> spin_ci (CASSCF or QD)
    cas_energy: dict
    energy_pt2: dict
    qdres: object | None  # H_eff and QD states, only with --qd/--mixing
    state: dict
    c_act: np.ndarray  # active MO coefficients, AO x active
    s: np.ndarray
    labels: list[str]
    frag: np.ndarray
    dip: np.ndarray | None  # AO dipole integrals at the molecular centre, with --dipole
    mult_note: str = ""  # "several multiplicity blocks [...]: using --mult 1" or ""


@dataclass
class PairAnalysis:
    i: int
    j: int
    metrics: dict  # the summary entry (Omega, CT, PR_frag, nto_lambda, promotion, dE_*, mu, ...)
    omega: np.ndarray  # Ω_AB
    gamma: np.ndarray  # spin-traced γ (goes to .npy)
    nto: dict
    ndo: dict
    mu: np.ndarray | None = None
    dipole_fragments: tuple | None = None  # (per_frag, q_frag)


@dataclass
class MixingAnalysis:
    e: np.ndarray
    u: np.ndarray
    u_ov: np.ndarray
    energies_orca: dict
    composition: list
    inconsistent: bool  # |U| from H_eff vs printed vectors differ by > 0.1


@dataclass
class EhmapResult:
    system: EhmapSystem
    pairs: list[PairAnalysis]
    mixing: MixingAnalysis | None = None
    orbitals: tuple | None = None  # (occ, pop)


def select_states(
    orca_out: Path, state: dict, labels, groups, opts: EhmapOptions
) -> EhmapSystem:
    cas = parse_orca_output(orca_out)
    text = Path(orca_out).read_text()
    pt2 = detect_pt2(text)
    mult, note = opts.mult, ""
    if mult is None:
        mult = cas.mults[0]
        if len(cas.mults) > 1:
            note = f"several multiplicity blocks {cas.mults}: using --mult {mult}"
    elif mult not in cas.mults:
        raise ValueError(
            f"no multiplicity {mult} block in {orca_out} (found {cas.mults})"
        )

    ci = {r.index: r.spin_ci for r in cas.roots if r.mult == mult}
    cas_energy = {r.index: r.energy for r in cas.roots if r.mult == mult}
    energy_pt2 = (
        {r: e for (m, r), e in parse_pt2_energies(text, "NEVPT2").items() if m == mult}
        if pt2
        else {}
    )
    qdres = None
    if opts.qd or opts.mixing:
        if pt2 != "QD-NEVPT2":
            raise ValueError(
                f"--qd/--mixing need a QD-NEVPT2 run; this output has {pt2 or 'no PT2'} "
                "(plain NEVPT2 does not mix roots — analyse the CASSCF roots)"
            )
        qdres = parse_qdnevpt2(orca_out)[mult]
    if opts.qd:
        ci, energy_pt2 = {r.index: r.spin_ci for r in qdres.roots}, qdres.energies

    c_act = active_mo_coefficients(state, cas.norb)
    s = np.array(state["Molecule"]["S-Matrix"])
    frag = fragment_of_ao(ao_to_atom(state), groups)
    dip = centre_origin(ao_dipole(state), s, state) if opts.dipole else None
    return EhmapSystem(
        cas=cas,
        mult=mult,
        pt2=pt2,
        tag=f"m{mult}_" + ("qd_" if opts.qd else ""),
        ci=ci,
        cas_energy=cas_energy,
        energy_pt2=energy_pt2,
        qdres=qdres,
        state=state,
        c_act=c_act,
        s=s,
        labels=labels,
        frag=frag,
        dip=dip,
        mult_note=note,
    )


def analyse_pair(sys: EhmapSystem, i: int, j: int, rdm: dict) -> PairAnalysis:
    td = TransitionDensity(sys.ci[i], sys.ci[j], sys.cas.norb)
    g_spin = td.gamma_spin()
    g = g_spin.sum(axis=0)
    d_ao = td.to_ao(sys.c_act, g)
    om = sum(omega_matrix(td.to_ao(sys.c_act, gs).T, sys.s, sys.frag) for gs in g_spin)
    m = metrics(om)
    m["Omega_spin_traced"] = float((g * g).sum())
    nto = nto_info(g, sys.c_act, omega=m["Omega"])
    m["nto_sigma"], m["nto_lambda"] = nto["sigma"][:5].tolist(), nto["lam"][:5].tolist()
    m["PR_NTO"], m["parsed_norms"] = nto["pr_nto"], list(td.parsed_norms)
    for r in (i, j):
        if r not in rdm:
            t_r = TransitionDensity(sys.ci[r], sys.ci[r], sys.cas.norb)
            rdm[r] = t_r.gamma() / t_r.parsed_norms[0]
    ndo = ndo_info(rdm[j], rdm[i], sys.c_act)
    m["ndo_kappa"], m["promotion"] = ndo["kappa"][:6].tolist(), ndo["p"]
    m["dE_casscf_eV"] = (sys.cas_energy[j] - sys.cas_energy[i]) * HARTREE_EV
    if sys.energy_pt2:
        m["dE_pt2_eV"] = (sys.energy_pt2[j] - sys.energy_pt2[i]) * HARTREE_EV
    pa = PairAnalysis(i, j, m, om, g, nto, ndo)
    if sys.dip is not None:
        pa.mu = transition_dipole(d_ao, sys.dip)
        m["mu"] = pa.mu.tolist()
        pa.dipole_fragments = (
            fragment_contributions(d_ao, sys.dip, sys.frag),
            fragment_charges(d_ao, sys.s, sys.frag),
        )
    return pa


def run_ehmap(
    orca_out, state, labels, groups, pairs, opts: EhmapOptions
) -> EhmapResult:
    sys = select_states(orca_out, state, labels, groups, opts)
    if pairs is None:  # --all-pairs
        pairs = [(i, j) for i in sys.ci for j in sys.ci if i < j]
    rdm = {}
    result = EhmapResult(sys, [analyse_pair(sys, i, j, rdm) for i, j in pairs])
    if opts.mixing:
        e, u = mixing_matrix(sys.qdres.heff)
        u_ov = overlap_matrix(
            {r.index: r.spin_ci for r in sys.cas.roots if r.mult == sys.mult},
            {r.index: r.spin_ci for r in sys.qdres.roots},
        )
        result.mixing = MixingAnalysis(
            e,
            u,
            u_ov,
            sys.qdres.energies,
            list(composition(u)),
            bool(np.abs(np.abs(u) - np.abs(u_ov)).max() > 0.1),
        )
    if opts.orbitals:
        result.orbitals = (
            active_occupations(state, sys.cas.norb),
            orbital_fragment_populations(sys.c_act, sys.s, sys.frag),
        )
    return result


def cube_set(pa: PairAnalysis, k: int, tag: str) -> tuple[list[str], np.ndarray]:
    """Names and column vectors for the top-k NTO pairs and NDOs of one pair."""
    i, j, nto, ndo = pa.i, pa.j, pa.nto, pa.ndo
    roles = ("hole", "particle")
    has_nto = pa.metrics["Omega"] > 1e-6  # σ = 0 → NTO cubes would be noise
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
    return names, vecs
