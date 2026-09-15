import os
import numpy as np
import pytest

from dyson_orca_tools.tdm import TransitionDensity, occ_array, _flip
from dyson_orca_tools.io.orca_output import parse_orca_output, parse_nto_occupations
from conftest import README_INITIAL

R2 = 1 / np.sqrt(2)
SINGLET = {"[ud]": R2, "[du]": -R2}
TRIPLET = {"[ud]": R2, "[du]": R2}
CLOSED = {"[20]": 1.0}


def test_homo_lumo_singlet_transition_density_is_sqrt2():
    g = TransitionDensity(SINGLET, CLOSED, 2).gamma()
    assert g[1, 0] == pytest.approx(np.sqrt(2))  # electron leaves 0, lands in 1
    assert abs(g).sum() == pytest.approx(np.sqrt(2))  # nothing else


def test_triplet_is_spin_forbidden():
    assert np.allclose(TransitionDensity(TRIPLET, CLOSED, 2).gamma(), 0)


def test_same_state_gives_occupations():
    np.testing.assert_allclose(
        TransitionDensity(CLOSED, CLOSED, 2).gamma(), np.diag([2.0, 0.0])
    )


def test_rdm_trace_and_bounds():
    td = TransitionDensity(README_INITIAL, README_INITIAL, 4)
    g = td.gamma() / td.parsed_norms[0]
    assert np.trace(g) == pytest.approx(4.0)  # nelc
    assert np.allclose(g, g.T)
    assert np.all(
        (np.linalg.eigvalsh(g) > -1e-12) & (np.linalg.eigvalsh(g) < 2 + 1e-12)
    )


def test_annihilation_sign_matches_dyson():
    """Our a_q half must give the same sign Dyson.apply_operator gives."""
    for det in ("[2u0d]", "[ud2u]", "[du2u]"):
        occ = occ_array(det)
        for q in np.flatnonzero(occ):
            s_q, _ = _flip(occ, q, 0)
            assert s_q == (-1 if occ[:q].sum() % 2 else 1)


@pytest.mark.skipif("CAS_OUT" not in os.environ, reason="local casscf output only")
def test_report_against_orca_ntos():
    cas = parse_orca_output(os.environ["CAS_OUT"])
    nto = parse_nto_occupations(os.environ["CAS_OUT"])
    ci = {r.index: r.spin_ci for r in cas.roots}
    for state in range(1, len(ci)):
        td = TransitionDensity(ci[0], ci[state], cas.norb)
        g = td.gamma()
        assert np.allclose(g, TransitionDensity(ci[state], ci[0], cas.norb).gamma().T)
        w = td.ntos(g)[0]
        print(
            f"root {state}: ours σ²={w[:3]}  σ={np.sqrt(w[:3])}  ORCA n={nto[(state, 3)][:3]}  norms={td.parsed_norms}"
        )
