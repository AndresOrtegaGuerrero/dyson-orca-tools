import os

import numpy as np
import pytest

from dyson_orca_tools.excited import nto_info, ndo_info
from dyson_orca_tools.tdm import TransitionDensity

R2 = 1 / np.sqrt(2)
SINGLET = {"[ud]": R2, "[du]": -R2}
CLOSED = {"[20]": 1.0}


def test_nto_of_homo_lumo_singlet():
    info = nto_info(TransitionDensity(SINGLET, CLOSED, 2).gamma(), np.eye(2))
    assert info["sigma"][0] == pytest.approx(np.sqrt(2))
    assert info["sigma"][1] == pytest.approx(0, abs=1e-12)
    assert info["lam"][0] == pytest.approx(2.0) and info["pr_nto"] == pytest.approx(1.0)
    assert abs(info["hole"][0, 0]) == pytest.approx(1)  # hole in MO 0
    assert abs(info["particle"][1, 0]) == pytest.approx(1)  # electron in MO 1


def test_ndo_of_homo_lumo_singlet():
    g0 = TransitionDensity(CLOSED, CLOSED, 2).gamma()
    g1 = TransitionDensity(SINGLET, SINGLET, 2).gamma()
    info = ndo_info(g1, g0, np.eye(2))
    np.testing.assert_allclose(sorted(info["kappa"]), [-1, 1])
    assert info["p"] == pytest.approx(1)


@pytest.mark.skipif("CAS1414_OUT" not in os.environ, reason="local data only")
def test_nto_lambdas_match_orca_n():
    """ORCA's printed NTO n is σ_k; ours agree within the CI truncation."""
    from dyson_orca_tools.io.orca_output import parse_orca_output, parse_nto_occupations

    out = os.environ["CAS1414_OUT"]
    cas = parse_orca_output(out)
    orca = parse_nto_occupations(out)
    mult = min(cas.mults)
    ci = {r.index: r.spin_ci for r in cas.roots if r.mult == mult}
    (root, _), n = max(
        ((k, v) for k, v in orca.items() if k[1] == mult), key=lambda kv: kv[1][0]
    )
    lam = nto_info(
        TransitionDensity(ci[0], ci[root], cas.norb).gamma(), np.eye(cas.norb)
    )["sigma"]
    print(f"mult {mult} root {root}: ours σ={lam[:3]}  ORCA n={n[:3]}")
    assert lam[0] == pytest.approx(n[0], rel=0.06)
