import json
import os
from pathlib import Path

import numpy as np
import pytest

from dyson_orca_tools.dipole import ao_dipole, transition_dipole, fragment_contributions
from conftest import fake_state


def test_fragment_contributions_sum_to_total():
    rng = np.random.default_rng(3)
    d_ao = rng.normal(size=(6, 6))
    dip = np.array(
        [(a := rng.normal(size=(6, 6))) + a.T for _ in range(3)]
    )  # symmetric
    per = fragment_contributions(d_ao, dip, np.array([0, 0, 1, 1, 2, 2]))
    np.testing.assert_allclose(per.sum(axis=0), transition_dipole(d_ao, dip))


def test_missing_dipole_raises():
    with pytest.raises(KeyError):
        ao_dipole(fake_state(3, 1, 0))


@pytest.mark.skipif("CAS1414_OUT" not in os.environ, reason="local cas1414 data only")
def test_report_against_orca_dipoles():
    """0→4: dominant y component within 5 % of ORCA's DY = -0.00696 (truncated CI vectors)."""
    from dyson_orca_tools.tdm import TransitionDensity
    from dyson_orca_tools.ehmap import active_mo_coefficients
    from dyson_orca_tools.io.orca_output import parse_orca_output

    out = Path(os.environ["CAS1414_OUT"])
    cas = parse_orca_output(out)
    st = json.load(open(out.with_name("mol.json")))
    ci = {r.index: r.spin_ci for r in cas.roots}
    td = TransitionDensity(ci[0], ci[4], cas.norb)
    d_ao = td.to_ao(active_mo_coefficients(st, cas.norb), td.gamma())
    mu = transition_dipole(d_ao, ao_dipole(st))
    print("0->4 mu:", mu)
    assert abs(mu[1]) == pytest.approx(0.00696, rel=0.05)
