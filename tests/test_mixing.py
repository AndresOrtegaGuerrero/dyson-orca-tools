import os

import numpy as np
import pytest

from dyson_orca_tools.mixing import mixing_matrix, overlap_matrix, composition
from dyson_orca_tools.io.orca_output import (
    detect_pt2,
    parse_pt2_energies,
    parse_qdnevpt2,
)

NEVPT2_TABLE = """ NEVPT2 TOTAL ENERGIES
-----------------------
STATE   ROOT MULT  Energy/a.u.   MRCI SOC BLOCK INPUT (Eh)
   0:    0    3  -100.500000    EDIAG[0]  -100.5
   1:    1    3  -100.400000    EDIAG[1]  -100.4
   2:    0    1  -100.450000    EDIAG[2]  -100.45
-----------------------------
 NEVPT2 TRANSITION ENERGIES
"""


def test_mixing_matrix_is_orthogonal_and_sorted():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(4, 4))
    e, u = mixing_matrix(
        h + h.T + 0.01 * rng.normal(size=(4, 4))
    )  # slightly non-Hermitian
    np.testing.assert_allclose(u.T @ u, np.eye(4), atol=1e-12)
    assert np.all(np.diff(e) >= 0)


def test_overlap_of_identical_vectors_is_norm():
    v = {0: {"[20]": 0.8, "[02]": -0.6}}
    assert overlap_matrix(v, v)[0, 0] == pytest.approx(1.0)


def test_composition_orders_by_weight():
    u = np.array([[0.6, 0.8], [0.8, -0.6]])
    assert composition(u)[0] == [(1, pytest.approx(0.64)), (0, pytest.approx(0.36))]


def test_detect_pt2():
    assert detect_pt2("") is None
    assert detect_pt2(NEVPT2_TABLE) == "NEVPT2"
    assert detect_pt2("QD-NEVPT2 Results\n" + NEVPT2_TABLE) == "QD-NEVPT2"


def test_parse_pt2_energies_keyed_by_mult_and_root():
    assert parse_pt2_energies(NEVPT2_TABLE, "NEVPT2") == {
        (3, 0): -100.5,
        (3, 1): -100.4,
        (1, 0): -100.45,
    }


def test_parse_qdnevpt2_empty_without_qd(tmp_path):
    p = tmp_path / "plain.out"
    p.write_text(NEVPT2_TABLE)
    assert parse_qdnevpt2(p) == {}


@pytest.mark.skipif("CAS1414_OUT" not in os.environ, reason="local cas1414 data only")
def test_report_qd_against_orca():
    from dyson_orca_tools.io.orca_output import parse_orca_output

    out = os.environ["CAS1414_OUT"]
    cas = parse_orca_output(out)
    for mult, qd in parse_qdnevpt2(out).items():
        e, u = mixing_matrix(qd.heff)
        np.testing.assert_allclose(
            e, [qd.energies[k] for k in range(len(e))], atol=2e-6
        )
        u_ov = overlap_matrix(
            {r.index: r.spin_ci for r in cas.roots if r.mult == mult},
            {r.index: r.spin_ci for r in qd.roots},
        )
        assert np.abs(np.abs(u) - np.abs(u_ov)).max() < 0.1
        print(f"mult {mult}:", composition(u))
