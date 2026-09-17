import pytest

from dyson_orca_tools.io.orca_output import (
    parse_orca_output,
    runs_from_output,
    initial_from_output,
)

FAKE_OUT = """\
| 12>   Multiplicity           Mult            ....    2
| 13>   Total Charge           Charge          ....   -1
  Number of active electrons           ...    5
  Number of active orbitals            ...    4
---------------------------------------------
CAS-SCF STATES FOR BLOCK  1 MULT= 2 NROOTS= 2
---------------------------------------------
ROOT   0:  E=    -230.4901234 Eh
      0.98776 [    3]: 22u0
ROOT   1:  E=    -230.4012345 Eh
      0.96040 [    5]: 2u20
---------------------------------------------
CAS-SCF STATES FOR BLOCK  2 MULT= 4 NROOTS= 1
---------------------------------------------
ROOT   0:  E=    -230.3000000 Eh
      1.00000 [    0]: 2uuu
---------------------------------------------
Spin-Determinant CI Printing
---------------------------------------------
CAS-SCF STATES FOR BLOCK  1 MULT= 2 NROOTS= 2
ROOT   0:  E=    -230.4901234 Eh
      [22u0]     0.993890846
      [20u2]    -0.052026571
ROOT   1:  E=    -230.4012345 Eh
      [2u20]     0.980000000
CAS-SCF STATES FOR BLOCK  2 MULT= 4 NROOTS= 1
ROOT   0:  E=    -230.3000000 Eh
      [2uuu]     1.000000000
---------------------------------------------
DENSITY MATRIX
---------------------------------------------
"""


@pytest.fixture
def out_file(tmp_path):
    p = tmp_path / "anion.out"
    p.write_text(FAKE_OUT)
    return p


def test_parse_info_and_roots(out_file):
    cas = parse_orca_output(out_file)
    assert (cas.nelc, cas.norb, cas.mult, cas.charge) == (5, 4, 2, -1)
    assert cas.mults == [2, 4]
    r0, r1, q0 = cas.roots
    assert r0.spin_ci == {"[22u0]": 0.993890846, "[20u2]": -0.052026571}
    assert r0.energy == pytest.approx(-230.4901234)
    assert r1.spin_ci == {"[2u20]": 0.98}
    assert (q0.mult, q0.energy) == (4, pytest.approx(-230.3))


def test_runs_one_per_multiplicity(out_file):
    runs = runs_from_output(parse_orca_output(out_file), "anion.json")
    assert [r["mult"] for r in runs] == [2, 4]
    assert len(runs[0]["roots"]) == 2 and len(runs[1]["roots"]) == 1
    assert runs[0]["file"] == "anion.json"


def test_initial_picks_root(out_file):
    cas = parse_orca_output(out_file)
    assert initial_from_output(cas, root=1)["energy"] == pytest.approx(-230.4012345)
    with pytest.raises(ValueError):
        initial_from_output(cas, root=5)


def test_missing_det_block(tmp_path):
    p = tmp_path / "bad.out"
    p.write_text("ROOT   0:  E=    -1.0 Eh\n")
    with pytest.raises(ValueError, match="PrintWF det"):
        parse_orca_output(p)
