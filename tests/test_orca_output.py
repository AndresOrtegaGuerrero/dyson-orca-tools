import pytest

from dyson_orca_tools.io.orca_output import (
    build_parameters,
    ground_state,
    initial_from_output,
    parse_orca_output,
    runs_from_output,
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
    assert initial_from_output(cas, mult=2, root=1)["energy"] == pytest.approx(
        -230.4012345
    )
    with pytest.raises(ValueError):
        initial_from_output(cas, mult=2, root=5)


# ORCA 6 layout: blocks from the highest multiplicity down, each CSF listing followed
# by its own determinant section (no BLOCK header inside the determinant section).
FAKE_OUT_PER_BLOCK = """\
| 12>   Multiplicity           Mult            ....    2
| 13>   Total Charge           Charge          ....   -1
  Number of active electrons           ...    5
  Number of active orbitals            ...    4
---------------------------------------------
CAS-SCF STATES FOR BLOCK  0 MULT= 4 NROOTS= 1
---------------------------------------------
ROOT   0:  E=    -230.3000000 Eh
      1.00000 [    0]: 2uuu
----------------------------------------
  Spin-Determinant CI Printing
----------------------------------------
ROOT   0:  E=    -230.3000000 Eh
      [2uuu]     1.000000000
---------------------------------------------
CAS-SCF STATES FOR BLOCK  1 MULT= 2 NROOTS= 2
---------------------------------------------
ROOT   0:  E=    -230.4901234 Eh
      0.98776 [    3]: 22u0
ROOT   1:  E=    -230.4012345 Eh
      0.96040 [    5]: 2u20
----------------------------------------
  Spin-Determinant CI Printing
----------------------------------------
ROOT   0:  E=    -230.4901234 Eh
      [22u0]     0.993890846
      [20u2]    -0.052026571
ROOT   1:  E=    -230.4012345 Eh
      [2u20]     0.980000000
---------------------------------------------
SA-CASSCF TRANSITION ENERGIES
---------------------------------------------
"""


def test_per_block_layout_has_no_ghost_roots(tmp_path):
    p = tmp_path / "ion.out"
    p.write_text(FAKE_OUT_PER_BLOCK)
    cas = parse_orca_output(p)
    assert cas.mults == [2, 4]
    assert [(r.mult, r.index, len(r.spin_ci)) for r in cas.roots] == [
        (4, 0, 1),
        (2, 0, 2),
        (2, 1, 1),
    ]
    assert cas.roots[1].energy == pytest.approx(-230.4901234)
    assert ground_state(cas) == (2, 0)


def test_missing_det_block(tmp_path):
    p = tmp_path / "bad.out"
    p.write_text("ROOT   0:  E=    -1.0 Eh\n")
    with pytest.raises(ValueError, match="PrintWF det"):
        parse_orca_output(p)


def test_ground_state_picks_lowest_energy_across_blocks(out_file):
    cas = parse_orca_output(out_file)
    assert ground_state(cas) == (2, 0)  # -230.490 beats the quartet's -230.300


def test_initial_from_output_records_root_and_label(out_file):
    cas = parse_orca_output(out_file)
    d = initial_from_output(cas, mult=4, root=0)
    assert (d["mult"], d["root"], d["label"]) == (4, 0, "m4r0")


def test_build_parameters_writes_initial_list(out_file):
    cas = parse_orca_output(out_file)
    raw = build_parameters(cas, [], initial_states=[(2, 0), (4, 0)])
    assert [s["label"] for s in raw["parameters"]["initial"]] == ["m2r0", "m4r0"]
    assert build_parameters(cas, [], [])["parameters"]["initial"][0]["label"] == "m2r0"
