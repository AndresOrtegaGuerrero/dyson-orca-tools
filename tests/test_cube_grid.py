import pytest
import typer

from dyson_orca_tools.cli.main import _cube_grid


def test_cube_grid_single_value_is_cubic():
    assert _cube_grid("80", None, 14.0) == {
        "nx": 80,
        "ny": 80,
        "nz": 80,
        "resolution": None,
        "margin": 14.0,
    }


def test_cube_grid_per_axis_and_spacing():
    grid = _cube_grid("40,50,60", 0.2, 5.0)
    assert (grid["nx"], grid["ny"], grid["nz"]) == (40, 50, 60)
    assert grid["resolution"] == 0.2
    assert grid["margin"] == 5.0


@pytest.mark.parametrize(
    "points, spacing, margin",
    [
        ("abc", None, 14.0),
        ("10,10", None, 14.0),
        ("0", None, 14.0),
        ("80", 0.0, 14.0),
        ("80", None, -1.0),
    ],
)
def test_cube_grid_rejects_bad_values(points, spacing, margin):
    with pytest.raises(typer.Exit):
        _cube_grid(points, spacing, margin)


def _h2_state():
    """H2 in a minimal s basis, enough for PySCF to build a molecule."""
    shell = {"Shell": "s", "Exponents": [1.0], "Coefficients": [1.0]}
    atoms = [
        {"ElementLabel": "H", "Coords": [0.0, 0.0, z], "Basis": [shell]}
        for z in (0.0, 0.74)
    ]
    return {
        "Molecule": {
            "Atoms": atoms,
            "Charge": 0,
            "Multiplicity": 1,
            "MolecularOrbitals": {"OrbitalLabels": ["0H 1s", "1H 1s"]},
        }
    }


def _header_counts(path):
    lines = path.read_text().splitlines()
    return [lines[i].split() for i in (3, 4, 5)]


def test_write_cube_uses_grid(tmp_path):
    pytest.importorskip("pyscf")
    from dyson_orca_tools.io.cube import write_cube

    out = tmp_path / "h2.cube"
    write_cube(_h2_state(), [1.0, 1.0], str(out), nx=10, ny=12, nz=14, margin=2.0)
    assert [int(r[0]) for r in _header_counts(out)] == [10, 12, 14]


def test_write_cube_spacing_overrides_points(tmp_path):
    pytest.importorskip("pyscf")
    from dyson_orca_tools.io.cube import write_cube

    out = tmp_path / "h2.cube"
    write_cube(_h2_state(), [1.0, 1.0], str(out), resolution=0.5, margin=2.0)
    rows = _header_counts(out)
    assert all(int(r[0]) != 80 for r in rows)
    for axis, row in enumerate(rows):  # PySCF rounds the step to fit the box
        assert float(row[1 + axis]) == pytest.approx(0.5, rel=0.2)
