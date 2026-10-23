import copy
import json
from typer.testing import CliRunner

from dyson_orca_tools.cli.main import app
from conftest import fake_state


def _write_inputs(tmp_path, raw):
    ini = raw["parameters"]["initial"]
    nelc0 = (ini[0] if isinstance(ini, list) else ini)["nelc"]
    (tmp_path / "ref.json").write_text(json.dumps(fake_state(4, 1, 0)))
    for k, run in enumerate(raw["parameters"]["final"]):
        charge = nelc0 - run["nelc"]  # one more electron -> -1, one fewer -> +1
        (tmp_path / f"run{k}.json").write_text(
            json.dumps(fake_state(4, run["mult"], charge))
        )
        run["file"] = f"run{k}.json"
    (tmp_path / "params.json").write_text(json.dumps(raw))


def test_spectrum_one_folder_per_reference(tmp_path, readme_v3):
    _write_inputs(tmp_path, readme_v3)
    res = CliRunner().invoke(
        app,
        [
            "spectrum",
            "-i",
            str(tmp_path / "ref.json"),
            "-p",
            str(tmp_path / "params.json"),
            "-o",
            str(tmp_path / "out"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "out" / "m1" / "spectral_function.dat").exists()
    assert (tmp_path / "out" / "T1" / "dyson_peaks.csv").exists()
    assert (
        "initial" in (tmp_path / "out" / "dyson_peaks.csv").read_text().splitlines()[0]
    )


def test_spectrum_single_reference_layout_unchanged(tmp_path, readme_v2):
    _write_inputs(tmp_path, readme_v2)
    res = CliRunner().invoke(
        app,
        [
            "spectrum",
            "-i",
            str(tmp_path / "ref.json"),
            "-p",
            str(tmp_path / "params.json"),
            "-o",
            str(tmp_path / "out"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "out" / "spectral_function.dat").exists()


def test_spectrum_folders_per_reference_and_mult(tmp_path, readme_v3):
    raw = copy.deepcopy(readme_v3)
    raw["parameters"][
        "final"
    ].append(  # quartet cation: forbidden from m1, allowed from T1
        {
            "nelc": 3,
            "norb": 4,
            "mult": 4,
            "roots": [{"energy": -230.1, "spin_ci": {"[uuu0]": 1.0}}],
        }
    )
    _write_inputs(tmp_path, raw)
    res = CliRunner().invoke(
        app,
        [
            "spectrum",
            "-i",
            str(tmp_path / "ref.json"),
            "-p",
            str(tmp_path / "params.json"),
            "-o",
            str(tmp_path / "out"),
        ],
    )
    assert res.exit_code == 0, res.output
    out = tmp_path / "out"
    assert (
        out / "m1" / "spectral_function.dat"
    ).exists()  # doublets only: no sub-folder
    assert not (out / "m1" / "m4").exists()
    assert (
        out / "T1" / "m2" / "dyson_peaks.csv"
    ).exists()  # one folder per ion multiplicity
    assert (out / "T1" / "m4" / "dyson_peaks.csv").exists()
    assert "initial" in (out / "dyson_peaks.csv").read_text().splitlines()[0]
