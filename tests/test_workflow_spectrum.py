from conftest import load_params
from dyson_orca_tools.workflows.spectrum import run_spectrum, SpectrumOptions
import pytest


def test_run_spectrum_leaves_and_layout(readme_v3):
    initial, params = load_params(readme_v3)
    res = run_spectrum(initial, params, SpectrumOptions())
    assert {str(leaf.subdir) for leaf in res.leaves} >= {"m1", "T1"}  # mirrors test_cli
    assert all(len(leaf.omega) == 2000 for leaf in res.leaves)
    assert res.n_dropped == 0 and res.empty == []


def test_run_spectrum_align_fermi_moves_zero(readme_v2):
    initial, params = load_params(readme_v2)
    plain = run_spectrum(initial, params, SpectrumOptions())
    fermi = run_spectrum(initial, params, SpectrumOptions(align_fermi=True))
    assert fermi.mu != 0.0
    assert all(p.omega < 0 for p in fermi.peaks if p.side == "-")
    assert all(p.omega > 0 for p in fermi.peaks if p.side == "+")
    moved = {
        (p.side, p.label): q.omega - p.omega for p, q in zip(plain.peaks, fermi.peaks)
    }
    assert all(d == pytest.approx(-fermi.mu) for d in moved.values())  # rigid shift
