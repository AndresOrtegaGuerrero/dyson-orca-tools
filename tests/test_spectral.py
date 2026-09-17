import numpy as np
import pytest

from dyson_orca_tools.io.params import parse_parameters
from dyson_orca_tools.spectral import (
    HARTREE_EV,
    build_peaks,
    omega_grid,
    peak_energy,
    spectral_function,
)
from conftest import fake_state


def _load(raw, seed_final=0, rotate=None):
    params = parse_parameters(raw)
    for run in params.runs:
        run.data = fake_state(4, run.mult, 0, seed=seed_final, rotate_active=rotate)
    return fake_state(4, params.initial.mult, 0), params


def test_peak_energy_signs():
    assert peak_energy(-10.0, -9.5, removal=True) == pytest.approx(-0.5 * HARTREE_EV)
    assert peak_energy(-10.0, -9.5, removal=False) == pytest.approx(0.5 * HARTREE_EV)


def test_lorentzian_area_is_pi_times_strength(readme_v2):
    initial, params = _load(readme_v2)
    peaks = build_peaks(initial, params)
    omega = np.linspace(-200, 200, 400_001)
    area = np.trapezoid(spectral_function(peaks, omega, eta=0.05), omega)
    assert area == pytest.approx(np.pi * sum(p.strength for p in peaks), rel=1e-3)


def test_readme_strength_and_labels(readme_v2):
    initial, params = _load(readme_v2)
    peaks = build_peaks(initial, params)
    by_label = {p.label: p for p in peaks}
    assert set(by_label) == {"+,0", "+,1", "-,0"}
    assert by_label["+,0"].strength == pytest.approx(0.894872, abs=1e-5)
    assert all(0.0 <= p.strength <= 1.0 + 1e-12 for p in peaks)
    assert all(p.casci for p in peaks)
    assert [p.omega for p in peaks] == sorted(p.omega for p in peaks)


def test_forbidden_spin_has_zero_strength(readme_v2):
    readme_v2["parameters"]["final"][1] = {
        "nelc": 3,
        "norb": 4,
        "mult": 4,  # quartet from a singlet: ΔS = 3/2
        "roots": [{"energy": -230.2, "spin_ci": {"[uuu0]": 1.0}}],
    }
    initial, params = _load(readme_v2)
    peaks = build_peaks(initial, params)
    assert [p for p in peaks if p.side == "-"][0].strength == pytest.approx(
        0.0, abs=1e-12
    )


def test_non_orthogonal_pair_takes_casscf_branch(readme_v2):
    initial, params = _load(
        readme_v2, seed_final=1
    )  # different MO set for the charged runs
    peaks = build_peaks(initial, params)
    assert not any(p.casci for p in peaks)
    assert all(0.0 <= p.strength <= 1.0 + 1e-12 for p in peaks)


def test_omega_grid_pads_both_sides(readme_v2):
    initial, params = _load(readme_v2)
    peaks = build_peaks(initial, params)
    grid = omega_grid(peaks, pad=1.0, npts=11)
    assert grid[0] == pytest.approx(min(p.omega for p in peaks) - 1.0)
    assert grid[-1] == pytest.approx(max(p.omega for p in peaks) + 1.0)


def test_missing_energy_is_rejected(readme_v2):
    del readme_v2["parameters"]["final"][0]["roots"][1]["energy"]
    initial, params = _load(readme_v2)
    with pytest.raises(ValueError, match="energy"):
        build_peaks(initial, params)
