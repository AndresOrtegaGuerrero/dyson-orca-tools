import numpy as np
import pytest
import copy

from dyson_orca_tools.io.params import parse_parameters
from dyson_orca_tools.spectral import (
    HARTREE_EV,
    build_peaks,
    omega_grid,
    peak_energy,
    spectral_function,
    DysonPeak,
    midgap,
    shift_peaks,
)
from conftest import load_params


def test_peak_energy_signs():
    assert peak_energy(-10.0, -9.5, removal=True) == pytest.approx(-0.5 * HARTREE_EV)
    assert peak_energy(-10.0, -9.5, removal=False) == pytest.approx(0.5 * HARTREE_EV)


def test_lorentzian_area_is_pi_times_strength(readme_v2):
    initial, params = load_params(readme_v2)
    peaks = build_peaks(initial, params)
    omega = np.linspace(-200, 200, 400_001)
    area = np.trapezoid(spectral_function(peaks, omega, eta=0.05), omega)
    assert area == pytest.approx(np.pi * sum(p.strength for p in peaks), rel=1e-3)


def test_readme_strength_and_labels(readme_v2):
    initial, params = load_params(readme_v2)
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
    initial, params = load_params(readme_v2)
    peaks = build_peaks(initial, params)
    assert [p for p in peaks if p.side == "-"][0].strength == pytest.approx(
        0.0, abs=1e-12
    )
    kept = build_peaks(initial, params, min_strength=1e-8)
    assert not [p for p in kept if p.mult == 4]


def test_non_orthogonal_pair_takes_casscf_branch(readme_v2):
    initial, params = load_params(
        readme_v2, seed_final=1
    )  # different MO set for the charged runs
    peaks = build_peaks(initial, params)
    assert not any(p.casci for p in peaks)
    assert all(0.0 <= p.strength <= 1.0 + 1e-12 for p in peaks)


def test_omega_grid_pads_both_sides(readme_v2):
    initial, params = load_params(readme_v2)
    peaks = build_peaks(initial, params)
    grid = omega_grid(peaks, pad=1.0, npts=11)
    assert grid[0] == pytest.approx(min(p.omega for p in peaks) - 1.0)
    assert grid[-1] == pytest.approx(max(p.omega for p in peaks) + 1.0)


def test_missing_energy_is_rejected(readme_v2):
    del readme_v2["parameters"]["final"][0]["roots"][1]["energy"]
    initial, params = load_params(readme_v2)
    with pytest.raises(ValueError, match="energy"):
        build_peaks(initial, params)


def test_readme_v2_parses(readme_v2):
    params = parse_parameters(readme_v2)
    assert params.initial.mult == 1
    assert [run.mult for run in params.runs] == [2, 2]


def test_determinant_ms_must_match_mult(readme_v2):
    raw = copy.deepcopy(readme_v2)
    raw["parameters"]["final"][0]["mult"] = 4  # doublet strings declared quartet
    with pytest.raises(ValueError, match="2m_s"):
        parse_parameters(raw)


def test_two_initials_give_two_spectra(readme_v3):
    initial, params = load_params(readme_v3)
    peaks = build_peaks(initial, params)
    by_init = {lab: [p for p in peaks if p.initial == lab] for lab in ("m1", "T1")}
    assert len(by_init["m1"]) == len(by_init["T1"]) == 3  # 3 roots in the fixture
    assert [p.label for p in by_init["m1"] if p.side == "-"] == ["-,0"]
    assert [p.label for p in by_init["T1"] if p.side == "-"] == ["-,0"]


def test_single_initial_peaks_unchanged(readme_v2):
    # labels and energies exactly as before PR 2
    initial, params = load_params(readme_v2)
    labels = [(p.initial, p.label) for p in build_peaks(initial, params)]
    assert labels == [("m1", "-,0"), ("m1", "+,0"), ("m1", "+,1")]


def test_omega_is_relative_to_each_initial(readme_v3):
    initial, params = load_params(readme_v3)
    peaks = build_peaks(initial, params)
    anion0 = {p.initial: p.omega for p in peaks if p.side == "+" and p.label == "+,0"}
    # T1 sits 0.1 Eh above S0 in the fixture, so its addition peak is 0.1 Eh lower
    assert anion0["m1"] - anion0["T1"] == pytest.approx(0.1 * HARTREE_EV)


def _peak(side, omega, initial="g"):
    return DysonPeak(
        label="",
        initial=initial,
        side=side,
        mult=2,
        omega=omega,
        strength=1.0,
        coeff_ao=np.zeros(1),
        coeff_mo=np.zeros(1),
        mo_labels=["HOMO"],
        ci_norm=1.0,
        leading_det="",
        casci=True,
    )


def test_midgap_uses_gap_edges():
    peaks = [_peak("-", -9.0), _peak("-", -7.0), _peak("+", -1.0), _peak("+", 2.0)]
    assert midgap(peaks, "g") == pytest.approx(-4.0)  # (−7 + −1)/2


def test_midgap_ignores_other_references():
    peaks = [_peak("-", -7.0), _peak("+", -1.0), _peak("-", -3.0, initial="x")]
    assert midgap(peaks, "g") == pytest.approx(-4.0)


def test_midgap_needs_both_sides():
    with pytest.raises(ValueError):
        midgap([_peak("-", -7.0)], "g")


def test_shift_peaks_per_side():
    peaks = [_peak("-", -7.0), _peak("+", -1.0)]
    shift_peaks(peaks, mu=-4.0, shift=0.1, shift_removal=0.5, shift_addition=-0.5)
    assert [p.omega for p in peaks] == pytest.approx([-2.4, 2.6])
