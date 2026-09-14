import copy
import numpy as np
import pytest

from dyson_orca_tools.dyson import Dyson
from conftest import fake_state, README_INITIAL, README_ANION

N_ELEC = {"2": 2, "u": 1, "d": 1, "0": 0}


def _params(final_ci):
    return {
        "parameters": {
            "initial": {"nelc": 4, "norb": 4, "mult": 1, "spin_ci": README_INITIAL},
            "final": {"nelc": 5, "norb": 4, "mult": 2, "spin_ci": final_ci},
        }
    }


@pytest.fixture
def pair():
    return fake_state(4, 1, 0), fake_state(4, 2, -1)


def test_both_branches_agree_on_shared_orbitals(pair):
    d = Dyson(*pair, _params(README_ANION))
    assert d.calculation_is_casci()
    np.testing.assert_allclose(
        d.casci_dyson_coefficients(), d.dyson_coefficients(), atol=1e-12
    )


def test_strength_equals_sum_of_squares_for_orthonormal_mos(pair):
    d = Dyson(*pair, _params(README_ANION))
    coeff = d.casci_dyson_coefficients()
    assert d.strength(d.dyson_orbital()) == pytest.approx(float(coeff @ coeff))


@pytest.mark.parametrize("q", [0, 1, 2])
def test_dyson_orbital_invariant_under_active_orbital_swap(pair, q):
    """Swapping final orbitals q,q+1 (with the fermionic sign (-1)^(n_q n_{q+1}) on the
    CI coefficients) must leave the Dyson orbital unchanged: exercises the
    non-orthogonal branch, including the interleaved -> α-first reorder signs."""
    initial, final = pair
    reference = Dyson(initial, final, _params(README_ANION)).dyson_orbital()

    swapped = copy.deepcopy(final)
    mos = swapped["Molecule"]["MolecularOrbitals"]["MOs"]
    mos[q], mos[q + 1] = mos[q + 1], mos[q]
    ci = {}
    for key, c in README_ANION.items():
        s = list(key.strip("[]"))
        sign = (-1) ** (N_ELEC[s[q]] * N_ELEC[s[q + 1]])
        s[q], s[q + 1] = s[q + 1], s[q]
        ci["[" + "".join(s) + "]"] = sign * c

    d = Dyson(initial, swapped, _params(ci))
    assert not d.calculation_is_casci()
    np.testing.assert_allclose(d.dyson_orbital(), reference, atol=1e-12)


def test_spin_forbidden_gives_zero(pair):
    initial, final = pair
    final = copy.deepcopy(final)
    final["Molecule"]["Multiplicity"] = 4
    params = _params({"[uuu0]": 1.0})
    params["parameters"]["final"].update(nelc=3, mult=4)
    d = Dyson(initial, final, params)
    assert d.strength(d.dyson_orbital()) == 0.0


def test_mo_weights_sum_to_strength(pair):
    d = Dyson(*pair, _params(README_ANION))
    mo = d.dyson_mo_coefficients()
    assert float(mo @ mo) == pytest.approx(d.strength(d.dyson_orbital(mo)))
    assert d.active_labels() == ["HOMO-1", "HOMO", "LUMO", "LUMO+1"]
