import numpy as np
import pytest
from dyson_orca_tools.ehmap import (
    ao_to_atom,
    fragment_of_ao,
    omega_matrix,
    metrics,
    active_mo_coefficients,
    load_fragments,
)
from conftest import fake_state


def test_omega_sum_rule():
    """Σ_AB Ω_AB must equal ‖γ‖²_F when the MOs are S-orthonormal."""
    st = fake_state(6, 1, 0)
    c = np.array(
        [mo["MOCoefficients"] for mo in st["Molecule"]["MolecularOrbitals"]["MOs"]]
    ).T
    s = np.array(st["Molecule"]["S-Matrix"])
    g = np.random.default_rng(1).normal(size=(4, 4))  # any 4×4 "γ" on active MOs 1..4
    d = c[:, 1:5] @ g @ c[:, 1:5].T
    frag = np.array([0, 0, 1, 1, 2, 2])
    om = omega_matrix(d, s, frag)
    assert om.sum() == pytest.approx((g**2).sum())
    assert metrics(om)["Omega"] == pytest.approx((g**2).sum())


def test_fragment_grouping_and_labels():
    st = {
        "Molecule": {
            "S-Matrix": np.eye(5).tolist(),
            "MolecularOrbitals": {
                "OrbitalLabels": ["0C 1s", "0C 2s", "1H 1s", "2H 1s", "2H 2s"]
            },
        }
    }
    atoms = ao_to_atom(st)
    assert atoms.tolist() == [0, 0, 1, 2, 2]
    assert fragment_of_ao(atoms, [[0], [1, 2]]).tolist() == [0, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        fragment_of_ao(atoms, [[0, 1]])  # atom 2 unassigned


def test_local_excitation_is_diagonal():
    """γ on one fragment's own orbitals → Ω lives on that fragment's diagonal block."""
    st = fake_state(4, 1, 0)
    c = np.array(
        [mo["MOCoefficients"] for mo in st["Molecule"]["MolecularOrbitals"]["MOs"]]
    ).T
    s = np.array(st["Molecule"]["S-Matrix"])
    g = np.zeros((4, 4))
    g[1, 0] = 1.0  # MO 0 → MO 1
    om = omega_matrix(c @ g @ c.T, s, np.arange(4))
    assert om.sum() == pytest.approx(1.0)


def test_active_block_and_default_fragments():
    st = fake_state(5, 1, 0)
    for k, mo in enumerate(st["Molecule"]["MolecularOrbitals"]["MOs"]):
        mo["Occupancy"] = [2.0, 2.0, 1.3, 0.7, 0.0][k]
    st["Molecule"]["Atoms"] = [
        {"Idx": 0, "ElementLabel": "C"},
        {"Idx": 1, "ElementLabel": "H"},
    ]
    assert active_mo_coefficients(st, 2).shape == (5, 2)
    with pytest.raises(ValueError):
        active_mo_coefficients(st, 3)  # would include the empty MO
    assert load_fragments(None, st) == (["0C", "1H"], None)


def test_load_fragments_index_base(tmp_path):
    st = {"Molecule": {"Atoms": [{"Idx": i, "ElementLabel": "C"} for i in range(6)]}}
    p = tmp_path / "frags.json"
    p.write_text('{"index_base": 1, "fragments": {"a": ["1-3"], "b": [4, "5-6"]}}')
    labels, groups = load_fragments(p, st)
    assert labels == ["a", "b"] and groups == [[0, 1, 2], [3, 4, 5]]
