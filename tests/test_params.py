import copy

import pytest

from dyson_orca_tools.io.params import parse_parameters

TRIPLET = {"[2uu0]": 0.9, "[u2u0]": -0.1}


def _two_initials(readme_v2):
    raw = copy.deepcopy(readme_v2)
    singlet = raw["parameters"]["initial"]
    triplet = {
        **singlet,
        "mult": 3,
        "energy": -230.4,
        "spin_ci": TRIPLET,
        "label": "T1",
    }
    raw["parameters"]["initial"] = [singlet, triplet]
    return raw


def test_v2_dict_initial_becomes_one_state(readme_v2):
    params = parse_parameters(readme_v2)
    assert len(params.initials) == 1 and params.initial is params.initials[0]
    assert params.initial.label == "m1"


def test_v3_list_of_initials(readme_v2):
    params = parse_parameters(_two_initials(readme_v2))
    assert [s.label for s in params.initials] == ["m1", "T1"]
    run, root = params.runs[0], params.runs[0].roots[0]
    pair = params.pair(run, root, params.initials[1])
    assert pair["parameters"]["initial"]["mult"] == 3
    assert pair["parameters"]["initial"]["spin_ci"] == TRIPLET


def test_pair_defaults_to_first_initial(readme_v2):
    params = parse_parameters(_two_initials(readme_v2))
    run, root = params.runs[0], params.runs[0].roots[0]
    assert params.pair(run, root)["parameters"]["initial"]["mult"] == 1


def test_initials_must_share_active_space(readme_v2):
    raw = _two_initials(readme_v2)
    raw["parameters"]["initial"][1]["nelc"] = 6
    with pytest.raises(ValueError, match="nelc"):
        parse_parameters(raw)


def test_initial_labels_must_be_unique(readme_v2):
    raw = _two_initials(readme_v2)
    raw["parameters"]["initial"][1]["label"] = "m1"
    with pytest.raises(ValueError, match="label"):
        parse_parameters(raw)


def test_parity_checked_against_every_initial(readme_v2):
    raw = _two_initials(readme_v2)
    raw["parameters"]["final"][1].update(
        mult=3, roots=[{"energy": -230.2, "spin_ci": {"[uu00]": 1.0}}]
    )
    with pytest.raises(ValueError, match="parity"):
        parse_parameters(raw)
