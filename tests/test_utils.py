import pytest

from dyson_orca_tools.utils import sanity_check
from conftest import fake_state


def test_sanity_check_ignores_json_multiplicity():
    sanity_check(fake_state(4, 1, 0), fake_state(4, 4, -1))


def test_sanity_check_still_rejects_wrong_charge():
    with pytest.raises(ValueError):
        sanity_check(fake_state(4, 1, 0), fake_state(4, 2, -2))


def test_utils_does_not_import_typer():
    import dyson_orca_tools.utils as u

    assert "typer" not in vars(u)
