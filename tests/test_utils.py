import pytest
import typer

from dyson_orca_tools.utils import sannity_check
from conftest import fake_state


def test_sannity_check_ignores_json_multiplicity():
    sannity_check(fake_state(4, 1, 0), fake_state(4, 4, -1))


def test_sannity_check_still_rejects_wrong_charge():
    with pytest.raises(typer.Exit):
        sannity_check(fake_state(4, 1, 0), fake_state(4, 2, -2))
