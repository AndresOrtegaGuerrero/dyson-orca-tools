import copy
import numpy as np
import pytest
from dyson_orca_tools.io.params import parse_parameters


def fake_state(
    nbas: int, mult: int, charge: int, seed: int = 0, rotate_active=None
) -> dict:
    """Minimal ORCA-like JSON: random SPD overlap S and MOs with C^T S C = 1."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nbas, nbas))
    S = A @ A.T + nbas * np.eye(nbas)
    C = np.linalg.inv(np.linalg.cholesky(S)).T
    if (
        rotate_active is not None
    ):  # orthogonal mix inside the active block -> "CASSCF" pair
        C = C @ rotate_active
    mos = [{"Occupancy": 0.0, "MOCoefficients": list(C[:, i])} for i in range(nbas)]
    return {
        "Molecule": {
            "Multiplicity": mult,
            "Charge": charge,
            "Atoms": [],
            "S-Matrix": S.tolist(),
            "MolecularOrbitals": {"MOs": mos},
        }
    }


README_INITIAL = {
    "[2200]": 0.957520133,
    "[2020]": -0.224387606,
    "[dduu]": -0.018039314,
    "[dudu]": -0.066820934,
    "[uddu]": 0.084860248,
    "[duud]": 0.084860248,
    "[udud]": -0.066820934,
    "[uudd]": -0.018039314,
    "[0202]": -0.063982267,
}
README_ANION = {
    "[02u2]": -0.051431580,
    "[du2u]": 0.055430255,
    "[ud2u]": -0.061033518,
    "[uu2d]": 0.005603263,
    "[20u2]": -0.052026571,
    "[22u0]": 0.993890846,
}

TRIPLET = {"[2uu0]": 0.9, "[u2u0]": -0.1}


@pytest.fixture
def readme_v2() -> dict:
    return {
        "parameters": {
            "initial": {
                "nelc": 4,
                "norb": 4,
                "mult": 1,
                "energy": -230.5,
                "spin_ci": README_INITIAL,
            },
            "final": [
                {
                    "nelc": 5,
                    "norb": 4,
                    "mult": 2,
                    "roots": [
                        {"energy": -230.45, "spin_ci": README_ANION},
                        {"energy": -230.30, "spin_ci": {"[2u20]": 1.0}},
                    ],
                },
                {
                    "nelc": 3,
                    "norb": 4,
                    "mult": 2,
                    "roots": [{"energy": -230.20, "spin_ci": {"[2u00]": 1.0}}],
                },
            ],
        }
    }


@pytest.fixture
def readme_v3(readme_v2) -> dict:
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


def load_params(raw, seed_final=0, rotate=None):
    params = parse_parameters(raw)
    for run in params.runs:
        run.data = fake_state(4, run.mult, 0, seed=seed_final, rotate_active=rotate)
    return fake_state(4, params.initial.mult, 0), params
