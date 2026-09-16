"""Transition dipole from the AO transition density and orca_2json dipole integrals (a.u.)."""

import numpy as np

from .ehmap import _projector


def ao_dipole(state: dict) -> np.ndarray:
    """(3, nbas, nbas) integrals; needs "1elPropertyIntegrals": ["dipole"] in mol.json.conf."""
    d = state["Molecule"].get("dipole")
    if d is None:
        raise KeyError(
            'no "dipole" in the JSON: add "1elPropertyIntegrals": ["dipole"] '
            "to mol.json.conf and rerun orca_2json"
        )
    return np.asarray(d, dtype=float)


def transition_dipole(d_ao: np.ndarray, dip: np.ndarray) -> np.ndarray:
    """μ = -Σ_μν D_μν d_μν per axis (electron charge -1)."""
    return -np.einsum("ij,kij->k", d_ao, dip)


def fragment_contributions(
    d_ao: np.ndarray, dip: np.ndarray, frag: np.ndarray
) -> np.ndarray:
    """(nfrag, 3): μ_A = -½ Σ_{μ∈A} [(D d)_μμ + (d D)_μμ]; rows sum to transition_dipole."""
    per_ao = -0.5 * (
        np.einsum("ij,kji->ki", d_ao, dip) + np.einsum("kij,ji->ki", dip, d_ao)
    )
    return (per_ao @ _projector(frag)).T


def fragment_charges(d_ao: np.ndarray, s: np.ndarray, frag: np.ndarray) -> np.ndarray:
    """Mulliken transition charge per fragment, ½ Σ_{μ∈A} [(DS)_μμ + (SD)_μμ]; sums to 0 for I ≠ J."""
    per_ao = 0.5 * (np.einsum("ij,ji->i", d_ao, s) + np.einsum("ij,ji->i", s, d_ao))
    return per_ao @ _projector(frag)


def centre_origin(dip: np.ndarray, s: np.ndarray, state: dict) -> np.ndarray:
    """Dipole integrals with the origin moved to the mean atomic position (bohr): d' = d - R·S."""
    xyz = np.array([a["Coords"] for a in state["Molecule"]["Atoms"]])
    if state["Molecule"].get("CoordinateUnits", "Angs").lower().startswith("ang"):
        xyz = xyz / 0.529177210903
    r = xyz.mean(axis=0) - np.asarray(state["Molecule"].get("Origin", [0.0, 0.0, 0.0]))
    return dip - r[:, None, None] * s
