"""NTOs (SVD of γ^{IJ}) and NDOs (eigenvectors of γ^{JJ} − γ^{II}) in the AO basis."""

from pathlib import Path

import numpy as np


def nto_info(gamma: np.ndarray, c_act: np.ndarray, omega: float | None = None) -> dict:
    """NTOs of the (spin-traced) γ. sigma = singular values (ORCA's NTO n); lam = NTO weights,
    σ² rescaled so that Σλ = omega (Plasser's λ_i in the spin-orbital convention) when omega is given;
    pr_nto = (Σλ)²/Σλ² (NTO pairs involved); hole/particle = AO coefficients, one column per pair."""
    u, s, vt = np.linalg.svd(gamma)
    lam = s**2
    if omega is not None and lam.sum() > 1e-14:
        lam = lam * (omega / lam.sum())
    pr = float(lam.sum() ** 2 / (lam**2).sum()) if lam.sum() > 1e-6 else float("nan")
    return {
        "sigma": s,
        "lam": lam,
        "pr_nto": pr,
        "hole": c_act @ vt.T,
        "particle": c_act @ u,
    }


def ndo_info(g_state: np.ndarray, g_ground: np.ndarray, c_act: np.ndarray) -> dict:
    """kappa < 0 detachment, > 0 attachment (sorted by |kappa|); p = promotion number."""
    kappa, w = np.linalg.eigh(g_state - g_ground)
    order = np.argsort(-np.abs(kappa))
    kappa, w = kappa[order], w[:, order]
    return {"kappa": kappa, "coeff": c_act @ w, "p": float(-kappa[kappa < 0].sum())}


def write_orbital_cubes(
    state: dict, coeffs: np.ndarray, names: list[str], outdir, **grid
):
    """One cube per column of coeffs (AO, ORCA order); needs the [cube] extra.

    ``grid`` (nx, ny, nz, resolution, margin) is passed on to ``write_cube``.
    """
    from .io.cube import write_cube  # ImportError -> caller warns and skips

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, name in enumerate(names):
        paths.append(outdir / f"{name}.cube")
        write_cube(state, coeffs[:, k], str(paths[-1]), **grid)
    return paths
