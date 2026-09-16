"""QD-NEVPT2 state mixing: perturbed states as combinations of CASSCF roots."""

import numpy as np


def mixing_matrix(heff: np.ndarray):
    """(energies, U): eigenpairs of the symmetrized H_eff; column K of U = QD root K in CAS roots."""
    return np.linalg.eigh(0.5 * (heff + heff.T))


def overlap_matrix(cas_ci: dict[int, dict], qd_ci: dict[int, dict]) -> np.ndarray:
    """U_JK = <Ψ_J|Φ_K> from the printed (truncated) determinant vectors — check of mixing_matrix."""
    n = len(cas_ci)
    return np.array(
        [
            [
                sum(cas_ci[J].get(d, 0.0) * c for d, c in qd_ci[K].items())
                for K in range(n)
            ]
            for J in range(n)
        ]
    )


def composition(
    u: np.ndarray, threshold: float = 0.01
) -> list[list[tuple[int, float]]]:
    """Per QD root K: [(CAS root J, U_JK²), ...] above threshold, largest first."""
    return [
        sorted(
            [(J, float(w)) for J, w in enumerate(u[:, K] ** 2) if w > threshold],
            key=lambda t: -t[1],
        )
        for K in range(u.shape[1])
    ]
