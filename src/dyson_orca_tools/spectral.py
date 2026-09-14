"""Multireference spectral function (Kumar et al., JACS 2025, 147, 24993, eq. 10).

    rho(w) = eta * sum_j |<Ψ_{±,j}| a / a† |Ψ_0>|² / ((w - E_j)² + eta²)

One peak per charged root j: its Dyson orbital, strength <ϱ|ϱ> and energy E_j
relative to the neutral ground state.
"""

from dataclasses import dataclass
import numpy as np

from .dyson import Dyson
from .io.params import Parameters

HARTREE_EV = 27.211386245988


@dataclass
class DysonPeak:
    label: str  # "-,0", "+,1" ... numbered by energy within each side
    side: str  # "-" removal (N-1), "+" addition (N+1)
    mult: int
    omega: float  # eV, relative to the neutral ground state
    strength: float
    coeff_ao: np.ndarray
    coeff_mo: np.ndarray  # d_p over the initial active MOs; Σ d_p² = strength
    mo_labels: list[str]  # HOMO-k / LUMO+k names of those MOs
    ci_norm: float  # Σ c² of the printed final-root CI vector (1 = nothing truncated)
    leading_det: str  # determinant with the largest |c| in the final root
    casci: bool  # True if the pair shared the active orbitals

    def composition(self, n: int = 3) -> list[tuple[str, float]]:
        """Largest MO weights d_p² / strength, e.g. [('HOMO', 0.97), ('HOMO-2', 0.02)]."""
        if self.strength == 0:
            return []
        w = self.coeff_mo**2 / self.strength
        top = np.argsort(w)[::-1][:n]
        return [(self.mo_labels[i], float(w[i])) for i in top if w[i] > 0.005]


def peak_energy(e_initial: float, e_final: float, removal: bool) -> float:
    """eV. Removal peaks sit at E_N - E_{N-1} < 0, addition at E_{N+1} - E_N."""
    de = (e_final - e_initial) * HARTREE_EV
    return -de if removal else de


def _relabel(peaks: list[DysonPeak]) -> list[DysonPeak]:
    """Number peaks 0, 1, ... by increasing |omega| within each side."""
    for side in ("-", "+"):
        same_side = sorted(
            (p for p in peaks if p.side == side), key=lambda p: abs(p.omega)
        )
        for j, p in enumerate(same_side):
            p.label = f"{side},{j}"
    return sorted(peaks, key=lambda p: p.omega)


def build_peaks(initial_data: dict, params: Parameters) -> list[DysonPeak]:
    """One Dyson orbital per charged root; requires energies on every state."""
    if params.initial.energy is None:
        raise ValueError("initial: 'energy' is required for the spectral function")

    peaks = []
    for i, run in enumerate(params.runs):
        if run.data is None:
            raise ValueError(f"final[{i}]: ORCA JSON not loaded")
        removal = params.is_removal(run)
        for j, root in enumerate(run.roots):
            if root.energy is None:
                raise ValueError(f"final[{i}].roots[{j}]: 'energy' is required")
            dyson = Dyson(initial_data, run.data, params.pair(run, root))
            coeff_mo = dyson.dyson_mo_coefficients()
            coeff_ao = dyson.dyson_orbital(coeff_mo)
            leading = max(root.spin_ci, key=lambda k: abs(root.spin_ci[k]))
            peaks.append(
                DysonPeak(
                    label="",
                    side="-" if removal else "+",
                    mult=run.mult,
                    omega=peak_energy(params.initial.energy, root.energy, removal),
                    strength=dyson.strength(coeff_ao),
                    coeff_ao=coeff_ao,
                    coeff_mo=coeff_mo,
                    mo_labels=dyson.active_labels(),
                    ci_norm=sum(c * c for c in root.spin_ci.values()),
                    leading_det=leading,
                    casci=dyson.calculation_is_casci(),
                )
            )
    return _relabel(peaks)


def spectral_function(
    peaks: list[DysonPeak], omega: np.ndarray, eta: float = 0.05
) -> np.ndarray:
    """rho(omega) on a grid (eV); each peak integrates to pi * strength."""
    omega = np.asarray(omega, dtype=float)
    rho = np.zeros_like(omega)
    for p in peaks:
        rho += eta * p.strength / ((omega - p.omega) ** 2 + eta**2)
    return rho


def omega_grid(
    peaks: list[DysonPeak], pad: float = 1.0, npts: int = 2000
) -> np.ndarray:
    lo = min(p.omega for p in peaks) - pad
    hi = max(p.omega for p in peaks) + pad
    return np.linspace(lo, hi, npts)
