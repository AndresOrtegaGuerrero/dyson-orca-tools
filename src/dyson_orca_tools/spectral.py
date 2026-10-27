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
    initial: str  # label of the reference state this peak belongs to
    side: str  # "-" removal (N-1), "+" addition (N+1)
    mult: int
    omega: float  # eV, relative to the reference state of this peak
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


def relabel(peaks: list[DysonPeak]) -> list[DysonPeak]:
    """Number peaks 0, 1, ... by increasing energy of the N±1 state within each side
    and reference state: removal ω = E_N − E_{N−1} (descending ω), addition
    ω = E_{N+1} − E_N (ascending ω). Output keeps the reference order of the file."""
    for ini, side in {(p.initial, p.side) for p in peaks}:
        sign = -1 if side == "-" else 1
        same = sorted(
            (p for p in peaks if (p.initial, p.side) == (ini, side)),
            key=lambda p: sign * p.omega,
        )
        for j, p in enumerate(same):
            p.label = f"{side},{j}"

    order = {lab: k for k, lab in enumerate(dict.fromkeys(p.initial for p in peaks))}
    return sorted(peaks, key=lambda p: (order[p.initial], p.omega))


def build_peaks(
    initial_data: dict, params: Parameters, min_strength: float = 0.0
) -> list[DysonPeak]:
    """One Dyson orbital per charged root; requires energies on every state."""

    peaks = []

    for ini in params.initials:
        if ini.energy is None:
            raise ValueError(f"initial '{ini.label}': 'energy' is required")

        for i, run in enumerate(params.runs):
            if run.data is None:
                raise ValueError(f"final[{i}]: ORCA JSON not loaded")
            removal = params.is_removal(run)
            for j, root in enumerate(run.roots):
                if root.energy is None:
                    raise ValueError(f"final[{i}].roots[{j}]: 'energy' is required")
                dyson = Dyson(initial_data, run.data, params.pair(run, root, ini))
                coeff_mo = dyson.dyson_mo_coefficients()
                coeff_ao = dyson.dyson_orbital(coeff_mo)
                leading = max(root.spin_ci, key=lambda k: abs(root.spin_ci[k]))
                peaks.append(
                    DysonPeak(
                        label="",
                        initial=ini.label,
                        side="-" if removal else "+",
                        mult=run.mult,
                        omega=peak_energy(ini.energy, root.energy, removal),
                        strength=dyson.strength(coeff_ao),
                        coeff_ao=coeff_ao,
                        coeff_mo=coeff_mo,
                        mo_labels=dyson.active_labels(),
                        ci_norm=sum(c * c for c in root.spin_ci.values()),
                        leading_det=leading,
                        casci=dyson.calculation_is_casci(),
                    )
                )
    peaks = [p for p in peaks if p.strength >= min_strength]
    return relabel(peaks)


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


def midgap(peaks: list[DysonPeak], initial: str) -> float:
    """eV midgap between the highest removal and lowest addition peak for a given reference state."""
    removal = [p.omega for p in peaks if p.initial == initial and p.side == "-"]
    addition = [p.omega for p in peaks if p.initial == initial and p.side == "+"]
    if not removal or not addition:
        raise ValueError(f"midgap: no removal or addition peaks for '{initial}'")
    return 0.5 * (max(removal) + min(addition))


def shift_peaks(
    peaks: list[DysonPeak],
    mu: float = 0.0,
    shift: float = 0.0,
    shift_removal: float = 0.0,
    shift_addition: float = 0.0,
) -> None:
    """ω → ω − mu + shift + (shift_removal | shift_addition by side)."""
    per_side = {"-": shift_removal, "+": shift_addition}
    for p in peaks:
        p.omega = p.omega - mu + shift + per_side[p.side]
