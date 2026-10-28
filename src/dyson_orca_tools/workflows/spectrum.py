"""spectrum command: peaks -> shifts -> one spectrum per (reference, ion multiplicity)"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..io.params import Parameters
from ..spectral import (
    DysonPeak,
    HARTREE_EV,
    build_peaks,
    midgap,
    omega_grid,
    relabel,
    shift_peaks,
    spectral_function,
)


@dataclass
class SpectrumOptions:
    eta: float = 0.05
    omega_min: float | None = None
    omega_max: float | None = None
    npts: int = 2000
    shift: float = 0.0
    shift_removal: float = 0.0
    shift_addition: float = 0.0
    align_fermi: bool = False
    common_reference: bool = False
    min_strength: float = 1e-8


@dataclass
class SpectrumLeaf:
    initial: str
    mult: int
    subdir: Path  # Path("") | Path("T1") | Path("T1/m2"), relative to output_dir
    peaks: list[DysonPeak]  # relabelled, this leaf only
    omega: np.ndarray
    rho: np.ndarray


@dataclass
class SpectrumResult:
    peaks: list[DysonPeak]  # every peak after shifts (for the all-references table)
    leaves: list[SpectrumLeaf]
    empty: list[str]  # references that lost all peaks to min_strength
    de_ref: dict[str, float]  # E_ref − E_0 in eV
    ground: str
    mu: float
    n_dropped: int
    zero_note: str


def spectrum_curve(peaks, opts: SpectrumOptions) -> tuple[np.ndarray, np.ndarray]:
    omegas = omega_grid(peaks, npts=opts.npts)
    lo = omegas[0] if opts.omega_min is None else opts.omega_min
    hi = omegas[-1] if opts.omega_max is None else opts.omega_max
    omega = np.linspace(lo, hi, opts.npts)
    return omega, spectral_function(peaks, omega, eta=opts.eta)


def run_spectrum(
    initial: dict, params: Parameters, opts: SpectrumOptions
) -> SpectrumResult:
    peaks = build_peaks(initial, params, min_strength=opts.min_strength)
    n_all = sum(len(r.roots) for r in params.runs) * len(params.initials)

    e0 = min(s.energy for s in params.initials)
    de_ref = {s.label: (s.energy - e0) * HARTREE_EV for s in params.initials}
    ground = min(params.initials, key=lambda s: s.energy).label
    if opts.common_reference:
        for p in peaks:
            p.omega += de_ref[p.initial]

    mu = midgap(peaks, ground) if opts.align_fermi else 0.0  # ValueError propagates
    shift_peaks(
        peaks,
        mu=mu,
        shift=opts.shift,
        shift_removal=opts.shift_removal,
        shift_addition=opts.shift_addition,
    )

    labels = [s.label for s in params.initials]
    leaves, empty = [], []
    for lab in labels:
        mine = [p for p in peaks if p.initial == lab]
        if not mine:
            empty.append(lab)
            continue
        ref_dir = Path("") if len(labels) == 1 else Path(lab)
        mults = sorted({p.mult for p in mine})
        for m in mults:
            subdir = ref_dir if len(mults) == 1 else ref_dir / f"m{m}"
            subset = relabel([p for p in mine if p.mult == m])  # in place
            omega, rho = spectrum_curve(subset, opts)
            leaves.append(SpectrumLeaf(lab, m, subdir, subset, omega, rho))

    zero_note = (
        f"mu = {mu:.4f} eV, shift = {opts.shift} eV, shift_removal = {opts.shift_removal} eV, "
        f"shift_addition = {opts.shift_addition} eV, common_reference = {opts.common_reference}"
    )
    return SpectrumResult(
        peaks, leaves, empty, de_ref, ground, mu, n_all - len(peaks), zero_note
    )
