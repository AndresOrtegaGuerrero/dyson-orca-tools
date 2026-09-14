"""Publication-style plot of the multireference spectral function (matplotlib, optional)."""

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Plotting needs matplotlib: pip install 'dyson-orca-tools[plot]'"
    ) from exc

COLOR = {"-": "#2a78d6", "+": "#eb6834"}  # removal (N-1) blue, addition (N+1) orange
INK = {"primary": "#1a1a19", "secondary": "#5c5b55", "grid": "#d9d8d2"}

STYLE = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK["secondary"],
    "axes.labelcolor": INK["primary"],
    "xtick.color": INK["secondary"],
    "ytick.color": INK["secondary"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


@dataclass
class PeakInfo:
    label: str
    side: str
    mult: int
    omega: float
    strength: float


def read_outputs(out_dir: Path):
    """dyson_peaks.csv + spectral_function.dat -> (peaks, omega, rho, eta)."""
    out_dir = Path(out_dir)
    with open(out_dir / "dyson_peaks.csv") as f:
        peaks = [
            PeakInfo(
                r["label"],
                r["side"],
                int(r["mult"]),
                float(r["omega_eV"]),
                float(r["strength"]),
            )
            for r in csv.DictReader(f)
        ]
    with open(out_dir / "spectral_function.dat") as f:
        header = f.readline()
    eta = float(header.split("eta =")[1].split()[0]) if "eta =" in header else None
    data = np.loadtxt(out_dir / "spectral_function.dat")
    return peaks, data[:, 0], data[:, 1], eta


def _spread(positions, min_gap, lo, hi):
    """Push sorted label positions apart to at least min_gap, staying inside [lo, hi]."""
    order = np.argsort(positions)
    pos = np.array(positions, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    pos = np.minimum(pos, hi)  # if we ran past the top, walk back down
    for i in range(len(pos) - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - min_gap)
    pos = np.maximum(pos, lo)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def _side_curve(peaks, side, omega, rho, eta):
    """ρ(ω) restricted to one side. Rebuilt from the peaks when η is known (sides may
    overlap in energy, e.g. a bound anion); otherwise split the total curve at ω = 0."""
    if eta:
        out = np.zeros_like(omega)
        for p in peaks:
            if p.side == side:
                out += eta * p.strength / ((omega - p.omega) ** 2 + eta**2)
        return out
    mask = omega <= 0 if side == "-" else omega >= 0
    return np.where(mask, rho, 0.0)


def plot_spectrum(
    peaks,
    omega,
    rho,
    path,
    eta=None,
    label_threshold=0.0,
    reference="E_0",
    title=None,
    orientation="horizontal",
    figsize=None,
):
    """ρ(ω) with removal side blue, addition side orange, peaks labelled ϱ±,j (strength).
    orientation='vertical' puts the energy on the y axis (as in the paper's figures)."""
    omega, rho = np.asarray(omega), np.asarray(rho)
    vertical = orientation == "vertical"
    figsize = figsize or ((2.6, 4.0) if vertical else (3.5, 2.5))
    ymax = rho.max()
    x_lab, y_lab = r"$\omega - " + reference + r"$ (eV)", r"$\rho_s(\omega)$"

    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        for side, name in (("-", r"$N-1$ (removal)"), ("+", r"$N+1$ (addition)")):
            rho_side = _side_curve(peaks, side, omega, rho, eta)
            if vertical:
                ax.fill_betweenx(
                    omega, rho_side, 0, color=COLOR[side], alpha=0.15, lw=0
                )
                ax.plot(rho_side, omega, color=COLOR[side], lw=1.2, label=name)
            else:
                ax.fill_between(omega, rho_side, 0, color=COLOR[side], alpha=0.15, lw=0)
                ax.plot(omega, rho_side, color=COLOR[side], lw=1.2, label=name)

        ref_line = ax.axhline if vertical else ax.axvline
        ref_line(0, color=INK["secondary"], lw=0.6, ls=(0, (3, 2)))
        if vertical:
            ax.set_xlabel(y_lab), ax.set_ylabel(x_lab)
            ax.set_ylim(omega[0], omega[-1]), ax.set_xlim(0, ymax * 1.9)
            ax.grid(axis="x", color=INK["grid"], lw=0.5)
        else:
            ax.set_xlabel(x_lab), ax.set_ylabel(y_lab)
            ax.set_xlim(omega[0], omega[-1]), ax.set_ylim(0, ymax * 1.75)
            ax.grid(axis="y", color=INK["grid"], lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_axisbelow(True)

        # tick marker on the energy axis at every root, so weak peaks stay locatable
        for p in peaks:
            if vertical:
                ax.plot(
                    0,
                    p.omega,
                    marker=4,
                    ms=4,
                    color=COLOR[p.side],
                    clip_on=False,
                    zorder=3,
                )
            else:
                ax.plot(
                    p.omega,
                    0,
                    marker=6,
                    ms=4,
                    color=COLOR[p.side],
                    clip_on=False,
                    zorder=3,
                )

        # direct labels on a common baseline beyond the tallest peak, thin leader to the tip
        leader = dict(arrowstyle="-", lw=0.4, color=INK["grid"], shrinkA=0, shrinkB=1)
        shown = [p for p in peaks if p.strength >= label_threshold]
        span = omega[-1] - omega[0]
        gap = span * (
            0.045 if vertical else 0.035
        )  # ~ one label height/width in energy units
        slots = _spread(
            [p.omega for p in shown], gap, omega[0] + gap / 2, omega[-1] - gap / 2
        )
        for p, slot in zip(shown, slots):
            side, j = p.label.split(",")
            text = rf"$\varrho_{{{side},{j}}}$ ({p.strength:.2f})"
            height = np.interp(p.omega, omega, rho)
            if vertical:
                ax.annotate(
                    text,
                    xy=(height, p.omega),
                    xytext=(ymax * 1.12, slot),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color=INK["primary"],
                    arrowprops=leader,
                )
            else:
                ax.annotate(
                    text,
                    xy=(p.omega, height),
                    xytext=(slot, ymax * 1.12),
                    textcoords="data",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=INK["primary"],
                    arrowprops=leader,
                )

        ax.legend(loc="center right" if vertical else "upper center", handlelength=1.2)
        if eta:
            ax.text(
                0.99,
                0.01 if vertical else 0.98,
                rf"$\eta$ = {eta:g} eV",
                transform=ax.transAxes,
                ha="right",
                va="bottom" if vertical else "top",
                fontsize=7,
                color=INK["secondary"],
            )
        if title:
            ax.set_title(title, fontsize=9, color=INK["primary"], loc="left")

        path = Path(path)
        fig.savefig(path)
        if path.suffix.lower() != ".pdf":
            fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    return path
