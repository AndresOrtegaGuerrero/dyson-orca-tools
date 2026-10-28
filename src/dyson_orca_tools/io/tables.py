"""CSV tables written by the commands"""

import csv
from pathlib import Path


def write_peaks_csv(path: Path, peaks, with_initial: bool = False):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["label", "side", "mult", "omega_eV", "strength", "branch"]
        w.writerow((["initial"] if with_initial else []) + head)
        for p in peaks:
            row = [
                p.label,
                p.side,
                p.mult,
                f"{p.omega:.6f}",
                f"{p.strength:.6f}",
                "CASCI" if p.casci else "CASSCF",
            ]
            w.writerow(([p.initial] if with_initial else []) + row)


def write_composition_csv(path: Path, peaks):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "strength", "ci_norm", "leading_det", *peaks[0].mo_labels])
        for p in peaks:  # d_p² per initial active MO; the row sums to the strength
            w.writerow(
                [
                    p.label,
                    f"{p.strength:.6f}",
                    f"{p.ci_norm:.6f}",
                    p.leading_det,
                    *(f"{d * d:.6f}" for d in p.coeff_mo),
                ]
            )


def write_omega_csv(path: Path, om, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hole\\electron", *labels])
        for lab, row in zip(labels, om):
            w.writerow([lab, *(f"{x:.6e}" for x in row)])


def write_pair_csv(path: Path, summary, roots):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["I\\J", *roots])
        for i in roots:
            w.writerow(
                [
                    i,
                    *(
                        f"{summary[f'{min(i, j)}:{max(i, j)}']['Omega']:.4e}"
                        if i != j
                        else "-"
                        for j in roots
                    ),
                ]
            )


def write_mixing_csv(path: Path, u, e, e_orca):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "QD_root",
                "E_eigh_Eh",
                "E_orca_Eh",
                *(f"U2_CAS{J}" for J in range(u.shape[0])),
            ]
        )
        for K in range(u.shape[1]):
            w.writerow(
                [
                    K,
                    f"{e[K]:.6f}",
                    f"{e_orca[K]:.6f}",
                    *(f"{x:.4f}" for x in u[:, K] ** 2),
                ]
            )


def write_dipole_csv(path: Path, per_frag, q_frag, mu, labels):
    """Origin = molecular centre; a fragment's μ_A shifts by q_A·ΔR under an origin change."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fragment", "mu_x", "mu_y", "mu_z", "transition_charge"])
        for lab, row, q in zip(labels, per_frag, q_frag):
            w.writerow([lab, *(f"{x:+.6f}" for x in row), f"{q:+.4f}"])
        w.writerow(["total", *(f"{x:+.6f}" for x in mu), f"{q_frag.sum():+.4f}"])


def write_orbital_csv(path: Path, occ, pop, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["active_mo", "occupation", *labels])
        for i, (o, row) in enumerate(zip(occ, pop), start=1):
            w.writerow([i, f"{o:.4f}", *(f"{x:.4f}" for x in row)])
