"""Fragment electron-hole map from an AO transition density (Plasser & Lischka J. Chem. Theory Comput. (2012) 8 (8): 2777–2789.)"""

import re
import numpy as np
import json

_LABEL = re.compile(r"^(\d+)[A-Za-z]+")  # '12C  1pz' -> 12


def ao_to_atom(state: dict) -> np.ndarray:
    """Return the atom index associated with each AO."""
    labels = state["Molecule"]["MolecularOrbitals"]["OrbitalLabels"]
    s = np.asarray(state["Molecule"]["S-Matrix"])

    atoms = []
    for label in labels:
        match = _LABEL.match(label)
        if match is None:
            raise ValueError(f"Could not determine atom index from AO label: {label!r}")
        atoms.append(int(match.group(1)))

    atoms = np.asarray(atoms, dtype=int)

    if s.shape != (len(atoms), len(atoms)):
        raise ValueError(
            f"S-matrix shape {s.shape} is inconsistent with {len(atoms)} AO labels"
        )

    return atoms


def fragment_of_ao(
    atom_map: np.ndarray, groups: list[list[int]] | None = None
) -> np.ndarray:
    """Fragment index per AO. groups = [[atoms of frag 0], [atoms of frag 1], ...]; default one atom = one fragment."""
    if groups is None:
        return atom_map
    frag_of_atom = {a: i for i, g in enumerate(groups) for a in g}
    missing = set(atom_map) - frag_of_atom.keys()
    if missing:
        raise ValueError(f"atoms not assigned to any fragment: {sorted(missing)}")
    return np.array([frag_of_atom[a] for a in atom_map])


def _atoms(spec) -> list[int]:
    """int, 'a-b' (inclusive) or a list of those -> atom indices."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [i for item in spec for i in _atoms(item)]


def load_fragments(path, state: dict):
    """(labels, groups) from {"index_base": 0|1, "fragments": {"porphyrin": ["1-24", 117], ...}}.
    index_base 1 = viewer numbering, 0 (default) = ORCA labels. Default: one fragment per atom."""
    atoms = state["Molecule"]["Atoms"]
    if path is None:
        return [f"{a['Idx']}{a['ElementLabel']}" for a in atoms], None
    with open(path) as f:
        spec = json.load(f)
    shift = spec.get("index_base", 0)
    groups = [[i - shift for i in _atoms(v)] for v in spec["fragments"].values()]
    return list(spec["fragments"]), groups


def active_mo_coefficients(state: dict, norb: int) -> np.ndarray:
    """AO × norb block of the active MOs (after the doubly occupied ones)."""
    mos = state["Molecule"]["MolecularOrbitals"]["MOs"]
    n_inactive = sum(mo["Occupancy"] == 2.0 for mo in mos)
    active = mos[n_inactive : n_inactive + norb]
    occ = [mo["Occupancy"] for mo in active]
    if len(active) != norb or not all(0.0 < o < 2.0 for o in occ):
        raise ValueError(f"active block does not look fractional: {occ}")
    return np.column_stack([mo["MOCoefficients"] for mo in active])


def _projector(frag: np.ndarray) -> np.ndarray:
    """0/1 matrix AO × fragment: x @ _projector(frag) sums an AO quantity per fragment."""
    p = np.zeros((len(frag), int(frag.max()) + 1))
    p[np.arange(len(frag)), frag] = 1.0
    return p


def omega_matrix(
    d_ao: np.ndarray,
    s: np.ndarray,
    frag: np.ndarray,
) -> np.ndarray:
    """Compute the fragment omega matrix.

    Implements the omega expression given in Appendix B of
    J. Chem. Phys. 141, 024106 (2014).
    """
    d_ao = np.asarray(d_ao)
    s = np.asarray(s)
    frag = np.asarray(frag, dtype=int)

    if frag.size == 0:
        raise ValueError("fragment map must contain at least one AO")

    if d_ao.shape != s.shape:
        raise ValueError(f"d_ao shape {d_ao.shape} does not match S shape {s.shape}")

    if d_ao.shape != (frag.size, frag.size):
        raise ValueError(
            f"d_ao shape {d_ao.shape} is inconsistent with {frag.size} AOs"
        )

    if np.any(frag < 0):
        raise ValueError("fragment indices must be non-negative")

    ds = d_ao @ s
    sd = s @ d_ao

    per_ao = 0.5 * (ds * sd + d_ao * (sd @ s))

    p = _projector(frag)
    return p.T @ per_ao @ p


def metrics(omega: np.ndarray) -> dict[str, float]:
    """Total Ω, charge-transfer number and participation ratio (NaN if Ω is exactly zero)."""
    total = float(omega.sum())
    denom = float((omega**2).sum())
    return {
        "Omega": total,
        "CT": float((total - np.trace(omega)) / total) if total else float("nan"),
        "PR": float(total**2 / denom) if denom else float("nan"),
    }


def plot_ehmap(
    omega: np.ndarray,
    labels: list[str],
    png: str,
    title: str = "",
    vmax: float | None = None,
) -> None:
    """Plot and save the fragment electron-hole omega map."""
    import matplotlib.pyplot as plt

    if omega.shape != (len(labels), len(labels)):
        raise ValueError(
            f"omega shape {omega.shape} is inconsistent with {len(labels)} labels"
        )

    if omega.size == 0:
        raise ValueError("omega matrix must not be empty")

    if vmax is None:
        vmax = float(omega.max())

    fig, ax = plt.subplots(figsize=(0.35 * len(labels) + 2, 0.35 * len(labels) + 2))

    im = ax.imshow(
        omega,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
    )

    ax.set_xlabel("Electron on")
    ax.set_ylabel("Hole on")

    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)

    fig.colorbar(im, ax=ax, label=r"$\Omega_{AB}$")
    ax.set_title(title)

    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def active_occupations(state: dict, norb: int) -> list[float]:
    """Occupation numbers of the active MOs, same slice as active_mo_coefficients."""
    mos = state["Molecule"]["MolecularOrbitals"]["MOs"]
    n_inactive = sum(mo["Occupancy"] == 2.0 for mo in mos)
    return [mo["Occupancy"] for mo in mos[n_inactive : n_inactive + norb]]


def orbital_fragment_populations(
    c: np.ndarray, s: np.ndarray, frag: np.ndarray
) -> np.ndarray:
    """Mulliken population of each MO (column of c) on each fragment: rows sum to 1."""
    per_ao = c * (s @ c)  # nbas × norb, C_μi (SC)_μi
    return per_ao.T @ _projector(frag)
