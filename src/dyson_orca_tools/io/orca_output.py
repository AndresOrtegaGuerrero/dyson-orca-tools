"""Read CI vectors, root energies and active-space info from an ORCA CASSCF/CASCI output.

Needs ``PrintWF det`` in ``%casscf`` (and a small ``TPrintWF`` so no determinant is
dropped). Expected block::

    Spin-Determinant CI Printing
    ...
    CAS-SCF STATES FOR BLOCK  1 MULT= 2 NROOTS= 3
    ROOT   0:  E=    -230.4901234 Eh
        [22u0]     0.993890846
        [20u2]    -0.052026571
    ROOT   1:  E=    ...
"""

from dataclasses import dataclass, field
from pathlib import Path
import re

_RE_ROOT = re.compile(r"^ROOT\s+(\d+):(?:\s+E=\s*(-?\d+\.\d+))?")
_RE_BLOCK = re.compile(r"BLOCK\s+\d+\s+MULT=\s*(\d+)")
_RE_DET = re.compile(r"^\[([2ud0]+)\]\s+(-?\d+\.\d+)")
_RE_INFO = {
    "norb": re.compile(r"Number of active orbitals\s+\.+\s+(\d+)"),
    "nelc": re.compile(r"Number of active electrons\s+\.+\s+(\d+)"),
    "mult": re.compile(r"Multiplicity\s+Mult\s+\.+\s+(\d+)"),
    "charge": re.compile(r"Total Charge\s+Charge\s+\.+\s+(-?\d+)"),
}


@dataclass
class OrcaRoot:
    index: int
    mult: int
    energy: float | None
    spin_ci: dict[str, float] = field(default_factory=dict)


@dataclass
class OrcaCasOutput:
    path: Path
    nelc: int | None = None
    norb: int | None = None
    mult: int | None = None  # multiplicity of the input (first block)
    charge: int | None = None
    roots: list[OrcaRoot] = field(default_factory=list)

    def roots_of_mult(self, mult: int) -> list[OrcaRoot]:
        return [r for r in self.roots if r.mult == mult]

    @property
    def mults(self) -> list[int]:
        return sorted({r.mult for r in self.roots})


def parse_orca_output(path: Path) -> OrcaCasOutput:
    out = OrcaCasOutput(Path(path))
    energies: dict[tuple[int, int], float] = {}  # (mult, root) -> E, from any ROOT line
    in_det_block, mult, root = False, None, None

    with open(path) as f:
        for raw in f:
            line = raw.strip()

            for key, rx in _RE_INFO.items():
                if getattr(out, key) is None and (m := rx.search(raw)):
                    setattr(out, key, int(m.group(1)))

            if m := _RE_BLOCK.search(line):
                mult = int(m.group(1))
                continue
            if m := _RE_ROOT.match(line):
                idx = int(m.group(1))
                if m.group(2) is not None and mult is not None:
                    energies[(mult, idx)] = float(m.group(2))
                if in_det_block and mult is not None:
                    root = OrcaRoot(idx, mult, energies.get((mult, idx)))
                    out.roots.append(root)
                continue

            if "Spin-Determinant CI Printing" in line:
                in_det_block, root = True, None
                continue
            if in_det_block and ("DENSITY MATRIX" in line or "TRANSITION" in line):
                in_det_block, root = False, None
                continue

            if in_det_block and root is not None and (m := _RE_DET.match(line)):
                root.spin_ci[f"[{m.group(1)}]"] = float(m.group(2))

    for r in out.roots:  # energy may have been printed only in the CSF block, earlier
        if r.energy is None:
            r.energy = energies.get((r.mult, r.index))
    if out.mult is None and out.roots:
        out.mult = out.roots[0].mult
    if not out.roots:
        raise ValueError(
            f"{path}: no 'Spin-Determinant CI Printing' block (add 'PrintWF det' to %casscf)"
        )
    return out


# ----------------------------------------------------- parameters v2 assembly
def _state_dict(cas: OrcaCasOutput, root: OrcaRoot) -> dict:
    return {
        "nelc": cas.nelc,
        "norb": cas.norb,
        "mult": root.mult,
        "energy": root.energy,
        "spin_ci": root.spin_ci,
    }


def initial_from_output(
    cas: OrcaCasOutput, root: int = 0, mult: int | None = None
) -> dict:
    mult = mult or cas.mult
    roots = cas.roots_of_mult(mult)
    if root >= len(roots):
        raise ValueError(f"{cas.path}: root {root} not found for mult {mult}")
    return _state_dict(cas, roots[root])


def runs_from_output(cas: OrcaCasOutput, json_file: str | Path) -> list[dict]:
    """One run per multiplicity block found in the output, all pointing to json_file."""
    return [
        {
            "file": str(json_file),
            "nelc": cas.nelc,
            "norb": cas.norb,
            "mult": m,
            "roots": [
                {"energy": r.energy, "spin_ci": r.spin_ci} for r in cas.roots_of_mult(m)
            ],
        }
        for m in cas.mults
    ]


def build_parameters(
    initial_out: Path, finals: list[tuple[Path, Path]], initial_root: int = 0
) -> dict:
    """finals: (orca .out, orca_2json .json) pairs for the N±1 calculations."""
    cas_initial = parse_orca_output(initial_out)
    runs = []
    for out_file, json_file in finals:
        runs.extend(runs_from_output(parse_orca_output(out_file), json_file))
    return {
        "parameters": {
            "initial": initial_from_output(cas_initial, initial_root),
            "final": runs,
        }
    }
