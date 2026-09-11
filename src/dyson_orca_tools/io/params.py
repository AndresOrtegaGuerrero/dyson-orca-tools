"""Parameters file: v1 (one charged state) and v2 (list of charged runs with roots).

v2 layout::

    {"parameters": {
        "initial": {"nelc": 4, "norb": 4, "mult": 1, "energy": -230.51, "spin_ci": {...}},
        "final": [
            {"file": "cation_d.json", "nelc": 3, "norb": 4, "mult": 2,
             "roots": [{"energy": -230.24, "spin_ci": {...}}, ...]},
            ...
        ]}}

v1 (``final`` is a dict with ``spin_ci``) is wrapped into one run with one root.
"""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class State:
    nelc: int
    norb: int
    mult: int
    spin_ci: dict
    energy: float | None = None


@dataclass
class Root:
    spin_ci: dict
    energy: float | None = None


@dataclass
class ChargedRun:
    nelc: int
    norb: int
    mult: int
    roots: list[Root]
    file: Path | None = None
    data: dict | None = None  # ORCA JSON, filled by the loader


@dataclass
class Parameters:
    initial: State
    runs: list[ChargedRun]

    def is_removal(self, run: ChargedRun) -> bool:
        return run.nelc < self.initial.nelc

    def pair(self, run: ChargedRun, root: Root) -> dict:
        """Legacy dict consumed by ``Dyson`` for one (initial, root) pair."""
        return {
            "parameters": {
                "initial": {
                    "nelc": self.initial.nelc,
                    "norb": self.initial.norb,
                    "mult": self.initial.mult,
                    "spin_ci": self.initial.spin_ci,
                },
                "final": {
                    "nelc": run.nelc,
                    "norb": run.norb,
                    "mult": run.mult,
                    "spin_ci": root.spin_ci,
                },
            }
        }


# ----------------------------------------------------------------- validation
def _check_determinants(spin_ci: dict, norb: int, nelc: int, where: str):
    for det in spin_ci:
        s = det.strip("[]")
        if len(s) != norb:
            raise ValueError(f"{where}: '{det}' has {len(s)} orbitals, expected {norb}")
        n = 2 * s.count("2") + s.count("u") + s.count("d")
        if n != nelc:
            raise ValueError(f"{where}: '{det}' has {n} electrons, expected {nelc}")


def _require(d: dict, keys, where: str):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{where}: missing {', '.join(missing)}")


def validate(params: Parameters):
    ini = params.initial
    _check_determinants(ini.spin_ci, ini.norb, ini.nelc, "initial")
    if not params.runs:
        raise ValueError("final: no charged runs given")

    for i, run in enumerate(params.runs):
        where = f"final[{i}]"
        if abs(run.nelc - ini.nelc) != 1:
            raise ValueError(f"{where}: nelc must differ from initial by 1")
        if run.norb != ini.norb:
            raise ValueError(f"{where}: norb must equal initial norb")
        if abs(run.mult - ini.mult) != 1:
            raise ValueError(f"{where}: mult must differ from initial by 1 (ΔS = ±1/2)")
        if not run.roots:
            raise ValueError(f"{where}: no roots")
        for j, root in enumerate(run.roots):
            if not root.spin_ci:
                raise ValueError(f"{where}.roots[{j}]: empty spin_ci")
            _check_determinants(root.spin_ci, run.norb, run.nelc, f"{where}.roots[{j}]")


# -------------------------------------------------------------------- loading
def _parse_run(raw: dict, where: str) -> ChargedRun:
    _require(raw, ["nelc", "norb", "mult"], where)
    if "roots" in raw:
        roots = [Root(r["spin_ci"], r.get("energy")) for r in raw["roots"]]
    elif "spin_ci" in raw:  # v1 style, single root
        roots = [Root(raw["spin_ci"], raw.get("energy"))]
    else:
        raise ValueError(f"{where}: needs 'roots' or 'spin_ci'")
    file = Path(raw["file"]) if "file" in raw else None
    return ChargedRun(raw["nelc"], raw["norb"], raw["mult"], roots, file=file)


def parse_parameters(raw: dict) -> Parameters:
    """Dict (v1 or v2) -> Parameters. Does not touch the filesystem."""
    _require(raw, ["parameters"], "root")
    p = raw["parameters"]
    _require(p, ["initial", "final"], "parameters")
    _require(p["initial"], ["nelc", "norb", "mult", "spin_ci"], "initial")

    ini = p["initial"]
    initial = State(
        ini["nelc"], ini["norb"], ini["mult"], ini["spin_ci"], ini.get("energy")
    )

    finals = p["final"] if isinstance(p["final"], list) else [p["final"]]
    runs = [_parse_run(r, f"final[{i}]") for i, r in enumerate(finals)]

    params = Parameters(initial, runs)
    validate(params)
    return params


def load_parameters(path: Path, final_json: Path | None = None) -> Parameters:
    """Read the parameters file and the ORCA JSON of every charged run.

    ``final_json`` covers the v1 CLI, where the charged JSON is a CLI option
    instead of a ``file`` entry; run files are resolved relative to ``path``.
    """
    path = Path(path)
    with open(path) as f:
        params = parse_parameters(json.load(f))

    for i, run in enumerate(params.runs):
        file = run.file or final_json
        if file is None:
            raise ValueError(
                f"final[{i}]: no 'file' given and no charged JSON on the CLI"
            )
        file = Path(file) if Path(file).is_absolute() else path.parent / file
        if not file.is_file():
            raise FileNotFoundError(f"final[{i}]: {file} not found")
        with open(file) as f:
            run.data = json.load(f)
        run.file = file
    return params
