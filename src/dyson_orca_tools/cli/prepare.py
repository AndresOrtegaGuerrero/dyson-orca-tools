"""prepare: Prepare the json an ci parameters for Dyson orbital calculation"""

from pathlib import Path
from typing import List
import typer
import json
import os
from .console import ok, fail

from ..io.orca_output import build_parameters, parse_orca_output
from ..io.params import parse_parameters

NAME = "prepare"


def run(
    initial_out: Path = typer.Option(
        ...,
        "-i",
        "--initial-out",
        help="ORCA output of the initial state (PrintWF det).",
    ),
    final: List[str] = typer.Option(
        ...,
        "-f",
        "--final",
        help="ORCA output of one N±1 calculation, repeatable. "
        "Its orca_2json file is <name>.json next to it, or give 'run.out:file.json'.",
    ),
    output: Path = typer.Option(
        "params.json", "-o", "--output", help="Parameters file to write."
    ),
    initial_state: List[str] = typer.Option(
        None,
        "--initial-state",
        help="Reference state as MULT:ROOT, repeatable (e.g. 1:0 3:0). "
        "Default: the lowest-energy root of the output.",
    ),
    all_initial_roots: bool = typer.Option(
        False, "--all-initial-roots", help="Every root of every block as a reference."
    ),
):
    """Build the parameters file (v3) from ORCA outputs (CI vectors, energies, active space)."""
    pairs = []
    for item in final:
        out_file, _, json_file = item.partition(":")
        out_file = Path(out_file)
        pairs.append(
            (out_file, Path(json_file) if json_file else out_file.with_suffix(".json"))
        )
    try:
        cas = parse_orca_output(initial_out)
        if all_initial_roots:
            states = [(r.mult, r.index) for r in cas.roots]
        else:
            states = [_parse_state_spec(s) for s in initial_state or []]

        raw = build_parameters(cas, pairs, states)
        params = parse_parameters(raw)  # validates before writing
    except (ValueError, FileNotFoundError) as exc:
        fail(str(exc))

    for run in raw["parameters"]["final"]:  # paths relative to the parameters file
        run["file"] = os.path.relpath(
            Path(run["file"]).resolve(), output.resolve().parent
        )
    with open(output, "w") as f:
        json.dump(raw, f, indent=2)
    for ini in params.initials:
        ok(
            f"initial {ini.label}: CAS({ini.nelc},{ini.norb}) mult {ini.mult}, "
            f"E = {ini.energy} Eh, {len(ini.spin_ci)} determinants"
        )
    for run in params.runs:
        norm = [sum(c * c for c in r.spin_ci.values()) for r in run.roots]
        typer.echo(
            f"   final {run.file}: mult {run.mult}, {len(run.roots)} root(s), "
            f"Σc² = {', '.join(f'{n:.3f}' for n in norm)}"
        )
    ok(f"written {output}")


def _parse_state_spec(spec: str) -> tuple[int, int]:
    """'3:0' -> (3, 0)."""
    try:
        mult, root = spec.split(":")
        return int(mult), int(root)
    except ValueError:
        raise ValueError(f"--initial-state '{spec}': expected MULT:ROOT, e.g. 1:0")
