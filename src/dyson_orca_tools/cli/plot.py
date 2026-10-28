from pathlib import Path

import typer

from .console import plot_or_warn

NAME = "plot"


def run(
    output_dir: Path = typer.Option(
        ".",
        "-o",
        "--output-dir",
        help="Folder with dyson_peaks.csv and spectral_function.dat.",
    ),
    title: str = typer.Option(
        None, "--title", help="Optional title, e.g. 'pentacene CASCI(12,12)'."
    ),
    vertical: bool = typer.Option(
        False, "--vertical", help="Energy on the y axis (as in the paper's figures)."
    ),
    legend_outside: bool = typer.Option(
        False, "--legend-outside", help="Place the legend outside the axes."
    ),
    label_threshold: float = typer.Option(
        0.0,
        "--label-threshold",
        help="Label only peaks with strength >= this (curves keep all roots).",
    ),
    label_energy: bool = typer.Option(
        False,
        "--label-energy",
        help="Append ω (eV, relative to the chosen zero) to each peak label.",
    ),
):
    """Re-plot a spectrum from the files written by `spectrum` (no recomputation)."""
    plot_or_warn(
        output_dir, title, vertical, legend_outside, label_threshold, label_energy
    )
