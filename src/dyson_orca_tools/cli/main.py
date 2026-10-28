import typer

from . import dyson, prepare, plot, spectrum, ehmap

app = typer.Typer(
    help="Dyson orbitals and multireference spectral functions from ORCA CASCI/CASSCF JSON."
)

for module in (dyson, prepare, plot, spectrum, ehmap):
    app.command(module.NAME)(module.run)


if __name__ == "__main__":
    app()
