# Contributing

## Development setup

```shell
git clone https://github.com/AndresOrtegaGuerrero/dyson-orca-tools.git
cd dyson-orca-tools
pip install -e ".[dev,plot]"
pre-commit install
pytest
```

Tests that need real ORCA data read it from a folder given by an environment variable
(e.g. `CAS1414_OUT`) and are skipped when it is unset; no calculation output is kept in the
repository.

## Releasing

Releases are published to PyPI by GitHub Actions when a `v*` tag is pushed. From an up-to-date `main`:

```shell
bumpver update --patch    # or --minor / --major: bumps pyproject.toml + version.py, commits, tags and pushes
```

The `release` workflow then checks that the tag matches `dyson_orca_tools.__version__`, builds the sdist and wheel, uploads them to PyPI via trusted publishing, and creates a GitHub release with auto-generated notes.
