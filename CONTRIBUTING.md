# Contributing to SCD Edition

Thank you for helping improve SCD Edition. Bug reports, loader requests,
documentation improvements and focused pull requests are welcome.

## Report a problem

Use the appropriate GitHub issue form. Include:

- SCD Edition version or commit;
- installation method, operating system and Python version;
- CPU or GPU model and, for CUDA problems, driver information;
- acquisition system and file format;
- exact steps, expected behaviour, actual behaviour and the complete traceback.

Do not upload participant recordings, identifying metadata or confidential lab
data to a public issue. If a minimal reproducer is necessary, use synthetic or
explicitly redistributable data.

## Development setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create an
environment using either the CPU or CUDA dependency source:

```bash
uv sync --extra cpu
```

```bash
uv sync --extra cuda
```

On Windows, add `--python 3.13 --managed-python` if the system Python comes
from Conda or otherwise causes Qt DLL conflicts.

The `cpu` and `cuda` extras are mutually exclusive. `uv` does not remember the
extra used by an earlier `uv sync`, and a bare `uv run` can therefore replace a
CUDA-enabled PyTorch installation with the default CPU build. Always select the
same backend when allowing `uv` to sync the environment.

Run all checks through the backend-safe wrapper before opening a pull request:

```bash
python scripts/dev_check.py --backend cpu
```

or:

```bash
python scripts/dev_check.py --backend cuda
```

The wrapper performs a locked sync with the selected extra, then runs Ruff and
pytest with `uv run --no-sync`. For an individual command, include the backend
explicitly, for example:

```bash
uv run --extra cuda pytest tests/test_otb4_loader.py
```

The GUI smoke tests run headlessly by setting `QT_QPA_PLATFORM=offscreen`.
Changes to loaders should include a small synthetic fixture and tests for
malformed input. Changes to scientific calculations should include numerical
regression tests and a clear source or rationale.

For a new acquisition format, first read the
[data-import guide](docs/importing-data.md). Generic MATLAB/HDF5 layouts should
use the inspector; a built-in loader is most useful when the format has stable
metadata, units and channel-order semantics that the generic path cannot
capture.

## Pull requests

Keep each pull request focused. Explain the user-visible change, tests run and
any compatibility implications for existing `.pkl` sessions. Update the README
or quick-start when behaviour or supported formats change.

The files under `src/scd_app/_vendor/` come from Motor Unit Toolbox and should
not be edited casually. If they need updating, record the new upstream commit,
preserve its MIT licence, and compare the affected numerical outputs against
the previous version.

## Releases

1. Update the version in `pyproject.toml`, `CITATION.cff` and `CHANGELOG.md`.
2. Run the full test suite and `python -m build` from a clean checkout.
3. Run `python scripts/check_distribution.py`.
4. Push a matching tag such as `v0.1.0`.
5. Approve the protected `pypi` environment when the release workflow asks.
6. After Zenodo archives the GitHub release, add its concept DOI to the README
   and `CITATION.cff`.
