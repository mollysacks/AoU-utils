# AoU-utils

Helper functions for working with data on the [All of Us Researcher Workbench](https://workbench.researchallofus.org/).

The import name is `aou_utils`.

## Workflow

- Code is written and edited **only on Expanse**, pushed to GitHub, then installed into a Workbench environment with pip. The repo is never edited from inside the Workbench.
- `main` holds stable, shared code.
- Each project gets its own branch, `dev/<project-name>`. Useful pieces get merged back into `main`.
- This repo is **code only**. No participant-level data, outputs, or counts derived from the Workbench go in here (see `.gitignore`).

## Installing on the Workbench

In a notebook cell:

```python
%pip install --user --no-deps --force-reinstall "git+https://github.com/mollysacks/AoU-utils.git@main"
```

For a project branch, swap `@main` for `@dev/<project-name>`. To freeze the exact code an analysis used, pin a tag or commit instead (`@v0.1.0` or `@<commit-sha>`).

Restart the kernel after installing, then:

```python
import aou_utils
from aou_utils import bq

aou_utils.cdr()                 # CDR BigQuery dataset
aou_utils.workspace_bucket()    # gs:// workspace bucket
df = bq.query("SELECT COUNT(*) AS n FROM `{cdr}.person`")
```

`--no-deps` keeps pip from touching the Workbench's preinstalled pandas/BigQuery. `--force-reinstall` makes sure you get the latest commit, since the version number doesn't change between pushes to a branch.

### Installing from a wheel

pip builds a wheel when installing from git, so the command above is usually all you need. CI also builds one on every push (downloadable from the Actions run). Pushing a `v*` tag attaches the wheel to a GitHub release, which you can install with:

```python
%pip install --user --no-deps "https://github.com/mollysacks/AoU-utils/releases/download/v0.1.0/aou_utils-0.1.0-py3-none-any.whl"
```

## Developing on Expanse

```bash
git checkout -b dev/<project-name>
pip install -e ".[dev]"
pytest
git push -u origin dev/<project-name>
```

Code that calls Workbench services (BigQuery, the workspace bucket) can't run on Expanse, so keep those imports inside functions and test the logic around them with mocked env vars or small synthetic data.

## Releasing

1. Bump `version` in `pyproject.toml` on `main`.
2. `git tag v0.X.Y && git push origin v0.X.Y`
