# AoU-utils: notes for agents

- Python package for the All of Us Researcher Workbench. Source is in `src/aou_utils/`, tests are in `tests/`.
- Edited only on Expanse and installed on the Workbench with pip from GitHub. The Workbench can't be reached from here, so code that needs BigQuery or GCS can't be run here. Keep `google.cloud` and similar imports inside functions so the package still imports, and test with mocks or synthetic data.
- Workbench 2.0 (Verily) does NOT set `WORKSPACE_CDR` or `WORKSPACE_BUCKET`. `aou_utils.env` falls back to `wb resource list --format=json`. The billing project comes from `GOOGLE_PROJECT` or `GOOGLE_CLOUD_PROJECT`. Always use the accessors in `aou_utils.env`.
- Keep `dependencies` in `pyproject.toml` empty or minimal. Never pin packages the Workbench already provides.
- NEVER commit data, outputs, or anything derived from participant data. This includes hard-coded counts (AoU policy forbids sharing counts of 1–20).
- Branches: `main` is stable. Use `dev/<project>` for project work. Don't commit to `main` directly when working on a project.
- Notebooks in `notebooks/` are templates. Commit them with no outputs (enforced by `tests/test_notebooks.py`).
- Run `pytest` before committing.
