"""Notebooks in the repo must not carry outputs, which could contain participant data."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path):
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"], f"{path.name} has saved outputs; clear them before committing"
