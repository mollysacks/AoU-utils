import pytest

import aou_utils
from aou_utils import env


def test_env_vars_read(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "proj.cdr")
    monkeypatch.setenv("WORKSPACE_BUCKET", "gs://bucket")
    monkeypatch.setenv("GOOGLE_PROJECT", "proj")
    assert aou_utils.cdr() == "proj.cdr"
    assert aou_utils.workspace_bucket() == "gs://bucket"
    assert aou_utils.google_project() == "proj"


def test_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    with pytest.raises(EnvironmentError):
        env.cdr()
