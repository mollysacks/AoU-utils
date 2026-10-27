import json
import subprocess

import pytest

import aou_utils
from aou_utils import env

# Trimmed from real `wb resource list --format=json` output, with ids anonymized
RESOURCES = [
    {"id": "C2024Q3R9", "resourceType": "BQ_DATASET",
     "projectId": "wb-cdr-project", "datasetId": "C2024Q3R9"},
    {"id": "spark-cluster", "resourceType": "DATAPROC_CLUSTER"},
    {"id": "dataproc-staging-ws", "resourceType": "GCS_BUCKET",
     "bucketName": "dataproc-staging-wb-ws-project"},
    {"id": "dataproc-temp-ws", "resourceType": "GCS_BUCKET",
     "bucketName": "dataproc-temp-wb-ws-project"},
    {"id": "rw-migration-ws", "resourceType": "GCS_BUCKET",
     "bucketName": "rw-migration-ws"},
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("WORKSPACE_CDR", "WORKSPACE_BUCKET", "GOOGLE_PROJECT", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    env.resources.cache_clear()
    yield
    env.resources.cache_clear()


@pytest.fixture
def fake_wb(monkeypatch):
    def run(cmd, **kwargs):
        assert cmd[:3] == ["wb", "resource", "list"]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(RESOURCES))
    monkeypatch.setattr(env.subprocess, "run", run)


def test_env_vars_take_precedence(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "proj.cdr")
    monkeypatch.setenv("WORKSPACE_BUCKET", "gs://bucket")
    monkeypatch.setenv("GOOGLE_PROJECT", "proj")
    assert aou_utils.cdr() == "proj.cdr"
    assert aou_utils.workspace_bucket() == "gs://bucket"
    assert aou_utils.google_project() == "proj"


def test_cdr_from_wb(fake_wb):
    assert env.cdr() == "wb-cdr-project.C2024Q3R9"
    assert env.os.environ["WORKSPACE_CDR"] == "wb-cdr-project.C2024Q3R9"


def test_bucket_from_wb_skips_dataproc(fake_wb):
    assert env.workspace_bucket() == "gs://rw-migration-ws"


def test_bucket_by_resource_id(fake_wb):
    assert env.workspace_bucket("dataproc-temp-ws") == "gs://dataproc-temp-wb-ws-project"


def test_ambiguous_bucket_raises(fake_wb, monkeypatch):
    extra = RESOURCES + [{"id": "mine", "resourceType": "GCS_BUCKET", "bucketName": "my-bucket"}]
    monkeypatch.setattr(env.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(extra)))
    with pytest.raises(EnvironmentError, match="rw-migration-ws"):
        env.workspace_bucket()


def test_google_project_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    assert env.google_project() == "proj"


def test_missing_wb_raises(monkeypatch):
    def run(cmd, **kwargs):
        raise FileNotFoundError
    monkeypatch.setattr(env.subprocess, "run", run)
    with pytest.raises(EnvironmentError, match="wb"):
        env.cdr()
