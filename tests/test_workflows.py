import json
import subprocess

import pytest

from aou_utils import env, workflows


@pytest.fixture
def calls(monkeypatch):
    monkeypatch.setenv("WORKSPACE_BUCKET", "gs://ws-bucket")
    monkeypatch.setattr(env, "workspace_bucket_id", lambda: "ws-bucket-id")
    monkeypatch.setattr(workflows, "workspace_bucket_id", lambda: "ws-bucket-id")
    seen = []

    def run(cmd, input=None, **kwargs):
        seen.append((cmd, input))
        out = json.dumps({"ok": True}) if cmd[0] == "wb" else ""
        if cmd[:3] == ["gcloud", "storage", "ls"]:
            out = "gs://ws-bucket/runs/a/call-x/p.score.tsv\ngs://ws-bucket/runs/a/call-x/p.log\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out)

    monkeypatch.setattr(workflows.subprocess, "run", run)
    return seen


def test_create_strips_bucket(calls):
    workflows.create("prscs", "gs://ws-bucket/wdl/main/prscs.wdl")
    cmd = calls[0][0]
    assert cmd[:3] == ["wb", "workflow", "create"]
    assert "--path=wdl/main/prscs.wdl" in cmd and "--bucket-id=ws-bucket-id" in cmd


def test_create_rejects_other_bucket(calls):
    with pytest.raises(ValueError):
        workflows.create("prscs", "gs://other/prscs.wdl")


def test_run_writes_inputs_then_submits(calls):
    workflows.run("prscs", {"prscs.n_gwas": 10}, "runs/a", job_id="height")
    (cp, stdin), (wb, _) = calls
    assert cp == ["gcloud", "storage", "cp", "-", "gs://ws-bucket/runs/a/inputs.json"]
    assert json.loads(stdin) == {"prscs.n_gwas": 10}
    assert "--inputs-uri=gs://ws-bucket/runs/a/inputs.json" in wb
    assert "--output-path=runs/a" in wb and "--job-id=height" in wb


def test_find_outputs(calls):
    assert workflows.find_outputs("runs/a", ".score.tsv") == ["gs://ws-bucket/runs/a/call-x/p.score.tsv"]
