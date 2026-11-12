"""Run this repo's WDL workflows (src/wdl/) on Workbench 2.0 with the `wb workflow` CLI.

Typical use: `upload_wdl` copies a WDL from GitHub into the workspace bucket, `create` registers
it with the workspace, `run` starts a job, and `describe` / `tasks` check on it.
Jobs run on the Workbench's own Cromwell server (no cromshell or Cromwell app needed). Cromwell's
execution and status files go to <bucket>/cromwell-execution/<folder> (one folder per kind of
analysis, e.g. 'PGS'; Cromwell makes a subfolder per job). The WDLs copy their final outputs to
the `output_dir` they're given.
"""

import json
import subprocess
import time
import urllib.request

from .env import workspace_bucket, workspace_bucket_id

REPO = "mollysacks/AoU-utils"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/src/wdl/{name}"
EXECUTION_DIR = "cromwell-execution"


def _run(cmd, stdin: str = None) -> str:
    try:
        return subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{e.stderr}") from None


def _wb(*args):
    out = _run(["wb", *args, "--format=json"])
    return json.loads(out) if out.strip() else None


def _bucket_path(path: str) -> str:
    """Path inside the workspace bucket, accepting either 'dir/file' or 'gs://bucket/dir/file'."""
    bucket = workspace_bucket()
    if path.startswith("gs://"):
        if not path.startswith(bucket + "/"):
            raise ValueError(f"{path} isn't in the workspace bucket {bucket}")
        path = path[len(bucket) + 1:]
    return path.strip("/")


def write_text(text: str, gs_path: str) -> str:
    """Write a string to a gs:// path."""
    _run(["gcloud", "storage", "cp", "-", gs_path], stdin=text)
    return gs_path


def upload_wdl(name: str, ref: str = "main", folder: str = "wdl") -> str:
    """Copy src/wdl/<name> at branch/tag/commit `ref` from GitHub to <bucket>/<folder>/<ref>/<name>.

    Returns the path inside the bucket, which is what `create` wants.
    """
    url = RAW_URL.format(repo=REPO, ref=ref, name=name)
    with urllib.request.urlopen(url) as r:
        text = r.read().decode()
    path = f"{folder}/{ref}/{name}"
    write_text(text, f"{workspace_bucket()}/{path}")
    return path


def create(workflow_id: str, wdl_path: str, bucket_id: str = None, replace: bool = True):
    """Register a WDL in the workspace bucket as workflow `workflow_id`.

    There's no `wb workflow update`, so with replace=True an existing workflow of the same id is
    deleted and registered again (picking up the new WDL).
    """
    args = ["workflow", "create", f"--workflow={workflow_id}", "--workflow-type=WDL",
            f"--bucket-id={bucket_id or workspace_bucket_id()}", f"--path={_bucket_path(wdl_path)}"]
    try:
        return _wb(*args)
    except RuntimeError as e:
        if not (replace and "already exists" in str(e)):
            raise
    delete(workflow_id)
    return _wb(*args)


def delete(workflow_id: str):
    """Remove a registered workflow (not its jobs' outputs)."""
    _run(["wb", "workflow", "delete", "--quiet", f"--workflow={workflow_id}"])


def run(workflow_id: str, inputs: dict, folder: str, job_id: str = None, bucket_id: str = None,
        description: str = None, keep_intermediate: bool = True):
    """Start a job; returns what `wb workflow job run` prints (pass it to `describe` / `tasks`).

    `inputs` keys are '<workflow name>.<input>' as in the WDL, e.g. {'prscs.n_gwas': 500000}.
    Cromwell's files go to <bucket>/cromwell-execution/<folder>. The inputs are saved to
    <that folder>/inputs/<job>-<timestamp>.json and passed by URI. keep_intermediate keeps
    Cromwell's per-task files (logs, stderr) instead of the Workbench default of deleting them.
    """
    output_path = f"{EXECUTION_DIR}/{folder.strip('/')}"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    inputs_uri = write_text(json.dumps(inputs, indent=2),
                            f"{workspace_bucket()}/{output_path}/inputs/{job_id or workflow_id}-{stamp}.json")
    args = ["workflow", "job", "run", f"--workflow={workflow_id}",
            f"--output-bucket-id={bucket_id or workspace_bucket_id()}", f"--output-path={output_path}",
            f"--inputs-uri={inputs_uri}", f"--delete-intermediate-outputs={str(not keep_intermediate).lower()}"]
    if job_id:
        args.append(f"--job-id={job_id}")
    if description:
        args.append(f"--description={description}")
    return _wb(*args)


def _job_uuid(job) -> str:
    """The job's UUID, from the dict `run` returned or a UUID string."""
    if isinstance(job, str):
        return job
    for key in ("runId", "id", "jobId"):
        if job.get(key):
            return job[key]
    raise ValueError(f"No job id found in {list(job)}; pass the UUID shown by `wb workflow job list`")


def describe(job):
    """Status and details of a job (`wb workflow job describe`). `job`: run()'s result or the UUID."""
    return _wb("workflow", "job", "describe", f"--job-id={_job_uuid(job)}")


def tasks(job):
    """The job's tasks with their status (`wb workflow job task list`), as a DataFrame."""
    import pandas as pd

    out = _wb("workflow", "job", "task", "list", f"--job-id={_job_uuid(job)}")
    return pd.json_normalize(out) if isinstance(out, list) else out


def list_jobs():
    return _wb("workflow", "job", "list")


def find_outputs(output_path: str, suffix: str) -> list:
    """gs:// paths under <bucket>/<output_path> ending in `suffix` (e.g. '.score.tsv')."""
    prefix = f"{workspace_bucket()}/{_bucket_path(output_path)}"
    out = _run(["gcloud", "storage", "ls", "-r", f"{prefix}/**"])
    return sorted(line for line in out.split() if line.endswith(suffix))
