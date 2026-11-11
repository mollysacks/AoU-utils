"""Run this repo's WDL workflows (src/wdl/) on Workbench 2.0 with the `wb workflow` CLI.

Typical use: `upload_wdl` copies a WDL from GitHub into the workspace bucket, `create` registers
it with the workspace, `run` writes the inputs JSON next to the outputs and starts a job, and
`describe` / `find_outputs` check on it.
"""

import json
import subprocess
import urllib.request

from .env import workspace_bucket, workspace_bucket_id

REPO = "mollysacks/AoU-utils"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/src/wdl/{name}"


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


def create(workflow_id: str, wdl_path: str, bucket_id: str = None):
    """Register a WDL in the workspace bucket as workflow `workflow_id`."""
    return _wb("workflow", "create", f"--workflow={workflow_id}", "--workflow-type=WDL",
               f"--bucket-id={bucket_id or workspace_bucket_id()}", f"--path={_bucket_path(wdl_path)}")


def run(workflow_id: str, inputs: dict, output_path: str, job_id: str = None, bucket_id: str = None,
        description: str = None):
    """Start a job. Inputs are saved to <bucket>/<output_path>/inputs.json and passed by URI.

    `inputs` keys are '<workflow name>.<input>' as in the WDL, e.g. {'prscs.n_gwas': 500000}.
    """
    output_path = _bucket_path(output_path)
    inputs_uri = write_text(json.dumps(inputs, indent=2), f"{workspace_bucket()}/{output_path}/inputs.json")
    args = ["workflow", "job", "run", f"--workflow={workflow_id}",
            f"--output-bucket-id={bucket_id or workspace_bucket_id()}", f"--output-path={output_path}",
            f"--inputs-uri={inputs_uri}"]
    if job_id:
        args.append(f"--job-id={job_id}")
    if description:
        args.append(f"--description={description}")
    return _wb(*args)


def describe(job_id: str):
    """Status and details of a job (`wb workflow job describe`)."""
    return _wb("workflow", "job", "describe", f"--job-id={job_id}")


def list_jobs():
    return _wb("workflow", "job", "list")


def find_outputs(output_path: str, suffix: str) -> list:
    """gs:// paths under <bucket>/<output_path> ending in `suffix` (e.g. '.score.tsv')."""
    prefix = f"{workspace_bucket()}/{_bucket_path(output_path)}"
    out = _run(["gcloud", "storage", "ls", "-r", f"{prefix}/**"])
    return sorted(line for line in out.split() if line.endswith(suffix))
