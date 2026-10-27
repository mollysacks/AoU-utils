"""Accessors for workspace settings (CDR dataset, bucket, billing project).

The legacy Workbench set WORKSPACE_CDR / WORKSPACE_BUCKET as environment variables. Workbench 2.0
(Verily) doesn't, so we fall back to the workspace's resources as listed by the `wb` CLI.
Resolved values are written back to os.environ so other code expecting the old variables works too.
"""

import functools
import json
import os
import re
import subprocess

# CDR dataset ids look like C2024Q3R9
_CDR_ID = re.compile(r"^C\d{4}Q\dR\d+$")
# Buckets the platform creates for Dataproc; never what you want for your own files
_SYSTEM_BUCKET_PREFIXES = ("dataproc-staging-", "dataproc-temp-")


@functools.lru_cache(maxsize=None)
def resources() -> tuple:
    """The workspace's resources from `wb resource list` (cached for the session)."""
    try:
        out = subprocess.run(
            ["wb", "resource", "list", "--format=json"], capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError:
        raise EnvironmentError(
            "The `wb` CLI isn't available. Are you running inside an All of Us Workbench environment?"
        ) from None
    return tuple(json.loads(out))


def _only(matches: list, what: str) -> dict:
    if len(matches) != 1:
        ids = [r["id"] for r in matches]
        raise EnvironmentError(f"Expected one {what}, found {len(matches)}: {ids}")
    return matches[0]


def cdr() -> str:
    """BigQuery dataset of the workspace's Curated Data Repository, as 'project.dataset'."""
    if os.environ.get("WORKSPACE_CDR"):
        return os.environ["WORKSPACE_CDR"]
    datasets = [r for r in resources() if r["resourceType"] == "BQ_DATASET"]
    cdrs = [r for r in datasets if _CDR_ID.match(r["datasetId"])] or datasets
    r = _only(cdrs, "CDR BigQuery dataset")
    os.environ["WORKSPACE_CDR"] = f"{r['projectId']}.{r['datasetId']}"
    return os.environ["WORKSPACE_CDR"]


def workspace_bucket(resource_id: str = None) -> str:
    """The workspace's GCS bucket as 'gs://...'.

    With no argument, uses WORKSPACE_BUCKET if set, else the single non-Dataproc bucket resource.
    Pass a resource id (as shown by `wb resource list`) to pick one when there are several.
    """
    if resource_id is None and os.environ.get("WORKSPACE_BUCKET"):
        return os.environ["WORKSPACE_BUCKET"]
    buckets = [r for r in resources() if r["resourceType"] == "GCS_BUCKET"]
    if resource_id is not None:
        r = _only([b for b in buckets if b["id"] == resource_id], f"bucket with id {resource_id!r}")
        return f"gs://{r['bucketName']}"
    user_buckets = [b for b in buckets if not b["bucketName"].startswith(_SYSTEM_BUCKET_PREFIXES)]
    r = _only(user_buckets, "non-Dataproc GCS bucket (pass resource_id to choose)")
    os.environ["WORKSPACE_BUCKET"] = f"gs://{r['bucketName']}"
    return os.environ["WORKSPACE_BUCKET"]


def google_project() -> str:
    """The workspace's Google billing project."""
    for var in ("GOOGLE_PROJECT", "GOOGLE_CLOUD_PROJECT"):
        if os.environ.get(var):
            return os.environ[var]
    raise EnvironmentError(
        "Neither GOOGLE_PROJECT nor GOOGLE_CLOUD_PROJECT is set. "
        "Are you running inside an All of Us Workbench environment?"
    )
