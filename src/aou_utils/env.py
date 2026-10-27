"""Accessors for the environment variables the Workbench sets in every workspace."""

import os


def _require(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise EnvironmentError(
            f"{var} is not set. Are you running inside an All of Us Workbench environment?"
        )
    return value


def cdr() -> str:
    """BigQuery dataset of the workspace's Curated Data Repository, e.g. 'project.dataset'."""
    return _require("WORKSPACE_CDR")


def workspace_bucket() -> str:
    """The workspace's GCS bucket, e.g. 'gs://fc-secure-...'."""
    return _require("WORKSPACE_BUCKET")


def google_project() -> str:
    """The workspace's Google billing project."""
    return _require("GOOGLE_PROJECT")
