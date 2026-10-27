"""Utilities for the All of Us Researcher Workbench."""

from importlib.metadata import PackageNotFoundError, version

from .env import cdr, google_project, workspace_bucket

try:
    __version__ = version("aou-utils")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0+unknown"

__all__ = ["cdr", "google_project", "workspace_bucket", "__version__"]
