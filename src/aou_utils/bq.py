"""BigQuery helpers. Imports are lazy so the package installs and imports off-Workbench."""

from .env import cdr, google_project


def query(sql: str, **format_kwargs):
    """Run a query against the workspace CDR and return a pandas DataFrame.

    `{cdr}` in the SQL is replaced with the CDR dataset name, so queries can be written as
    ``SELECT ... FROM `{cdr}.person` ``. Extra keyword args are also substituted.
    """
    from google.cloud import bigquery

    sql = sql.format(cdr=cdr(), **format_kwargs)
    # The CDR lives in another project, so bill queries to the workspace project explicitly
    return bigquery.Client(project=google_project()).query(sql).to_dataframe()
