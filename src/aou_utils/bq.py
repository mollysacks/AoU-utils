"""BigQuery helpers. Imports are lazy so the package installs and imports off-Workbench."""

import numbers

from .env import cdr, google_project


def _param_spec(value):
    """(BigQuery type, is_array, python value) for a query parameter value."""
    is_array = hasattr(value, "__iter__") and not isinstance(value, (str, bytes))
    values = list(value) if is_array else [value]
    if not values:
        raise ValueError("Empty array query parameters aren't supported")
    first = values[0]
    if isinstance(first, bool):
        bq_type, cast = "BOOL", bool
    elif isinstance(first, numbers.Integral):
        bq_type, cast = "INT64", int
    elif isinstance(first, numbers.Real):
        bq_type, cast = "FLOAT64", float
    elif isinstance(first, str):
        bq_type, cast = "STRING", str
    else:
        raise TypeError(f"Unsupported query parameter type: {type(first)}")
    values = [cast(v) for v in values]
    return bq_type, is_array, values if is_array else values[0]


def query(sql: str, params: dict = None, **format_kwargs):
    """Run a query against the workspace CDR and return a pandas DataFrame.

    `{cdr}` in the SQL is replaced with the CDR dataset name, so queries can be written as
    ``SELECT ... FROM `{cdr}.person` ``. Extra keyword args are also substituted.

    `params` become BigQuery query parameters, referenced as `@name` in the SQL. Lists become
    array parameters, e.g. ``WHERE person_id IN UNNEST(@person_ids)``. Use these for long ID
    lists rather than formatting them into the SQL.
    """
    from google.cloud import bigquery

    sql = sql.format(cdr=cdr(), **format_kwargs)
    query_params = []
    for name, value in (params or {}).items():
        bq_type, is_array, value = _param_spec(value)
        if is_array:
            query_params.append(bigquery.ArrayQueryParameter(name, bq_type, value))
        else:
            query_params.append(bigquery.ScalarQueryParameter(name, bq_type, value))
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    # The CDR lives in another project, so bill queries to the workspace project explicitly
    client = bigquery.Client(project=google_project())
    return client.query(sql, job_config=job_config).to_dataframe()
