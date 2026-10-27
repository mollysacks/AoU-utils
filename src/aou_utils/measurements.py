"""Pull, summarize and transform EHR measurements (including physical measurements).

Physical measurements (PM) live in the same OMOP `measurement` table as EHR labs and vitals; their
source concepts are in the 'PPI' vocabulary, which is how we count PM records.
"""

import numpy as np
import pandas as pd

from . import bq

PERSON_MEDIANS_SQL = """
WITH records AS (
  SELECT
    m.person_id,
    m.measurement_concept_id AS concept_id,
    IFNULL(m.unit_concept_id, 0) AS unit_concept_id,
    m.value_as_number AS value,
    DATE_DIFF(m.measurement_date, DATE(p.birth_datetime), DAY) / 365.25 AS age,
    IFNULL(src.vocabulary_id = 'PPI', FALSE) AS is_pm
  FROM `{cdr}.measurement` AS m
  JOIN `{cdr}.person` AS p USING (person_id)
  LEFT JOIN `{cdr}.concept` AS src ON src.concept_id = m.measurement_source_concept_id
  WHERE m.person_id IN UNNEST(@person_ids)
    AND m.value_as_number IS NOT NULL
    AND m.measurement_concept_id != 0
),
-- Keep only each concept's most common unit, rather than trying to convert between units
top_unit AS (
  SELECT concept_id, unit_concept_id
  FROM records
  GROUP BY concept_id, unit_concept_id
  QUALIFY ROW_NUMBER() OVER (PARTITION BY concept_id ORDER BY COUNT(*) DESC, unit_concept_id) = 1
),
unit_records AS (
  SELECT r.* FROM records AS r JOIN top_unit USING (concept_id, unit_concept_id)
),
-- Drop implausible records: those outside the concept's [trim, 1000 - trim] per-mille range
bounds AS (
  SELECT concept_id, q[OFFSET(@trim_permille)] AS lo, q[OFFSET(1000 - @trim_permille)] AS hi
  FROM (SELECT concept_id, APPROX_QUANTILES(value, 1000) AS q FROM unit_records GROUP BY concept_id)
),
kept AS (
  SELECT r.* FROM unit_records AS r JOIN bounds AS b USING (concept_id)
  WHERE r.value BETWEEN b.lo AND b.hi
),
medians AS (
  SELECT DISTINCT
    person_id,
    concept_id,
    unit_concept_id,
    PERCENTILE_CONT(value, 0.5) OVER w AS value,
    PERCENTILE_CONT(age, 0.5) OVER w AS age,
    COUNT(*) OVER w AS n_records,
    COUNTIF(is_pm) OVER w AS n_pm_records
  FROM kept
  WINDOW w AS (PARTITION BY person_id, concept_id)
)
SELECT * FROM medians
WHERE concept_id IN (
  SELECT concept_id FROM medians GROUP BY concept_id HAVING COUNT(*) >= @min_people
)
"""


def person_medians(person_ids, min_people: int = 1, trim_permille: int = 1) -> pd.DataFrame:
    """Per-person median of every numeric measurement, computed in BigQuery.

    For each concept, only records in its most common unit are used, and records outside its
    [trim_permille, 1000 - trim_permille] per-mille range are dropped as implausible. `age` is the
    median age across the same records. Concepts measured in fewer than `min_people` people are
    dropped. Returns one row per (person_id, concept_id).
    """
    return bq.query(
        PERSON_MEDIANS_SQL,
        params={
            "person_ids": list(person_ids),
            "min_people": int(min_people),
            "trim_permille": int(trim_permille),
        },
    )


def concept_names(concept_ids) -> pd.DataFrame:
    """concept_id -> concept_name and vocabulary_id."""
    return bq.query(
        "SELECT concept_id, concept_name, vocabulary_id FROM `{cdr}.concept` "
        "WHERE concept_id IN UNNEST(@ids)",
        params={"ids": [int(i) for i in set(concept_ids)]},
    )


def to_wide(medians: pd.DataFrame, column: str = "value") -> pd.DataFrame:
    """Long per-person medians -> person_id x concept_id matrix of `column` (float32)."""
    return medians.pivot(index="person_id", columns="concept_id", values=column).astype("float32")


def boxcox(wide: pd.DataFrame):
    """Box-Cox transform each column, with lambda estimated by maximum likelihood.

    Box-Cox needs positive values, so columns with a minimum <= 0 are shifted by (1 - min) first.
    Columns with fewer than 3 distinct values are left out. Returns (transformed, params), where
    params is indexed by column with `lambda` and `shift`.
    """
    from scipy import stats

    out, params = {}, {}
    for col in wide.columns:
        x = wide[col]
        observed = x.dropna().astype("float64")
        if observed.nunique() < 3:
            continue
        shift = 1.0 - observed.min() if observed.min() <= 0 else 0.0
        transformed, lam = stats.boxcox(observed + shift)
        out[col] = pd.Series(transformed, index=observed.index).reindex(x.index)
        params[col] = {"lambda": lam, "shift": shift}
    transformed = pd.DataFrame(out, index=wide.index, dtype="float32")
    transformed.columns.name = wide.columns.name
    return transformed, pd.DataFrame.from_dict(params, orient="index")


def residualize(values: pd.DataFrame, ages: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Regress covariates out of each column and z-score the residuals.

    Each column j is regressed on [1, age_j, age_j^2, covariates], where age_j is the person's age
    for that measurement (same shape as `values`). Rows missing the value, age or any covariate
    get NaN.
    """
    covariates = covariates.reindex(values.index).astype("float64")
    out = {}
    for col in values.columns:
        age = ages[col].astype("float64")
        X = pd.concat([age.rename("age"), (age ** 2).rename("age2"), covariates], axis=1)
        X.insert(0, "intercept", 1.0)
        y = values[col].astype("float64")
        ok = y.notna() & X.notna().all(axis=1)
        resid = pd.Series(np.nan, index=values.index)
        if ok.sum() > X.shape[1]:
            beta, *_ = np.linalg.lstsq(X[ok].to_numpy(), y[ok].to_numpy(), rcond=None)
            r = y[ok].to_numpy() - X[ok].to_numpy() @ beta
            resid[ok] = (r - r.mean()) / r.std(ddof=1)
        out[col] = resid
    result = pd.DataFrame(out, index=values.index, dtype="float32")
    result.columns.name = values.columns.name
    return result


def summarize(medians: pd.DataFrame) -> pd.DataFrame:
    """Per-measurement table: name, vocabulary, unit, n_people, n_records, pm_fraction.

    pm_fraction is the share of records that came from physical measurements rather than EHR.
    """
    concepts = medians.groupby("concept_id").agg(
        unit_concept_id=("unit_concept_id", "first"),
        n_people=("person_id", "size"),
        n_records=("n_records", "sum"),
        n_pm_records=("n_pm_records", "sum"),
    )
    names = concept_names(pd.concat([medians["concept_id"], medians["unit_concept_id"]]))
    names = names.set_index("concept_id")
    concepts["name"] = names["concept_name"].reindex(concepts.index).to_numpy()
    concepts["vocabulary"] = names["vocabulary_id"].reindex(concepts.index).to_numpy()
    concepts["unit"] = names["concept_name"].reindex(concepts["unit_concept_id"]).to_numpy()
    concepts["pm_fraction"] = concepts["n_pm_records"] / concepts["n_records"]
    return concepts


def coverage(present: pd.DataFrame, groups: pd.Series):
    """(counts, fractions) of people with each measurement, as concept x group tables.

    `present` is a person x concept boolean matrix; `groups` maps person_id to a group label.
    """
    groups = groups.reindex(present.index)
    counts = present.groupby(groups).sum().T
    fractions = counts / groups.value_counts()[counts.columns]
    return counts, fractions


def complete_cases(present: pd.DataFrame, score: pd.Series, groups: pd.Series, cutoffs) -> pd.DataFrame:
    """For each cutoff, keep measurements with score >= cutoff and count people with all of them.

    Returns one row per cutoff: n_measurements, then n complete in each group.
    """
    groups = groups.reindex(present.index)
    rows = []
    for cutoff in cutoffs:
        keep = score.index[score >= cutoff]
        complete = present[keep].all(axis=1)
        n = groups[complete].value_counts().reindex(groups.unique(), fill_value=0)
        rows.append({"cutoff": cutoff, "n_measurements": len(keep), **n.add_prefix("complete_")})
    return pd.DataFrame(rows).set_index("cutoff")
