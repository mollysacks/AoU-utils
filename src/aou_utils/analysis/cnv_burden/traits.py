"""Quantitative traits for the burden tests: pull per-person medians from the measurement table
(EHR + physical measurements) and transform them for regression."""

import pandas as pd

from aou_utils.analysis.lda import measurements

# OMOP measurement_concept_ids (LOINC): Body height (8302-2), Body mass index (39156-5)
HEIGHT = 3036277
BMI = 3038553
TRAITS = {"height": HEIGHT, "bmi": BMI}


def fetch(person_ids, traits: dict = None, min_age: float = 18, trim_permille: int = 1):
    """Per-person median of each trait, from measurements taken at age >= `min_age`.

    `traits` maps a column name to its measurement_concept_id (default `TRAITS`). Returns
    (values, ages): person_id x trait DataFrames of the median value and the median age at measurement.
    Records in a concept's less common units, and those outside its [trim_permille, 1000 - trim_permille]
    per-mille range, are dropped (see `measurements.person_medians`).
    """
    traits = TRAITS if traits is None else traits
    medians = measurements.person_medians(person_ids, concept_ids=list(traits.values()),
                                          min_age=min_age, trim_permille=trim_permille)
    names = {concept: name for name, concept in traits.items()}
    values = measurements.to_wide(medians).rename(columns=names)
    ages = measurements.to_wide(medians, "age").rename(columns=names)
    values.columns.name = ages.columns.name = "trait"
    return values, ages


def transform(values: pd.DataFrame, ages: pd.DataFrame, covariates: pd.DataFrame, groups: pd.Series = None):
    """Regress age, age² and `covariates` (e.g. female indicator + PC1-10) out of each trait, Box-Cox
    the residuals, then z-score.

    Residuals are centered on 0, so Box-Cox shifts them by (1 - min) first (see `measurements.boxcox`).
    `groups` (e.g. sex) fits lambda separately within each group; by default one lambda per trait.
    Returns (z, boxcox_params). People missing a covariate (or a group, if given) get NaN.
    """
    resid = measurements.residualize(values, ages[values.columns], covariates)
    transformed, params = measurements.boxcox(resid, groups=groups)
    z = (transformed - transformed.mean()) / transformed.std()
    return z.astype("float32"), params
