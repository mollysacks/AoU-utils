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


def transform(values: pd.DataFrame, ages: pd.DataFrame, sex: pd.Series, covariates: pd.DataFrame):
    """Box-Cox each trait with lambda fit separately by sex, then regress out age, age² and
    `covariates` (e.g. female indicator + PC1-10) and z-score the residuals.

    Returns (z, boxcox_params). People with no sex, or missing a covariate, get NaN.
    """
    transformed, params = measurements.boxcox(values, groups=sex)
    z = measurements.residualize(transformed, ages[transformed.columns], covariates)
    return z, params
