import numpy as np
import pandas as pd
import pytest

from aou_utils import measurements


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_to_wide():
    long = pd.DataFrame({"person_id": [1, 1, 2], "concept_id": [10, 11, 10], "value": [1.0, 2.0, 3.0]})
    wide = measurements.to_wide(long)
    assert wide.loc[1, 10] == 1.0 and wide.loc[2, 10] == 3.0 and np.isnan(wide.loc[2, 11])


def test_boxcox_recovers_log_for_lognormal(rng):
    wide = pd.DataFrame({"a": np.exp(rng.normal(size=5000)), "b": rng.normal(size=5000)})
    wide.loc[:9, "a"] = np.nan
    transformed, params = measurements.boxcox(wide)
    assert abs(params.loc["a", "lambda"]) < 0.1  # lognormal -> lambda near 0 (log)
    assert params.loc["a", "shift"] == 0
    assert params.loc["b", "shift"] > 0  # normal data has negatives, so it gets shifted
    assert transformed["a"].isna().sum() == 10


def test_boxcox_skips_near_constant_columns():
    wide = pd.DataFrame({"a": [1.0, 2.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0, 4.5]})
    transformed, params = measurements.boxcox(wide)
    assert list(transformed.columns) == ["b"]


def test_residualize_removes_covariate_effects(rng):
    n = 2000
    idx = pd.Index(range(n), name="person_id")
    age = pd.Series(rng.uniform(20, 80, n), index=idx)
    sex = pd.Series(rng.integers(0, 2, n), index=idx, dtype=float)
    y = 0.05 * age + 2 * sex + rng.normal(size=n)
    values = pd.DataFrame({"m": y})
    ages = pd.DataFrame({"m": age})
    covs = pd.DataFrame({"sex": sex})
    covs.iloc[0] = np.nan

    resid = measurements.residualize(values, ages, covs)["m"]
    assert np.isnan(resid.iloc[0])
    ok = resid.notna()
    assert abs(np.corrcoef(resid[ok], age[ok])[0, 1]) < 0.01
    assert abs(np.corrcoef(resid[ok], sex[ok])[0, 1]) < 0.01
    assert resid.std() == pytest.approx(1, abs=1e-3)
