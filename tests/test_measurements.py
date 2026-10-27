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


def test_coverage_and_complete_cases():
    present = pd.DataFrame(
        {10: [True, True, False, True], 11: [True, False, False, True]},
        index=pd.Index([1, 2, 3, 4], name="person_id"),
    )
    groups = pd.Series({1: "DEL", 2: "DEL", 3: "NC", 4: "NC"})
    counts, fractions = measurements.coverage(present, groups)
    assert counts.loc[10, "DEL"] == 2 and counts.loc[11, "DEL"] == 1
    assert fractions.loc[10, "NC"] == 0.5

    score = fractions["DEL"]
    table = measurements.complete_cases(present, score, groups, [0.5, 1.0])
    assert table.loc[1.0, "n_measurements"] == 1  # only concept 10 has full DEL coverage
    assert table.loc[1.0, "complete_DEL"] == 2
    assert table.loc[0.5, "complete_DEL"] == 1  # needs both 10 and 11


def test_boxcox_by_group_fits_separate_lambdas(rng):
    n = 4000
    sex = pd.Series(np.where(np.arange(n) < n // 2, "female", "male"))
    sex.iloc[0] = np.nan
    x = np.where(sex == "female", np.exp(rng.normal(size=n)), rng.normal(10, 1, size=n) ** 2)
    wide = pd.DataFrame({"a": x})
    transformed, params = measurements.boxcox(wide, groups=sex)
    assert set(params.columns) == {"lambda_female", "shift_female", "lambda_male", "shift_male"}
    assert abs(params.loc["a", "lambda_female"]) < 0.1  # lognormal -> log
    assert params.loc["a", "lambda_male"] > 0.3  # squared normal -> closer to sqrt
    assert np.isnan(transformed.loc[0, "a"])  # no group -> not transformed
    assert transformed["a"].notna().sum() == n - 1


def test_boxcox_by_group_skips_degenerate_group():
    wide = pd.DataFrame({"a": [1.0, 2.0, 3.5, 4.0, 1.0, 1.0]})
    groups = pd.Series(["f", "f", "f", "f", "m", "m"])
    transformed, params = measurements.boxcox(wide, groups=groups)
    assert transformed["a"].iloc[:4].notna().all() and transformed["a"].iloc[4:].isna().all()
    assert "lambda_m" not in params.columns
