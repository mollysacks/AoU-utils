import numpy as np
import pandas as pd
import pytest

from aou_utils.analysis.cnv_burden import models, traits


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    n = 20_000
    carrier_del = rng.random(n) < 0.3
    carrier_dup = rng.random(n) < 0.3
    d = pd.DataFrame({
        "del_mb": np.where(carrier_del, rng.exponential(0.5, n), 0.0),
        "dup_mb": np.where(carrier_dup, rng.exponential(0.5, n), 0.0),
        "female": rng.integers(0, 2, n).astype(float),
        "PC1": rng.normal(size=n),
    })
    d["cnv_mb"] = d["del_mb"] + d["dup_mb"]
    # DEL lowers the trait, DUP has no effect
    d["y"] = -0.4 * d["del_mb"] + 0.5 * d["female"] + 0.3 * d["PC1"] + rng.normal(size=n)
    return d


def test_burden_test_recovers_effects(data):
    res = models.burden_test(data, "y", covariates=["female", "PC1"]).set_index(["model", "predictor"])
    assert set(res.index.get_level_values("model")) == set(models.DEFAULT_MODELS)
    d = res.loc[("DEL", "del_mb")]
    assert d["ci_low"] < -0.4 < d["ci_high"] and d["p"] < 1e-10
    assert abs(res.loc[("DUP", "dup_mb"), "beta"]) < 0.05 and res.loc[("DUP", "dup_mb"), "p"] > 1e-3
    joint = res.loc["DEL+DUP"]
    assert len(joint) == 2 and joint["delta_r2"].nunique() == 1
    assert joint.loc["del_mb", "beta"] == pytest.approx(-0.4, abs=0.05)
    assert res.loc[("DEL", "del_mb"), "delta_r2"] > res.loc[("DUP", "dup_mb"), "delta_r2"]


def test_burden_test_matches_lstsq_and_r2(data):
    res = models.burden_test(data, "y", covariates=["female", "PC1"], models={"DEL": ["del_mb"]})
    X = np.column_stack([np.ones(len(data)), data[["del_mb", "female", "PC1"]]])
    beta, *_ = np.linalg.lstsq(X, data["y"].to_numpy(), rcond=None)
    assert res.loc[0, "beta"] == pytest.approx(beta[1])
    y = data["y"].to_numpy()
    r2 = lambda X: 1 - ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert res.loc[0, "delta_r2"] == pytest.approx(r2(X) - r2(X[:, [0, 2, 3]]))
    # with one predictor the F test and the t test agree
    assert res.loc[0, "f_p"] == pytest.approx(res.loc[0, "p"], rel=1e-6)


def test_burden_test_complete_cases_and_counts(data):
    data.loc[:99, "y"] = np.nan
    res = models.burden_test(data, "y", covariates=["female"], models={"DEL": ["del_mb"]})
    assert res.loc[0, "n"] == len(data) - 100
    assert res.loc[0, "n_carriers"] == int((data.loc[100:, "del_mb"] > 0).sum())


def test_burden_test_bootstrap(data):
    small = data.iloc[:3000]
    res = models.burden_test(small, "y", covariates=["female"], models={"DEL": ["del_mb"]}, n_boot=50)
    r = res.loc[0]
    assert r["boot_beta_low"] < r["beta"] < r["boot_beta_high"]
    assert 0 <= r["boot_delta_r2_low"] < r["delta_r2"] < r["boot_delta_r2_high"]


def test_burden_test_too_few_rows():
    d = pd.DataFrame({"y": [1.0, 2.0], "del_mb": [0.0, 1.0]})
    with pytest.raises(ValueError, match="complete rows"):
        models.burden_test(d, "y", models={"DEL": ["del_mb"]})


def test_fetch_queries_trait_concepts(monkeypatch):
    seen = {}

    def fake_query(sql, params=None, **fmt):
        seen.update(sql=sql, params=params, fmt=fmt)
        return pd.DataFrame({"person_id": [1, 1, 2], "concept_id": [traits.HEIGHT, traits.BMI, traits.HEIGHT],
                             "unit_concept_id": 0, "value": [170.0, 25.0, 160.0], "age": [40.0, 41.0, 50.0],
                             "n_records": 1, "n_pm_records": 1})

    monkeypatch.setattr(traits.measurements.bq, "query", fake_query)
    values, ages = traits.fetch([1, 2])
    assert seen["params"]["concept_ids"] == [traits.HEIGHT, traits.BMI]
    assert seen["params"]["min_age"] == 18.0
    assert "@concept_ids" in seen["fmt"]["extra_filters"] and "@min_age" in seen["fmt"]["extra_filters"]
    assert values.loc[1, "height"] == 170.0 and np.isnan(values.loc[2, "bmi"])
    assert ages.loc[2, "height"] == 50.0


def test_person_medians_without_filters_is_unchanged(monkeypatch):
    seen = {}
    monkeypatch.setattr(traits.measurements.bq, "query",
                        lambda sql, params=None, **fmt: seen.update(params=params, fmt=fmt))
    traits.measurements.person_medians([1, 2])
    assert seen["fmt"]["extra_filters"] == "" and set(seen["params"]) == {"person_ids", "min_people", "trim_permille"}


@pytest.fixture
def trait_data():
    rng = np.random.default_rng(2)
    n = 4000
    idx = pd.Index(range(n), name="person_id")
    sex = pd.Series(np.where(rng.random(n) < 0.5, "Female", "Male"), index=idx)
    female = (sex == "Female").astype(float).rename("female")
    age = pd.Series(rng.uniform(20, 80, n), index=idx)
    values = pd.DataFrame({"height": 175 - 13 * female + rng.normal(0, 7, n),
                           "bmi": np.exp(rng.normal(3.3, 0.25, n) + 0.1 * female)}, index=idx)
    ages = pd.DataFrame({"height": age, "bmi": age})
    return values, ages, sex, female


def test_transform_residualizes_then_boxcox(trait_data):
    from scipy import stats

    values, ages, _, female = trait_data
    z, params = traits.transform(values, ages, female.to_frame())
    assert set(params.columns) == {"lambda", "shift"} and (params["shift"] > 0).all()  # residuals shifted
    assert z.mean().abs().max() < 1e-4 and (z.std() - 1).abs().max() < 1e-3
    assert abs(np.corrcoef(z["height"], female)[0, 1]) < 0.02  # sex regressed out before Box-Cox
    assert abs(stats.skew(z["bmi"])) < abs(stats.skew(values["bmi"])) / 3  # skew of log-normal BMI removed


def test_transform_boxcox_by_group(trait_data):
    values, ages, sex, female = trait_data
    z, params = traits.transform(values, ages, female.to_frame(), groups=sex)
    assert set(params.columns) == {"lambda_Female", "lambda_Male", "shift_Female", "shift_Male"}
    assert z.notna().all().all()
