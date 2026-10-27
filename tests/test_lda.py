import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from aou_utils import lda  # noqa: E402


@pytest.fixture
def data():
    """3 groups; feature 'a' separates DEL < NC < DUP, 'b' separates carriers from NC, 'c' is noise."""
    rng = np.random.default_rng(0)
    sizes = {"DEL": 150, "DUP": 200, "NC": 3000}
    shifts = {"DEL": (-1.5, 1.0), "DUP": (1.5, 1.0), "NC": (0.0, 0.0)}
    groups, rows = [], []
    for g, n in sizes.items():
        a = rng.normal(shifts[g][0], 1, n)
        b = rng.normal(shifts[g][1], 1, n)
        rows.append(np.column_stack([a, b, rng.normal(size=n)]))
        groups += [g] * n
    X = pd.DataFrame(np.vstack(rows), columns=["a", "b", "c"])
    X.index.name = "person_id"
    y = pd.Series(groups, index=X.index)
    return X, y


def test_split_fit_holdout(data):
    _, y = data
    fit = lda.split_fit_holdout(y, fit_frac=0.2, seed=1)
    assert fit[y != "NC"].all()
    assert fit[y == "NC"].sum() == 600
    assert fit.equals(lda.split_fit_holdout(y, fit_frac=0.2, seed=1))


def test_fit_orients_and_loads_on_informative_features(data):
    X, y = data
    X.iloc[0, 2] = np.nan  # incomplete rows are dropped from the fit and get NaN scores
    result = lda.fit(X, y)
    scores = result.transform(X)
    assert scores.iloc[0].isna().all()
    means = scores.groupby(y).mean()
    assert (means.loc["DUP"] > means.loc["DEL"]).all()

    loadings = lda.structure(X, scores)
    ld1 = loadings["LD1"].abs()
    assert ld1["a"] > 0.8 and ld1["c"] < 0.1
    assert result.explained_variance_ratio.sum() == pytest.approx(1)


def test_cross_validate(data):
    X, y = data
    cv = lda.cross_validate(X, y, n_permutations=5)
    assert cv["auc_pair"] > 0.9
    assert cv["auc_ovr"] > 0.7
    assert cv["permutation_p"] == pytest.approx(1 / 6)
    assert len(cv["permutation_aucs"]) == 5
