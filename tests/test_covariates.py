import numpy as np
import pandas as pd
import pytest

from aou_utils import covariates


def test_read_ancestry(tmp_path):
    path = tmp_path / "ancestry_preds.tsv"
    path.write_text(
        "research_id\tancestry_pred\tprobabilities\tpca_features\n"
        "1\teur\t[0.9, 0.1]\t[0.1, -0.2, 3e-3, 4.0]\n"
        "2\tafr\t[0.2, 0.8]\t[1.0, 2.0, 3.0, 4.0]\n"
    )
    out = covariates.read_ancestry(path, n_pcs=3)
    assert out.index.tolist() == [1, 2] and out.index.name == "person_id"
    assert list(out.columns) == ["ancestry", "PC1", "PC2", "PC3"]
    assert out.loc[1, "PC3"] == pytest.approx(0.003) and out.loc[2, "ancestry"] == "afr"
    with pytest.raises(ValueError, match="only 4 PCs"):
        covariates.read_ancestry(path, n_pcs=10)


def test_sex_at_birth_and_indicator(monkeypatch):
    monkeypatch.setattr(covariates.bq, "query", lambda sql, params=None: pd.DataFrame(
        {"person_id": params["person_ids"], "sex_at_birth": ["Female", "Male", "Intersex"]}))
    sex = covariates.sex_at_birth([1, 2, 3])
    assert sex.loc[2] == "Male"
    female = covariates.female_indicator(sex)
    assert female.loc[1] == 1.0 and female.loc[2] == 0.0 and np.isnan(female.loc[3])
