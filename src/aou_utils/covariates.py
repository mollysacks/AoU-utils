"""Standard covariates: sex at birth (from the CDR) and genetic ancestry PCs (from the srWGS
ancestry predictions file)."""

import re

import numpy as np
import pandas as pd

from . import bq

SEX_AT_BIRTH_SQL = """
SELECT p.person_id, c.concept_name AS sex_at_birth
FROM `{cdr}.person` AS p
LEFT JOIN `{cdr}.concept` AS c ON c.concept_id = p.sex_at_birth_concept_id
WHERE p.person_id IN UNNEST(@person_ids)
"""


def sex_at_birth(person_ids) -> pd.Series:
    """Sex at birth concept name per person ('Male', 'Female', or another answer / skip / None)."""
    df = bq.query(SEX_AT_BIRTH_SQL, params={"person_ids": list(person_ids)})
    return df.set_index("person_id")["sex_at_birth"]


def female_indicator(sex: pd.Series) -> pd.Series:
    """1.0 for 'Female', 0.0 for 'Male', NaN otherwise (for use as a covariate)."""
    return sex.map({"Female": 1.0, "Male": 0.0}).astype("float64").rename("female")


def _parse_vector(text) -> np.ndarray:
    """'[0.1, -2e-3, ...]' (or space-separated) -> float array."""
    return np.array([float(x) for x in re.split(r"[\s,]+", str(text).strip("[] \n")) if x])


def read_ancestry(path, n_pcs: int = 10, id_col: str = "research_id",
                  pcs_col: str = "pca_features", ancestry_col: str = "ancestry_pred") -> pd.DataFrame:
    """Ancestry predictions file (e.g. the srWGS ancestry_preds.tsv) -> DataFrame indexed by person_id
    with `ancestry` and PC1..PC<n_pcs>.

    The PCs are stored as one list-valued column (`pcs_col`). `path` is a local file: copy a
    requester-pays gs:// file locally first (gsutil -u <billing project> cp ...).
    """
    df = pd.read_csv(path, sep="\t", dtype={id_col: "int64"})
    pcs = np.vstack(df[pcs_col].map(_parse_vector).to_numpy())
    if pcs.shape[1] < n_pcs:
        raise ValueError(f"{path} has only {pcs.shape[1]} PCs")
    out = pd.DataFrame(pcs[:, :n_pcs], columns=[f"PC{i + 1}" for i in range(n_pcs)],
                       index=pd.Index(df[id_col].to_numpy(), name="person_id"))
    out.insert(0, "ancestry", df[ancestry_col].to_numpy())
    return out
