"""Per-sample CNV summary from the array CNV pipeline (e.g. results/QC/<array>/<array>.txt).

One row per called sample (`identifier` = person_id), with PennCNV-style sample QC metrics (LRR_SD,
BAF_DRIFT, WF, GCWF, ...), per-chromosome CNV counts and lengths, and autosomal totals
(Num_Auto_Del, Len_Auto_Del, ...). People with no CNVs are rows too, so the file defines who was called.
"""

import pandas as pd

QC_METRICS = ["LRR_mean", "LRR_median", "LRR_SD", "BAF_mean", "BAF_median", "BAF_SD", "BAF_DRIFT",
              "WF", "GCWF", "Num_CNVs"]

# burden column -> summary column (lengths are in bp)
_BURDEN = {"del_n": "Num_Auto_Del", "dup_n": "Num_Auto_Dup",
           "del_len": "Len_Auto_Del", "dup_len": "Len_Auto_Dup"}


def read(path, id_col: str = "identifier") -> pd.DataFrame:
    """Read the tab-separated summary, indexed by person_id (int64). Raises on duplicate IDs."""
    df = pd.read_csv(path, sep="\t")
    df.index = pd.Index(df.pop(id_col).astype("int64"), name="person_id")
    if not df.index.is_unique:
        raise ValueError(f"{path} has duplicate {id_col} values")
    return df


def burden(summary: pd.DataFrame, people=None) -> pd.DataFrame:
    """Autosomal CNV burden in the same columns as `aou_utils.cnv.segments.burden`:
    del_mb, dup_mb, cnv_mb, del_n, dup_n, cnv_n. Optionally limited to (and ordered by) `people`,
    who must all be in the summary.
    """
    if people is not None:
        people = pd.Index(people, name="person_id")
        missing = people.difference(summary.index)
        if len(missing):
            raise KeyError(f"{len(missing)} people aren't in the CNV summary")
        summary = summary.loc[people]
    out = pd.DataFrame(index=summary.index)
    out["del_mb"] = summary[_BURDEN["del_len"]].astype("float64") / 1e6
    out["dup_mb"] = summary[_BURDEN["dup_len"]].astype("float64") / 1e6
    out["cnv_mb"] = out["del_mb"] + out["dup_mb"]
    out["del_n"] = summary[_BURDEN["del_n"]].astype("int64")
    out["dup_n"] = summary[_BURDEN["dup_n"]].astype("int64")
    out["cnv_n"] = out["del_n"] + out["dup_n"]
    return out


def qc_pass(summary: pd.DataFrame, max_lrr_sd: float = None, max_baf_drift: float = None,
            max_abs_wf: float = None, max_abs_gcwf: float = None, max_cnvs: int = None) -> pd.Series:
    """Boolean per sample: passes every given sample-QC threshold (None = not applied).
    Samples with a missing value for an applied metric fail."""
    ok = pd.Series(True, index=summary.index)
    checks = [("LRR_SD", max_lrr_sd, False), ("BAF_DRIFT", max_baf_drift, False),
              ("WF", max_abs_wf, True), ("GCWF", max_abs_gcwf, True), ("Num_CNVs", max_cnvs, False)]
    for col, limit, absolute in checks:
        if limit is None:
            continue
        x = summary[col].abs() if absolute else summary[col]
        ok &= x.le(limit)  # NaN -> False
    return ok
