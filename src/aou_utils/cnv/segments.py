"""Segment-level CNV calls and per-person CNV burden.

Calls are a DataFrame with one row per CNV and the columns in `COLUMNS`:
person_id, chrom ('1'..'22', 'X', 'Y'; no 'chr' prefix), start, end (bp, end > start), type ('DEL'/'DUP').
`standardize` builds that from a table with other column names or codings.
"""

import numpy as np
import pandas as pd

COLUMNS = ["person_id", "chrom", "start", "end", "type"]
AUTOSOMES = tuple(str(c) for c in range(1, 23))
TYPES = ("DEL", "DUP")


def _chrom(values: pd.Series) -> pd.Series:
    """'chr1' / 1 / '1' -> '1'; 'chrX' / 23 -> 'X'; 24 -> 'Y' (PLINK numbering)."""
    c = values.astype(str).str.strip().str.replace(r"^chr", "", regex=True, case=False).str.upper()
    return c.replace({"23": "X", "24": "Y"})


def standardize(calls: pd.DataFrame, person_id: str = "person_id", chrom: str = "chrom",
                start: str = "start", end: str = "end", type: str = "type",
                del_values=("DEL",), dup_values=("DUP",)) -> pd.DataFrame:
    """Rename and recode a table of CNV calls to the standard `COLUMNS`.

    `del_values` / `dup_values` are the values of the `type` column that mean a deletion or a
    duplication (e.g. copy numbers (0, 1) and (3, 4) for a PLINK .cnv TYPE column). They are compared
    as strings, case-insensitively. Rows of any other type are dropped. Raises if a CNV has end <= start.
    """
    out = pd.DataFrame({
        "person_id": calls[person_id].to_numpy(),
        "chrom": _chrom(calls[chrom]).to_numpy(),
        "start": calls[start].astype("int64").to_numpy(),
        "end": calls[end].astype("int64").to_numpy(),
    })
    raw_type = calls[type].astype(str).str.strip().str.upper().to_numpy()
    dels = {str(v).upper() for v in del_values}
    dups = {str(v).upper() for v in dup_values}
    out["type"] = np.select([np.isin(raw_type, list(dels)), np.isin(raw_type, list(dups))],
                            ["DEL", "DUP"], default="")
    out = out[out["type"] != ""].reset_index(drop=True)
    bad = out["end"] <= out["start"]
    if bad.any():
        raise ValueError(f"{int(bad.sum())} CNVs have end <= start")
    return out


def sizes(calls: pd.DataFrame) -> pd.Series:
    """Length of each CNV in bp."""
    return calls["end"] - calls["start"]


def filter_calls(calls: pd.DataFrame, min_size: int = 0, max_size: int = None,
                 chroms=AUTOSOMES, types=TYPES, people=None) -> pd.DataFrame:
    """Keep CNVs with min_size <= length (bp) [<= max_size], on `chroms`, of `types`, and (if given)
    carried by `people`."""
    length = sizes(calls)
    keep = (length >= min_size) & calls["chrom"].isin([str(c) for c in chroms]) & calls["type"].isin(types)
    if max_size is not None:
        keep &= length <= max_size
    if people is not None:
        keep &= calls["person_id"].isin(people)
    return calls[keep].reset_index(drop=True)


def merge_overlaps(calls: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping CNVs of the same type in the same person into one CNV.

    Callers can emit overlapping or nested calls; merging keeps those bases from being counted twice
    in the Mb burden. CNVs that only touch (start == previous end) are merged too.
    """
    if calls.empty:
        return calls[COLUMNS].copy()
    key = ["person_id", "type", "chrom"]
    s = calls[COLUMNS].sort_values(key + ["start", "end"]).reset_index(drop=True)
    grouped = [s[k] for k in key]
    # Furthest end seen so far in this person/type/chrom, before the current CNV
    prev_end = s.groupby(key, sort=False)["end"].cummax().groupby(grouped, sort=False).shift()
    block = (prev_end.isna() | (s["start"] > prev_end)).cumsum()
    merged = s.groupby(block, sort=False).agg(
        person_id=("person_id", "first"), chrom=("chrom", "first"),
        start=("start", "min"), end=("end", "max"), type=("type", "first"),
    )
    return merged[COLUMNS].reset_index(drop=True)


def burden(calls: pd.DataFrame, people) -> pd.DataFrame:
    """Per-person CNV burden, after merging overlapping calls.

    Returns a DataFrame indexed by person_id (every person in `people`, in order, 0 for people with
    no CNVs) with:
      del_mb, dup_mb, cnv_mb  total Mb deleted / duplicated / either
      del_n, dup_n, cnv_n     number of CNVs (merged)
    Filter the calls first (size, chromosomes, QC) to get the burden of a subset, e.g. CNVs > 1 Mb.
    CNVs carried by people not in `people` are ignored.
    """
    people = pd.Index(people, name="person_id")
    if not people.is_unique:
        raise ValueError("people has duplicate IDs")
    merged = merge_overlaps(calls[calls["person_id"].isin(people)])
    merged["mb"] = sizes(merged) / 1e6
    mb = merged.pivot_table(index="person_id", columns="type", values="mb", aggfunc="sum")
    n = merged.pivot_table(index="person_id", columns="type", values="mb", aggfunc="count")
    out = pd.DataFrame(index=people)
    for t in TYPES:
        out[f"{t.lower()}_mb"] = mb[t].reindex(people).fillna(0.0).to_numpy() if t in mb else 0.0
        out[f"{t.lower()}_n"] = n[t].reindex(people).fillna(0).astype("int64").to_numpy() if t in n else 0
    out["cnv_mb"] = out["del_mb"] + out["dup_mb"]
    out["cnv_n"] = out["del_n"] + out["dup_n"]
    return out[["del_mb", "dup_mb", "cnv_mb", "del_n", "dup_n", "cnv_n"]]
