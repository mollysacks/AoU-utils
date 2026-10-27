"""Recurrent CNV calls: loading, matching locus names, and grouping people by CNV class."""

import re

import pandas as pd

# CNV columns are named by cytoband, e.g. "16p11.2_BP4_BP5"; metadata columns aren't
_LOCUS_COLUMN = re.compile(r"^\d{1,2}[pq]\d")


def normalize(name: str) -> str:
    """Locus name with runs of '-', '_' and '.' collapsed to '_', for matching across sources."""
    return re.sub(r"[-_.]+", "_", str(name).strip())


def load_calls(path, id_col: str = "identifier", skip_suffix: str = "_oneway") -> pd.DataFrame:
    """Copy-number calls (person_id x locus) from a recurrent CNV genomatrix CSV.

    Keeps columns named by cytoband, minus those ending in `skip_suffix` (the '_oneway' columns can
    assign one CNV to several loci; the plain columns are mutually exclusive).
    """
    header = pd.read_csv(path, nrows=0).columns
    loci = [c for c in header if _LOCUS_COLUMN.match(c) and not c.endswith(skip_suffix)]
    calls = pd.read_csv(path, usecols=[id_col, *loci]).rename(columns={id_col: "person_id"})
    calls = calls.set_index("person_id")
    if not calls.index.is_unique:
        raise ValueError(f"{path} has duplicate {id_col} values")
    return calls.astype("Int8")


def match_loci(names, columns) -> dict:
    """Map each locus name to the column it refers to, ignoring '-', '_' and '.' differences."""
    by_key = {normalize(c): c for c in columns}
    missing = [n for n in names if normalize(n) not in by_key]
    if missing:
        raise KeyError(f"No CNV column for: {missing}")
    return {n: by_key[normalize(n)] for n in names}


def classify(calls: pd.DataFrame, classes: dict, split_copy: bool = True) -> pd.Series:
    """Group label per person from copy-number calls (2 = normal).

    `classes` maps a class name to its loci (names are matched to columns with `match_loci`).
    A carrier of any of a class's loci is labeled '<class>_DEL' or '<class>_DUP' (copy < 2 or > 2),
    or just '<class>' with split_copy=False. People with copy number 2 at every classified locus
    are 'NC'. People carrying CNVs with more than one label are 'multiple'; people with no CNV but
    a missing call at some classified locus get NaN.
    """
    carrier = {}
    all_columns = []
    for name, loci in classes.items():
        columns = list(match_loci(loci, calls.columns).values())
        all_columns += columns
        cn = calls[columns]
        if split_copy:
            carrier[f"{name}_DEL"] = (cn < 2).fillna(False).any(axis=1)
            carrier[f"{name}_DUP"] = (cn > 2).fillna(False).any(axis=1)
        else:
            carrier[name] = (cn != 2).fillna(False).any(axis=1)
    carrier = pd.DataFrame(carrier)
    n_labels = carrier.sum(axis=1)

    groups = pd.Series(pd.NA, index=calls.index, dtype="object")
    groups[n_labels == 1] = carrier[n_labels == 1].idxmax(axis=1)
    groups[n_labels > 1] = "multiple"
    no_cnv = (n_labels == 0) & calls[all_columns].notna().all(axis=1)
    groups[no_cnv] = "NC"
    return groups
