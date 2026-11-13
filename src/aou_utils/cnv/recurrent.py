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
    # float32 holds whole-number copy numbers exactly and allows NaN for missing calls
    return calls.astype("float32")


def match_loci(names, columns) -> dict:
    """Map each locus name to the column it refers to, ignoring '-', '_' and '.' differences."""
    by_key = {normalize(c): c for c in columns}
    missing = [n for n in names if normalize(n) not in by_key]
    if missing:
        raise KeyError(f"No CNV column for: {missing}")
    return {n: by_key[normalize(n)] for n in names}


def _locus_allele(entry: str):
    """Split a locus entry like '22q11.21_A-D:DEL' into ('22q11.21_A-D', 'DEL'); no suffix means both."""
    name, _, allele = str(entry).partition(":")
    if allele not in ("", "DEL", "DUP"):
        raise ValueError(f"Allele must be DEL or DUP, not {allele!r} (in {entry!r})")
    return name, allele or None


def classify(calls: pd.DataFrame, classes: dict, split_copy: bool = True, exclude=()) -> pd.Series:
    """Group label per person from copy-number calls (2 = normal).

    `classes` maps a class name to its loci (names are matched to columns with `match_loci`).
    A carrier of any of a class's loci is labeled '<class>_DEL' or '<class>_DUP' (copy < 2 or > 2),
    or just '<class>' with split_copy=False. People with copy number 2 at every classified locus
    are 'NC'. People carrying CNVs with more than one label are 'multiple'; people with no CNV but
    a missing call at some classified locus get NaN.

    A locus can be limited to one allele with a suffix, e.g. '22q11.21_A-D:DEL'. Carriers of a
    classified locus's other allele are labeled 'excluded'. So are carriers of any locus in
    `exclude`, whatever else they carry, so they end up in neither a carrier group nor NC (e.g.
    common CNVs that would dominate a class).
    """
    carrier = {}
    all_columns, del_columns, dup_columns = [], set(), set()
    for name, loci in classes.items():
        parsed = [_locus_allele(entry) for entry in loci]
        columns = match_loci([locus for locus, _ in parsed], calls.columns)
        dels = [columns[locus] for locus, allele in parsed if allele != "DUP"]
        dups = [columns[locus] for locus, allele in parsed if allele != "DEL"]
        all_columns += list(columns.values())
        del_columns.update(dels)
        dup_columns.update(dups)
        is_del = (calls[dels] < 2).fillna(False).any(axis=1)
        is_dup = (calls[dups] > 2).fillna(False).any(axis=1)
        if split_copy:
            carrier[f"{name}_DEL"] = is_del
            carrier[f"{name}_DUP"] = is_dup
        else:
            carrier[name] = is_del | is_dup
    carrier = pd.DataFrame(carrier)
    n_labels = carrier.sum(axis=1)

    groups = pd.Series(pd.NA, index=calls.index, dtype="object")
    groups[n_labels == 1] = carrier[n_labels == 1].idxmax(axis=1)
    groups[n_labels > 1] = "multiple"
    no_cnv = (n_labels == 0) & calls[all_columns].notna().all(axis=1)
    groups[no_cnv] = "NC"
    all_columns = list(dict.fromkeys(all_columns))
    other_allele = (calls[[c for c in all_columns if c not in del_columns]] < 2).fillna(False).any(axis=1) | \
                   (calls[[c for c in all_columns if c not in dup_columns]] > 2).fillna(False).any(axis=1)
    groups[other_allele] = "excluded"
    if exclude:
        cn = calls[list(match_loci(exclude, calls.columns).values())]
        groups[((cn != 2) & cn.notna()).any(axis=1)] = "excluded"
    return groups
