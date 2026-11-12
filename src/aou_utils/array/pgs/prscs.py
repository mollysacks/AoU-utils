"""Inputs for PRS-CS (src/wdl/prscs.wdl): summary statistics and the GRCh38 HapMap3 SNP list.

PRS-CS identifies SNPs by rsID. Its LD reference (snpinfo_ukbb_hm3 / snpinfo_1kg_hm3) gives GRCh37
positions; the AoU genotype data are GRCh38. `lift_snpinfo` lifts the reference SNPs to GRCh38 so
the WDL can find them in the target data by position and alleles.
"""

import numpy as np
import pandas as pd

from .liftover import lift

AUTOSOMES = range(1, 23)
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_ATGC = {"A", "C", "G", "T"}


def read_snpinfo(path) -> pd.DataFrame:
    """The LD reference's snpinfo file: CHR SNP BP A1 A2 MAF (GRCh37 positions)."""
    return pd.read_csv(path, sep=r"\s+", dtype={"CHR": int, "SNP": str, "BP": int, "A1": str, "A2": str})


def lift_snpinfo(snpinfo: pd.DataFrame, chain: pd.DataFrame) -> pd.DataFrame:
    """Reference SNPs with GRCh38 positions: CHR SNP BP A1 A2, ready for the WDL's `hm3_grch38`.

    Drops SNPs that don't lift or that land on another chromosome or contig. Alleles are
    complemented where the chain maps to the minus strand, so they're on the GRCh38 plus strand.
    """
    lifted = lift(snpinfo["CHR"], snpinfo["BP"], chain)
    same_chrom = lifted["chrom"].to_numpy() == ("chr" + snpinfo["CHR"].astype(str)).to_numpy()
    keep = same_chrom & lifted["pos"].notna().to_numpy()
    out = snpinfo.loc[keep, ["CHR", "SNP", "A1", "A2"]].copy()
    out["BP"] = lifted.loc[keep, "pos"].astype("int64").to_numpy()
    minus = (lifted.loc[keep, "strand"] == "-").to_numpy()
    for a in ("A1", "A2"):
        out.loc[minus, a] = out.loc[minus, a].str.translate(_COMPLEMENT)
    out = out.drop_duplicates("SNP", keep=False)
    return out[["CHR", "SNP", "BP", "A1", "A2"]].sort_values(["CHR", "BP"]).reset_index(drop=True)


def format_sumstats(df: pd.DataFrame, snp: str, a1: str, a2: str, beta: str = None,
                    odds_ratio: str = None, se: str = None, p: str = None) -> pd.DataFrame:
    """GWAS summary statistics in PRS-CS format: SNP A1 A2 BETA|OR SE|P.

    a1 is the effect allele. Give one of beta / odds_ratio and one of se / p (PRS-CS recommends SE;
    with P, p-values under 1e-323 get truncated). Keeps single-nucleotide A/C/G/T variants with
    complete values, and drops rsIDs that appear more than once.
    """
    if (beta is None) == (odds_ratio is None):
        raise ValueError("Give exactly one of beta or odds_ratio")
    if (se is None) == (p is None):
        raise ValueError("Give exactly one of se or p")
    effect, eff_name = (beta, "BETA") if beta is not None else (odds_ratio, "OR")
    stat, stat_name = (se, "SE") if se is not None else (p, "P")
    out = pd.DataFrame({
        "SNP": df[snp].astype(str),
        "A1": df[a1].astype(str).str.upper(),
        "A2": df[a2].astype(str).str.upper(),
        eff_name: pd.to_numeric(df[effect], errors="coerce"),
        stat_name: pd.to_numeric(df[stat], errors="coerce"),
    })
    ok = out["A1"].isin(_ATGC) & out["A2"].isin(_ATGC) & out[[eff_name, stat_name]].notna().all(axis=1)
    ok &= out["SNP"].str.startswith("rs")
    if eff_name == "OR":
        ok &= out["OR"] > 0
    out = out[ok]
    return out[~out["SNP"].duplicated(keep=False)].reset_index(drop=True)


def _position_keys(chrom, pos, x, y) -> pd.Series:
    """'chrom:pos:alleleA/alleleB' with the alleles sorted, so allele order doesn't matter."""
    c = pd.Series(chrom).astype(str).str.replace("^chr", "", regex=True).reset_index(drop=True)
    p = pd.Series(pos).astype("int64").astype(str).reset_index(drop=True)
    x = pd.Series(x).astype(str).str.upper().reset_index(drop=True)
    y = pd.Series(y).astype(str).str.upper().reset_index(drop=True)
    lo, hi = x.where(x < y, y), y.where(x < y, x)
    return c + ":" + p + ":" + lo + "/" + hi


def add_rsids(df: pd.DataFrame, chrom: str, pos: str, a1: str, a2: str, reference: pd.DataFrame) -> pd.Series:
    """rsIDs for summary statistics that only have positions, matched to the reference SNP list.

    `reference` is `read_snpinfo(...)` for GRCh37 positions or `lift_snpinfo(...)` for GRCh38;
    use whichever build the summary statistics are on. Matches on chromosome, position and alleles
    (either order, either strand). Returns a Series aligned to df, NaN where nothing matched.
    """
    ref_a1, ref_a2 = reference["A1"].str.upper(), reference["A2"].str.upper()
    snps = reference["SNP"].reset_index(drop=True)
    ref_keys = pd.concat([
        pd.DataFrame({"key": _position_keys(reference["CHR"], reference["BP"], ref_a1, ref_a2), "SNP": snps}),
        pd.DataFrame({"key": _position_keys(reference["CHR"], reference["BP"], ref_a1.str.translate(_COMPLEMENT),
                                            ref_a2.str.translate(_COMPLEMENT)), "SNP": snps}),
    ]).drop_duplicates()
    # A key that points at two different rsIDs is ambiguous; don't use it
    ref_keys = ref_keys[~ref_keys["key"].duplicated(keep=False)]
    lookup = pd.Series(ref_keys["SNP"].to_numpy(), index=ref_keys["key"].to_numpy())

    out = pd.Series(np.nan, index=df.index, dtype=object)
    has_pos = df[pos].notna()
    sub = df[has_pos]
    out[has_pos] = _position_keys(sub[chrom], sub[pos], sub[a1], sub[a2]).map(lookup).to_numpy()
    return out


def chrom_paths(template: str, chroms=AUTOSOMES) -> list:
    """Per-chromosome paths from a template with a {chrom} field, e.g. 'gs://b/imputed/chr{chrom}.pvar'."""
    return [template.format(chrom=c) for c in chroms]
