import gzip

import pandas as pd
import pytest

from aou_utils.array.pgs import liftover, prscs

# Two chains for chr1: a '+' chain with a gap, and a '-' chain onto chr1 of a 1000 bp query.
# Source positions 101-150 -> 1001-1050; 161-180 -> 1056-1075 (5 bp query gap);
# 501-510 -> minus strand, q_start 100, q_size 1000 (0-based q 100..109 on '-')
CHAIN = """chain 100 chr1 2000 + 100 180 chr1 3000 + 1000 1075 1
50 10 5
20

chain 50 chr1 2000 + 500 510 chr1 1000 - 100 110 2
10

chain 10 chr2 2000 + 0 10 chr5_alt 100 + 0 10 3
10
"""


@pytest.fixture
def chain(tmp_path):
    p = tmp_path / "test.over.chain.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(CHAIN)
    return liftover.read_chain(p)


def test_read_chain(chain):
    assert list(chain["start"]) == [100, 160, 500, 0]
    assert list(chain["q_start"]) == [1000, 1055, 100, 0]


def test_lift_plus_minus_and_unmapped(chain):
    out = liftover.lift(["1", "chr1", "1", "1", "1", "1"], [101, 150, 155, 161, 501, 999], chain)
    assert list(out["chrom"][:2]) == ["chr1", "chr1"]
    assert list(out["pos"][:2]) == [1001, 1050]
    assert pd.isna(out["pos"][2])          # in the gap
    assert out["pos"][3] == 1056
    # minus strand: 0-based q = 100 on '-' of a 1000 bp chrom -> plus-strand 0-based 899 -> 900
    assert out["pos"][4] == 900 and out["strand"][4] == "-"
    assert pd.isna(out["pos"][5]) and pd.isna(out["chrom"][5])


def test_lift_snpinfo(chain):
    snpinfo = pd.DataFrame({"CHR": [1, 1, 1, 2], "SNP": ["rs1", "rs2", "rs3", "rs4"],
                            "BP": [101, 155, 501, 5], "A1": ["A", "C", "A", "G"],
                            "A2": ["G", "T", "C", "T"], "MAF": [0.1, 0.2, 0.3, 0.4]})
    out = prscs.lift_snpinfo(snpinfo, chain)
    # rs2 in a gap, rs4 lands on an alt contig
    assert list(out["SNP"]) == ["rs3", "rs1"]
    assert list(out.columns) == ["CHR", "SNP", "BP", "A1", "A2"]
    rs3 = out.set_index("SNP").loc["rs3"]
    assert (rs3["BP"], rs3["A1"], rs3["A2"]) == (900, "T", "G")   # complemented on '-' strand


def test_format_sumstats():
    df = pd.DataFrame({"rsid": ["rs1", "rs2", "rs3", "rs3", "rs4", "1:5"],
                       "ea": ["a", "AT", "C", "C", "G", "A"], "oa": ["g", "A", "T", "T", "T", "C"],
                       "b": [0.1, 0.2, 0.3, 0.3, None, 0.1], "se": [0.01] * 6})
    out = prscs.format_sumstats(df, "rsid", "ea", "oa", beta="b", se="se")
    assert list(out.columns) == ["SNP", "A1", "A2", "BETA", "SE"]
    assert list(out["SNP"]) == ["rs1"]
    assert out.loc[0, "A1"] == "A"
    with pytest.raises(ValueError):
        prscs.format_sumstats(df, "rsid", "ea", "oa", beta="b", odds_ratio="b", se="se")


def test_format_sumstats_or_p():
    df = pd.DataFrame({"SNP": ["rs1", "rs2"], "A1": ["A", "C"], "A2": ["G", "T"],
                       "OR": [1.2, -1], "P": [0.01, 0.2]})
    out = prscs.format_sumstats(df, "SNP", "A1", "A2", odds_ratio="OR", p="P")
    assert list(out.columns) == ["SNP", "A1", "A2", "OR", "P"]
    assert list(out["SNP"]) == ["rs1"]


def test_add_rsids():
    ref = pd.DataFrame({"CHR": [1, 1, 2, 2], "SNP": ["rs1", "rs2", "rs3", "rs4"], "BP": [100, 200, 300, 300],
                        "A1": ["A", "C", "A", "A"], "A2": ["G", "T", "C", "C"]})
    gwas = pd.DataFrame({"chr": ["chr1", "1", "1", "2", "1"], "pos": [100, 200, 200, 300, None],
                         "ea": ["G", "G", "C", "A", "A"], "oa": ["A", "A", "G", "C", "G"]})
    out = prscs.add_rsids(gwas, "chr", "pos", "ea", "oa", ref)
    # swapped alleles; strand flip (C/T -> G/A); allele mismatch; ambiguous position (rs3/rs4); no pos
    assert out[0] == "rs1"
    assert out[1] == "rs2"
    assert pd.isna(out[2]) and pd.isna(out[3]) and pd.isna(out[4])


def test_chrom_paths():
    assert prscs.chrom_paths("gs://b/chr{chrom}.pvar", [1, 22]) == ["gs://b/chr1.pvar", "gs://b/chr22.pvar"]
