import pandas as pd
import pytest

from aou_utils.cnv import segments


def calls(rows):
    return pd.DataFrame(rows, columns=segments.COLUMNS)


def test_standardize_plink_style():
    raw = pd.DataFrame({"IID": [1, 1, 2, 3], "CHR": [1, 23, "chr2", 5], "BP1": [100, 200, 300, 400],
                        "BP2": [200, 400, 500, 900], "TYPE": [1, 3, 0, 2]})
    out = segments.standardize(raw, person_id="IID", chrom="CHR", start="BP1", end="BP2", type="TYPE",
                               del_values=(0, 1), dup_values=(3, 4))
    assert list(out.columns) == segments.COLUMNS
    assert out["type"].tolist() == ["DEL", "DUP", "DEL"]  # copy number 2 dropped
    assert out["chrom"].tolist() == ["1", "X", "2"]


def test_standardize_rejects_bad_coordinates():
    raw = pd.DataFrame({"person_id": [1], "chrom": ["1"], "start": [500], "end": [500], "type": ["del"]})
    with pytest.raises(ValueError, match="end <= start"):
        segments.standardize(raw)


def test_filter_calls():
    c = calls([[1, "1", 0, 50_000, "DEL"], [1, "1", 0, 2_000_000, "DUP"], [2, "X", 0, 100_000, "DEL"]])
    assert len(segments.filter_calls(c, min_size=10_000)) == 2  # chrX dropped by default
    assert len(segments.filter_calls(c, min_size=1_000_000)) == 1
    assert len(segments.filter_calls(c, max_size=60_000, chroms=segments.AUTOSOMES + ("X",))) == 1
    assert len(segments.filter_calls(c, types=("DEL",), people=[2], chroms=("X",))) == 1


def test_merge_overlaps_merges_within_person_type_chrom_only():
    c = calls([
        [1, "1", 100, 300, "DEL"], [1, "1", 200, 400, "DEL"], [1, "1", 150, 250, "DEL"],  # -> 100-400
        [1, "1", 500, 600, "DEL"],                                                        # separate
        [1, "1", 250, 350, "DUP"],                                                        # other type
        [1, "2", 100, 300, "DEL"],                                                        # other chrom
        [2, "1", 100, 300, "DEL"],                                                        # other person
        [3, "1", 100, 1000, "DEL"], [3, "1", 200, 300, "DEL"], [3, "1", 900, 1100, "DEL"],  # nested, then overlaps the first
    ])
    m = segments.merge_overlaps(c)
    got = set(map(tuple, m[segments.COLUMNS].itertuples(index=False)))
    assert got == {(1, "1", 100, 400, "DEL"), (1, "1", 500, 600, "DEL"), (1, "1", 250, 350, "DUP"),
                   (1, "2", 100, 300, "DEL"), (2, "1", 100, 300, "DEL"), (3, "1", 100, 1100, "DEL")}


def test_burden_includes_people_without_cnvs():
    c = calls([[1, "1", 0, 1_500_000, "DEL"], [1, "1", 1_000_000, 2_000_000, "DEL"],
               [1, "3", 0, 500_000, "DUP"], [2, "2", 0, 250_000, "DUP"], [99, "1", 0, 10, "DEL"]])
    b = segments.burden(c, people=[1, 2, 3])
    assert b.index.tolist() == [1, 2, 3]
    assert b.loc[1, "del_mb"] == pytest.approx(2.0) and b.loc[1, "del_n"] == 1  # overlap merged
    assert b.loc[1, "dup_mb"] == pytest.approx(0.5) and b.loc[1, "cnv_mb"] == pytest.approx(2.5)
    assert b.loc[2, "del_mb"] == 0 and b.loc[2, "dup_n"] == 1 and b.loc[2, "cnv_n"] == 1
    assert (b.loc[3] == 0).all()


def test_burden_with_no_calls_or_one_type():
    empty = calls([])
    assert (segments.burden(empty, people=[1, 2]) == 0).all().all()
    only_dels = calls([[1, "1", 0, 1_000_000, "DEL"]])
    b = segments.burden(only_dels, people=[1, 2])
    assert b.loc[1, "del_mb"] == pytest.approx(1.0) and (b["dup_mb"] == 0).all()


def test_burden_rejects_duplicate_people():
    with pytest.raises(ValueError, match="duplicate"):
        segments.burden(calls([]), people=[1, 1])
