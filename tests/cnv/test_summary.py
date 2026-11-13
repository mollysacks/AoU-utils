import numpy as np
import pandas as pd
import pytest

from aou_utils.cnv import summary


@pytest.fixture
def path(tmp_path):
    df = pd.DataFrame({
        "identifier": [101, 102, 103],
        "LRR_SD": [0.15, 0.40, np.nan], "BAF_DRIFT": [0.0, 0.002, 0.0], "WF": [-0.01, 0.06, 0.0],
        "GCWF": [0.0, 0.0, 0.0], "Num_CNVs": [12, 150, 0],
        "Num_Auto_Del": [3, 100, 0], "Num_Auto_Dup": [1, 40, 0],
        "Len_Auto_Del": [1_500_000, 9_000_000, 0], "Len_Auto_Dup": [250_000, 2_000_000, 0],
        "Freq_Calc_Reason": ["", "chr aberration", ""],
    })
    p = tmp_path / "GDA.txt"
    df.to_csv(p, sep="\t", index=False)
    return p


def test_read_and_burden(path):
    s = summary.read(path)
    assert s.index.name == "person_id" and s.index.tolist() == [101, 102, 103]
    b = summary.burden(s, people=[103, 101])
    assert b.index.tolist() == [103, 101]
    assert list(b.columns) == ["del_mb", "dup_mb", "cnv_mb", "del_n", "dup_n", "cnv_n"]
    assert b.loc[101, "del_mb"] == pytest.approx(1.5) and b.loc[101, "cnv_mb"] == pytest.approx(1.75)
    assert b.loc[101, "cnv_n"] == 4 and (b.loc[103] == 0).all()


def test_burden_requires_people_in_summary(path):
    with pytest.raises(KeyError, match="1 people"):
        summary.burden(summary.read(path), people=[101, 999])


def test_qc_pass(path):
    s = summary.read(path)
    assert summary.qc_pass(s).all()
    ok = summary.qc_pass(s, max_lrr_sd=0.35, max_abs_wf=0.05, max_cnvs=100)
    assert ok.tolist() == [True, False, False]  # 102 fails all three; 103 has no LRR_SD


def test_read_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.txt"
    pd.DataFrame({"identifier": [1, 1], "LRR_SD": [0.1, 0.2]}).to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        summary.read(p)
