import pandas as pd
import pytest

from aou_utils import cnv


@pytest.fixture
def calls():
    # people: 1 NC, 2 mirror DEL (via the second mirror locus), 3 asym DUP, 4 mirror DUP + asym DEL,
    # 5 missing at a classified locus, 6 carries only an unclassified CNV
    return pd.DataFrame(
        {"1q21.1": [2, 2, 2, 3, 2, 2], "16p11.2_BP4_BP5": [2, 1, 2, 2, 2, 2],
         "17q12": [2, 2, 3, 1, None, 2], "3q29": [2, 2, 2, 2, 2, 1]},
        index=pd.Index([1, 2, 3, 4, 5, 6], name="person_id"),
    ).astype("Int8")


def test_classify_split_by_copy(calls):
    classes = {"mirror": ["1q21.1", "16p11.2_BP4-BP5"], "asymmetric": ["17q12"]}
    groups = cnv.classify(calls, classes)
    assert groups.tolist()[:4] == ["NC", "mirror_DEL", "asymmetric_DUP", "multiple"]
    assert pd.isna(groups[5])
    assert groups[6] == "NC"  # unclassified CNVs don't count against NC


def test_classify_pooled(calls):
    groups = cnv.classify(calls, {"mirror": ["1q21.1", "16p11.2_BP4_BP5"]}, split_copy=False)
    assert groups[2] == "mirror" and groups[4] == "mirror"


def test_match_loci_reports_missing(calls):
    with pytest.raises(KeyError, match="22q11.21"):
        cnv.match_loci(["22q11.21_A-D"], calls.columns)


def test_load_calls_skips_metadata_and_oneway(tmp_path):
    path = tmp_path / "calls.csv"
    pd.DataFrame({"identifier": [1, 2], "PLATFORM": ["x", "y"], "16p11.2_BP4_BP5": [1, 2],
                  "16p11.2_BP4_BP5_oneway": [1, 2], "15q11.2-13.1_BP2_BP3": [2, 3]}).to_csv(path)
    calls = cnv.load_calls(path)
    assert list(calls.columns) == ["16p11.2_BP4_BP5", "15q11.2-13.1_BP2_BP3"]
    assert calls.index.name == "person_id"


def test_classify_exclude_loci(calls):
    # person 6 carries 3q29 only: NC without exclusion, 'excluded' when 3q29 is excluded
    classes = {"mirror": ["1q21.1", "16p11.2_BP4_BP5"], "asymmetric": ["17q12"]}
    groups = cnv.classify(calls, classes, exclude=["3q29"])
    assert groups[6] == "excluded"
    assert groups[1] == "NC" and groups[2] == "mirror_DEL"
