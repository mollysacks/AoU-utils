import pandas as pd

from aou_utils.privacy import mask_fractions, suppress


def test_suppress_small_counts_only():
    s = suppress(pd.Series([0, 1, 20, 21, 500]))
    assert s.tolist() == [0, "<=20", "<=20", 21, 500]


def test_mask_fractions():
    counts = pd.Series([0, 5, 50])
    fractions = pd.Series([0.0, 0.05, 0.5])
    masked = mask_fractions(fractions, counts)
    assert masked[0] == 0.0 and pd.isna(masked[1]) and masked[2] == 0.5
