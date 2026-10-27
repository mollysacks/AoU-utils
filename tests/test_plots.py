import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aou_utils.plots import dodged_hist  # noqa: E402


def test_dodged_hist_counts_and_layout():
    values = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, np.nan])
    groups = pd.Series(["a", "a", "b", "b", "b", "a", "b"])
    fig, ax = plt.subplots()
    edges = dodged_hist(ax, values, groups, order=["a", "b"], bins=3, clip=(0, 1), density=False)
    a_bars, b_bars = ax.containers
    assert [p.get_height() for p in a_bars] == [1, 1, 1]
    assert [p.get_height() for p in b_bars] == [0, 1, 2]
    # second group's bars sit just to the right of the first group's within each bin
    assert b_bars[0].get_x() == pytest.approx(a_bars[0].get_x() + (edges[1] - edges[0]) / 2)
    plt.close(fig)


def test_forest_draws_one_series_per_group():
    from aou_utils.plots import forest

    effects = pd.DataFrame({
        "concept_id": [1, 2, 1, 2], "group": ["a", "a", "b", "b"],
        "effect": [0.5, -0.2, 0.1, np.nan], "ci_low": [0.3, -0.4, -0.1, np.nan], "ci_high": [0.7, 0.0, 0.3, np.nan],
    })
    fig, ax = plt.subplots()
    forest(ax, effects, [1, 2], ["a", "b"], labels=["one", "two"])
    assert len(ax.containers) == 2
    assert [t.get_text() for t in ax.get_yticklabels()] == ["one", "two"]
    plt.close(fig)
