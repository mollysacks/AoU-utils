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


def test_profile_heatmap_orders_rows_and_labels_cells():
    from aou_utils.plots import profile_heatmap

    effects = pd.DataFrame({
        "concept_id": [1, 1, 2, 2, 3, 3], "group": ["a", "b"] * 3,
        "effect": [0.8, 0.7, -0.5, 0.4, 0.75, 0.72], "q": [0.01, 0.2, 0.01, 0.01, 0.5, np.nan],
    })
    fig, ax = plt.subplots()
    rows = profile_heatmap(ax, effects, [1, 2, 3], ["a", "b"], labels=["x", "y", "z"])
    assert sorted(rows) == [1, 2, 3]
    assert abs(rows.index(1) - rows.index(3)) == 1  # similar profiles end up adjacent
    texts = {t.get_text() for t in ax.texts}
    assert "+0.80*" in texts and "+0.70" in texts
    plt.close(fig)


def test_group_boxes_skips_empty_groups():
    from aou_utils.plots import group_boxes

    values = pd.Series([1.0, 2.0, 3.0, 10.0, 0.5, np.nan])
    groups = pd.Series(["NC", "NC", "NC", "a", "a", "b"])
    fig, ax = plt.subplots()
    group_boxes(ax, values, groups, ["NC", "a", "b", "c"])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["NC\n(n=3)", "a\n(n=2)"]
    plt.close(fig)


def test_lda_biplot_bars_and_means():
    from aou_utils.plots import lda_biplot

    rng = np.random.default_rng(0)
    scores = pd.DataFrame(rng.normal(size=(60, 2)), columns=["LD1", "LD2"])
    scores.iloc[0] = np.nan
    groups = pd.Series(["NC"] * 40 + ["a"] * 10 + ["b"] * 10)
    loadings = pd.DataFrame({"LD1": [0.9, -0.5, 0.1, 0.0], "LD2": [0.0, 0.2, -0.8, 0.6]}, index=[11, 12, 13, 14])
    fig = plt.figure()
    axes = lda_biplot(fig, scores, loadings, groups, ["a", "b", "NC"],
                      labels={11: "w", 12: "x", 13: "y", 14: "z"}, top=3)
    assert [t.get_text() for t in axes["LD1"].get_yticklabels()] == ["w", "y", "x"]  # largest |loading|, sorted by value
    assert [t.get_text() for t in axes["LD2"].get_xticklabels()] == ["z", "x", "y"]
    assert {t.get_text() for t in axes["scatter"].texts} == {"a", "b", "NC"}
    plt.close(fig)
