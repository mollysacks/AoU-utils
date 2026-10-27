"""Plotting helpers. matplotlib is imported lazily so the package imports without it."""

import numpy as np
import pandas as pd

# Fixed categorical order (blue, orange, aqua, yellow, magenta, green); assign in order, never cycle
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def dodged_hist(ax, values: pd.Series, groups: pd.Series, order, bins: int = 30,
                clip: tuple = (0.001, 0.999), density: bool = True):
    """Histogram of `values` by group, with each group's bars side by side within a bin.

    Bins are shared across groups and span the pooled `clip` quantiles, so long tails don't squash
    the plot; values outside are left out. With `density`, each group is normalized to area 1 so
    small groups are comparable with large ones. Returns the bin edges.
    """
    values = values.dropna()
    groups = groups.reindex(values.index)
    lo, hi = values.quantile(list(clip))
    edges = np.histogram_bin_edges(values, bins=bins, range=(lo, hi))
    widths = np.diff(edges)
    k = len(order)
    for i, (level, color) in enumerate(zip(order, SERIES_COLORS)):
        heights, _ = np.histogram(values[groups == level], bins=edges, density=density)
        ax.bar(edges[:-1] + widths * i / k, heights, width=widths / k, align="edge",
               color=color, label=level, linewidth=0)
    ax.set_ylabel("density" if density else "count")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return edges


def forest(ax, effects: pd.DataFrame, concepts, order, labels=None):
    """Effect and 95% CI per concept (rows) for each group (colored points, offset within a row).

    `effects` has columns concept_id, group, effect, ci_low, ci_high (as from
    measurements.group_effects); `concepts` gives the row order and `order` the groups.
    """
    concepts = list(concepts)
    offsets = np.linspace(-0.3, 0.3, len(order)) if len(order) > 1 else [0.0]
    rows = np.arange(len(concepts))
    for g, color, offset in zip(order, SERIES_COLORS, offsets):
        e = effects[effects["group"] == g].set_index("concept_id").reindex(concepts)
        xerr = np.vstack([e["effect"] - e["ci_low"], e["ci_high"] - e["effect"]])
        ax.errorbar(e["effect"], rows + offset, xerr=xerr, fmt="o", color=color, markersize=4,
                    elinewidth=1, capsize=0, label=g)
    ax.axvline(0, color="#999999", linewidth=1, zorder=0)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels if labels is not None else concepts)
    ax.invert_yaxis()
    ax.set_xlabel("difference from reference (SD)")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
