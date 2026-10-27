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


# Diverging blue <-> red with a neutral gray midpoint
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"]


def profile_heatmap(ax, effects: pd.DataFrame, concepts, order, labels=None, q_cut: float = 0.05,
                    cluster: bool = True, vmax: float = None):
    """Heatmap of per-trait effects: rows = concepts, columns = groups, cells labeled with the effect.

    `effects` has columns concept_id, group, effect, q (as from measurements.group_effects).
    Cells with q < q_cut get a '*'. With `cluster`, rows are ordered by hierarchical clustering of
    their effect profiles so traits that behave alike sit together. Returns the row order used.
    """
    from matplotlib.colors import LinearSegmentedColormap

    matrix = effects.pivot(index="concept_id", columns="group", values="effect").reindex(index=list(concepts), columns=order)
    q = effects.pivot(index="concept_id", columns="group", values="q").reindex(index=matrix.index, columns=order)
    labels = pd.Series(labels if labels is not None else matrix.index, index=matrix.index)
    if cluster and len(matrix) > 2:
        from scipy.cluster.hierarchy import leaves_list, linkage

        rows = leaves_list(linkage(matrix.fillna(0).to_numpy(), method="average"))
        matrix, q, labels = matrix.iloc[rows], q.iloc[rows], labels.iloc[rows]

    vmax = vmax or np.nanmax(np.abs(matrix.to_numpy()))
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            if pd.isna(value):
                continue
            star = "*" if q.iat[i, j] < q_cut else ""
            ax.text(j, i, f"{value:+.2f}{star}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 * vmax else "#0b0b0b")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(labels)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.figure.colorbar(image, ax=ax, shrink=0.6, label="difference from reference (SD)")
    return list(matrix.index)
