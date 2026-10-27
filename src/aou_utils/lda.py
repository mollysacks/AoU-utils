"""Linear discriminant analysis of measurements across genotype groups (e.g. CNV DEL / DUP / NC).

sklearn is imported lazily so the package imports without it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def split_fit_holdout(groups: pd.Series, holdout_group: str = "NC", fit_frac: float = 0.2,
                      seed: int = 0) -> pd.Series:
    """True for people used to fit the LDA, False for people held out.

    Everyone outside `holdout_group` is in the fit set, plus a random `fit_frac` of
    `holdout_group`. The rest of `holdout_group` is held out (e.g. for GWAS of the scores).
    """
    rng = np.random.default_rng(seed)
    fit = groups.notna() & (groups != holdout_group)
    pool = groups.index[groups == holdout_group]
    chosen = rng.choice(pool, size=int(round(fit_frac * len(pool))), replace=False)
    fit.loc[chosen] = True
    return fit


def _model(shrinkage):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # The eigen solver supports both shrinkage and projecting onto the discriminant axes
    return LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage)


@dataclass
class LDAFit:
    model: object
    columns: list
    axes: list
    signs: np.ndarray

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scores on the discriminant axes; rows missing any measurement get NaN."""
        X = X[self.columns]
        ok = X.notna().all(axis=1)
        scores = pd.DataFrame(np.nan, index=X.index, columns=self.axes)
        if ok.any():
            projected = self.model.transform(X[ok].to_numpy())[:, : len(self.axes)]
            scores.loc[ok] = projected * self.signs
        return scores

    @property
    def explained_variance_ratio(self) -> pd.Series:
        """Share of between-group separation on each axis, among the (n_groups - 1) axes.

        With shrinkage, sklearn also gives tiny nonzero ratios to the leftover axes, so we renormalize.
        """
        ratio = self.model.explained_variance_ratio_[: len(self.axes)]
        return pd.Series(ratio / ratio.sum(), index=self.axes)

    @property
    def coefficients(self) -> pd.DataFrame:
        """Raw discriminant coefficients (scalings), oriented like the scores."""
        return pd.DataFrame(self.model.scalings_[:, : len(self.axes)] * self.signs,
                            index=self.columns, columns=self.axes)


def fit(X: pd.DataFrame, y: pd.Series, orient: tuple = ("DEL", "DUP"), shrinkage="auto") -> LDAFit:
    """Fit a shrinkage LDA of X (complete cases only) on group labels y.

    Each axis is flipped if needed so that orient[1] has a higher mean score than orient[0]
    (by default DUP > DEL), which makes signs comparable across runs.
    """
    X, y = complete_cases(X, y)
    model = _model(shrinkage).fit(X.to_numpy(), y.to_numpy())
    n_axes = len(model.classes_) - 1
    axes = [f"LD{i + 1}" for i in range(n_axes)]
    raw = pd.DataFrame(model.transform(X.to_numpy())[:, :n_axes], index=X.index, columns=axes)
    means = raw.groupby(y).mean()
    signs = np.sign(means.loc[orient[1]] - means.loc[orient[0]]).replace(0, 1).to_numpy()
    return LDAFit(model=model, columns=list(X.columns), axes=axes, signs=signs)


def complete_cases(X: pd.DataFrame, y: pd.Series):
    """Rows of X and y with no missing measurements and a group label."""
    y = y.reindex(X.index)
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def structure(X: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Loadings: correlation of each measurement with each discriminant score."""
    X = X.loc[scores.dropna().index]
    return pd.DataFrame({axis: X.corrwith(scores[axis]) for axis in scores.columns})


def cross_validate(X: pd.DataFrame, y: pd.Series, pair: tuple = ("DEL", "DUP"), n_splits: int = 5,
                   n_permutations: int = 0, shrinkage="auto", seed: int = 0) -> dict:
    """Out-of-fold AUCs, and optionally a permutation p-value for the overall AUC.

    AUC is used rather than accuracy because it doesn't depend on the (very unbalanced) group sizes.
    Returns:
      auc_ovr: macro one-vs-rest AUC over all groups
      auc_pair: AUC separating pair[0] from pair[1], among people in those two groups
      permutation_p, permutation_aucs: if n_permutations > 0, shuffling labels
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score

    X, y = complete_cases(X, y)
    model = _model(shrinkage)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X.to_numpy(), y.to_numpy(), cv=cv, method="predict_proba")
    classes = sorted(y.unique())
    proba = pd.DataFrame(proba, index=y.index, columns=classes)

    in_pair = y.isin(pair)
    pair_score = proba.loc[in_pair, pair[1]] / proba.loc[in_pair, list(pair)].sum(axis=1)
    result = {
        "auc_ovr": roc_auc_score(y, proba[classes], multi_class="ovr", average="macro", labels=classes),
        "auc_pair": roc_auc_score(y[in_pair] == pair[1], pair_score),
    }
    if n_permutations:
        _, perm_scores, p = permutation_test_score(
            model, X.to_numpy(), y.to_numpy(), cv=cv, scoring="roc_auc_ovr",
            n_permutations=n_permutations, random_state=seed, n_jobs=-1,
        )
        result["permutation_p"] = p
        result["permutation_aucs"] = perm_scores
    return result
