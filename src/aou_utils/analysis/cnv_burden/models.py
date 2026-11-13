"""Linear CNV-burden models for quantitative traits (e.g. height, BMI).

For each model (a set of burden predictors such as del_mb), the trait is regressed on
[1, predictors, covariates] and on [1, covariates] alone, using the same people (complete cases).
Reported per predictor: beta (trait units per predictor unit, e.g. SD per Mb), SE, t-test p and 95% CI.
Reported per model: incremental R² (delta_r2 = R²(full) - R²(covariates only)), the F-test p for all
of the model's predictors jointly, and optional percentile bootstrap CIs for beta and delta_r2.
"""

import numpy as np
import pandas as pd

# Default models: each burden measure on its own, then DEL and DUP Mb jointly
DEFAULT_MODELS = {
    "DEL": ["del_mb"],
    "DUP": ["dup_mb"],
    "CNV": ["cnv_mb"],
    "DEL+DUP": ["del_mb", "dup_mb"],
}


def _lstsq(X: np.ndarray, y: np.ndarray):
    """(beta, residual sum of squares) of y on X."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def _design(data: pd.DataFrame, columns) -> np.ndarray:
    X = data[list(columns)].to_numpy(dtype="float64")
    return np.column_stack([np.ones(len(X)), X])


def _fit(y: np.ndarray, X_full: np.ndarray, X_null: np.ndarray, k: int) -> dict:
    """OLS of y on X_full, whose columns 1..k are the tested predictors, vs X_null (without them)."""
    from scipy import stats

    n, p = X_full.shape
    beta, rss = _lstsq(X_full, y)
    _, rss_null = _lstsq(X_null, y)
    tss = float(((y - y.mean()) ** 2).sum())
    df_resid = n - p
    sigma2 = rss / df_resid
    xtx_inv = np.linalg.pinv(X_full.T @ X_full)
    se = np.sqrt(np.diag(xtx_inv)[1:k + 1] * sigma2)
    t = beta[1:k + 1] / se
    tcrit = stats.t.ppf(0.975, df_resid)
    f = ((rss_null - rss) / k) / sigma2
    return {
        "beta": beta[1:k + 1], "se": se, "p": 2 * stats.t.sf(np.abs(t), df_resid),
        "ci_low": beta[1:k + 1] - tcrit * se, "ci_high": beta[1:k + 1] + tcrit * se,
        "r2_null": 1 - rss_null / tss, "delta_r2": (rss_null - rss) / tss,
        "f_p": float(stats.f.sf(f, k, df_resid)),
    }


def _bootstrap(y, X_full, X_null, k, n_boot, rng):
    """(n_boot x k betas, n_boot delta_r2s) from resampling people with replacement."""
    n = len(y)
    betas, d_r2 = np.empty((n_boot, k)), np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        yb, Xf, Xn = y[i], X_full[i], X_null[i]
        beta, rss = _lstsq(Xf, yb)
        _, rss_null = _lstsq(Xn, yb)
        betas[b] = beta[1:k + 1]
        d_r2[b] = (rss_null - rss) / float(((yb - yb.mean()) ** 2).sum())
    return betas, d_r2


def burden_test(data: pd.DataFrame, trait: str, covariates=(), models: dict = None,
                n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
    """Test each model's burden predictors against `trait`, adjusting for `covariates`.

    `data` has one row per person with the trait, the burden columns and the covariates. `models`
    maps a model name to its predictor columns (default `DEFAULT_MODELS`). Each model uses the people
    with no missing value in the trait, its predictors or the covariates.

    Returns one row per (model, predictor) with: n, n_carriers (people with predictor > 0), beta, se,
    p, ci_low, ci_high, and the model-level delta_r2, r2_null and f_p (repeated on the model's rows).
    With n_boot > 0, also boot_beta_low/high and boot_delta_r2_low/high (2.5th and 97.5th percentiles).
    n and n_carriers are raw counts: pass them through `aou_utils.privacy.suppress` before showing them.
    """
    models = DEFAULT_MODELS if models is None else models
    covariates = list(covariates)
    rng = np.random.default_rng(seed)
    rows = []
    for name, predictors in models.items():
        predictors = list(predictors)
        cols = [trait] + predictors + covariates
        d = data[cols].dropna()
        k = len(predictors)
        if len(d) <= k + len(covariates) + 1:
            raise ValueError(f"Model {name!r}: only {len(d)} complete rows")
        y = d[trait].to_numpy(dtype="float64")
        X_full = _design(d, predictors + covariates)
        X_null = _design(d, covariates)
        fit = _fit(y, X_full, X_null, k)
        if n_boot:
            boot_betas, boot_d_r2 = _bootstrap(y, X_full, X_null, k, n_boot, rng)
        for j, pred in enumerate(predictors):
            row = {
                "model": name, "predictor": pred, "n": len(d), "n_carriers": int((d[pred] > 0).sum()),
                "beta": fit["beta"][j], "se": fit["se"][j], "p": fit["p"][j],
                "ci_low": fit["ci_low"][j], "ci_high": fit["ci_high"][j],
                "delta_r2": fit["delta_r2"], "r2_null": fit["r2_null"], "f_p": fit["f_p"],
            }
            if n_boot:
                row["boot_beta_low"], row["boot_beta_high"] = np.percentile(boot_betas[:, j], [2.5, 97.5])
                row["boot_delta_r2_low"], row["boot_delta_r2_high"] = np.percentile(boot_d_r2, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)
