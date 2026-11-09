"""Model interpretability: TreeSHAP attributions, Partial Dependence Plots (PDP), and Area of Applicability (AOA)."""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def compute_shap_importance(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute sample-level SHAP values `(n_samples, n_features)` and global importance summary.

    Uses `shap.TreeExplainer` when `shap` is installed; falls back to an exact
    single-feature marginal ablation / interventional attribution when `shap`
    is unavailable or when `model` is a custom wrapper.
    """
    if isinstance(X, pd.DataFrame):
        f_names = list(X.columns) if feature_names is None else list(feature_names)
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        f_names = (
            [f"f_{i}" for i in range(X_arr.shape[1])]
            if feature_names is None
            else list(feature_names)
        )

    n_samples, n_features = X_arr.shape
    shap_vals: np.ndarray | None = None

    try:
        import shap

        target_est = getattr(model, "_backend", model)
        explainer = shap.TreeExplainer(target_est)
        raw_shap = explainer.shap_values(X_arr)
        shap_vals = np.asarray(raw_shap, dtype=float)
    except Exception:
        # Interventional marginal attribution relative to feature medians
        base_pred = np.asarray(model.predict(X_arr), dtype=float)
        medians = np.nanmedian(X_arr, axis=0)
        shap_vals = np.zeros((n_samples, n_features), dtype=float)
        for j in range(n_features):
            X_abl = X_arr.copy()
            X_abl[:, j] = medians[j]
            pred_abl = np.asarray(model.predict(X_abl), dtype=float)
            shap_vals[:, j] = base_pred - pred_abl

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    # Directional sign: correlation between feature value and its SHAP attribution
    signs: list[float] = []
    for j in range(n_features):
        xj = X_arr[:, j]
        sj = shap_vals[:, j]
        if np.std(xj) > 1e-12 and np.std(sj) > 1e-12:
            signs.append(float(np.corrcoef(xj, sj)[0, 1]))
        else:
            signs.append(0.0)

    summary = (
        pd.DataFrame(
            {
                "feature": f_names,
                "mean_abs_shap": mean_abs,
                "rel_importance_pct": 100.0 * mean_abs / max(float(np.sum(mean_abs)), 1e-12),
                "direction_corr": signs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return summary, shap_vals


def compute_pdp_1d(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    feature: str | int,
    feature_names: Sequence[str] | None = None,
    grid_resolution: int = 25,
    grid_points: int | None = None,
) -> pd.DataFrame:
    """Compute 1D Partial Dependence Plot (PDP) mean and ICE standard deviation."""
    if grid_points is not None:
        grid_resolution = grid_points

    if isinstance(X, pd.DataFrame):
        f_names = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        f_names = (
            [f"f_{i}" for i in range(X_arr.shape[1])]
            if feature_names is None
            else list(feature_names)
        )

    f_idx = f_names.index(feature) if isinstance(feature, str) else int(feature)
    f_label = f_names[f_idx]
    col_vals = X_arr[:, f_idx]
    grid = np.linspace(float(np.nanpercentile(col_vals, 2)), float(np.nanpercentile(col_vals, 98)), grid_resolution)

    pdp_mean: list[float] = []
    ice_sd: list[float] = []
    ice_q10: list[float] = []
    ice_q90: list[float] = []
    for g_val in grid:
        X_mod = X_arr.copy()
        X_mod[:, f_idx] = g_val
        preds = np.asarray(model.predict(X_mod), dtype=float)
        pdp_mean.append(float(np.mean(preds)))
        ice_sd.append(float(np.std(preds, ddof=1)) if preds.size > 1 else 0.0)
        ice_q10.append(float(np.percentile(preds, 10)) if preds.size > 0 else float(np.mean(preds)))
        ice_q90.append(float(np.percentile(preds, 90)) if preds.size > 0 else float(np.mean(preds)))

    return pd.DataFrame(
        {
            "feature": f_label,
            "grid_value": grid,
            "feature_value": grid,
            "pdp_mean": pdp_mean,
            "ice_sd": ice_sd,
            "ice_q10": ice_q10,
            "ice_q90": ice_q90,
        }
    )


compute_partial_dependence_1d = compute_pdp_1d


def compute_aoa(
    X_train: np.ndarray | pd.DataFrame,
    X_pred: np.ndarray | pd.DataFrame,
    weights: np.ndarray | None = None,
    quantile_thresh: float = 0.95,
) -> dict[str, np.ndarray | float]:
    """Compute Meyer & Pebesma (2021) Area of Applicability (AOA) and Dissimilarity Index (DI).

    Parameters
    ----------
    X_train : array-like of shape (n_train, n_features)
        Training covariate matrix.
    X_pred : array-like of shape (n_pred, n_features)
        Prediction covariate matrix.
    weights : np.ndarray of shape (n_features,), optional
        Feature importance weights (e.g. `mean_abs_shap`). Defaults to equal weights.
    quantile_thresh : float, default 0.95
        Quantile of within-training nearest-neighbour DI used as AOA threshold.
    """
    X_tr = np.asarray(X_train, dtype=float)
    X_pr = np.asarray(X_pred, dtype=float)
    n_tr, p = X_tr.shape

    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)

    Z_tr = (X_tr - mu) / sd
    Z_pr = (X_pr - mu) / sd

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / max(float(np.sum(w)), 1e-12) * p
    else:
        w = np.ones(p, dtype=float)

    Zw_tr = Z_tr * w[None, :]
    Zw_pr = Z_pr * w[None, :]

    # Average pairwise distance in weighted standardised training space
    tr_dists = cdist(Zw_tr, Zw_tr, metric="euclidean")
    mean_tr_dist = float(np.mean(tr_dists[np.triu_indices(n_tr, k=1)])) if n_tr > 1 else 1.0
    mean_tr_dist = max(mean_tr_dist, 1e-12)

    # Leave-one-out nearest training neighbour distance for threshold calibration
    np.fill_diagonal(tr_dists, np.inf)
    tr_min_dist = np.min(tr_dists, axis=1)
    tr_di = tr_min_dist / mean_tr_dist
    di_threshold = float(np.quantile(tr_di, quantile_thresh))

    # Prediction points DI
    pr_dists = cdist(Zw_pr, Zw_tr, metric="euclidean")
    pr_min_dist = np.min(pr_dists, axis=1)
    di_pred = pr_min_dist / mean_tr_dist
    aoa_mask = di_pred <= di_threshold

    return {
        "di": di_pred,
        "aoa_mask": aoa_mask,
        "di_threshold": di_threshold,
        "train_di": tr_di,
    }
