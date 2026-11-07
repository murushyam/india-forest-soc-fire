"""Nested Cross-Validation comparing optimistic random K-fold against honest spatial hexagonal block CV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..cv.spatial_blocks import hex_block_folds
from . import build_model
from .metrics import interval_metrics, regression_metrics


@dataclass
class NestedCVResult:
    """Results from a nested cross-validation experiment."""

    model_name: str
    scheme: str
    diameter_km: float | None
    oof_predictions: np.ndarray
    oof_quantiles: np.ndarray | None  # shape (n_samples, 3) for [0.05, 0.50, 0.95] when qrf
    metrics: dict[str, float]
    fold_metrics: pd.DataFrame
    best_params_per_fold: list[dict[str, Any]]


def project_lonlat_to_meters(
    lon: Sequence[float] | np.ndarray,
    lat: Sequence[float] | np.ndarray,
    ref_lon: float | None = None,
    ref_lat: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert decimal degrees `(lon, lat)` to planar metric coordinates `(x_m, y_m)`."""
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lon0 = float(np.mean(lon_arr)) if ref_lon is None else float(ref_lon)
    lat0 = float(np.mean(lat_arr)) if ref_lat is None else float(ref_lat)
    cos_lat = np.cos(np.radians(lat0))
    x_m = (lon_arr - lon0) * 111_320.0 * cos_lat
    y_m = (lat_arr - lat0) * 110_574.0
    return x_m, y_m


_PARAM_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "rf": [
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 4},
    ],
    "xgb": [
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 350, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.9},
    ],
    "qrf": [
        {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
    ],
}


def _make_splits(
    n_samples: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
    scheme: str,
    diameter_km: float | None,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    eff_splits = min(n_splits, n_samples)
    if eff_splits < 2:
        raise ValueError(f"Need >= 2 samples for cross-validation, got {n_samples}")

    if scheme == "random_kfold":
        kf = KFold(n_splits=eff_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in kf.split(np.arange(n_samples))]
    if scheme == "spatial_hex_block":
        diam_km = float(diameter_km or 50.0)
        fold_ids = hex_block_folds(x_m, y_m, diameter_km=diam_km, n_folds=eff_splits, seed=seed)
        splits = [
            (np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f))
            for f in range(eff_splits)
        ]
        # Filter out any empty test fold (if all points fell into fewer hex blocks than eff_splits)
        valid_splits = [(tr, te) for tr, te in splits if te.size > 0 and tr.size > 0]
        if len(valid_splits) >= 2:
            return valid_splits
        # Fallback if spatial extent is smaller than single hex block
        kf = KFold(n_splits=eff_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in kf.split(np.arange(n_samples))]
    raise ValueError(f"Unknown CV scheme '{scheme}'. Choose 'random_kfold' or 'spatial_hex_block'.")


def run_nested_cv(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | Sequence[float],
    x_m: np.ndarray,
    y_m: np.ndarray,
    model_name: str = "rf",
    scheme: str = "spatial_hex_block",
    diameter_km: float | None = 50.0,
    outer_folds: int = 5,
    inner_folds: int = 3,
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95),
    seed: int = 42,
) -> NestedCVResult:
    """Execute 5x3 nested cross-validation with inner hyperparameter selection."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    xm = np.asarray(x_m, dtype=float)
    ym = np.asarray(y_m, dtype=float)
    n = y_arr.size

    outer_splits = _make_splits(n, xm, ym, scheme, diameter_km, outer_folds, seed)
    oof_preds = np.full(n, np.nan, dtype=float)
    oof_q = np.full((n, len(quantiles)), np.nan, dtype=float) if model_name.lower() == "qrf" else None

    fold_rows: list[dict[str, float]] = []
    best_params_list: list[dict[str, Any]] = []
    candidates = _PARAM_CANDIDATES.get(model_name.lower(), [{}])

    for f_idx, (train_idx, test_idx) in enumerate(outer_splits):
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        xm_tr, ym_tr = xm[train_idx], ym[train_idx]
        X_te, y_te = X_arr[test_idx], y_arr[test_idx]

        # Inner hyperparameter tuning loop
        best_score = np.inf
        best_params = candidates[0]
        if len(train_idx) >= 6 and len(candidates) > 1:
            inner_splits = _make_splits(
                len(train_idx),
                xm_tr,
                ym_tr,
                scheme,
                diameter_km,
                min(inner_folds, len(train_idx) // 2),
                seed + f_idx + 1,
            )
            for cand in candidates:
                inner_errs: list[float] = []
                for itr, ite in inner_splits:
                    est = build_model(model_name, random_state=seed, **cand)
                    est.fit(X_tr[itr], y_tr[itr])
                    pred_val = est.predict(X_tr[ite])
                    inner_errs.append(float(np.sqrt(np.mean((y_tr[ite] - pred_val) ** 2))))
                mean_rmse = float(np.mean(inner_errs))
                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_params = cand

        best_params_list.append(dict(best_params))
        final_fold_model = build_model(model_name, random_state=seed + f_idx, **best_params)
        final_fold_model.fit(X_tr, y_tr)

        if model_name.lower() == "qrf" and oof_q is not None:
            q_preds = final_fold_model.predict(X_te, quantiles=quantiles)
            oof_q[test_idx] = q_preds
            fold_pred = q_preds[:, 1]  # median q0.50 as point prediction
        else:
            fold_pred = np.asarray(final_fold_model.predict(X_te), dtype=float)

        oof_preds[test_idx] = fold_pred
        fm = regression_metrics(y_te, fold_pred)
        fm["fold"] = float(f_idx)
        fm["n_test"] = float(len(test_idx))
        fold_rows.append(fm)

    valid_mask = np.isfinite(oof_preds)
    pooled_metrics = regression_metrics(y_arr[valid_mask], oof_preds[valid_mask])
    if oof_q is not None and valid_mask.any():
        im = interval_metrics(
            y_arr[valid_mask],
            oof_q[valid_mask, 0],
            oof_q[valid_mask, 2],
            nominal=float(quantiles[2] - quantiles[0]),
        )
        pooled_metrics.update(im)

    return NestedCVResult(
        model_name=model_name,
        scheme=scheme,
        diameter_km=diameter_km if scheme == "spatial_hex_block" else None,
        oof_predictions=oof_preds,
        oof_quantiles=oof_q,
        metrics=pooled_metrics,
        fold_metrics=pd.DataFrame(fold_rows),
        best_params_per_fold=best_params_list,
    )


def compare_random_vs_spatial_cv(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | Sequence[float],
    x_m: np.ndarray,
    y_m: np.ndarray,
    model_names: Sequence[str] = ("rf", "xgb", "qrf"),
    block_diameters_km: Sequence[float] = (25.0, 50.0, 100.0),
    outer_folds: int = 5,
    inner_folds: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark random K-fold CV vs hexagonal spatial block CV across block sizes and models."""
    rows: list[dict[str, Any]] = []
    for m_name in model_names:
        res_rand = run_nested_cv(
            X,
            y,
            x_m,
            y_m,
            model_name=m_name,
            scheme="random_kfold",
            diameter_km=None,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=seed,
        )
        rows.append(
            {
                "model": m_name,
                "scheme": "random_kfold",
                "diameter_km": 0.0,
                **res_rand.metrics,
                "r2_optimism_gap": 0.0,
                "rmse_inflation": 0.0,
            }
        )
        for d_km in block_diameters_km:
            res_sp = run_nested_cv(
                X,
                y,
                x_m,
                y_m,
                model_name=m_name,
                scheme="spatial_hex_block",
                diameter_km=float(d_km),
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                seed=seed,
            )
            rows.append(
                {
                    "model": m_name,
                    "scheme": "spatial_hex_block",
                    "diameter_km": float(d_km),
                    **res_sp.metrics,
                    "r2_optimism_gap": float(res_rand.metrics["r2"] - res_sp.metrics["r2"]),
                    "rmse_inflation": float(res_sp.metrics["rmse"] - res_rand.metrics["rmse"]),
                }
            )
    return pd.DataFrame(rows)
