"""Empirical variogram range estimation and Moran's I spatial autocorrelation diagnostics."""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform


def empirical_variogram(
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
    max_lag_m: float = 150_000.0,
    n_bins: int = 12,
    n_lags: Optional[int] = None,
) -> pd.DataFrame:
    """Compute isotropic empirical semivariance gamma(h) across distance bins."""
    if n_lags is not None:
        n_bins = n_lags

    coords = np.column_stack([np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)])
    z = np.asarray(values, dtype=float)
    if coords.shape[0] != z.size or z.size < 4:
        raise ValueError("Need >= 4 observations with matching coordinate and value lengths.")

    dists = pdist(coords, metric="euclidean")
    sq_diffs = 0.5 * (pdist(z[:, None], metric="sqeuclidean"))

    edges = np.linspace(0.0, max_lag_m, n_bins + 1)
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (dists > lo) & (dists <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            rows.append(
                {
                    "lag_center_m": float(0.5 * (lo + hi)),
                    "semivariance": float(np.mean(sq_diffs[mask])),
                    "n_pairs": float(cnt),
                }
            )
    return pd.DataFrame(rows, columns=["lag_center_m", "semivariance", "n_pairs"])


compute_empirical_variogram = empirical_variogram


def fit_variogram_range(
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
    max_lag_m: float = 150_000.0,
    n_bins: int = 12,
    candidate_blocks_km: tuple[float, ...] = (25.0, 50.0, 100.0),
    n_lags: Optional[int] = None,
) -> dict[str, float]:
    """Fit exponential variogram model and recommend minimum spatial block diameter (km).

    Exponential model: gamma(h) = nugget + partial_sill * (1 - exp(-3 * h / effective_range)).
    """
    if n_lags is not None:
        n_bins = n_lags

    emp = empirical_variogram(x_m, y_m, values, max_lag_m=max_lag_m, n_bins=n_bins)
    if emp.empty or "n_pairs" not in emp.columns:
        emp = pd.DataFrame(columns=["lag_center_m", "semivariance", "n_pairs"])
    else:
        emp = emp.loc[emp["n_pairs"] >= 2]

    if len(emp) < 3:
        return {
            "nugget": 0.0,
            "partial_sill": float(np.nanvar(values)) if len(values) > 0 else 0.0,
            "effective_range_m": 50_000.0,
            "effective_range_km": 50.0,
            "range_param_m": 50_000.0 / 3.0,
            "recommended_block_km": 50.0,
        }

    h = emp["lag_center_m"].to_numpy(dtype=float)
    g = emp["semivariance"].to_numpy(dtype=float)
    var_z = max(float(np.var(values)), 1e-6)

    def exp_model(lag: np.ndarray, c0: float, c1: float, a_eff: float) -> np.ndarray:
        return c0 + c1 * (1.0 - np.exp(-3.0 * lag / np.maximum(a_eff, 10.0)))

    lb_a = min(1_000.0, 0.1 * max_lag_m)
    ub_a = max(3.0 * max_lag_m, lb_a * 2.0)
    p0_a = float(np.clip(0.4 * max_lag_m, lb_a * 1.05, ub_a * 0.95))

    try:
        popt, _ = curve_fit(
            exp_model,
            h,
            g,
            p0=[0.1 * var_z, 0.9 * var_z, p0_a],
            bounds=([0.0, 0.0, lb_a], [2.0 * var_z, 4.0 * var_z, ub_a]),
            maxfev=5000,
        )
        c0, c1, a_eff = float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        c0, c1, a_eff = float(g.min()), float(max(g.max() - g.min(), 0.0)), 50_000.0

    a_km = a_eff / 1000.0
    cands = sorted(candidate_blocks_km)
    rec_km = next((b for b in cands if b >= a_km), cands[-1])
    return {
        "nugget": c0,
        "partial_sill": c1,
        "effective_range_m": a_eff,
        "effective_range_km": a_km,
        "range_param_m": a_eff / 3.0,
        "recommended_block_km": float(rec_km),
    }


def morans_i(
    x_m: np.ndarray,
    y_m: np.ndarray,
    residuals: np.ndarray,
    bandwidth_m: float = 35_000.0,
) -> dict[str, float]:
    """Compute global Moran's I statistic on model residuals using inverse-distance weights within `bandwidth_m`."""
    coords = np.column_stack([np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)])
    z = np.asarray(residuals, dtype=float)
    n = z.size
    if n < 4:
        return {"morans_i": 0.0, "expected_i": 0.0}

    z_dev = z - np.mean(z)
    denom = float(np.sum(z_dev**2))
    if denom <= 1e-12:
        return {"morans_i": 0.0, "expected_i": float(-1.0 / (n - 1))}

    dist_mat = squareform(pdist(coords, metric="euclidean"))
    with np.errstate(divide="ignore"):
        w = np.where((dist_mat > 0.0) & (dist_mat <= bandwidth_m), 1.0 / (1.0 + dist_mat / 1000.0), 0.0)

    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return {"morans_i": 0.0, "expected_i": float(-1.0 / (n - 1))}

    num = float(n * (z_dev @ w @ z_dev))
    i_stat = num / (w_sum * denom)
    return {
        "morans_i": float(i_stat),
        "expected_i": float(-1.0 / (n - 1)),
    }
