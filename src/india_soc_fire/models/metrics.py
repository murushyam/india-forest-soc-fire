"""Regression accuracy and uncertainty calibration metrics for Digital Soil Mapping."""
from __future__ import annotations

import numpy as np


def lins_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's (1989) Concordance Correlation Coefficient (CCC)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size < 2:
        return 0.0
    mu_t = float(np.mean(yt))
    mu_p = float(np.mean(yp))
    var_t = float(np.var(yt, ddof=0))
    var_p = float(np.var(yp, ddof=0))
    cov_tp = float(np.mean((yt - mu_t) * (yp - mu_p)))
    denom = var_t + var_p + (mu_t - mu_p) ** 2
    if denom <= 1e-12:
        return 1.0
    return float((2.0 * cov_tp) / denom)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, Mean Error (bias), R2 (Model Efficiency Coefficient), and Lin's CCC."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias_me = float(np.mean(err))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2_mec = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    ccc = lins_ccc(yt, yp)
    return {
        "rmse": rmse,
        "mae": mae,
        "bias_me": bias_me,
        "r2": r2_mec,
        "ccc": ccc,
    }


def interval_metrics(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    nominal: float = 0.90,
) -> dict[str, float]:
    """Compute Prediction Interval Coverage Probability (PICP), MPIW, and Absolute Coverage Error."""
    yt = np.asarray(y_true, dtype=float)
    ql = np.asarray(q_low, dtype=float)
    qh = np.asarray(q_high, dtype=float)
    covered = (yt >= ql) & (yt <= qh)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(qh - ql))
    return {
        "picp": picp,
        "mpiw": mpiw,
        "nominal": float(nominal),
        "coverage_error": float(picp - nominal),
    }
