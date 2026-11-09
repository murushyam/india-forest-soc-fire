"""Domain-constrained metrics, scale-normalized asymmetric LinEx loss, finite-sample
conformal calibration diagnostics (PICP, PINAW, ACE), and Wilcoxon-Holm-Bonferroni
statistical hypothesis testing for the ``india-soc-fire`` framework.

Implements all Four-Pillar Mathematical & Physical Support Guarantees from the
``scientific-research-writing`` and ``code-review-and-graph-validation`` skills.
"""

from __future__ import annotations

from typing import Dict, List
import math
import numpy as np
from scipy import stats


def project_non_negative_support(y_hat: np.ndarray) -> np.ndarray:
    """Enforce physical non-negativity Pi_[0, inf)(y_hat) = max(0, y_hat)."""
    return np.maximum(0.0, np.asarray(y_hat, dtype=float))


def project_discrete_count_support(y_hat: np.ndarray) -> np.ndarray:
    """Enforce discrete integer count support Pi_N0(y_hat) = max(0, floor(y_hat + 0.5))."""
    arr = np.asarray(y_hat, dtype=float)
    return np.maximum(0, np.floor(arr + 0.5)).astype(int)


def compute_scale_normalized_linex_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    b: float = 1.2,
    clip_val: float = 10.0,
) -> float:
    """Compute scale-normalized asymmetric LinEx loss (Eq. 33).

    L_asym(y, y_hat; b) = (2 / b^2) * [exp(b * u_i) - b * u_i - 1],
    where u_i = clip((y_i - y_hat_i) / sigma_Y, -clip_val, clip_val).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = project_non_negative_support(y_pred)
    sigma_y = max(float(np.std(yt)), 1e-6)
    u = np.clip((yt - yp) / sigma_y, -clip_val, clip_val)
    loss = (2.0 / (b ** 2)) * (np.exp(b * u) - b * u - 1.0)
    return float(np.mean(loss))


def compute_bounded_conformal_intervals(
    cal_residuals: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """Compute finite-sample conformalized prediction intervals and PICP/PINAW/ACE.

    Applies finite-sample quantile level min(1.0, ceil((N_cal + 1)*(1 - alpha)) / N_cal)
    and projects lower bounds onto [0, inf).
    """
    r_cal = np.asarray(cal_residuals, dtype=float)
    n_cal = len(r_cal)
    q_level = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / max(n_cal, 1))
    q_hat = float(np.quantile(r_cal, q_level))

    lower = np.maximum(0.0, np.asarray(y_pred_lower, dtype=float) - q_hat)
    upper = np.maximum(lower, np.asarray(y_pred_upper, dtype=float) + q_hat)
    yt = np.asarray(y_test, dtype=float)

    covered = (yt >= lower) & (yt <= upper)
    picp = float(np.mean(covered))
    y_range = max(float(np.max(yt) - np.min(yt)), 1e-6)
    pinaw = float(np.mean(upper - lower) / y_range)
    ace = float(picp - (1.0 - alpha))

    return {
        "q_hat": q_hat,
        "picp": picp,
        "pinaw": pinaw,
        "ace": ace,
        "nominal_coverage": 1.0 - alpha,
    }


def wilcoxon_holm_bonferroni(
    errors_baseline: List[np.ndarray],
    errors_model: List[np.ndarray],
) -> List[Dict[str, float]]:
    """Run two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni correction."""
    raw_pvals: List[float] = []
    stats_list: List[float] = []
    for eb, em in zip(errors_baseline, errors_model):
        diff = np.abs(np.asarray(eb, dtype=float)) - np.abs(np.asarray(em, dtype=float))
        if np.allclose(diff, 0.0):
            stat, pval = 0.0, 1.0
        else:
            res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            stat, pval = float(res.statistic), float(res.pvalue)
        stats_list.append(stat)
        raw_pvals.append(pval)

    m = len(raw_pvals)
    order = np.argsort(raw_pvals)
    adj_pvals = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = min(1.0, raw_pvals[idx] * (m - rank))
        running_max = max(running_max, corrected)
        adj_pvals[idx] = running_max

    return [
        {
            "statistic": stats_list[i],
            "p_raw": raw_pvals[i],
            "p_holm_bonferroni": float(adj_pvals[i]),
            "significant_005": bool(adj_pvals[i] < 0.05),
        }
        for i in range(m)
    ]
