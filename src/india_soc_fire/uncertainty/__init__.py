"""Spatial prediction uncertainty: 100x spatial-block bootstrap (mean, SD, CV) and QRF 90% intervals."""
from __future__ import annotations

from typing import Any, NamedTuple, Sequence
import numpy as np

from ..models import build_model


class BootstrapSummary(NamedTuple):
    """Supports both tuple unpacking `m, sd, cv = summarize_bootstrap(preds)`
    and dict-style key lookup `res['mean']`, `res['sd']`, `res['cv']`.
    """

    mean: np.ndarray
    sd: np.ndarray
    cv: np.ndarray

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        return tuple.__getitem__(self, key)


def summarize_bootstrap(preds: np.ndarray) -> BootstrapSummary:
    """Given predictions of shape `(n_boot, ...)`, return `(mean, sd, cv)`."""
    arr = np.asarray(preds, dtype=float)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean > 0, sd / mean, np.nan)
    return BootstrapSummary(mean=mean, sd=sd, cv=cv)


def bootstrap_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    model_name: str = "rf",
    n_boot: int = 100,
    block_ids: np.ndarray | None = None,
    seed: int = 42,
    **model_params: Any,
) -> BootstrapSummary:
    """Fit `n_boot` spatial-block (or row) bootstrap models and return `BootstrapSummary(mean, sd, cv)`."""
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    X_pr = np.asarray(X_pred, dtype=float)
    n_samples = y_tr.size
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")

    rng = np.random.default_rng(seed)
    preds = np.empty((n_boot, X_pr.shape[0]), dtype=float)

    uniq_blocks = np.unique(block_ids) if block_ids is not None else None
    use_blocks = uniq_blocks is not None and uniq_blocks.size >= 3

    for b in range(n_boot):
        if use_blocks and block_ids is not None and uniq_blocks is not None:
            sampled_blocks = rng.choice(uniq_blocks, size=uniq_blocks.size, replace=True)
            idx_list = [np.flatnonzero(block_ids == blk) for blk in sampled_blocks]
            boot_idx = np.concatenate(idx_list)
        else:
            boot_idx = rng.integers(0, n_samples, size=n_samples)

        est = build_model(model_name, random_state=seed + b, **model_params)
        est.fit(X_tr[boot_idx], y_tr[boot_idx])
        preds[b] = np.asarray(est.predict(X_pr), dtype=float)

    return summarize_bootstrap(preds)


def qrf_interval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    quantiles: Sequence[float] = (0.05, 0.50, 0.95),
    seed: int = 42,
    **model_params: Any,
) -> dict[str, np.ndarray]:
    """Fit Quantile Regression Forest and return `q05`, `q50`, `q95`, `piw90`, and `rpiw90`."""
    if len(quantiles) != 3:
        raise ValueError("quantiles must be a 3-tuple (q_low, q_med, q_high), e.g. (0.05, 0.50, 0.95)")

    qrf = build_model("qrf", random_state=seed, **model_params)
    qrf.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
    q_preds = qrf.predict(np.asarray(X_pred, dtype=float), quantiles=quantiles)

    q_low = q_preds[:, 0]
    q_med = q_preds[:, 1]
    q_high = q_preds[:, 2]
    piw = np.maximum(q_high - q_low, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rpiw = np.where(q_med > 0, piw / q_med, np.nan)

    return {
        "q05": q_low,
        "q50": q_med,
        "q95": q_high,
        "piw90": piw,
        "rpiw90": rpiw,
    }
