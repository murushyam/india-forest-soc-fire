"""Uncertainty (STUB): bootstrap mean/SD/CV rasters and QRF 90% PI (q05, q95).

Planned API: bootstrap_predict(model_factory, X, y, X_grid, n) -> mean, sd, cv;
qrf_interval(model, X_grid, (0.05, 0.95)) -> lower, upper, width.
"""
import numpy as np

def summarize_bootstrap(preds: np.ndarray):
    """preds: (n_boot, n_cells) -> mean, sd, cv."""
    mean = preds.mean(0); sd = preds.std(0, ddof=1)
    return mean, sd, np.divide(sd, mean, out=np.full_like(mean, np.nan), where=mean != 0)
