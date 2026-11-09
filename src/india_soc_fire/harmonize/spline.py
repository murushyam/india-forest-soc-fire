"""Mass-preserving equal-area spline depth harmonisation benchmark (Bishop et al., 1999; Malone et al., 2009)."""
from __future__ import annotations

from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def equal_area_spline_harmonize(
    horizon_tops_cm: Sequence[float] | pd.DataFrame,
    horizon_bottoms_cm: Sequence[float] | None = None,
    horizon_stocks_mg_ha: Sequence[float] | None = None,
    target_intervals_cm: Iterable[tuple[int, int]] = ((0, 30), (30, 100), (0, 100)),
    max_grid_cm: float | None = None,
) -> dict[Any, Any]:
    """Mass-preserving monotonic cubic Hermite (PCHIP) cumulative mass spline.

    Supports either:
    1. horizon_tops_cm, horizon_bottoms_cm, horizon_stocks_mg_ha sequences
    2. horizon_tops_cm as a DataFrame containing 'top_cm', 'bottom_cm', 'stock_mg_ha'
    """
    if isinstance(horizon_tops_cm, pd.DataFrame):
        df = horizon_tops_cm
        tops = df["top_cm"].to_numpy(dtype=float)
        bottoms = df["bottom_cm"].to_numpy(dtype=float)
        stocks = df["stock_mg_ha"].to_numpy(dtype=float)
    else:
        tops = np.asarray(horizon_tops_cm, dtype=float)
        bottoms = np.asarray(horizon_bottoms_cm, dtype=float)
        stocks = np.asarray(horizon_stocks_mg_ha, dtype=float)
    if not (tops.size == bottoms.size == stocks.size) or tops.size == 0:
        raise ValueError("Horizon arrays must be non-empty and equal length")

    order = np.argsort(tops)
    tops = tops[order]
    bottoms = bottoms[order]
    stocks = stocks[order]

    z_knots = np.concatenate([[0.0], bottoms])
    m_knots = np.concatenate([[0.0], np.cumsum(stocks)])

    # Deduplicate any identical knot depths
    uniq_z, uniq_idx = np.unique(z_knots, return_index=True)
    uniq_m = m_knots[uniq_idx]

    if uniq_z.size < 2:
        raise ValueError("Need at least one positive depth horizon")

    spline = PchipInterpolator(uniq_z, uniq_m, extrapolate=True)
    d_max = float(uniq_z[-1])

    def eval_cum(z: float) -> float:
        if z <= 0.0:
            return 0.0
        if z <= d_max:
            return float(spline(z))
        # Beyond observed max depth, decay density exponentially to avoid runaway extrapolation
        slope_end = max(float(spline.derivative()(d_max)), 0.0)
        decay_k = 0.035  # cm-1 (~beta = 0.966)
        extra = (slope_end / decay_k) * (1.0 - np.exp(-decay_k * (z - d_max)))
        return float(uniq_m[-1] + extra)

    out: dict[Any, Any] = {}
    for (lo, hi) in target_intervals_cm:
        val = max(0.0, eval_cum(float(hi)) - eval_cum(float(lo)))
        out[(lo, hi)] = val
        out[f"{lo}_{hi}"] = {"soc_stock_mg_ha": val}
    return out
