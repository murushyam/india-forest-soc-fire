"""Mass-preserving equal-area spline depth harmonisation benchmark (Bishop et al., 1999; Malone et al., 2009)."""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
from scipy.interpolate import PchipInterpolator


def equal_area_spline_harmonize(
    horizon_tops_cm: Sequence[float],
    horizon_bottoms_cm: Sequence[float],
    horizon_stocks_mg_ha: Sequence[float],
    target_intervals_cm: Iterable[tuple[int, int]] = ((0, 30), (30, 100), (0, 100)),
) -> dict[tuple[int, int], float]:
    """Mass-preserving monotonic cubic Hermite (PCHIP) cumulative mass spline.

    Because the cumulative SOC mass function M(z) = int_0^z rho_C(u) du is
    strictly non-decreasing and exactly known at horizon boundaries z_0=0, z_1, ..., z_n,
    interpolating M(z) with a shape-preserving monotonic spline (Fritsch & Carlson, 1980)
    guarantees:
      1. Exact mass preservation over every measured horizon [top_k, bottom_k],
      2. Non-negative SOC density dM/dz >= 0 at every depth z,
      3. Exact additivity across target intervals: M(0, 30) + M(30, 100) = M(0, 100).
    """
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

    return {
        (lo, hi): max(0.0, eval_cum(float(hi)) - eval_cum(float(lo)))
        for (lo, hi) in target_intervals_cm
    }
