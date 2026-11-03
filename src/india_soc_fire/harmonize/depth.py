"""Jobbagy & Jackson (2000) exponential depth model.

Cumulative fraction of the 0-dmax SOC stock above depth d (cm):
    Y(d) = 1 - beta**d,  0 < beta < 1
beta is fitted per biome on Indian profiles measured to ~1 m (do NOT reuse
Madagascar or global beta values).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar


def cumulative_fraction(depth_cm, beta: float):
    d = np.asarray(depth_cm, dtype=float)
    return 1.0 - beta ** d


def interval_fraction(top_cm: float, bottom_cm: float, beta: float) -> float:
    """Fraction of the infinite-depth stock lying between top and bottom."""
    return float(beta ** top_cm - beta ** bottom_cm)


def fit_beta(tops, bottoms, stocks) -> float:
    """Fit beta from layer stocks of one or more profiles.

    tops, bottoms: layer boundaries (cm); stocks: SOC stock per layer (any unit).
    Fits normalised cumulative stock at each layer bottom by least squares,
    using profile total down to max bottom as normaliser.
    Pass a list of arrays per argument to pool several profiles.
    """
    if not isinstance(tops[0], (list, tuple, np.ndarray)):
        tops, bottoms, stocks = [tops], [bottoms], [stocks]
    obs = []
    for t, b, s in zip(tops, bottoms, stocks):
        t, b, s = map(lambda x: np.asarray(x, float), (t, b, s))
        order = np.argsort(t)
        b, s = b[order], s[order]
        cum = np.cumsum(s)
        dmax = b[-1]
        obs.append((b, cum / cum[-1], dmax))

    def loss(beta):
        err = 0.0
        for b, y, dmax in obs:
            pred = (1 - beta ** b) / (1 - beta ** dmax)
            err += np.sum((pred - y) ** 2)
        return err

    res = minimize_scalar(loss, bounds=(0.80, 0.9999), method="bounded")
    return float(res.x)


def harmonize_profile(tops, bottoms, stocks, beta: float,
                      targets=((0, 30), (30, 100), (0, 100))) -> dict:
    """Estimate stocks for target depth intervals from an observed profile.

    Total stock is scaled from the observed depth range, then split with the
    fitted exponential. Extrapolation below the deepest sample is flagged.
    """
    t = np.asarray(tops, float); b = np.asarray(bottoms, float); s = np.asarray(stocks, float)
    obs_frac = interval_fraction(t.min(), b.max(), beta)
    total_inf = s.sum() / obs_frac
    out = {}
    for top, bot in targets:
        out[f"{top}-{bot}"] = {
            "stock": total_inf * interval_fraction(top, bot, beta),
            "extrapolated": bool(bot > b.max() or top < t.min()),
        }
    return out
