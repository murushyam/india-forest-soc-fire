"""Jobbagy & Jackson (2000) exponential depth model.

Cumulative fraction of the 0-dmax SOC stock above depth d (cm):
    Y(d) = 1 - beta**d,  0 < beta < 1
beta is fitted per biome on Indian profiles measured to ~1 m (do NOT reuse
Madagascar or global beta values).
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
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


def harmonize_profile(
    tops,
    bottoms,
    stocks,
    beta: float,
    targets=((0, 30), (30, 100), (0, 100)),
) -> dict:
    """Estimate stocks for target depth intervals from an observed profile.

    Uses mass-preserving Jobbágy-Jackson exponential disaggregation within
    measured horizons and exponential tail scaling beyond `b.max()`.
    This preserves exact measured horizon stocks when target boundaries align
    with horizon boundaries (e.g. 0-30 cm and 30-100 cm) while maintaining
    exact additivity across intervals.
    """
    t = np.asarray(tops, float)
    b = np.asarray(bottoms, float)
    s = np.asarray(stocks, float)
    order = np.argsort(t)
    t, b, s = t[order], b[order], s[order]

    t_min = float(t.min())
    b_max = float(b.max())
    obs_frac = interval_fraction(t_min, b_max, beta)
    total_inf = float(s.sum()) / obs_frac

    out = {}
    for top, bot in targets:
        top_f, bot_f = float(top), float(bot)
        stock_val = 0.0

        # 1. Mass-preserving integration across observed horizons
        for tk, bk, sk in zip(t, b, s):
            a_k = max(tk, top_f)
            c_k = min(bk, bot_f)
            if c_k > a_k:
                hz_frac = interval_fraction(tk, bk, beta)
                sub_frac = interval_fraction(a_k, c_k, beta)
                w_k = sub_frac / hz_frac if hz_frac > 1e-12 else (c_k - a_k) / max(bk - tk, 1e-12)
                stock_val += float(sk) * w_k

        # 2. Extrapolation above t_min (if top_f < t_min)
        if top_f < t_min:
            hi_pre = min(bot_f, t_min)
            if hi_pre > top_f:
                stock_val += total_inf * interval_fraction(top_f, hi_pre, beta)

        # 3. Extrapolation below b_max (if bot_f > b_max)
        if bot_f > b_max:
            lo_post = max(top_f, b_max)
            if bot_f > lo_post:
                stock_val += total_inf * interval_fraction(lo_post, bot_f, beta)

        out[f"{top}-{bot}"] = {
            "stock": float(stock_val),
            "extrapolated": bool(bot_f > b_max or top_f < t_min),
        }
    return out


def fit_beta_by_biome(
    profiles_df: pd.DataFrame,
    min_depth_cm: float = 50.0,
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Calibrate biome-specific Jobbágy-Jackson beta parameters on profiles >= min_depth_cm.

    Returns a DataFrame with columns:
    ``biome, n_profiles, beta_pooled, beta_mean, beta_median, beta_ci_low, beta_ci_high, rmse_cum``
    """
    required = {"profile_id", "top_cm", "bottom_cm", "stock_mg_ha"}
    if not required.issubset(profiles_df.columns):
        raise ValueError(f"profiles_df must contain {required}")

    df = profiles_df.copy()
    if "biome" not in df.columns:
        df["biome"] = "default_forest"

    rng = np.random.default_rng(seed)
    profile_betas: list[dict[str, object]] = []
    biome_layers: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}

    for pid, grp in df.groupby("profile_id", sort=False):
        grp_s = grp.sort_values("top_cm")
        d_max = float(grp_s["bottom_cm"].max())
        if len(grp_s) < 2 or d_max < min_depth_cm:
            continue
        tops = grp_s["top_cm"].to_numpy(dtype=float)
        bottoms = grp_s["bottom_cm"].to_numpy(dtype=float)
        stocks = grp_s["stock_mg_ha"].to_numpy(dtype=float)
        cum = np.cumsum(stocks)
        if cum[-1] <= 0:
            continue

        b_val = fit_beta(tops, bottoms, stocks)
        y_obs = cum / cum[-1]
        y_hat = (1.0 - np.power(b_val, bottoms)) / (1.0 - np.power(b_val, d_max))
        biome = str(grp_s["biome"].iloc[0])
        profile_betas.append(
            {
                "profile_id": pid,
                "biome": biome,
                "beta": b_val,
                "sq_err": float(np.mean((y_obs - y_hat) ** 2)),
            }
        )
        if biome not in biome_layers:
            biome_layers[biome] = ([], [], [])
        biome_layers[biome][0].append(tops)
        biome_layers[biome][1].append(bottoms)
        biome_layers[biome][2].append(stocks)

    prof_df = pd.DataFrame(profile_betas)
    if prof_df.empty:
        return pd.DataFrame(
            columns=[
                "biome",
                "n_profiles",
                "beta_pooled",
                "beta_mean",
                "beta_median",
                "beta_ci_low",
                "beta_ci_high",
                "rmse_cum",
            ]
        )

    rows: list[dict[str, object]] = []
    for biome, bgrp in prof_df.groupby("biome", sort=True):
        betas = bgrp["beta"].to_numpy(dtype=float)
        n_p = len(betas)
        t_list, b_list, s_list = biome_layers[biome]
        beta_pooled = fit_beta(t_list, b_list, s_list)

        if n_p >= 2 and n_boot > 0:
            boot_idx = rng.integers(0, n_p, size=(n_boot, n_p))
            boot_means = betas[boot_idx].mean(axis=1)
            ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
        else:
            ci_low = ci_high = float(betas.mean())

        rows.append(
            {
                "biome": biome,
                "n_profiles": int(n_p),
                "beta_pooled": float(beta_pooled),
                "beta_mean": float(betas.mean()),
                "beta_median": float(np.median(betas)),
                "beta_ci_low": float(ci_low),
                "beta_ci_high": float(ci_high),
                "rmse_cum": float(np.sqrt(bgrp["sq_err"].mean())),
            }
        )
    return pd.DataFrame(rows)


def harmonize_all_profiles(
    profiles_df: pd.DataFrame,
    beta_by_biome: dict[str, float] | None = None,
    target_intervals_cm: Sequence[tuple[int, int]] = ((0, 30), (30, 100), (0, 100)),
    use_profile_beta_when_deep: bool = False,
    min_depth_cm: float = 50.0,
) -> pd.DataFrame:
    """Harmonise all soil profiles in a horizon DataFrame into 1-row-per-profile stocks."""
    df = profiles_df.copy()
    if beta_by_biome is None:
        beta_table = fit_beta_by_biome(df, min_depth_cm=min_depth_cm)
        beta_by_biome = dict(zip(beta_table["biome"], beta_table["beta_pooled"]))

    records: list[dict[str, object]] = []
    for pid, grp in df.groupby("profile_id", sort=False):
        grp_s = grp.sort_values("top_cm")
        biome = str(grp_s["biome"].iloc[0]) if "biome" in grp_s.columns else "default_forest"
        d_max = float(grp_s["bottom_cm"].max())
        n_hz = len(grp_s)

        tops = grp_s["top_cm"].to_numpy(dtype=float)
        bottoms = grp_s["bottom_cm"].to_numpy(dtype=float)
        stocks = grp_s["stock_mg_ha"].to_numpy(dtype=float)

        if use_profile_beta_when_deep and n_hz >= 2 and d_max >= min_depth_cm:
            beta_use = fit_beta(tops, bottoms, stocks)
        else:
            beta_use = float(beta_by_biome.get(biome, 0.966))

        harm = harmonize_profile(
            tops,
            bottoms,
            stocks,
            beta=beta_use,
            targets=target_intervals_cm,
        )

        rec: dict[str, object] = {
            "profile_id": pid,
            "lon": float(grp_s["lon"].iloc[0]),
            "lat": float(grp_s["lat"].iloc[0]),
            "max_depth_cm": d_max,
            "n_horizons": int(n_hz),
            "beta": float(beta_use),
        }
        for optional_col in ("region", "biome", "burn_class", "sample_year", "source", "licence"):
            if optional_col in grp_s.columns:
                rec[optional_col] = grp_s[optional_col].iloc[0]

        for lo, hi in target_intervals_cm:
            key = f"{lo}-{hi}"
            rec[f"soc_{lo}_{hi}_mg_ha"] = float(harm[key]["stock"])
            rec[f"extrapolated_{lo}_{hi}"] = bool(harm[key]["extrapolated"])
        records.append(rec)

    return pd.DataFrame(records)
