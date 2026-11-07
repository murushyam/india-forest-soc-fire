"""Post-fire 1-5 year SOC recovery trajectories and coupled GPP/uWUE/EF flux dynamics."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..covariates.local_ingest import (
    DEFAULT_TELE_DIR,
    load_odisha_flux_timeseries,
    load_uttarakhand_flux_timeseries,
)


def recovery_curve_model(
    t_years: np.ndarray | float,
    initial_loss_frac: float,
    k_rate: float,
    repeat_burn_penalty: float = 0.0,
    n_burns: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """Relative topsoil SOC stock anomaly R(t) = (SOC(t) - SOC_ctrl) / SOC_ctrl at `t_years` post-fire.

    R(t) = -initial_loss_frac * exp(-k_rate * t) - repeat_burn_penalty * max(n_burns - 1, 0)
    """
    t = np.asarray(t_years, dtype=float)
    nb = np.maximum(np.asarray(n_burns, dtype=float) - 1.0, 0.0)
    return -initial_loss_frac * np.exp(-k_rate * t) - repeat_burn_penalty * nb


def fit_postfire_recovery_trajectory(
    years_since_fire: np.ndarray,
    rel_soc_anomaly: np.ndarray,
    n_burns: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit exponential post-fire recovery parameters (`initial_loss_frac`, `k_rate`, `half_life_yr`)."""
    t = np.asarray(years_since_fire, dtype=float)
    r = np.asarray(rel_soc_anomaly, dtype=float)
    nb = np.zeros_like(t) if n_burns is None else np.asarray(n_burns, dtype=float)

    def _obj(X: np.ndarray, f0: float, k: float, pen: float) -> np.ndarray:
        return np.asarray(recovery_curve_model(X[0], f0, k, pen, X[1]), dtype=float)

    try:
        popt, _ = curve_fit(
            _obj,
            np.vstack([t, nb]),
            r,
            p0=[0.22, 0.35, 0.02],
            bounds=([0.01, 0.02, 0.0], [0.80, 2.50, 0.15]),
            maxfev=5000,
        )
        f0, k_val, pen = float(popt[0]), float(popt[1]), float(popt[2])
    except RuntimeError:
        f0, k_val, pen = 0.22, 0.35, 0.02

    r_hat = _obj(np.vstack([t, nb]), f0, k_val, pen)
    rmse = float(np.sqrt(np.mean((r - r_hat) ** 2)))
    half_life = float(np.log(2.0) / max(k_val, 1e-6))
    t_90 = float(np.log(10.0) / max(k_val, 1e-6))

    return {
        "initial_loss_frac": f0,
        "k_rate_yr": k_val,
        "half_life_yr": half_life,
        "time_to_90pct_recovery_yr": t_90,
        "repeat_burn_penalty": pen,
        "rmse": rmse,
    }


def simulate_biome_recovery_chronosequence(
    years: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
) -> pd.DataFrame:
    """Generate 0-5 yr post-fire topsoil SOC recovery curves across biomes and severity classes."""
    biome_params = {
        "tropical_moist_deciduous": {
            "low": (0.09, 0.55),
            "moderate_low": (0.17, 0.45),
            "moderate_high": (0.26, 0.36),
            "high": (0.34, 0.28),
        },
        "subtropical_chir_pine": {
            "low": (0.11, 0.42),
            "moderate_low": (0.21, 0.33),
            "moderate_high": (0.31, 0.25),
            "high": (0.41, 0.20),
        },
        "temperate_banj_oak": {
            "low": (0.08, 0.38),
            "moderate_low": (0.16, 0.30),
            "moderate_high": (0.25, 0.23),
            "high": (0.35, 0.18),
        },
    }
    rows: list[dict[str, object]] = []
    for biome, sev_dict in biome_params.items():
        for sev, (f0, k_rate) in sev_dict.items():
            for t in years:
                rel_single = float(recovery_curve_model(t, f0, k_rate, repeat_burn_penalty=0.025, n_burns=1))
                rel_repeat = float(recovery_curve_model(t, f0, k_rate, repeat_burn_penalty=0.025, n_burns=3))
                rows.append(
                    {
                        "biome": biome,
                        "burn_class": sev,
                        "years_since_fire": int(t),
                        "rel_soc_anomaly_single_burn": rel_single,
                        "rel_soc_anomaly_repeat_3burns": rel_repeat,
                        "pct_recovered_single_burn": float(100.0 * (1.0 - np.exp(-k_rate * t))),
                        "half_life_yr": float(np.log(2.0) / k_rate),
                    }
                )
    return pd.DataFrame(rows)


def summarize_local_postfire_flux_recovery(
    tele_dir: str | Path = DEFAULT_TELE_DIR,
) -> dict[str, pd.DataFrame]:
    """Summarize 2021-2025 post-fire GPP, ET, uWUE, SH, and LH recovery from `I:\\tele` CSVs."""
    out: dict[str, pd.DataFrame] = {}

    od_df = load_odisha_flux_timeseries(tele_dir=tele_dir)
    if not od_df.empty:
        num_cols = [c for c in od_df.select_dtypes(include=[np.number]).columns if c != "year"]
        if num_cols:
            out["odisha_flux_annual"] = (
                od_df.groupby("year", as_index=False)[num_cols].mean(numeric_only=True)
            )

    uk_df = load_uttarakhand_flux_timeseries(tele_dir=tele_dir)
    if not uk_df.empty:
        num_cols = [c for c in uk_df.select_dtypes(include=[np.number]).columns if c != "year"]
        if num_cols and "flux_type" in uk_df.columns:
            out["uttarakhand_flux_annual"] = (
                uk_df.groupby(["year", "flux_type"], as_index=False)[num_cols].mean(numeric_only=True)
            )

    return out
