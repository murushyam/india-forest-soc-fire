"""Pedotransfer functions (PTFs) for bulk density and coarse fragment imputation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class BulkDensityPTF:
    """Calibrated organic-mineral mixing & depth-aware bulk density PTF.

    Combines the physical Adams (1973) / Alexander (1980) organic-mineral
    self-mixing formulation with a depth compaction term:

        BD(SOC, z_mid) = 100 / (OM / bd_om + (100 - OM) / (bd_min0 + k_z * ln(z_mid + 1)))

    where OM (%) = 0.1724 * soc_g_kg.
    """

    bd_om: float = 0.224
    bd_min0: float = 1.38
    k_z: float = 0.065
    rmse: float = 0.12
    n_calib: int = 0

    def predict(
        self,
        soc_g_kg: np.ndarray | float,
        z_mid_cm: np.ndarray | float,
    ) -> np.ndarray:
        """Predict fine-earth bulk density (g cm-3) from SOC (g kg-1) and midpoint depth (cm)."""
        soc = np.asarray(soc_g_kg, dtype=float)
        z = np.maximum(np.asarray(z_mid_cm, dtype=float), 0.5)
        om_pct = np.clip(0.1724 * soc, 0.01, 95.0)
        bd_min = np.clip(self.bd_min0 + self.k_z * np.log(z + 1.0), 0.85, 1.95)
        bd = 100.0 / (om_pct / self.bd_om + (100.0 - om_pct) / bd_min)
        return np.clip(bd, 0.25, 1.95)


def _adams_depth_model(X: np.ndarray, bd_om: float, bd_min0: float, k_z: float) -> np.ndarray:
    soc, z = X[0], X[1]
    om_pct = np.clip(0.1724 * soc, 0.01, 95.0)
    bd_min = bd_min0 + k_z * np.log(np.maximum(z, 0.5) + 1.0)
    return 100.0 / (om_pct / bd_om + (100.0 - om_pct) / bd_min)


def fit_bd_ptf(df: pd.DataFrame) -> BulkDensityPTF:
    """Calibrate BulkDensityPTF parameters on Indian forest horizons with measured BD."""
    valid = (
        df["bd_g_cm3"].notna()
        & df["soc_g_kg"].notna()
        & df["top_cm"].notna()
        & df["bottom_cm"].notna()
    )
    if "bd_imputed" in df.columns:
        valid &= ~df["bd_imputed"].fillna(False).astype(bool)

    sub = df.loc[valid]
    if len(sub) < 10:
        return BulkDensityPTF()

    soc = sub["soc_g_kg"].to_numpy(dtype=float)
    z_mid = 0.5 * (sub["top_cm"].to_numpy(dtype=float) + sub["bottom_cm"].to_numpy(dtype=float))
    bd_obs = sub["bd_g_cm3"].to_numpy(dtype=float)

    try:
        popt, _ = curve_fit(
            _adams_depth_model,
            np.vstack([soc, z_mid]),
            bd_obs,
            p0=[0.224, 1.38, 0.065],
            bounds=([0.12, 0.90, 0.0], [0.55, 1.85, 0.25]),
            maxfev=5000,
        )
        bd_pred = _adams_depth_model(np.vstack([soc, z_mid]), *popt)
        rmse = float(np.sqrt(np.mean((bd_obs - bd_pred) ** 2)))
        return BulkDensityPTF(
            bd_om=float(popt[0]),
            bd_min0=float(popt[1]),
            k_z=float(popt[2]),
            rmse=rmse,
            n_calib=int(len(sub)),
        )
    except RuntimeError:
        return BulkDensityPTF(n_calib=int(len(sub)))


def impute_missing_bd_and_cf(
    df: pd.DataFrame,
    ptf: BulkDensityPTF | None = None,
    default_coarse_frac_by_biome: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, BulkDensityPTF]:
    """Impute missing bulk density and coarse fragments with explicit provenance flags.

    Adds ``bd_imputed`` (bool), ``bd_se`` (standard error of BD in g cm-3), and
    ``cf_imputed`` (bool) columns so downstream stock calculations can propagate
    pedotransfer uncertainty.
    """
    out = df.copy()
    if ptf is None:
        ptf = fit_bd_ptf(out)

    z_mid = 0.5 * (out["top_cm"].astype(float) + out["bottom_cm"].astype(float))
    bd_missing = out["bd_g_cm3"].isna()

    if "bd_imputed" not in out.columns:
        out["bd_imputed"] = False
    out["bd_imputed"] = out["bd_imputed"].fillna(False).astype(bool) | bd_missing

    if bd_missing.any():
        out.loc[bd_missing, "bd_g_cm3"] = ptf.predict(
            out.loc[bd_missing, "soc_g_kg"].to_numpy(dtype=float),
            z_mid.loc[bd_missing].to_numpy(dtype=float),
        )

    # Assign standard error: 0.04 g/cm3 laboratory core measurement error vs PTF RMSE
    out["bd_se"] = np.where(out["bd_imputed"], ptf.rmse, 0.04)

    # Coarse fragments: default biome-informed gravel fraction if missing
    if default_coarse_frac_by_biome is None:
        default_coarse_frac_by_biome = {
            "tropical_moist_deciduous": 0.08,
            "subtropical_chir_pine": 0.16,
            "temperate_banj_oak": 0.14,
            "himalayan_moist_temperate": 0.14,
        }

    cf_missing = out["coarse_frac"].isna()
    out["cf_imputed"] = cf_missing
    if cf_missing.any():
        if "biome" in out.columns:
            mapped_cf = out.loc[cf_missing, "biome"].map(default_coarse_frac_by_biome).fillna(0.10)
            out.loc[cf_missing, "coarse_frac"] = mapped_cf.to_numpy(dtype=float)
        else:
            out.loc[cf_missing, "coarse_frac"] = 0.10

    return out, ptf


def propagate_horizon_stock_sd(
    soc_g_kg: np.ndarray | float,
    bd_g_cm3: np.ndarray | float,
    thick_cm: np.ndarray | float,
    coarse_frac: np.ndarray | float = 0.0,
    soc_rel_se: float = 0.05,
    bd_se: np.ndarray | float = 0.08,
    cf_se: float = 0.04,
) -> np.ndarray:
    """First-order Taylor propagation of analytical & PTF error into horizon SOC stock SD (Mg C ha-1)."""
    soc = np.asarray(soc_g_kg, dtype=float)
    bd = np.asarray(bd_g_cm3, dtype=float)
    dz = np.asarray(thick_cm, dtype=float)
    cf = np.asarray(coarse_frac, dtype=float)

    stock = soc * bd * dz * (1.0 - cf) * 0.1
    rel_var = (
        (soc_rel_se) ** 2
        + (np.asarray(bd_se, dtype=float) / np.maximum(bd, 0.1)) ** 2
        + (cf_se / np.maximum(1.0 - cf, 0.05)) ** 2
    )
    return stock * np.sqrt(rel_var)
