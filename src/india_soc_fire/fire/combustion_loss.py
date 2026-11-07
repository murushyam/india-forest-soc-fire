"""Space-for-time paired covariate matching for topsoil (0-30 cm) SOC combustion loss
and subsoil (30-100 cm) negative-control falsification testing.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ttest_1samp


DEFAULT_MATCH_COLS: tuple[str, ...] = (
    "elevation",
    "slope",
    "aspect",
    "twi",
    "era5land_mat",
    "era5land_map",
)

SEVERITY_ORDER: tuple[str, ...] = ("low", "moderate_low", "moderate_high", "high")


def estimate_paired_combustion_loss(
    df: pd.DataFrame,
    topsoil_col: str = "soc_0_30_mg_ha",
    subsoil_col: str = "soc_30_100_mg_ha",
    severity_col: str = "burn_class",
    control_label: str = "unburned",
    match_cols: Sequence[str] = DEFAULT_MATCH_COLS,
    k_matches: int = 3,
    n_boot: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate topsoil (0-30 cm) fire combustion loss and subsoil (30-100 cm) stability by severity class.

    For each burned observation in severity class `s`, matches `k_matches` unburned control
    observations within the same biome/region using standardised Euclidean / diagonal-Mahalanobis
    distance on non-fire environmental covariates (`match_cols`).

    Returns a DataFrame with columns:
    - ``burn_class``, ``n_burned``, ``n_controls``
    - ``topsoil_control_mg_ha``, ``topsoil_burned_mg_ha``
    - ``topsoil_delta_mg_ha`` (ATT = Burned - Matched Control)
    - ``topsoil_loss_frac`` (-delta / control), ``loss_frac_ci_low``, ``loss_frac_ci_high``
    - ``subsoil_delta_mg_ha``, ``subsoil_loss_frac``, ``subsoil_p_value`` (falsification check)
    """
    if topsoil_col not in df.columns or severity_col not in df.columns:
        raise ValueError(f"df must contain '{topsoil_col}' and '{severity_col}'")

    has_subsoil = subsoil_col in df.columns
    avail_match = [c for c in match_cols if c in df.columns]

    controls = df.loc[df[severity_col] == control_label].copy()
    if controls.empty:
        raise ValueError(f"No control observations with {severity_col} == '{control_label}'")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []

    # Standardise matching covariates across the dataset
    if avail_match:
        X_all = df[avail_match].to_numpy(dtype=float)
        mu = np.nanmean(X_all, axis=0)
        sd = np.nanstd(X_all, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Z_ctrl_all = (controls[avail_match].to_numpy(dtype=float) - mu) / sd
    else:
        Z_ctrl_all = None

    for sev in SEVERITY_ORDER:
        burned = df.loc[df[severity_col] == sev].copy()
        if burned.empty:
            continue

        top_b = burned[topsoil_col].to_numpy(dtype=float)
        sub_b = burned[subsoil_col].to_numpy(dtype=float) if has_subsoil else np.zeros_like(top_b)

        matched_top_ctrl = np.empty_like(top_b)
        matched_sub_ctrl = np.empty_like(sub_b)

        for i in range(len(burned)):
            # Prefer matching within the same biome if biome column is present
            if "biome" in df.columns:
                b_val = burned["biome"].iloc[i]
                cand_mask = (controls["biome"] == b_val).to_numpy()
                if not cand_mask.any():
                    cand_mask = np.ones(len(controls), dtype=bool)
            else:
                cand_mask = np.ones(len(controls), dtype=bool)

            ctrl_sub = controls.loc[cand_mask]
            k_eff = min(k_matches, len(ctrl_sub))

            if Z_ctrl_all is not None and avail_match:
                z_b = ((burned[avail_match].iloc[i].to_numpy(dtype=float) - mu) / sd)[None, :]
                z_c = Z_ctrl_all[cand_mask]
                dists = cdist(z_b, z_c, metric="euclidean")[0]
                nn_idx = np.argsort(dists)[:k_eff]
            else:
                nn_idx = np.arange(k_eff)

            matched_top_ctrl[i] = float(ctrl_sub[topsoil_col].iloc[nn_idx].mean())
            if has_subsoil:
                matched_sub_ctrl[i] = float(ctrl_sub[subsoil_col].iloc[nn_idx].mean())

        top_diffs = top_b - matched_top_ctrl
        sub_diffs = sub_b - matched_sub_ctrl

        mean_ctrl_top = float(np.mean(matched_top_ctrl))
        mean_burn_top = float(np.mean(top_b))
        delta_top = float(np.mean(top_diffs))
        loss_frac = float(-delta_top / max(mean_ctrl_top, 1e-6))

        # Bootstrap 95% CI on relative loss fraction
        if len(burned) >= 2 and n_boot > 0:
            boot_idx = rng.integers(0, len(burned), size=(n_boot, len(burned)))
            b_deltas = top_diffs[boot_idx].mean(axis=1)
            b_ctrls = np.maximum(matched_top_ctrl[boot_idx].mean(axis=1), 1e-6)
            b_loss = -b_deltas / b_ctrls
            ci_lo, ci_hi = np.quantile(b_loss, [0.025, 0.975])
        else:
            ci_lo, ci_hi = loss_frac, loss_frac

        mean_ctrl_sub = float(np.mean(matched_sub_ctrl)) if has_subsoil else np.nan
        delta_sub = float(np.mean(sub_diffs)) if has_subsoil else np.nan
        sub_loss_frac = float(-delta_sub / max(mean_ctrl_sub, 1e-6)) if has_subsoil else np.nan
        if has_subsoil and len(burned) >= 2 and np.std(sub_diffs) > 1e-12:
            _, p_val_sub = ttest_1samp(sub_diffs, popmean=0.0)
            p_sub = float(p_val_sub)
        else:
            p_sub = 1.0

        rows.append(
            {
                "burn_class": sev,
                "n_burned": int(len(burned)),
                "n_controls": int(len(controls)),
                "topsoil_control_mg_ha": mean_ctrl_top,
                "topsoil_burned_mg_ha": mean_burn_top,
                "topsoil_delta_mg_ha": delta_top,
                "topsoil_loss_frac": loss_frac,
                "loss_frac_ci_low": float(ci_lo),
                "loss_frac_ci_high": float(ci_hi),
                "subsoil_control_mg_ha": mean_ctrl_sub,
                "subsoil_delta_mg_ha": delta_sub,
                "subsoil_loss_frac": sub_loss_frac,
                "subsoil_p_value": p_sub,
            }
        )

    return pd.DataFrame(rows)
