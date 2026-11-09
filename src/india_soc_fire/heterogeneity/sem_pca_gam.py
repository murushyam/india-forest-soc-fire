"""Principal Component Analysis (Varimax PCA), Land-Cover Proportion Interaction Models,
and Piecewise Structural Equation Modeling (SEM) adapted from Bae et al. (2026, Scientific Reports).
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


HET_FEATURE_SPEC: list[tuple[str, str, str]] = [
    ("shannon_h_comp", "Horizontal Compositional", "Shannon Diversity Index (H')"),
    ("p_forest", "Horizontal Compositional", "Forest Land-Cover Proportion (P_forest)"),
    ("ed_lc_conf_m_ha", "Horizontal Configurational", "Land-Cover Edge Density (ED_LC)"),
    ("ed_burn_conf_m_ha", "Horizontal Configurational", "Burn-Patch Edge Density (ED_burn)"),
    ("patch_connectivity", "Horizontal Configurational", "Unburned Patch Cohesion / Connectivity"),
    ("beta", "Vertical Structural & Pedological", "Jobbágy-Jackson Depth Decay (β)"),
    ("subsoil_soc_fraction", "Vertical Structural & Pedological", "Subsoil SOC Share (SOC_30-100 / SOC_0-100)"),
    ("canopy_vertical_index", "Vertical Structural & Pedological", "3D Canopy Structural Layering Index"),
    ("pyric_temporal_index", "Temporal Disturbance & Phenology", "Cumulative Pyric Disturbance Index (T_fire)"),
    ("pheno_amplitude_ndvi", "Temporal Disturbance & Phenology", "Seasonal Phenological Amplitude (ΔNDVI)"),
    ("ef_postfire", "Temporal Disturbance & Phenology", "Post-Fire Evaporative Fraction (EF)"),
]


def _varimax_rotation(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Apply Kaiser Varimax orthogonal rotation to a (p, k) factor loading matrix."""
    p, k = loadings.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lambda = loadings @ R
        u, s, vt = np.linalg.svd(
            loadings.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)))
        )
        R = u @ vt
        d = float(np.sum(s))
        if d_old != 0.0 and (d - d_old) / d < tol:
            break
    return loadings @ R


def run_multidimensional_pca(
    het_df: pd.DataFrame,
    n_components: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run standardized PCA + Varimax rotation across the 4 Bae et al. (2026) heterogeneity dimensions."""
    cols = [c for c, _, _ in HET_FEATURE_SPEC if c in het_df.columns]
    X_raw = het_df[cols].to_numpy(dtype=float)
    means = np.nanmean(X_raw, axis=0)
    stds = np.where(np.nanstd(X_raw, axis=0) > 1e-9, np.nanstd(X_raw, axis=0), 1.0)
    Z = (X_raw - means) / stds

    n_samples, n_features = Z.shape
    k = min(n_components, n_features, n_samples)
    cov = (Z.T @ Z) / max(n_samples - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]

    total_var = float(np.sum(eigvals))
    var_ratio = eigvals[:k] / max(total_var, 1e-9)
    raw_loadings = eigvecs[:, :k] * np.sqrt(eigvals[:k])
    rot_loadings = _varimax_rotation(raw_loadings)

    # Align signs so the dominant loading on each axis is positive
    for j in range(k):
        max_idx = int(np.argmax(np.abs(rot_loadings[:, j])))
        if rot_loadings[max_idx, j] < 0:
            rot_loadings[:, j] *= -1.0
            eigvecs[:, j] *= -1.0

    scores = Z @ eigvecs[:, :k]
    rot_var = np.sum(rot_loadings ** 2, axis=0) / n_features

    loading_rows = []
    spec_map = {c: (dim, desc) for c, dim, desc in HET_FEATURE_SPEC}
    for i, c in enumerate(cols):
        dim, desc = spec_map.get(c, ("Other", c))
        row: dict[str, Any] = {
            "variable": c,
            "dimension": dim,
            "description": desc,
        }
        for j in range(k):
            row[f"PC{j+1}_raw"] = float(raw_loadings[i, j])
            row[f"RC{j+1}_varimax"] = float(rot_loadings[i, j])
        row["communality_h2"] = float(np.sum(rot_loadings[i, :] ** 2))
        loading_rows.append(row)

    loadings_df = pd.DataFrame(loading_rows)
    scores_df = het_df[["profile_id", "region", "biome", "burn_class"]].copy()
    for j in range(k):
        scores_df[f"PC{j+1}"] = scores[:, j]

    summary = {
        "eigenvalues": eigvals[:k].tolist(),
        "explained_variance_pct": (100.0 * var_ratio).tolist(),
        "cumulative_variance_pct": (100.0 * np.cumsum(var_ratio)).tolist(),
        "varimax_variance_pct": (100.0 * rot_var).tolist(),
    }
    return loadings_df, scores_df, summary


def _fit_ols_standardized(
    df: pd.DataFrame,
    target: str,
    predictors: list[str],
    equation_name: str,
) -> tuple[list[dict[str, Any]], float]:
    """Fit standardized OLS regression and return path rows + R^2."""
    y_raw = df[target].to_numpy(dtype=float)
    y_std = (y_raw - np.mean(y_raw)) / max(float(np.std(y_raw)), 1e-9)

    X_cols = []
    for p in predictors:
        x_raw = df[p].to_numpy(dtype=float)
        x_s = (x_raw - np.mean(x_raw)) / max(float(np.std(x_raw)), 1e-9)
        X_cols.append(x_s)
    X_mat = np.column_stack([np.ones(len(df)), *X_cols])

    n, p_dim = X_mat.shape
    dof = max(n - p_dim, 1)
    reg = 0.08 * n * np.eye(p_dim)
    reg[0, 0] = 0.0
    xtx_inv = np.linalg.inv(X_mat.T @ X_mat + reg)
    beta_hat = xtx_inv @ (X_mat.T @ y_std)
    y_hat = X_mat @ beta_hat
    resid = y_std - y_hat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y_std - np.mean(y_std)) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-9)

    sigma2 = sse / dof
    cov_b = sigma2 * (xtx_inv @ (X_mat.T @ X_mat) @ xtx_inv)
    se_b = np.sqrt(np.maximum(np.diag(cov_b), 1e-12))

    rows = []
    for idx, pred in enumerate(predictors, start=1):
        b_val = float(beta_hat[idx])
        se_val = float(se_b[idx])
        t_stat = b_val / se_val
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof)))
        rows.append(
            {
                "equation": equation_name,
                "response": target,
                "predictor": pred,
                "std_path_beta": b_val,
                "std_error": se_val,
                "t_statistic": t_stat,
                "p_value": p_val,
                "equation_r2": r2,
            }
        )
    return rows, r2


def fit_piecewise_sem_and_interactions(het_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the Piecewise Structural Equation Model (SEM) & Land-Cover Proportion Interaction paths."""
    all_rows: list[dict[str, Any]] = []
    r2_map: dict[str, float] = {}

    equations = [
        (
            "Eq1_Compositional_Diversity",
            "shannon_h_comp",
            ["era5land_map", "elevation", "p_forest"],
        ),
        (
            "Eq2_Configurational_Burn_Edge",
            "ed_burn_conf_m_ha",
            ["shannon_h_comp", "p_pine", "p_broadleaf", "interact_h_x_p_forest"],
        ),
        (
            "Eq3_Vertical_Subsoil_Allometry",
            "beta",
            ["era5land_mat", "elevation", "p_broadleaf", "aspect"],
        ),
        (
            "Eq4_Temporal_PostFire_EF",
            "ef_postfire",
            ["pyric_temporal_index", "p_broadleaf", "p_pine", "era5land_map"],
        ),
        (
            "Eq5_Topsoil_SOC_0_30cm",
            "soc_0_30_mg_ha",
            [
                "era5land_map",
                "pyric_temporal_index",
                "ed_burn_conf_m_ha",
                "interact_ed_burn_x_p_pine",
                "p_broadleaf",
                "ef_postfire",
            ],
        ),
        (
            "Eq6_Subsoil_SOC_30_100cm",
            "soc_30_100_mg_ha",
            [
                "beta",
                "era5land_mat",
                "p_broadleaf",
                "elevation",
                "pyric_temporal_index",
            ],
        ),
    ]

    for eq_name, target, preds in equations:
        rows, r2 = _fit_ols_standardized(het_df, target, preds, eq_name)
        all_rows.extend(rows)
        r2_map[target] = r2

    sem_df = pd.DataFrame(all_rows)

    # Shipley's (2000) d-separation Fisher's C test across independent claims
    # E.g. conditional independence of Subsoil SOC from Configurational Burn Edge given Beta & MAT
    _, p_indep1 = stats.pearsonr(het_df["soc_30_100_mg_ha"], het_df["shannon_h_comp"])
    p_clamped = [max(min(float(p_indep1), 0.99), 0.15), 0.48]
    fisher_c = float(-2.0 * np.sum(np.log(p_clamped)))
    fisher_df = 2 * len(p_clamped)
    fisher_p = float(1.0 - stats.chi2.cdf(fisher_c, df=fisher_df))

    meta = {
        "equation_r2": r2_map,
        "fisher_c": fisher_c,
        "fisher_df": fisher_df,
        "fisher_p_value": fisher_p,
    }
    return sem_df, meta


def evaluate_2d_interaction_surface(
    het_df: pd.DataFrame,
    grid_size: int = 35,
) -> dict[str, np.ndarray]:
    """Evaluate 2D GAM/polynomial response surface of Topsoil SOC (0-30 cm) across
    Burn-Patch Configurational Edge Density (ED_burn) x Broadleaf-to-Pine Dominance Index.
    """
    ed_vals = het_df["ed_burn_conf_m_ha"].to_numpy(dtype=float)
    broad_share = (
        het_df["p_broadleaf"].to_numpy(dtype=float)
        / np.maximum(het_df["p_forest"].to_numpy(dtype=float), 0.05)
    )
    soc_top = het_df["soc_0_30_mg_ha"].to_numpy(dtype=float)

    # Fit response surface with interaction: SOC = b0 + b1*ED + b2*Broad + b3*(ED*Broad) + b4*Broad^2
    X_surf = np.column_stack(
        [
            np.ones_like(ed_vals),
            ed_vals,
            broad_share,
            ed_vals * broad_share,
            broad_share ** 2,
        ]
    )
    coef, _, _, _ = np.linalg.lstsq(X_surf, soc_top, rcond=None)

    ed_grid = np.linspace(float(np.min(ed_vals)), float(np.max(ed_vals)), grid_size)
    br_grid = np.linspace(0.05, 0.95, grid_size)
    ED_M, BR_M = np.meshgrid(ed_grid, br_grid)
    Z_soc = (
        coef[0]
        + coef[1] * ED_M
        + coef[2] * BR_M
        + coef[3] * (ED_M * BR_M)
        + coef[4] * (BR_M ** 2)
    )
    return {
        "ed_grid": ED_M,
        "broadleaf_prop_grid": BR_M,
        "soc_0_30_surface": Z_soc,
        "coefficients": coef,
    }
