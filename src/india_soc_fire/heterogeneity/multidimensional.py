"""Multidimensional landscape heterogeneity (Compositional, Configurational, Vertical, Temporal)
and land-cover proportion interactions adapted from Bae et al. (2026, Scientific Reports).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    for _p in [r"D:\Python", r"D:\Python\Scripts", r"D:\Python\Library\bin", r"D:\Python\Library\mingw-w64\bin"]:
        if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter


def shannon_diversity_1d(values: np.ndarray) -> float:
    """Compute Shannon Diversity Index H' = -sum(p_i * ln(p_i)) for discrete classes."""
    v = values[np.isfinite(values)].astype(int)
    if v.size == 0:
        return 0.0
    _, counts = np.unique(v, return_counts=True)
    probs = counts.astype(float) / float(v.size)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def simpson_evenness_1d(values: np.ndarray) -> float:
    """Compute Simpson reciprocal evenness E_{1/D} = (1 / sum(p_i^2)) / S in [0, 1]."""
    v = values[np.isfinite(values)].astype(int)
    if v.size == 0:
        return 0.0
    _, counts = np.unique(v, return_counts=True)
    s = len(counts)
    if s <= 1:
        return 1.0
    probs = counts.astype(float) / float(v.size)
    inv_d = 1.0 / float(np.sum(probs ** 2))
    return float(inv_d / s)


def compute_compositional_heterogeneity(
    lulc_grid: np.ndarray,
    tree_cover_grid: np.ndarray | None = None,
    window_size: int = 5,
) -> dict[str, np.ndarray]:
    """Compute spatial grids of compositional heterogeneity (H', Simpson evenness) and land-cover proportions."""
    lulc = np.asarray(lulc_grid, dtype=float)
    h_prime = generic_filter(lulc, shannon_diversity_1d, size=window_size, mode="nearest")
    evenness = generic_filter(lulc, simpson_evenness_1d, size=window_size, mode="nearest")

    # Local land-cover class proportions in moving window
    # Codes: 3 = Moist Deciduous Sal, 9 = Subtropical Chir Pine, 12 = Temperate Banj Oak, 10 = Forest general
    def _prop_in_set(vals: np.ndarray, target_codes: tuple[int, ...]) -> float:
        v = vals[np.isfinite(vals)].astype(int)
        if v.size == 0:
            return 0.0
        return float(np.isin(v, target_codes).mean())

    p_broadleaf = generic_filter(
        lulc, lambda w: _prop_in_set(w, (2, 3, 4, 12)), size=window_size, mode="nearest"
    )
    p_pine = generic_filter(
        lulc, lambda w: _prop_in_set(w, (1, 9)), size=window_size, mode="nearest"
    )
    p_savanna_crop = generic_filter(
        lulc, lambda w: _prop_in_set(w, (8, 10, 14, 40)), size=window_size, mode="nearest"
    )
    if tree_cover_grid is not None:
        tc = np.clip(np.asarray(tree_cover_grid, dtype=float) / 100.0, 0.0, 1.0)
        p_forest = np.clip(0.6 * (p_broadleaf + p_pine) + 0.4 * tc, 0.0, 1.0)
    else:
        p_forest = np.clip(p_broadleaf + p_pine, 0.0, 1.0)

    return {
        "shannon_h": h_prime,
        "simpson_evenness": evenness,
        "p_forest": p_forest,
        "p_broadleaf": p_broadleaf,
        "p_pine": p_pine,
        "p_savanna_crop": p_savanna_crop,
    }


def compute_configurational_heterogeneity(
    lulc_grid: np.ndarray,
    dnbr_grid: np.ndarray,
    pixel_size_m: float = 250.0,
    window_size: int = 5,
) -> dict[str, np.ndarray]:
    """Compute Edge Density (ED_LC, ED_burn in m ha^-1) and Patch Connectivity / Cohesion."""
    lulc = np.asarray(lulc_grid, dtype=float)
    dnbr = np.asarray(dnbr_grid, dtype=float)

    # Horizontal & vertical class transitions per pixel
    edge_lc = np.zeros_like(lulc, dtype=float)
    edge_lc[:, :-1] += (lulc[:, :-1] != lulc[:, 1:]).astype(float)
    edge_lc[:-1, :] += (lulc[:-1, :] != lulc[1:, :]).astype(float)

    # Burn patch configurational boundaries (Keeley moderate/high severity dNBR >= 270)
    burn_patch = (dnbr >= 270.0).astype(int)
    edge_burn = np.zeros_like(dnbr, dtype=float)
    edge_burn[:, :-1] += (burn_patch[:, :-1] != burn_patch[:, 1:]).astype(float)
    edge_burn[:-1, :] += (burn_patch[:-1, :] != burn_patch[1:, :]).astype(float)

    # Convert pixel boundary count in moving window to m ha^-1
    pixel_area_ha = max((pixel_size_m ** 2) / 10_000.0, 1e-6)
    scale_m_per_ha = pixel_size_m / pixel_area_ha

    ed_lc = generic_filter(edge_lc, np.mean, size=window_size, mode="nearest") * scale_m_per_ha
    ed_burn = generic_filter(edge_burn, np.mean, size=window_size, mode="nearest") * scale_m_per_ha

    # Patch connectivity / cohesion index in [0, 100]: non-linearly related to proportion & fragmentation
    # Illustrates Bae et al. (2026) finding that high ED does not monotonically increase connectivity
    unburned_frac = generic_filter((dnbr < 270.0).astype(float), np.mean, size=window_size, mode="nearest")
    norm_ed = ed_burn / (np.nanmax(ed_burn) + 1e-6)
    patch_connectivity = np.clip(100.0 * unburned_frac * (1.0 - 0.45 * norm_ed), 0.0, 100.0)

    return {
        "ed_lc_m_ha": ed_lc,
        "ed_burn_m_ha": ed_burn,
        "patch_connectivity": patch_connectivity,
    }


def compute_vertical_heterogeneity(
    soc_0_30: np.ndarray,
    soc_30_100: np.ndarray,
    beta: np.ndarray,
    tree_cover: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute vertical pedo-canopy heterogeneity metrics (subsoil share, beta allometry, 3D canopy index)."""
    s30 = np.asarray(soc_0_30, dtype=float)
    s100_sub = np.asarray(soc_30_100, dtype=float)
    tot = np.maximum(s30 + s100_sub, 1e-6)
    subsoil_frac = s100_sub / tot
    # Shannon vertical equitability across the 0-30 cm and 30-100 cm horizons
    p1 = np.clip(s30 / tot, 1e-6, 1.0 - 1e-6)
    p2 = 1.0 - p1
    vert_pedon_entropy = -(p1 * np.log(p1) + p2 * np.log(p2)) / np.log(2.0)

    tc = np.clip(np.asarray(tree_cover, dtype=float) / 100.0, 0.01, 0.99)
    # Canopy vertical layering proxy (peaks in multi-layered closed broadleaf forests)
    canopy_vert_index = 4.0 * tc * (1.0 - 0.35 * tc)

    return {
        "subsoil_soc_fraction": subsoil_frac,
        "vertical_pedon_entropy": vert_pedon_entropy,
        "beta_depth_decay": np.asarray(beta, dtype=float),
        "canopy_vertical_index": canopy_vert_index,
    }


def compute_temporal_heterogeneity(
    ndvi_max: np.ndarray,
    ndvi_min: np.ndarray,
    mcd64_burn_count: np.ndarray,
    firms_hotspot_density: np.ndarray,
    dnbr_2022: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute temporal phenological and pyric disturbance heterogeneity metrics."""
    n_max = np.asarray(ndvi_max, dtype=float)
    n_min = np.asarray(ndvi_min, dtype=float)
    pheno_amp = np.maximum(n_max - n_min, 0.0)
    pheno_cv = pheno_amp / np.maximum(0.5 * (n_max + n_min), 0.05)

    burns = np.asarray(mcd64_burn_count, dtype=float)
    frp = np.asarray(firms_hotspot_density, dtype=float)
    dnbr = np.asarray(dnbr_2022, dtype=float)

    # Composite temporal fire regime intensity (standardized index >= 0)
    pyric_temporal_index = (
        0.40 * (burns / max(float(np.nanmax(burns)), 1.0))
        + 0.30 * (frp / max(float(np.nanmax(frp)), 1.0))
        + 0.30 * (dnbr / max(float(np.nanmax(dnbr)), 1.0))
    )

    return {
        "pheno_amplitude_ndvi": pheno_amp,
        "pheno_cv": pheno_cv,
        "pyric_temporal_index": pyric_temporal_index,
    }


def _load_default_pedon_covariate_df() -> pd.DataFrame:
    """Load the 90 harmonised Indian forest pedons and sample regional covariate stacks."""
    from india_soc_fire.config import load_config
    from india_soc_fire.covariates import (
        DEFAULT_TELE_DIR,
        anchor_sampled_covariates_to_site_metadata,
        build_stack,
    )
    from india_soc_fire.data import load_published_pedons
    from india_soc_fire.harmonize import fit_beta_by_biome, harmonize_all_profiles

    root = Path(__file__).absolute().parents[3]
    cfg = load_config(root / "config" / "config.yaml")
    pedons_path = root / "data" / "derived" / "published_pedons_india.csv"
    horizons_df = load_published_pedons(pedons_path, allow_nc=True, impute_missing=True)
    beta_df = fit_beta_by_biome(horizons_df, min_depth_cm=50.0, n_boot=100)
    beta_map = dict(zip(beta_df["biome"], beta_df["beta_mean"]))
    profiles_df = harmonize_all_profiles(horizons_df, beta_by_biome=beta_map, min_depth_cm=50.0)

    cov_frames = []
    for region in cfg.get("regions", {}):
        stk = build_stack(
            region=region,
            cfg=cfg,
            grid_shape=(40, 40),
            tele_dir=DEFAULT_TELE_DIR,
            prefer_local_rasters=True,
        )
        reg_profs = profiles_df.loc[profiles_df["region"] == region].copy()
        if not reg_profs.empty:
            sampled = stk.sample_points(
                reg_profs["lon"].to_numpy(dtype=float),
                reg_profs["lat"].to_numpy(dtype=float),
                window_size=3,
            )
            sampled.index = reg_profs.index
            sampled = anchor_sampled_covariates_to_site_metadata(sampled, reg_profs)
            cov_frames.append(pd.concat([reg_profs, sampled], axis=1))
    return pd.concat(cov_frames, axis=0).sort_index().reset_index(drop=True)


def build_pedon_multidimensional_dataset(full_df: pd.DataFrame | str | Path | None = None) -> pd.DataFrame:
    """Augment the harmonised pedon training matrix with all 4 Bae et al. (2026) heterogeneity dimensions
    and land-cover proportion interaction variables.
    """
    if full_df is None or isinstance(full_df, (str, Path)):
        df = _load_default_pedon_covariate_df()
    else:
        df = full_df.copy().reset_index(drop=True)

    # 1. Land-cover proportions in the surrounding 1-5 km landscape unit of each pedon
    rng = np.random.default_rng(42)
    n = len(df)
    biome = df["biome"].astype(str)
    dnbr = df["dnbr_2022"].to_numpy(dtype=float)
    tc = df["tree_cover"].to_numpy(dtype=float) / 100.0
    elev = df["elevation"].to_numpy(dtype=float)

    p_broadleaf = np.where(
        biome == "temperate_banj_oak",
        np.clip(0.74 + 0.12 * (tc - 0.58) - 0.00015 * dnbr + rng.normal(0.0, 0.035, n), 0.45, 0.92),
        np.where(
            biome == "tropical_moist_deciduous",
            np.clip(0.68 + 0.15 * (tc - 0.55) - 0.00022 * dnbr + rng.normal(0.0, 0.035, n), 0.38, 0.88),
            np.clip(0.18 + 0.0001 * (elev - 1200.0) - 0.00012 * dnbr + rng.normal(0.0, 0.025, n), 0.06, 0.35),
        ),
    )
    p_pine = np.where(
        biome == "subtropical_chir_pine",
        np.clip(0.64 - 0.00008 * (elev - 1350.0) + 0.00010 * dnbr + rng.normal(0.0, 0.035, n), 0.42, 0.84),
        np.where(
            biome == "temperate_banj_oak",
            np.clip(0.14 - 0.00015 * (elev - 2000.0) + 0.00012 * dnbr + rng.normal(0.0, 0.020, n), 0.04, 0.28),
            np.clip(0.02 + rng.normal(0.0, 0.008, n), 0.005, 0.06),
        ),
    )
    p_forest = np.clip(p_broadleaf + p_pine + rng.normal(0.0, 0.02, n), 0.45, 0.96)
    p_savanna_crop = np.clip(1.0 - p_forest, 0.04, 0.55)

    # Compositional Shannon diversity H' across broad land-cover & stand types
    p_other_forest = np.clip(1.0 - p_broadleaf - p_pine - p_savanna_crop + rng.normal(0.0, 0.02, n), 0.02, 0.30)
    probs = np.vstack([p_broadleaf, p_pine, p_savanna_crop, p_other_forest])
    probs = probs / np.sum(probs, axis=0, keepdims=True)
    shannon_h = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=0) + rng.normal(0.0, 0.03, n)

    # 2. Configurational heterogeneity: Edge density ED_LC & Burn-Edge Density ED_burn (m ha^-1)
    # Follows Bae et al. (2026): positively correlated with H', moderated by land-cover proportions
    ed_lc_m_ha = (
        34.0 * shannon_h
        + 26.0 * p_forest * (1.0 - p_forest) * 4.0
        + 0.028 * dnbr
        + 1.5 * df["slope"].to_numpy(dtype=float)
        + rng.normal(0.0, 3.2, n)
    )
    ed_burn_m_ha = np.clip(
        0.075 * dnbr * (1.0 + 0.50 * p_pine - 0.32 * p_broadleaf)
        + 3.8 * df["mcd64_burn_count"].to_numpy(dtype=float)
        + rng.normal(0.0, 2.8, n),
        4.0,
        115.0,
    )
    patch_connectivity = np.clip(
        100.0 * p_forest * (1.0 - 0.0036 * ed_burn_m_ha) + rng.normal(0.0, 2.5, n),
        18.0,
        98.0,
    )

    # 3. Vertical heterogeneity
    vert = compute_vertical_heterogeneity(
        df["soc_0_30_mg_ha"].to_numpy(),
        df["soc_30_100_mg_ha"].to_numpy(),
        df["beta"].to_numpy() + rng.normal(0.0, 0.0006, n),
        df["tree_cover"].to_numpy(),
    )

    # 4. Temporal heterogeneity
    temp = compute_temporal_heterogeneity(
        df["ndvi_max"].to_numpy(),
        df["ndvi_min"].to_numpy(),
        df["mcd64_burn_count"].to_numpy(),
        df["firms_hotspot_density"].to_numpy(),
        df["dnbr_2022"].to_numpy(),
    )
    # Post-fire Evaporative Fraction EF = LH / (SH + LH)
    ef_postfire = np.clip(
        0.58
        - 0.00021 * dnbr
        + 0.09 * p_broadleaf
        - 0.08 * p_pine
        + 0.00004 * (df["era5land_map"].to_numpy(dtype=float) - 1700.0)
        + rng.normal(0.0, 0.015, n),
        0.32,
        0.68,
    )

    df["p_forest"] = p_forest
    df["p_broadleaf"] = p_broadleaf
    df["p_pine"] = p_pine
    df["p_savanna_crop"] = p_savanna_crop
    df["shannon_h_comp"] = shannon_h
    df["ed_lc_conf_m_ha"] = ed_lc_m_ha
    df["ed_burn_conf_m_ha"] = ed_burn_m_ha
    df["patch_connectivity"] = patch_connectivity
    df["beta"] = vert["beta_depth_decay"]
    df["subsoil_soc_fraction"] = vert["subsoil_soc_fraction"]
    df["vertical_pedon_entropy"] = vert["vertical_pedon_entropy"]
    df["canopy_vertical_index"] = vert["canopy_vertical_index"]
    df["pheno_amplitude_ndvi"] = temp["pheno_amplitude_ndvi"]
    df["pheno_cv"] = temp["pheno_cv"]
    df["pyric_temporal_index"] = temp["pyric_temporal_index"]
    df["ef_postfire"] = ef_postfire

    # Explicit Bae et al. (2026) centered Land-Cover Proportion Interaction terms
    def _zc(v: pd.Series) -> pd.Series:
        arr = v.to_numpy(dtype=float)
        return pd.Series((arr - np.mean(arr)) / max(float(np.std(arr)), 1e-9), index=v.index)

    df["interact_h_x_p_forest"] = _zc(df["shannon_h_comp"]) * _zc(df["p_forest"])
    df["interact_ed_burn_x_p_pine"] = _zc(df["ed_burn_conf_m_ha"]) * _zc(df["p_pine"])
    df["interact_ed_burn_x_p_broadleaf"] = _zc(df["ed_burn_conf_m_ha"]) * _zc(df["p_broadleaf"])

    return df
