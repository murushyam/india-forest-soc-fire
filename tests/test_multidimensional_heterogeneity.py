"""Unit and empirical tests for the Bae et al. (2026) Multidimensional Landscape Heterogeneity,
Land-Cover Proportion Interaction, Piecewise SEM, Double-Negative-Control DML, and Pan-India/Global
Scaling modules.
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

from india_soc_fire.heterogeneity import (
    build_pedon_multidimensional_dataset,
    compute_compositional_heterogeneity,
    compute_configurational_heterogeneity,
    compute_temporal_heterogeneity,
    compute_vertical_heterogeneity,
    evaluate_2d_interaction_surface,
    fit_piecewise_sem_and_interactions,
    run_multidimensional_pca,
    shannon_diversity_1d,
    simpson_evenness_1d,
)
from india_soc_fire.scaling import (
    PAN_INDIA_CHAMPION_SETH_BIOMES,
    conformalize_spatial_intervals,
    estimate_double_negative_control_dml,
    generate_pan_india_and_global_tiles,
    synthesize_pan_india_sentinel_landscapes,
)

ROOT = Path(__file__).absolute().parents[1]
TELE_DIR = ROOT.parent


def test_shannon_and_simpson_diversity() -> None:
    equal_classes = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
    h = shannon_diversity_1d(equal_classes)
    j = simpson_evenness_1d(equal_classes)
    assert np.isclose(h, np.log(4.0), atol=1e-6)
    assert np.isclose(j, 1.0, atol=1e-6)

    mono_classes = np.array([3, 3, 3, 3], dtype=float)
    assert np.isclose(shannon_diversity_1d(mono_classes), 0.0, atol=1e-6)
    assert np.isclose(simpson_evenness_1d(mono_classes), 1.0, atol=1e-6)


def test_four_dimensional_heterogeneity_primitives() -> None:
    lulc = np.array([[3, 3, 9, 9], [3, 12, 9, 8], [12, 12, 8, 8], [3, 12, 9, 8]], dtype=float)
    tc = np.full_like(lulc, 65.0)
    comp = compute_compositional_heterogeneity(lulc, tree_cover_grid=tc, window_size=3)
    assert comp["shannon_h"].shape == lulc.shape
    assert float(np.mean(comp["shannon_h"])) > 0.2
    assert float(np.mean(comp["p_forest"])) > 0.2

    dnbr = np.array([[100, 320, 350, 120], [90, 290, 410, 150], [80, 110, 280, 95], [70, 85, 130, 60]], dtype=float)
    conf = compute_configurational_heterogeneity(lulc, dnbr_grid=dnbr, pixel_size_m=250.0, window_size=3)
    assert float(np.mean(conf["ed_lc_m_ha"])) > 0.0
    assert float(np.mean(conf["ed_burn_m_ha"])) > 0.0
    assert np.all((conf["patch_connectivity"] >= 0.0) & (conf["patch_connectivity"] <= 100.0))

    vert = compute_vertical_heterogeneity(
        np.array([58.6, 84.9]),
        np.array([62.1, 73.5]),
        np.array([0.961, 0.946]),
        np.array([72.0, 68.0]),
    )
    assert np.isclose(vert["subsoil_soc_fraction"][0], 62.1 / (58.6 + 62.1), atol=1e-5)
    assert np.all(vert["vertical_pedon_entropy"] > 0.9)

    temp = compute_temporal_heterogeneity(
        ndvi_max=np.array([0.78, 0.82]),
        ndvi_min=np.array([0.38, 0.44]),
        mcd64_burn_count=np.array([2.0, 4.0]),
        firms_hotspot_density=np.array([1.2, 2.5]),
        dnbr_2022=np.array([180.0, 340.0]),
    )
    assert np.all(temp["pheno_amplitude_ndvi"] > 0.3)
    assert temp["pyric_temporal_index"][1] > temp["pyric_temporal_index"][0]


def test_pedon_multidimensional_pca_and_sem() -> None:
    df = build_pedon_multidimensional_dataset()
    assert len(df) == 24
    required_cols = {
        "shannon_h_comp",
        "ed_burn_conf_m_ha",
        "beta",
        "subsoil_soc_fraction",
        "pyric_temporal_index",
        "ef_postfire",
        "p_forest",
        "p_broadleaf",
        "p_pine",
        "interact_h_x_p_forest",
        "interact_ed_burn_x_p_pine",
        "interact_ed_burn_x_p_broadleaf",
    }
    assert required_cols.issubset(set(df.columns))

    loadings_df, scores_df, pca_summary = run_multidimensional_pca(df, n_components=4)
    assert len(loadings_df) == 11
    assert len(scores_df) == 24
    assert sum(pca_summary["varimax_variance_pct"]) > 65.0

    sem_df, sem_meta = fit_piecewise_sem_and_interactions(df)
    assert len(sem_df) >= 20
    assert sem_meta["fisher_p_value"] > 0.05

    surf_res = evaluate_2d_interaction_surface(df, grid_size=20)
    assert surf_res["soc_0_30_surface"].shape == (20, 20)


def test_pan_india_global_scaling_and_dml_csqr() -> None:
    assert len(PAN_INDIA_CHAMPION_SETH_BIOMES) == 16
    sentinel_df = synthesize_pan_india_sentinel_landscapes(TELE_DIR)
    assert len(sentinel_df) == 7

    df = build_pedon_multidimensional_dataset()
    dml_res = estimate_double_negative_control_dml(df)
    assert dml_res["dml_topsoil_pct_per_fire_yr"] < 0.0
    assert dml_res["nco_subsoil_p_value"] > 0.10

    y = df["soc_0_30_mg_ha"].to_numpy(dtype=float)
    csqr = conformalize_spatial_intervals(y, y * 0.85, y * 1.15, alpha=0.10)
    assert csqr["conformal_picp_all"] >= 0.88

    ind_tiles = generate_pan_india_and_global_tiles("pan_india")
    glob_tiles = generate_pan_india_and_global_tiles("global")
    assert len(ind_tiles) > 30
    assert len(glob_tiles) == 60
