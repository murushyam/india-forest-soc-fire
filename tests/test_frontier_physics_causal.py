"""Unit and physical conservation tests for the Frontier Physics-Informed PDE,
Fourier Thermal Skin-Depth, POM/MAOM/PyC Pool Fractionation, Post-Fire Monsoon
Erosion Routing, Proximal DML-NCO Heterogeneous CATE, and CMIP6 Pyrogenic
Tipping-Point Index (Pi_tip) module.
"""
from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path

if sys.platform == "win32":
    for _p in [r"D:\Python", r"D:\Python\Scripts", r"D:\Python\Library\bin", r"D:\Python\Library\mingw-w64\bin"]:
        if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
    pathlib._WindowsFlavour.resolve = lambda self, path, strict=False: os.path.abspath(str(path))

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from india_soc_fire.advanced.frontier_physics_causal import (
    compute_fourier_thermal_skin_profile,
    compute_postfire_monsoon_erosion_routing,
    estimate_heterogeneous_cate_dml_nco,
    evaluate_cmip6_tipping_point_index,
    partition_soc_into_pom_maom_pyc,
    solve_continuous_advection_diffusion_soc_profile,
)
from india_soc_fire.heterogeneity import build_pedon_multidimensional_dataset
from india_soc_fire.metrics import (
    compute_bounded_conformal_intervals,
    compute_scale_normalized_linex_loss,
    project_discrete_count_support,
    project_non_negative_support,
    wilcoxon_holm_bonferroni,
)
from india_soc_fire.scaling import generate_global_spatial_grid, generate_pan_india_spatial_grid


def test_fourier_thermal_skin_depth_insulates_subsoil() -> None:
    z = np.linspace(0.0, 50.0, 501)
    res = compute_fourier_thermal_skin_profile(z, t_surface_c=550.0, t_ambient_c=25.0, duration_min=45.0)
    assert 4.0 < res["z_skin_cm"] < 6.5
    # At 30 cm depth (subsoil boundary), temperature must remain within 0.05 C of ambient
    idx_30 = int(np.argmin(np.abs(z - 30.0)))
    assert abs(float(res["temp_profile_c"][idx_30]) - 25.0) < 0.05
    # Pyrolysis loss fraction at 30 cm must be negligible (< 0.01%)
    assert float(res["pyrolysis_loss_frac"][idx_30]) < 1e-4


def test_continuous_advection_diffusion_pde_conservation() -> None:
    pde = solve_continuous_advection_diffusion_soc_profile(
        biome="tropical_moist_deciduous",
        map_mm=1890.0,
        mat_c=24.5,
        fire_freq_per_decade=2.5,
        target_total_stock_mg_ha=123.92,
    )
    assert np.isclose(pde["soc_0_30_mg_ha"] + pde["soc_30_100_mg_ha"], 123.92, atol=1e-3)
    assert 0.35 < pde["subsoil_fraction"] < 0.65
    assert np.all(pde["rho_c_mg_ha_cm"] > 0.0)


def test_pom_maom_pyc_mass_additivity_and_erosion_routing() -> None:
    df_het = build_pedon_multidimensional_dataset()
    df_pools = partition_soc_into_pom_maom_pyc(df_het)

    top_sum = df_pools["pom_0_30_mg_ha"] + df_pools["maom_0_30_mg_ha"] + df_pools["pyc_0_30_mg_ha"]
    sub_sum = df_pools["pom_30_100_mg_ha"] + df_pools["maom_30_100_mg_ha"] + df_pools["pyc_30_100_mg_ha"]
    assert np.allclose(top_sum.to_numpy(), df_pools["soc_0_30_mg_ha"].to_numpy(), atol=1e-5)
    assert np.allclose(sub_sum.to_numpy(), df_pools["soc_30_100_mg_ha"].to_numpy(), atol=1e-5)

    df_erosion = compute_postfire_monsoon_erosion_routing(df_pools)
    loss_sum = df_erosion["atm_volatilization_mg_ha"] + df_erosion["lateral_monsoon_erosion_mg_ha"]
    assert np.allclose(loss_sum.to_numpy(), df_erosion["gross_postfire_loss_mg_ha"].to_numpy(), atol=1e-5)

    df_cate = estimate_heterogeneous_cate_dml_nco(df_erosion)
    assert (df_cate["cate_dml_nco_pct_per_burn"] < 0.0).all()


def test_cmip6_tipping_point_index_pan_india_and_global() -> None:
    df_tip = evaluate_cmip6_tipping_point_index()
    assert len(df_tip) == 20  # 16 Champion & Seth Indian groups + 4 Global WWF benchmarks
    assert (df_tip["pi_tip_ssp585"] > df_tip["pi_tip_hist"]).all()
    # Subtropical pine (9/C1) must exceed tipping threshold Pi_tip > 1.0
    pine_row = df_tip[df_tip["group_code"] == "9/C1"].iloc[0]
    assert float(pine_row["pi_tip_hist"]) > 1.0
    assert float(pine_row["pi_tip_ssp585"]) > 2.0


def test_pan_india_and_global_spatial_grids() -> None:
    india_grid = generate_pan_india_spatial_grid(res_deg=0.5, seed=42)
    assert india_grid["soc_0_30"].shape == india_grid["lon"].shape
    assert np.nanmean(india_grid["soc_30_100"]) > 30.0
    assert np.nanmean(india_grid["pyc_30_100"]) > 2.5
    assert np.nanmax(india_grid["pi_tip_ssp585"]) > 1.5

    glob_grid = generate_global_spatial_grid(res_deg=2.0, seed=108)
    assert glob_grid["soc_0_30"].shape == glob_grid["lon"].shape
    assert np.nanmean(glob_grid["pi_tip_ssp585"]) > np.nanmean(glob_grid["pi_tip_hist"])


def test_domain_constrained_metrics_and_linex_loss() -> None:
    y_raw = np.array([-3.2, 0.0, 14.7, 28.4])
    assert np.all(project_non_negative_support(y_raw) >= 0.0)
    counts = project_discrete_count_support(np.array([-1.2, 0.4, 2.6, 5.1]))
    assert counts.tolist() == [0, 0, 3, 5]

    y_true = np.array([45.0, 62.0, 78.0, 91.0, 54.0])
    y_under = y_true - 8.0
    y_over = y_true + 8.0
    # Under-prediction must incur strictly higher LinEx penalty than over-prediction (b > 0)
    loss_under = compute_scale_normalized_linex_loss(y_true, y_under, b=1.2)
    loss_over = compute_scale_normalized_linex_loss(y_true, y_over, b=1.2)
    assert loss_under > loss_over > 0.0

    conf = compute_bounded_conformal_intervals(
        cal_residuals=np.array([2.0, 3.5, 4.0, 5.0, 6.5, 7.0, 8.0, 9.0, 10.0]),
        y_pred_lower=y_true - 4.0,
        y_pred_upper=y_true + 4.0,
        y_test=y_true,
        alpha=0.10,
    )
    assert conf["picp"] == 1.0
    assert conf["pinaw"] > 0.0

    w_res = wilcoxon_holm_bonferroni([np.full(12, 9.5)], [np.full(12, 4.2)])
    assert w_res[0]["significant_005"] is True
