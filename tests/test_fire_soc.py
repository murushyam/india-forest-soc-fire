"""Tests for Phase 6 (burn severity indices, paired combustion loss, subsoil falsification, recovery, and CLI)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from india_soc_fire.cli import run_full_pipeline
from india_soc_fire.fire import (
    apply_phenology_offset,
    classify_burn_severity,
    compute_dnbr,
    compute_rbr,
    compute_rdnbr,
    estimate_paired_combustion_loss,
    fit_postfire_recovery_trajectory,
    recovery_curve_model,
    simulate_biome_recovery_chronosequence,
)


def test_severity_indices_and_phenology_offset() -> None:
    dnbr = compute_dnbr(0.55, 0.15)
    assert dnbr == pytest.approx(400.0)
    rdnbr = compute_rdnbr(0.55, dnbr)
    rbr = compute_rbr(0.55, dnbr)
    assert float(rdnbr) > 400.0
    assert 200.0 < float(rbr) < 400.0

    arr = np.array([20.0, 35.0, 240.0, 480.0, 720.0])
    unburned_mask = np.array([True, True, False, False, False])
    corr, offset = apply_phenology_offset(arr, unburned_mask)
    assert offset == pytest.approx(27.5)
    classes = classify_burn_severity(corr, biome="tropical_moist_deciduous")
    assert classes[0] == "unburned"
    assert classes[-1] == "high"


def test_paired_combustion_loss_and_subsoil_falsification() -> None:
    rng = np.random.default_rng(123)
    # Create synthetic landscape with topographic confounding + true topsoil fire loss, zero subsoil loss
    n_ctrl = 20
    n_burn = 15
    elev_ctrl = rng.uniform(500, 1500, size=n_ctrl)
    elev_burn = rng.uniform(600, 1400, size=n_burn)

    df = pd.DataFrame(
        {
            "biome": ["subtropical_chir_pine"] * (n_ctrl + n_burn),
            "burn_class": ["unburned"] * n_ctrl + ["high"] * n_burn,
            "elevation": np.concatenate([elev_ctrl, elev_burn]),
            "slope": rng.uniform(5, 25, size=n_ctrl + n_burn),
            "aspect": rng.uniform(0, 360, size=n_ctrl + n_burn),
            "twi": rng.uniform(5, 10, size=n_ctrl + n_burn),
            "era5land_mat": 20.0 - 0.005 * np.concatenate([elev_ctrl, elev_burn]),
            "era5land_map": np.full(n_ctrl + n_burn, 1500.0),
        }
    )
    # True baseline SOC increases with elevation; high fire removes 15 Mg C/ha in topsoil (0-30 cm)
    # but 0 Mg C/ha in subsoil (30-100 cm)
    is_burn = (df["burn_class"] == "high").to_numpy(dtype=float)
    df["soc_0_30_mg_ha"] = 45.0 + 0.015 * df["elevation"] - 15.0 * is_burn + rng.normal(0, 1.0, len(df))
    df["soc_30_100_mg_ha"] = 50.0 + 0.012 * df["elevation"] + rng.normal(0, 1.0, len(df))

    loss_df = estimate_paired_combustion_loss(df, k_matches=3, n_boot=100)
    assert len(loss_df) == 1
    row = loss_df.iloc[0]
    assert row["burn_class"] == "high"
    # Topsoil delta should be close to -15 Mg C/ha
    assert -18.5 <= float(row["topsoil_delta_mg_ha"]) <= -11.5
    # Subsoil falsification test: delta near 0 and non-significant p-value (> 0.05)
    assert abs(float(row["subsoil_delta_mg_ha"])) < 3.0
    assert float(row["subsoil_p_value"]) > 0.05


def test_recovery_trajectory_fitting() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    r = np.asarray(recovery_curve_model(t, initial_loss_frac=0.30, k_rate=0.40, repeat_burn_penalty=0.0, n_burns=1))
    fit = fit_postfire_recovery_trajectory(t, r)
    assert fit["initial_loss_frac"] == pytest.approx(0.30, rel=1e-2)
    assert fit["k_rate_yr"] == pytest.approx(0.40, rel=1e-2)
    assert fit["half_life_yr"] == pytest.approx(np.log(2.0) / 0.40, rel=1e-2)

    chrono = simulate_biome_recovery_chronosequence()
    assert len(chrono) == 3 * 4 * 6  # 3 biomes * 4 severity classes * 6 years (0..5)


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    res = run_full_pipeline(
        config_path=root / "config" / "config.yaml",
        pedons_path=root / "data" / "derived" / "published_pedons_india.csv",
        output_dir=tmp_path / "out",
        grid_shape=(12, 12),
        n_boot=4,
        generate_figures=False,
    )
    assert res["n_profiles"] == 24
    assert (tmp_path / "out" / "phase2_biome_betas.csv").exists()
    assert (tmp_path / "out" / "phase4_cv_comparison_0_30cm.csv").exists()
    assert (tmp_path / "out" / "phase6_fire_combustion_loss.csv").exists()
