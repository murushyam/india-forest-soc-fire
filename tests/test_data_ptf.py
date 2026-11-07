"""Tests for Phase 1 (data loaders, QC, Walkley-Black, PTF, license gate) and Phase 2 (biome beta & spline)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from india_soc_fire.data import (
    apply_walkley_black_correction,
    enforce_licence_gate,
    fit_bd_ptf,
    impute_missing_bd_and_cf,
    load_published_pedons,
    load_wosis,
)
from india_soc_fire.harmonize import (
    equal_area_spline_harmonize,
    fit_beta_by_biome,
    harmonize_all_profiles,
)


PEDONS_CSV = Path(__file__).resolve().parents[1] / "data" / "derived" / "published_pedons_india.csv"


def test_licence_gate_rejects_soilgrids_target() -> None:
    bad_df = pd.DataFrame(
        {
            "profile_id": ["P1"],
            "source": ["ISRIC_SoilGrids2.0_extract"],
            "licence": ["CC-BY-4.0"],
        }
    )
    with pytest.raises(ValueError, match="SoilGrids"):
        enforce_licence_gate(bad_df)


def test_licence_gate_filters_unknown_and_nc() -> None:
    df = pd.DataFrame(
        {
            "profile_id": ["P1", "P2", "P3"],
            "source": ["WoSIS", "PaperA", "PaperB"],
            "licence": ["CC-BY-4.0", "CC-BY-NC-4.0", "UNKNOWN"],
        }
    )
    pub_only = enforce_licence_gate(df, allow_nc=False)
    assert list(pub_only["profile_id"]) == ["P1"]

    with_nc = enforce_licence_gate(df, allow_nc=True)
    assert list(with_nc["profile_id"]) == ["P1", "P2"]
    assert bool(with_nc.loc[with_nc["profile_id"] == "P2", "nc_restricted"].iloc[0]) is True


def test_walkley_black_correction() -> None:
    df = pd.DataFrame(
        {
            "soc_g_kg": [10.0, 20.0],
            "soc_method": ["WB_unheated", "dry_combustion"],
        }
    )
    corr = apply_walkley_black_correction(df, wb_recovery_factor=1.32)
    assert corr.loc[0, "soc_g_kg"] == pytest.approx(13.2)
    assert bool(corr.loc[0, "wb_corrected"]) is True
    assert corr.loc[1, "soc_g_kg"] == pytest.approx(20.0)
    assert bool(corr.loc[1, "wb_corrected"]) is False


def test_load_published_pedons_and_ptf_imputation() -> None:
    df = load_published_pedons(PEDONS_CSV)
    assert len(df) >= 80
    assert df["bd_g_cm3"].isna().sum() == 0
    assert df["coarse_frac"].isna().sum() == 0
    assert (df["stock_mg_ha"] > 0).all()
    assert (df["stock_sd_mg_ha"] > 0).all()
    # SIM_007 and UK_OAK_006 had missing BD in raw CSV -> should be flagged bd_imputed=True
    sim7 = df.loc[df["profile_id"] == "SIM_007"]
    assert sim7["bd_imputed"].all()
    assert sim7["wb_corrected"].all()

    ptf = fit_bd_ptf(df)
    assert 0.12 <= ptf.bd_om <= 0.55
    # Higher SOC at same depth must yield lower bulk density
    bd_low_soc = ptf.predict(5.0, 20.0)
    bd_high_soc = ptf.predict(45.0, 20.0)
    assert bd_high_soc < bd_low_soc


def test_load_wosis_roundtrip(tmp_path: Path) -> None:
    wosis_mock = pd.DataFrame(
        {
            "profile_code": ["W_IND_1", "W_IND_1"],
            "longitude": [86.30, 86.30],
            "latitude": [21.85, 21.85],
            "upper_depth": [0, 20],
            "lower_depth": [20, 60],
            "orgc_value_avg": [22.5, 9.5],
            "bdfiod_value_avg": [1.20, 1.40],
            "cfvo_value_avg": [8.0, 12.0],  # percentage -> should normalize to 0.08, 0.12
            "licence": ["CC-BY-4.0", "CC-BY-4.0"],
        }
    )
    tsv_path = tmp_path / "wosis_test.tsv"
    wosis_mock.to_csv(tsv_path, sep="\t", index=False)
    loaded = load_wosis(tsv_path)
    assert len(loaded) == 2
    assert loaded["coarse_frac"].iloc[0] == pytest.approx(0.08)
    assert (loaded["stock_mg_ha"] > 0).all()


def test_biome_beta_calibration_and_harmonize_all() -> None:
    df = load_published_pedons(PEDONS_CSV)
    beta_df = fit_beta_by_biome(df, min_depth_cm=50.0, n_boot=100)
    assert set(beta_df["biome"]) == {
        "tropical_moist_deciduous",
        "subtropical_chir_pine",
        "temperate_banj_oak",
    }
    assert ((beta_df["beta_mean"] > 0.90) & (beta_df["beta_mean"] < 0.995)).all()
    assert (beta_df["beta_ci_low"] <= beta_df["beta_mean"]).all()
    assert (beta_df["beta_mean"] <= beta_df["beta_ci_high"]).all()

    harm_df = harmonize_all_profiles(df)
    assert len(harm_df) == 24
    # Additivity check: 0-30 + 30-100 == 0-100 for every profile
    np.testing.assert_allclose(
        harm_df["soc_0_30_mg_ha"] + harm_df["soc_30_100_mg_ha"],
        harm_df["soc_0_100_mg_ha"],
        rtol=1e-10,
    )


def test_equal_area_spline_harmonize_additivity() -> None:
    res = equal_area_spline_harmonize(
        horizon_tops_cm=[0, 15, 30, 60],
        horizon_bottoms_cm=[15, 30, 60, 100],
        horizon_stocks_mg_ha=[28.0, 18.0, 22.0, 16.0],
    )
    assert res[(0, 30)] == pytest.approx(46.0, rel=1e-6)
    assert res[(30, 100)] == pytest.approx(38.0, rel=1e-6)
    assert res[(0, 30)] + res[(30, 100)] == pytest.approx(res[(0, 100)], rel=1e-9)
