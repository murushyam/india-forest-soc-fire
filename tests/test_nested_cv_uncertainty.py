"""Tests for Phase 4 (variogram, nested spatial CV, SHAP, PDP, AOA) and Phase 5 (bootstrap, QRF, benchmark)."""
from __future__ import annotations

import numpy as np

from india_soc_fire.cv import empirical_variogram, fit_variogram_range, morans_i
from india_soc_fire.models import build_model
from india_soc_fire.models.interpret import compute_aoa, compute_pdp_1d, compute_shap_importance
from india_soc_fire.models.metrics import interval_metrics, lins_ccc, regression_metrics
from india_soc_fire.models.nested_cv import compare_random_vs_spatial_cv, run_nested_cv
from india_soc_fire.uncertainty import bootstrap_predict, qrf_interval


def test_variogram_and_morans_i() -> None:
    rng = np.random.default_rng(7)
    x_m = rng.uniform(0, 120_000, size=40)
    y_m = rng.uniform(0, 120_000, size=40)
    vals = 50.0 + 0.0002 * x_m + rng.normal(0, 3.0, size=40)

    emp = empirical_variogram(x_m, y_m, vals, max_lag_m=100_000.0, n_bins=8)
    assert len(emp) >= 3
    v_fit = fit_variogram_range(x_m, y_m, vals, max_lag_m=100_000.0)
    assert v_fit["recommended_block_km"] in {25.0, 50.0, 100.0}

    mi = morans_i(x_m, y_m, vals - np.mean(vals), bandwidth_m=45_000.0)
    assert -1.0 <= mi["morans_i"] <= 1.5


def test_metrics_and_nested_cv() -> None:
    rng = np.random.default_rng(42)
    n = 36
    x_m = rng.uniform(0, 200_000, size=n)
    y_m = rng.uniform(0, 200_000, size=n)
    X = rng.normal(size=(n, 5))
    y = 60.0 + 12.0 * X[:, 0] - 8.0 * X[:, 1] + rng.normal(0, 2.0, size=n)

    assert lins_ccc(y, y) == 1.0
    m = regression_metrics(y, y + 1.0)
    assert m["rmse"] == 1.0 and m["bias_me"] == 1.0

    res_qrf = run_nested_cv(
        X, y, x_m, y_m, model_name="qrf", scheme="spatial_hex_block", diameter_km=50.0, outer_folds=3, inner_folds=2
    )
    assert res_qrf.oof_predictions.shape == (n,)
    assert res_qrf.oof_quantiles is not None and res_qrf.oof_quantiles.shape == (n, 3)
    assert "picp" in res_qrf.metrics

    comp = compare_random_vs_spatial_cv(
        X, y, x_m, y_m, model_names=("rf",), block_diameters_km=(25.0, 50.0), outer_folds=3, inner_folds=2
    )
    assert len(comp) == 3
    assert "r2_optimism_gap" in comp.columns


def test_shap_pdp_aoa_and_uncertainty() -> None:
    rng = np.random.default_rng(11)
    X_tr = rng.normal(size=(30, 4))
    y_tr = 45.0 + 10.0 * X_tr[:, 0] - 5.0 * X_tr[:, 1] + rng.normal(0, 1.5, size=30)
    X_pr = rng.normal(size=(12, 4))
    X_pr[-1] = np.array([8.0, -8.0, 8.0, -8.0])  # extreme outlier outside AOA

    rf = build_model("rf", n_estimators=80, random_state=11).fit(X_tr, y_tr)
    shap_df, shap_mat = compute_shap_importance(rf, X_tr, feature_names=["c1", "c2", "c3", "c4"])
    assert shap_mat.shape == (30, 4)
    assert shap_df.iloc[0]["feature"] == "c1"

    pdp_df = compute_pdp_1d(rf, X_tr, feature="c1", feature_names=["c1", "c2", "c3", "c4"], grid_resolution=10)
    assert len(pdp_df) == 10

    aoa = compute_aoa(X_tr, X_pr)
    assert bool(aoa["aoa_mask"][-1]) is False  # extreme outlier flagged outside AOA

    boot = bootstrap_predict(X_tr, y_tr, X_pr, model_name="rf", n_boot=8, n_estimators=50)
    assert boot["mean"].shape == (12,) and (boot["sd"] >= 0).all()

    qrf_res = qrf_interval(X_tr, y_tr, X_pr, n_estimators=80)
    assert (qrf_res["q05"] <= qrf_res["q50"]).all()
    assert (qrf_res["q50"] <= qrf_res["q95"]).all()

    im = interval_metrics(y_tr[:10], qrf_res["q05"][:10] - 50, qrf_res["q95"][:10] + 50)
    assert im["picp"] == 1.0
