"""Tests for Phase 3 (terrain derivatives, covariate stack builder, and focal point extraction)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from india_soc_fire.config import load_config
from india_soc_fire.covariates import (
    build_stack,
    compute_slope_aspect_twi,
    list_configured_covariates,
)

CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def test_terrain_derivatives_shapes_and_ranges() -> None:
    # South-facing inclined plane (elevation increases northward -> row 0 is highest)
    y = np.linspace(1000.0, 200.0, 40)[:, None]
    x = np.linspace(0.0, 50.0, 40)[None, :]
    dem = y + x

    res = compute_slope_aspect_twi(dem, res_m=30.0)
    assert set(res.keys()) == {
        "elevation",
        "slope",
        "aspect",
        "aspect_northness",
        "aspect_eastness",
        "twi",
    }
    for arr in res.values():
        assert arr.shape == (40, 40)
        assert np.isfinite(arr).all()
    assert (res["slope"] > 0.0).all() and (res["slope"] < 90.0).all()
    assert (res["aspect"] >= 0.0).all() and (res["aspect"] < 360.0).all()


def test_build_stack_and_point_sampling() -> None:
    cfg = load_config(CFG_PATH)
    expected_bands = list_configured_covariates(cfg)
    assert len(expected_bands) == 20

    for region in ("odisha_similipal", "uttarakhand"):
        stk = build_stack(region, cfg, grid_shape=(24, 24), prefer_local_rasters=False)
        assert stk.region == region
        assert stk.band_names == expected_bands
        assert stk.shape == (len(expected_bands), 24, 24)
        assert np.isfinite(stk.data).all()

        # Sample points inside region bbox
        b = stk.bounds
        xs = [0.5 * (b[0] + b[2]), b[0] + 0.25 * (b[2] - b[0])]
        ys = [0.5 * (b[1] + b[3]), b[1] + 0.75 * (b[3] - b[1])]
        sampled = stk.sample_points(xs, ys, window_size=3)
        assert sampled.shape == (2, len(expected_bands))
        assert list(sampled.columns) == expected_bands
        assert sampled.notna().all().all()
