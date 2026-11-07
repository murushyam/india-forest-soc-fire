"""Command-line pipeline orchestrating Phases 1-6 of india-forest-soc-fire."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .config import load_config
from .covariates import (
    DEFAULT_TELE_DIR,
    anchor_sampled_covariates_to_site_metadata,
    build_stack,
)
from .cv import fit_variogram_range, hex_block_ids
from .data import load_published_pedons
from .fire import (
    estimate_paired_combustion_loss,
    simulate_biome_recovery_chronosequence,
    summarize_local_postfire_flux_recovery,
)
from .harmonize import fit_beta_by_biome, harmonize_all_profiles
from .models import build_model
from .models.interpret import compute_shap_importance
from .models.nested_cv import compare_random_vs_spatial_cv, project_lonlat_to_meters, run_nested_cv
from .uncertainty.benchmark import benchmark_predictions
from .uncertainty.predict_raster import predict_soc_raster_stack, save_prediction_geotiffs
from .viz import (
    plot_biome_depth_curves,
    plot_cv_optimism_gap,
    plot_fire_combustion_and_recovery,
    plot_shap_topsoil_vs_subsoil,
)


def run_full_pipeline(
    config_path: str | Path = "config/config.yaml",
    pedons_path: str | Path = "data/derived/published_pedons_india.csv",
    output_dir: str | Path = "outputs",
    tele_dir: str | Path = DEFAULT_TELE_DIR,
    grid_shape: tuple[int, int] = (32, 32),
    n_boot: int = 12,
    generate_figures: bool = True,
) -> dict[str, Any]:
    """Execute the complete v1.0.0 pipeline across both pilot regions and all 3 depth intervals."""
    cfg = load_config(config_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load and QC soil profiles + pedotransfer BD/CF imputation
    horizons_df = load_published_pedons(pedons_path, allow_nc=True, impute_missing=True)
    horizons_df.to_csv(out_root / "phase1_qc_horizons.csv", index=False)

    # Phase 2: Biome-specific Jobbágy-Jackson depth harmonisation
    harm_cfg = cfg.get("harmonize", {})
    min_depth = float(harm_cfg.get("min_profile_depth_cm", 50.0)) if isinstance(harm_cfg, dict) else 50.0
    beta_df = fit_beta_by_biome(horizons_df, min_depth_cm=min_depth, n_boot=200)
    beta_df.to_csv(out_root / "phase2_biome_betas.csv", index=False)

    beta_map = dict(zip(beta_df["biome"], beta_df["beta_mean"]))
    profiles_df = harmonize_all_profiles(horizons_df, beta_by_biome=beta_map, min_depth_cm=min_depth)

    # Phase 3: Build multi-source covariate stacks per region & extract point values
    stacks = {}
    cov_frames: list[pd.DataFrame] = []
    for region in cfg.get("regions", {}):
        stk = build_stack(
            region=region,
            cfg=cfg,
            grid_shape=grid_shape,
            tele_dir=tele_dir,
            prefer_local_rasters=True,
        )
        stacks[region] = stk
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

    full_df = pd.concat(cov_frames, axis=0).sort_index().reset_index(drop=True)
    full_df.to_csv(out_root / "phase3_harmonized_training_matrix.csv", index=False)

    feature_cols = stacks[next(iter(stacks))].band_names
    X = full_df[feature_cols].to_numpy(dtype=float)
    x_m, y_m = project_lonlat_to_meters(full_df["lon"].to_numpy(), full_df["lat"].to_numpy())

    # Phase 4: Variogram diagnostics, Nested Spatial CV (Random vs Hex 25/50/100 km), and SHAP
    vario_info = fit_variogram_range(x_m, y_m, full_df["soc_0_30_mg_ha"].to_numpy(dtype=float))
    cv_comp_df = compare_random_vs_spatial_cv(
        X,
        full_df["soc_0_30_mg_ha"].to_numpy(dtype=float),
        x_m,
        y_m,
        model_names=("rf", "xgb", "qrf"),
        block_diameters_km=(25.0, 50.0, 100.0),
        outer_folds=4,
        inner_folds=2,
    )
    cv_comp_df.to_csv(out_root / "phase4_cv_comparison_0_30cm.csv", index=False)

    # Fit topsoil (0-30 cm) and subsoil (30-100 cm) RF models for SHAP comparison
    rf_top = build_model("rf", n_estimators=200, random_state=42).fit(
        X, full_df["soc_0_30_mg_ha"].to_numpy(dtype=float)
    )
    rf_sub = build_model("rf", n_estimators=200, random_state=42).fit(
        X, full_df["soc_30_100_mg_ha"].to_numpy(dtype=float)
    )
    shap_top, _ = compute_shap_importance(rf_top, X, feature_names=feature_cols)
    shap_sub, _ = compute_shap_importance(rf_sub, X, feature_names=feature_cols)
    shap_top.to_csv(out_root / "phase4_shap_topsoil_0_30cm.csv", index=False)
    shap_sub.to_csv(out_root / "phase4_shap_subsoil_30_100cm.csv", index=False)

    # Phase 5: Wall-to-wall raster prediction + uncertainty & external benchmark evaluation
    sp_cv_0_30 = run_nested_cv(
        X,
        full_df["soc_0_30_mg_ha"].to_numpy(dtype=float),
        x_m,
        y_m,
        model_name="rf",
        scheme="spatial_hex_block",
        diameter_km=50.0,
        outer_folds=4,
        inner_folds=2,
    )
    bench_df = benchmark_predictions(
        full_df,
        sp_cv_0_30.oof_predictions,
        depth_key="0_30",
        elevation_m=full_df["elevation"].to_numpy(dtype=float),
    )
    bench_df.to_csv(out_root / "phase5_benchmark_comparison.csv", index=False)

    blk_ids = hex_block_ids(x_m, y_m, diameter_km=50.0)
    raster_outputs: dict[str, list[Path]] = {}
    for region, stk in stacks.items():
        pred_maps = predict_soc_raster_stack(
            stk,
            X,
            full_df["soc_0_30_mg_ha"].to_numpy(dtype=float),
            model_name="rf",
            n_boot=n_boot,
            block_ids=blk_ids,
            n_estimators=120,
        )
        written = save_prediction_geotiffs(
            pred_maps,
            bounds=stk.bounds,
            crs=stk.crs,
            out_dir=out_root / "rasters" / region,
            prefix=f"{region}_soc_0_30cm",
        )
        raster_outputs[region] = written

    # Phase 6: Fire-SOC Nexus (Paired combustion loss & 1-5 yr recovery + local flux coupling)
    loss_df = estimate_paired_combustion_loss(full_df)
    loss_df.to_csv(out_root / "phase6_fire_combustion_loss.csv", index=False)

    rec_df = simulate_biome_recovery_chronosequence()
    rec_df.to_csv(out_root / "phase6_postfire_recovery_5yr.csv", index=False)

    flux_summaries = summarize_local_postfire_flux_recovery(tele_dir=tele_dir)
    for f_name, f_df in flux_summaries.items():
        f_df.to_csv(out_root / f"phase6_{f_name}.csv", index=False)

    figure_paths: list[Path] = []
    if generate_figures:
        fig_dir = out_root / "figures"
        figure_paths.append(plot_biome_depth_curves(beta_df, fig_dir / "fig1_biome_depth_curves.png"))
        figure_paths.append(plot_cv_optimism_gap(cv_comp_df, fig_dir / "fig2_cv_optimism_gap.png"))
        figure_paths.append(
            plot_shap_topsoil_vs_subsoil(shap_top, shap_sub, fig_dir / "fig3_shap_topsoil_vs_subsoil.png")
        )
        figure_paths.append(
            plot_fire_combustion_and_recovery(loss_df, rec_df, fig_dir / "fig4_fire_loss_and_recovery.png")
        )

    return {
        "n_horizons": len(horizons_df),
        "n_profiles": len(full_df),
        "biome_betas": beta_df,
        "variogram": vario_info,
        "cv_comparison": cv_comp_df,
        "benchmark": bench_df,
        "combustion_loss": loss_df,
        "recovery": rec_df,
        "flux_summaries": flux_summaries,
        "raster_files": raster_outputs,
        "figures": figure_paths,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="india-soc-fire",
        description="Harmonised multi-depth SOC mapping & fire-SOC recovery pipeline for Indian forest biomes.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--pedons",
        default="data/derived/published_pedons_india.csv",
        help="Path to published Indian forest pedons CSV.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for output tables, rasters, and figures.")
    parser.add_argument("--tele-dir", default=str(DEFAULT_TELE_DIR), help="Path to local I:\\tele data directory.")
    parser.add_argument("--grid-size", type=int, default=32, help="Grid width/height for raster prediction.")
    parser.add_argument("--n-boot", type=int, default=12, help="Number of bootstrap replicates.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = parser.parse_args(argv)

    res = run_full_pipeline(
        config_path=args.config,
        pedons_path=args.pedons,
        output_dir=args.output_dir,
        tele_dir=args.tele_dir,
        grid_shape=(args.grid_size, args.grid_size),
        n_boot=args.n_boot,
        generate_figures=not args.no_figures,
    )
    print(
        f"[india-soc-fire v1.0.0] Completed pipeline on {res['n_profiles']} profiles "
        f"({res['n_horizons']} horizons). Outputs written to '{args.output_dir}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
