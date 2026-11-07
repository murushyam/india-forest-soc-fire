"""Publication-grade figure generators for depth curves, spatial CV gap, SHAP, and fire-SOC recovery."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ..harmonize.depth import cumulative_fraction


def plot_biome_depth_curves(beta_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot Jobbágy-Jackson cumulative depth curves Y(d) = 1 - beta^d by Indian forest biome."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = np.linspace(0, 100, 200)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    for idx, row in beta_df.reset_index(drop=True).iterrows():
        c = colors[idx % len(colors)]
        biome = str(row["biome"])
        b_mean = float(row["beta_mean"])
        b_lo = float(row.get("beta_ci_low", b_mean))
        b_hi = float(row.get("beta_ci_high", b_mean))

        y_mean = cumulative_fraction(depths, b_mean)
        y_lo = cumulative_fraction(depths, b_hi)  # higher beta -> smaller cumulative fraction at shallow depth
        y_hi = cumulative_fraction(depths, b_lo)

        label = f"{biome} (β={b_mean:.4f}, n={int(row['n_profiles'])})"
        ax.plot(y_mean * 100.0, depths, lw=2.2, color=c, label=label)
        ax.fill_betweenx(depths, y_lo * 100.0, y_hi * 100.0, color=c, alpha=0.18)

    ax.axhline(30, color="gray", ls="--", lw=1.0, label="Topsoil boundary (30 cm)")
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative SOC Stock Fraction (%)")
    ax.set_ylabel("Soil Depth (cm)")
    ax.set_title("Recalibrated Jobbágy–Jackson Depth Profiles by Indian Forest Biome")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_cv_optimism_gap(cv_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot R2 and RMSE across Random K-fold vs Spatial Hex Block CV (25, 50, 100 km)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    models = list(cv_df["model"].unique())
    x_labels = [
        "Random KFold" if r["scheme"] == "random_kfold" else f"Hex {int(r['diameter_km'])} km"
        for _, r in cv_df.loc[cv_df["model"] == models[0]].iterrows()
    ]
    x = np.arange(len(x_labels))
    width = 0.25

    for i, m in enumerate(models):
        sub = cv_df.loc[cv_df["model"] == m]
        axes[0].bar(x + (i - len(models) / 2 + 0.5) * width, sub["r2"], width=width, label=m.upper())
        axes[1].bar(x + (i - len(models) / 2 + 0.5) * width, sub["rmse"], width=width, label=m.upper())

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=20)
    axes[0].set_ylabel("Model Efficiency / $R^2$")
    axes[0].set_title("Optimism Bias in $R^2$: Random vs Spatial Block CV")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=20)
    axes[1].set_ylabel("RMSE (Mg C ha$^{-1}$)")
    axes[1].set_title("Prediction Error Inflation Across Spatial Block Sizes")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_shap_topsoil_vs_subsoil(
    shap_top: pd.DataFrame,
    shap_sub: pd.DataFrame,
    out_path: str | Path,
    top_n: int = 10,
) -> Path:
    """Plot global SHAP importance ranking for Topsoil (0-30 cm) vs Subsoil (30-100 cm)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=180)
    for ax, df, title, color in (
        (axes[0], shap_top.head(top_n).iloc[::-1], "Topsoil (0–30 cm) Drivers", "#d95f02"),
        (axes[1], shap_sub.head(top_n).iloc[::-1], "Subsoil (30–100 cm) Drivers", "#1b9e77"),
    ):
        ax.barh(df["feature"], df["rel_importance_pct"], color=color, alpha=0.85)
        ax.set_xlabel("Relative Mean |SHAP| Importance (%)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_fire_combustion_and_recovery(
    loss_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Plot paired combustion loss (0-30 cm vs 30-100 cm control) and 0-5 yr recovery trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)

    # Panel A: Topsoil vs Subsoil relative loss by burn severity class
    x = np.arange(len(loss_df))
    w = 0.35
    axes[0].bar(
        x - w / 2,
        loss_df["topsoil_loss_frac"] * 100.0,
        width=w,
        color="#e6550d",
        label="Topsoil (0–30 cm)",
    )
    if "subsoil_loss_frac" in loss_df.columns:
        axes[0].bar(
            x + w / 2,
            loss_df["subsoil_loss_frac"].fillna(0.0) * 100.0,
            width=w,
            color="#3182bd",
            label="Subsoil (30–100 cm control)",
        )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(loss_df["burn_class"])
    axes[0].set_ylabel("SOC Stock Depletion Relative to Matched Unburned (%)")
    axes[0].set_title("Paired Covariate-Matched Fire Combustion Loss")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)

    # Panel B: 0-5 yr recovery curves for high severity across biomes
    high_rec = rec_df.loc[rec_df["burn_class"] == "moderate_high"]
    if high_rec.empty:
        high_rec = rec_df
    for biome, bgrp in high_rec.groupby("biome"):
        axes[1].plot(
            bgrp["years_since_fire"],
            bgrp["rel_soc_anomaly_single_burn"] * 100.0,
            marker="o",
            lw=2.0,
            label=f"{biome} (1 burn)",
        )
        axes[1].plot(
            bgrp["years_since_fire"],
            bgrp["rel_soc_anomaly_repeat_3burns"] * 100.0,
            marker="s",
            ls="--",
            lw=1.5,
            label=f"{biome} (3 burns)",
        )
    axes[1].axhline(0.0, color="gray", ls=":", lw=1.0)
    axes[1].set_xlabel("Years Since Fire ($t$)")
    axes[1].set_ylabel("Topsoil SOC Anomaly vs Unburned Control (%)")
    axes[1].set_title("1–5 Yr Post-Fire Topsoil SOC Recovery Trajectory")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p
