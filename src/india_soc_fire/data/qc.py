"""Quality control, Walkley-Black harmonisation, and license gate enforcement for soil profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd

STANDARD_COLUMNS: tuple[str, ...] = (
    "profile_id",
    " lon".strip(),
    "lat",
    "top_cm",
    "bottom_cm",
    "soc_g_kg",
    "bd_g_cm3",
    "coarse_frac",
    "source",
    "licence",
)

PUBLIC_LICENCES: frozenset[str] = frozenset(
    {
        "CC-BY-4.0",
        "CC-BY-3.0",
        "CC-BY",
        "CC0",
        "CC0-1.0",
        "ODbL",
        "ODbL-1.0",
        "GOV-OPEN",
        "NDSAP-OPEN",
        "OPEN-ACCESS",
    }
)

NC_LICENCES: frozenset[str] = frozenset(
    {
        "CC-BY-NC-4.0",
        "CC-BY-NC-3.0",
        "CC-BY-NC",
        "CC-BY-NC-SA-4.0",
    }
)

INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 97.5, 37.5)


def apply_walkley_black_correction(
    df: pd.DataFrame,
    wb_recovery_factor: float = 1.32,
) -> pd.DataFrame:
    """Harmonise unheated Walkley-Black wet-digestion organic carbon to total SOC.

    Unheated Walkley & Black (1934) oxidation typically recovers ~76% of total
    organic carbon in Indian forest mineral soils (factor = 1 / 0.76 ≈ 1.32;
    Walkley & Black 1934; Bhattacharyya et al., 2000). Rows with
    ``soc_method == 'WB_unheated'`` are multiplied by ``wb_recovery_factor``
    and flagged in ``wb_corrected``.
    """
    out = df.copy()
    if "soc_method" not in out.columns:
        out["soc_method"] = "dry_combustion_or_corrected"
    if "wb_corrected" not in out.columns:
        out["wb_corrected"] = False

    mask = (
        out["soc_method"].astype(str).str.upper().isin({"WB_UNHEATED", "WALKLEY_BLACK_UNHEATED", "WB_RAW"})
        & (~out["wb_corrected"].astype(bool))
    )
    if mask.any():
        out.loc[mask, "soc_g_kg"] = out.loc[mask, "soc_g_kg"].astype(float) * float(wb_recovery_factor)
        out.loc[mask, "wb_corrected"] = True
    return out


def enforce_licence_gate(
    df: pd.DataFrame,
    allow_nc: bool = False,
) -> pd.DataFrame:
    """Filter profile rows according to docs/DATA_PLAN.md licensing rules.

    Parameters
    ----------
    df : pd.DataFrame
        Input profile horizon table containing a ``licence`` column and ``source``.
    allow_nc : bool, default False
        If False, only public open-redistribution licences (e.g. CC-BY-4.0, CC0,
        ODbL, GOV-OPEN) are retained. If True, CC-BY-NC rows are also retained
        for internal research runs and flagged via ``nc_restricted=True``.
    """
    if "licence" not in df.columns:
        raise ValueError("Profile DataFrame must include a 'licence' column (see docs/DATA_PLAN.md).")

    out = df.copy()
    # Explicitly forbid SoilGrids 2.0 as training target (circularity rule in docs/DATA_PLAN.md)
    if "source" in out.columns:
        sg_mask = out["source"].astype(str).str.lower().str.contains("soilgrids", na=False)
        if sg_mask.any():
            raise ValueError(
                "SoilGrids 2.0 rows detected in training profiles. "
                "Per docs/DATA_PLAN.md, SoilGrids 2.0 must never be used as a training target."
            )

    lic_norm = out["licence"].fillna("UNKNOWN").astype(str).str.strip()
    is_public = lic_norm.isin(PUBLIC_LICENCES)
    is_nc = lic_norm.isin(NC_LICENCES)

    out["nc_restricted"] = is_nc
    if allow_nc:
        keep = is_public | is_nc
    else:
        keep = is_public

    return out.loc[keep].reset_index(drop=True)


def validate_and_clean_profiles(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None = INDIA_BBOX,
) -> pd.DataFrame:
    """Validate horizon schema, enforce physical bounds, and deduplicate horizons."""
    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required standard columns: {missing}")

    out = df.copy()
    for col in ("lon", "lat", "top_cm", "bottom_cm", "soc_g_kg", "bd_g_cm3", "coarse_frac"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Coordinate bounds check
    valid = out["lon"].notna() & out["lat"].notna()
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        valid &= (
            (out["lon"] >= min_lon)
            & (out["lon"] <= max_lon)
            & (out["lat"] >= min_lat)
            & (out["lat"] <= max_lat)
        )

    # Horizon geometry & physical SOC bounds (0 < soc_g_kg <= 580 g/kg organic soil ceiling)
    valid &= (
        (out["top_cm"] >= 0.0)
        & (out["bottom_cm"] > out["top_cm"])
        & (out["bottom_cm"] <= 300.0)
        & (out["soc_g_kg"] > 0.0)
        & (out["soc_g_kg"] <= 580.0)
    )

    # Where BD is present, enforce physical mineral/organic soil range [0.15, 2.20] g/cm3
    bd_present = out["bd_g_cm3"].notna()
    valid &= (~bd_present) | ((out["bd_g_cm3"] >= 0.15) & (out["bd_g_cm3"] <= 2.20))

    # Coarse fragments in [0, 0.95]
    cf_present = out["coarse_frac"].notna()
    valid &= (~cf_present) | ((out["coarse_frac"] >= 0.0) & (out["coarse_frac"] < 0.95))

    out = out.loc[valid].copy()

    # Round coordinates to 4 decimal places (~11 m) for spatial deduplication key
    out["_lon_r"] = out["lon"].round(4)
    out["_lat_r"] = out["lat"].round(4)
    out = (
        out.sort_values(["profile_id", "top_cm", "bottom_cm"])
        .drop_duplicates(subset=["_lon_r", "_lat_r", "top_cm", "bottom_cm"], keep="first")
        .drop(columns=["_lon_r", "_lat_r"])
        .reset_index(drop=True)
    )
    return out
