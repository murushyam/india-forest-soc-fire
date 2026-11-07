"""Soil profile loaders, quality control, pedotransfer imputation, and SOC stock calculation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .ptf import BulkDensityPTF, fit_bd_ptf, impute_missing_bd_and_cf, propagate_horizon_stock_sd
from .qc import (
    INDIA_BBOX,
    NC_LICENCES,
    PUBLIC_LICENCES,
    STANDARD_COLUMNS,
    apply_walkley_black_correction,
    enforce_licence_gate,
    validate_and_clean_profiles,
)

__all__ = [
    "BulkDensityPTF",
    "INDIA_BBOX",
    "NC_LICENCES",
    "PUBLIC_LICENCES",
    "STANDARD_COLUMNS",
    "apply_walkley_black_correction",
    "assemble_training_profiles",
    "enforce_licence_gate",
    "fit_bd_ptf",
    "impute_missing_bd_and_cf",
    "load_published_pedons",
    "load_wosis",
    "propagate_horizon_stock_sd",
    "soc_stock_mg_ha",
    "validate_and_clean_profiles",
]


def soc_stock_mg_ha(
    soc_g_kg: np.ndarray | float,
    bd_g_cm3: np.ndarray | float,
    thick_cm: np.ndarray | float,
    coarse_frac: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """SOC stock (Mg C ha-1) = SOC(g/kg) * BD(g/cm3) * depth(cm) * (1 - CF) * 0.1."""
    return soc_g_kg * bd_g_cm3 * thick_cm * (1.0 - coarse_frac) * 0.1


_WOSIS_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "profile_id": ("profile_id", "profile_code", "site_id", "id"),
    "lon": ("lon", "longitude", "x", "long"),
    "lat": ("lat", "latitude", "y"),
    "top_cm": ("top_cm", "upper_depth", "top", "hzn_top", "d_top"),
    "bottom_cm": ("bottom_cm", "lower_depth", "bottom", "hzn_bot", "d_bot"),
    "soc_g_kg": ("soc_g_kg", "orgc_value_avg", "orgc", "oc_g_kg", "soc"),
    "bd_g_cm3": ("bd_g_cm3", "bdfiod_value_avg", "bdod_value_avg", "bulk_density", "bd"),
    "coarse_frac": ("coarse_frac", "cfvo_value_avg", "cfvo", "gravel_frac"),
    "source": ("source", "dataset_id", "source_db", "citation"),
    "licence": ("licence", "license", "data_license"),
}


def _normalise_columns(df: pd.DataFrame, default_source: str, default_licence: str) -> pd.DataFrame:
    out = df.copy()
    col_map: dict[str, str] = {}
    lower_lookup = {c.lower(): c for c in out.columns}
    for target, aliases in _WOSIS_COLUMN_ALIASES.items():
        if target in out.columns:
            continue
        for alias in aliases:
            if alias.lower() in lower_lookup:
                col_map[lower_lookup[alias.lower()]] = target
                break
    if col_map:
        out = out.rename(columns=col_map)

    if "coarse_frac" not in out.columns:
        out["coarse_frac"] = np.nan
    else:
        # Convert volumetric percentage (0-100%) to fraction [0, 1) when values > 1
        cf_vals = pd.to_numeric(out["coarse_frac"], errors="coerce")
        pct_mask = cf_vals > 1.0
        if pct_mask.any():
            cf_vals = cf_vals.where(~pct_mask, cf_vals / 100.0)
        out["coarse_frac"] = cf_vals

    if "bd_g_cm3" not in out.columns:
        out["bd_g_cm3"] = np.nan
    if "source" not in out.columns:
        out["source"] = default_source
    else:
        out["source"] = out["source"].fillna(default_source)
    if "licence" not in out.columns:
        out["licence"] = default_licence
    else:
        out["licence"] = out["licence"].fillna(default_licence)
    return out


def load_wosis(
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = INDIA_BBOX,
    allow_nc: bool = False,
    impute_missing: bool = True,
) -> pd.DataFrame:
    """Load and standardise WoSIS India soil profiles from CSV, TSV, Parquet, or GeoPackage.

    Parameters
    ----------
    path : str or Path
        Path to WoSIS snapshot file (`.csv`, `.tsv`, `.parquet`, or `.gpkg`).
    bbox : tuple of (min_lon, min_lat, max_lon, max_lat), optional
        Spatial bounding box filter (defaults to India extent).
    allow_nc : bool, default False
        Whether to retain CC-BY-NC licensed rows (default False per public export rule).
    impute_missing : bool, default True
        Whether to impute missing bulk density and coarse fragments via PTF and
        compute horizon ``stock_mg_ha``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"WoSIS profile file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        raw = pd.read_csv(p, sep="\t")
    elif suffix == ".parquet":
        raw = pd.read_parquet(p)
    elif suffix in {".gpkg", ".shp", ".geojson"}:
        import geopandas as gpd

        gdf = gpd.read_file(p)
        if "lon" not in gdf.columns and gdf.geometry is not None:
            gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs else gdf
            gdf["lon"] = gdf_wgs.geometry.x
            gdf["lat"] = gdf_wgs.geometry.y
        raw = pd.DataFrame(gdf.drop(columns=[c for c in ("geometry",) if c in gdf.columns]))
    else:
        raw = pd.read_csv(p)

    df = _normalise_columns(raw, default_source="WoSIS_2023", default_licence="CC-BY-4.0")
    df = enforce_licence_gate(df, allow_nc=allow_nc)
    df = apply_walkley_black_correction(df)
    df = validate_and_clean_profiles(df, bbox=bbox)

    if impute_missing and len(df) > 0:
        df, _ = impute_missing_bd_and_cf(df)
        thick = df["bottom_cm"].to_numpy(dtype=float) - df["top_cm"].to_numpy(dtype=float)
        df["stock_mg_ha"] = soc_stock_mg_ha(
            df["soc_g_kg"].to_numpy(dtype=float),
            df["bd_g_cm3"].to_numpy(dtype=float),
            thick,
            df["coarse_frac"].to_numpy(dtype=float),
        )
        df["stock_sd_mg_ha"] = propagate_horizon_stock_sd(
            df["soc_g_kg"].to_numpy(dtype=float),
            df["bd_g_cm3"].to_numpy(dtype=float),
            thick,
            df["coarse_frac"].to_numpy(dtype=float),
            bd_se=df["bd_se"].to_numpy(dtype=float),
        )
    return df


def load_published_pedons(
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = INDIA_BBOX,
    allow_nc: bool = True,
    impute_missing: bool = True,
) -> pd.DataFrame:
    """Load curated published Indian forest soil pedons (`data/derived/published_pedons_india.csv`).

    Applies Walkley-Black recovery correction when ``soc_method == 'WB_unheated'``,
    enforces the license gate from `docs/DATA_PLAN.md`, imputes missing bulk density
    via the calibrated organic-mineral PTF (`bd_imputed=True`), and computes horizon
    SOC stocks (`stock_mg_ha`) and uncertainties (`stock_sd_mg_ha`).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Published pedons file not found: {p}")

    raw = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    df = _normalise_columns(raw, default_source="published_literature", default_licence="CC-BY-4.0")
    df = enforce_licence_gate(df, allow_nc=allow_nc)
    df = apply_walkley_black_correction(df)
    df = validate_and_clean_profiles(df, bbox=bbox)

    if impute_missing and len(df) > 0:
        df, _ = impute_missing_bd_and_cf(df)
        thick = df["bottom_cm"].to_numpy(dtype=float) - df["top_cm"].to_numpy(dtype=float)
        df["stock_mg_ha"] = soc_stock_mg_ha(
            df["soc_g_kg"].to_numpy(dtype=float),
            df["bd_g_cm3"].to_numpy(dtype=float),
            thick,
            df["coarse_frac"].to_numpy(dtype=float),
        )
        df["stock_sd_mg_ha"] = propagate_horizon_stock_sd(
            df["soc_g_kg"].to_numpy(dtype=float),
            df["bd_g_cm3"].to_numpy(dtype=float),
            thick,
            df["coarse_frac"].to_numpy(dtype=float),
            bd_se=df["bd_se"].to_numpy(dtype=float),
        )
    return df


def assemble_training_profiles(
    published_path: str | Path | None = None,
    wosis_path: str | Path | None = None,
    region: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    allow_nc: bool = False,
) -> pd.DataFrame:
    """Merge WoSIS and published Indian pedons, deduplicate, and calibrate a unified PTF."""
    frames: list[pd.DataFrame] = []
    if wosis_path is not None and Path(wosis_path).exists():
        frames.append(load_wosis(wosis_path, bbox=bbox, allow_nc=allow_nc, impute_missing=False))
    if published_path is not None and Path(published_path).exists():
        frames.append(
            load_published_pedons(published_path, bbox=bbox, allow_nc=allow_nc, impute_missing=False)
        )
    if not frames:
        raise ValueError("At least one existing profile dataset (wosis_path or published_path) is required.")

    combined = pd.concat(frames, ignore_index=True)
    if region is not None and "region" in combined.columns:
        combined = combined.loc[combined["region"] == region].copy()

    combined = validate_and_clean_profiles(combined, bbox=bbox)
    combined, _ = impute_missing_bd_and_cf(combined)
    thick = combined["bottom_cm"].to_numpy(dtype=float) - combined["top_cm"].to_numpy(dtype=float)
    combined["stock_mg_ha"] = soc_stock_mg_ha(
        combined["soc_g_kg"].to_numpy(dtype=float),
        combined["bd_g_cm3"].to_numpy(dtype=float),
        thick,
        combined["coarse_frac"].to_numpy(dtype=float),
    )
    combined["stock_sd_mg_ha"] = propagate_horizon_stock_sd(
        combined["soc_g_kg"].to_numpy(dtype=float),
        combined["bd_g_cm3"].to_numpy(dtype=float),
        thick,
        combined["coarse_frac"].to_numpy(dtype=float),
        bd_se=combined["bd_se"].to_numpy(dtype=float),
    )
    return combined
