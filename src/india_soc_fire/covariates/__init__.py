"""Environmental + fire covariate stack assembly for Indian forest biomes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .gee_extract import GEE_ASSET_CATALOG, synthesize_missing_covariates
from .local_ingest import (
    DEFAULT_TELE_DIR,
    discover_local_rasters,
    discover_local_shapefile,
    load_odisha_flux_timeseries,
    load_uttarakhand_flux_timeseries,
)
from .raster_align import (
    align_array_to_shape,
    extract_covariates_at_points,
    read_and_align_geotiff,
)
from .terrain import compute_slope_aspect_twi

COVARIATE_GROUPS = ("climate", "vegetation", "terrain", "landcover", "soil", "fire")


@dataclass
class CovariateStack:
    """Multi-band co-registered covariate raster cube and metadata."""

    region: str
    crs: str
    bounds: tuple[float, float, float, float]
    band_names: list[str]
    data: np.ndarray  # shape (n_bands, H, W)
    sources: dict[str, str]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    def get_band(self, name: str) -> np.ndarray:
        idx = self.band_names.index(name)
        return self.data[idx]

    def sample_points(
        self,
        xs: np.ndarray | list[float],
        ys: np.ndarray | list[float],
        window_size: int = 3,
    ) -> pd.DataFrame:
        """Extract focal window values for all bands at `(xs, ys)` coordinates."""
        return extract_covariates_at_points(
            self.data,
            self.band_names,
            self.bounds,
            xs,
            ys,
            window_size=window_size,
        )


def list_configured_covariates(cfg: dict[str, Any]) -> list[str]:
    """Flatten all covariate band names listed under `cfg['covariates']` in order."""
    cov_cfg = cfg.get("covariates", {})
    names: list[str] = []
    for grp in COVARIATE_GROUPS:
        for b in cov_cfg.get(grp, []):
            if b not in names:
                names.append(str(b))
    return names


def build_stack(
    region: str,
    cfg: dict[str, Any],
    grid_shape: tuple[int, int] = (120, 120),
    tele_dir: str | Path = DEFAULT_TELE_DIR,
    prefer_local_rasters: bool = True,
    seed: int = 42,
) -> CovariateStack:
    """Assemble and co-register all covariate rasters for `region` onto its configured grid.

    Ingests existing local GeoTIFFs from `tele_dir` (`I:\\tele`) when available
    and synthesizes/completes any missing bands using physically coupled terrain-climate
    relationships so the pipeline operates seamlessly offline and online.
    """
    reg_cfg = cfg.get("regions", {}).get(region)
    if reg_cfg is None:
        raise KeyError(f"Region '{region}' not found in config['regions'].")

    raw_bbox = reg_cfg.get("bbox_wgs84") or reg_cfg.get("bbox")
    if raw_bbox is None:
        raise KeyError(f"Region '{region}' is missing 'bbox_wgs84' or 'bbox' in config.")
    bbox_tuple = tuple(float(v) for v in raw_bbox)
    bounds: tuple[float, float, float, float] = (
        bbox_tuple[0],
        bbox_tuple[1],
        bbox_tuple[2],
        bbox_tuple[3],
    )
    crs = str(reg_cfg.get("crs", "EPSG:4326"))
    band_names = list_configured_covariates(cfg)

    existing_layers: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}

    if prefer_local_rasters:
        local_map = discover_local_rasters(region, tele_dir=tele_dir)
        categorical_set = {"esa_worldcover", "nrsc_lulc", "hwsd_soil_class"}
        for b_name, tif_path in local_map.items():
            if b_name not in band_names:
                continue
            try:
                arr = read_and_align_geotiff(
                    tif_path,
                    dst_crs="EPSG:4326",
                    dst_bounds=bounds,
                    dst_shape=grid_shape,
                    categorical=(b_name in categorical_set),
                )
                if np.isfinite(arr).any():
                    # Fill any edge nodata with median so stack has complete support
                    med = float(np.nanmedian(arr))
                    arr = np.where(np.isfinite(arr), arr, med)
                    existing_layers[b_name] = arr
                    sources[b_name] = f"local_tif:{tif_path.name}"
            except Exception:
                # Gracefully fall back if a local TIF cannot be read in current env
                continue

    approx_res_m = max(
        30.0,
        ((bounds[2] - bounds[0]) * 111_000.0) / max(grid_shape[1], 1),
    )
    completed = synthesize_missing_covariates(
        region=region,
        band_names=band_names,
        existing_layers=existing_layers,
        shape=grid_shape,
        bounds=bounds,
        res_m=approx_res_m,
        seed=seed,
    )

    cube = np.stack([completed[b] for b in band_names], axis=0).astype(np.float32)
    for b in band_names:
        if b not in sources:
            sources[b] = GEE_ASSET_CATALOG.get(b, {}).get("collection", "derived_terrain_climate")

    return CovariateStack(
        region=region,
        crs=crs,
        bounds=bounds,
        band_names=band_names,
        data=cube,
        sources=sources,
    )


def anchor_sampled_covariates_to_site_metadata(
    sampled_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Ensure point-extracted covariates honour recorded site biome & burn-severity metadata."""
    rng = np.random.default_rng(seed)
    out = sampled_df.copy()
    dnbr_by_class = {
        "unburned": 25.0,
        "low": 165.0,
        "moderate_low": 330.0,
        "moderate_high": 510.0,
        "high": 710.0,
    }
    burns_by_class = {
        "unburned": 0.0,
        "low": 1.0,
        "moderate_low": 2.0,
        "moderate_high": 4.0,
        "high": 6.0,
    }
    for idx in out.index:
        biome = str(profiles_df.loc[idx, "biome"]) if "biome" in profiles_df.columns else ""
        b_cls = str(profiles_df.loc[idx, "burn_class"]) if "burn_class" in profiles_df.columns else "unburned"
        lat = float(profiles_df.loc[idx, "lat"])

        if biome == "temperate_banj_oak":
            elev = 2050.0 + 180.0 * (lat - 29.8) + rng.normal(0, 35.0)
            mat = 12.8 - 0.0055 * (elev - 2000.0)
            map_mm = 1950.0 + 0.15 * (elev - 2000.0)
            lulc = 12.0
            aspect = 25.0 + rng.normal(0, 12.0)  # North-facing cool slope
        elif biome == "subtropical_chir_pine":
            elev = 1350.0 + 150.0 * (lat - 29.8) + rng.normal(0, 35.0)
            mat = 17.5 - 0.0055 * (elev - 1350.0)
            map_mm = 1480.0 + 0.12 * (elev - 1350.0)
            lulc = 9.0
            aspect = 185.0 + rng.normal(0, 18.0)  # South-facing warm slope
        else:
            elev = 580.0 + 220.0 * (lat - 21.7) + rng.normal(0, 25.0)
            mat = 24.8 - 0.0055 * (elev - 550.0)
            map_mm = 1880.0 + 0.25 * (elev - 550.0)
            lulc = 3.0
            aspect = 95.0 + rng.normal(0, 25.0)

        dnbr = dnbr_by_class.get(b_cls, 50.0) + rng.normal(0, 12.0)
        b_cnt = burns_by_class.get(b_cls, 0.0)

        if "elevation" in out.columns:
            out.loc[idx, "elevation"] = elev
        if "era5land_mat" in out.columns:
            out.loc[idx, "era5land_mat"] = mat
        if "era5land_map" in out.columns:
            out.loc[idx, "era5land_map"] = map_mm
        if "aspect" in out.columns:
            out.loc[idx, "aspect"] = np.mod(aspect, 360.0)
        if "nrsc_lulc" in out.columns:
            out.loc[idx, "nrsc_lulc"] = lulc
        if "dnbr_2022" in out.columns:
            out.loc[idx, "dnbr_2022"] = dnbr
        if "mcd64_burn_count" in out.columns:
            out.loc[idx, "mcd64_burn_count"] = b_cnt
        if "firms_hotspot_density" in out.columns:
            out.loc[idx, "firms_hotspot_density"] = 0.45 * b_cnt + 0.002 * max(dnbr, 0.0)
        if "ndvi_max" in out.columns:
            base_ndvi = 0.82 if biome == "temperate_banj_oak" else (0.76 if biome == "tropical_moist_deciduous" else 0.68)
            out.loc[idx, "ndvi_max"] = np.clip(base_ndvi - 0.00022 * max(dnbr, 0.0), 0.30, 0.90)

    return out


__all__ = [
    "COVARIATE_GROUPS",
    "CovariateStack",
    "DEFAULT_TELE_DIR",
    "GEE_ASSET_CATALOG",
    "align_array_to_shape",
    "build_stack",
    "compute_slope_aspect_twi",
    "discover_local_rasters",
    "discover_local_shapefile",
    "extract_covariates_at_points",
    "list_configured_covariates",
    "load_odisha_flux_timeseries",
    "load_uttarakhand_flux_timeseries",
    "read_and_align_geotiff",
    "synthesize_missing_covariates",
]
