"""Google Earth Engine (GEE) & Planetary Computer covariate extraction recipe & offline synthesizer."""
from __future__ import annotations

import numpy as np

from .terrain import compute_slope_aspect_twi


GEE_ASSET_CATALOG: dict[str, dict[str, str]] = {
    "era5land_mat": {
        "collection": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "band": "temperature_2m",
        "reducer": "mean_1991_2020_celsius",
    },
    "era5land_map": {
        "collection": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "band": "total_precipitation_sum",
        "reducer": "annual_sum_1991_2020_mm",
    },
    "imd_rain_monsoon_fraction": {
        "collection": "UCSB-CHG/CHIRPS/DAILY",
        "band": "precipitation",
        "reducer": "jjas_to_annual_ratio",
    },
    "pet": {
        "collection": "IDAHO_EPSCOR/TERRACLIMATE",
        "band": "pet",
        "reducer": "annual_sum_mm",
    },
    "elevation": {
        "collection": "USGS/SRTMGL1_003",
        "band": "elevation",
        "reducer": "mosaic",
    },
    "esa_worldcover": {
        "collection": "ESA/WorldCover/v200",
        "band": "Map",
        "reducer": "first",
    },
    "tree_cover": {
        "collection": "MODIS/061/MOD44B",
        "band": "Percent_Tree_Cover",
        "reducer": "median_2018_2022",
    },
    "mcd64_burn_count": {
        "collection": "MODIS/061/MCD64A1",
        "band": "BurnDate",
        "reducer": "count_annual_burns_2001_2022",
    },
    "firms_hotspot_density": {
        "collection": "FIRMS",
        "band": "T21",
        "reducer": "kernel_density_2012_2022",
    },
}


def synthesize_physically_consistent_dem(
    region: str,
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    seed: int = 42,
) -> np.ndarray:
    """Generate a physically realistic digital elevation model for offline/sandbox execution.

    - ``odisha_similipal``: Dome massif peaking at ~1150 m in the central-southern core
      with surrounding 150-400 m piedmont valleys.
    - ``uttarakhand``: SW-to-NE Himalayan orographic gradient rising from 400 m (Siwalik/Terai)
      to 3200 m (Lesser/Greater Himalaya) with incised river valleys.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    y_norm = np.linspace(1.0, 0.0, h)[:, None]  # 1 = North, 0 = South
    x_norm = np.linspace(0.0, 1.0, w)[None, :]  # 0 = West, 1 = East

    # Multi-scale ridge/valley harmonic structure
    ridges = (
        0.55 * np.sin(2.0 * np.pi * (2.5 * x_norm + 1.8 * y_norm))
        + 0.30 * np.cos(2.0 * np.pi * (5.0 * x_norm - 3.5 * y_norm))
        + 0.15 * np.sin(2.0 * np.pi * (9.0 * x_norm + 7.0 * y_norm))
    )
    noise = rng.normal(0.0, 15.0, size=shape)

    if region == "uttarakhand":
        # Elevation rises strongly toward North-East
        base_elev = 450.0 + 1950.0 * (0.65 * y_norm + 0.35 * x_norm) + 320.0 * ridges + noise
        return np.clip(base_elev, 300.0, 3500.0)
    else:
        # Similipal dome centered near (x=0.48, y=0.52)
        dist2 = (x_norm - 0.48) ** 2 + (y_norm - 0.52) ** 2
        dome = 820.0 * np.exp(-dist2 / 0.09)
        base_elev = 180.0 + dome + 110.0 * ridges + noise
        return np.clip(base_elev, 80.0, 1180.0)


def synthesize_missing_covariates(
    region: str,
    band_names: list[str],
    existing_layers: dict[str, np.ndarray],
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    res_m: float = 250.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Populate any covariates not present in `existing_layers` using physically coupled equations."""
    rng = np.random.default_rng(seed)
    layers = dict(existing_layers)

    if "elevation" not in layers:
        layers["elevation"] = synthesize_physically_consistent_dem(region, shape, bounds, seed=seed)

    terrain = compute_slope_aspect_twi(layers["elevation"], res_m=res_m)
    for k in ("slope", "aspect", "twi"):
        if k not in layers:
            layers[k] = terrain[k]
    northness = terrain["aspect_northness"]
    elev = layers["elevation"]

    if region == "uttarakhand":
        # Environmental lapse rate -6.2 C / km from 24.5 C at 300 m; cooler on north-facing slopes
        if "era5land_mat" not in layers:
            layers["era5land_mat"] = 24.5 - 0.0062 * (elev - 300.0) - 1.1 * northness
        if "era5land_map" not in layers:
            layers["era5land_map"] = np.clip(
                1450.0 + 0.25 * elev + 120.0 * northness + rng.normal(0.0, 35.0, shape),
                850.0,
                2600.0,
            )
        if "imd_rain_monsoon_fraction" not in layers:
            layers["imd_rain_monsoon_fraction"] = np.clip(
                0.76 - 0.00003 * elev + rng.normal(0.0, 0.015, shape),
                0.62,
                0.86,
            )
    else:
        # Odisha Similipal: moist tropical monsoon climate (MAT ~24-27 C, MAP ~1650-2200 mm)
        if "era5land_mat" not in layers:
            layers["era5land_mat"] = 27.2 - 0.0058 * (elev - 150.0) - 0.5 * northness
        if "era5land_map" not in layers:
            layers["era5land_map"] = np.clip(
                1680.0 + 0.55 * elev + 80.0 * northness + rng.normal(0.0, 40.0, shape),
                1400.0,
                2450.0,
            )
        if "imd_rain_monsoon_fraction" not in layers:
            layers["imd_rain_monsoon_fraction"] = np.clip(
                0.80 + 0.00002 * elev + rng.normal(0.0, 0.012, shape),
                0.74,
                0.88,
            )

    if "pet" not in layers:
        layers["pet"] = np.clip(850.0 + 38.0 * layers["era5land_mat"] - 45.0 * northness, 700.0, 1950.0)
    if "aridity" not in layers:
        layers["aridity"] = layers["era5land_map"] / np.maximum(layers["pet"], 100.0)

    # Fire layers if not already loaded from local I:\tele rasters
    if "mcd64_burn_count" not in layers:
        # Higher burn frequency on drier south-facing slopes (northness < 0) and mid-elevations
        fire_propensity = np.clip(1.2 - 0.9 * northness - 0.4 * (layers["aridity"] - 1.1), 0.0, 4.0)
        layers["mcd64_burn_count"] = np.round(np.clip(fire_propensity * 2.5, 0.0, 12.0))

    if "firms_hotspot_density" not in layers:
        layers["firms_hotspot_density"] = np.clip(
            0.35 * layers["mcd64_burn_count"] + 0.15 * np.maximum(-northness, 0.0),
            0.0,
            10.0,
        )

    if "dnbr_2022" not in layers:
        # Spectral burn severity correlated with burn count and south-aspect dryness
        layers["dnbr_2022"] = np.clip(
            45.0 * layers["mcd64_burn_count"] - 55.0 * northness + rng.normal(0.0, 25.0, shape),
            -150.0,
            780.0,
        )

    # Vegetation layers
    if "ndvi_max" not in layers:
        layers["ndvi_max"] = np.clip(
            0.68 + 0.06 * layers["aridity"] + 0.04 * northness - 0.00018 * np.maximum(layers["dnbr_2022"], 0.0),
            0.25,
            0.92,
        )
    if "ndvi_min" not in layers:
        layers["ndvi_min"] = np.clip(
            layers["ndvi_max"] - (0.22 if region != "uttarakhand" else 0.15) - 0.00025 * np.maximum(layers["dnbr_2022"], 0.0),
            0.12,
            0.78,
        )
    if "ndwi_dry" not in layers:
        layers["ndwi_dry"] = np.clip(
            0.18 + 0.08 * northness + 0.02 * (layers["twi"] - 6.0) - 0.015 * layers["mcd64_burn_count"],
            -0.25,
            0.55,
        )
    if "nbr_prefire" not in layers:
        layers["nbr_prefire"] = np.clip(0.85 * layers["ndvi_max"] - 0.05, 0.10, 0.85)

    # Landcover and soil classes
    if "tree_cover" not in layers:
        layers["tree_cover"] = np.clip(100.0 * (layers["ndvi_max"] - 0.15), 10.0, 95.0)
    if "esa_worldcover" not in layers:
        layers["esa_worldcover"] = np.where(layers["tree_cover"] >= 30.0, 10.0, 30.0)
    if "nrsc_lulc" not in layers:
        if region == "uttarakhand":
            # 12 = Temperate Banj Oak (>1800 m or cool north aspect), 9 = Subtropical Chir Pine
            layers["nrsc_lulc"] = np.where((elev >= 1750.0) | ((elev >= 1450.0) & (northness > 0.2)), 12.0, 9.0)
        else:
            # 3 = Tropical Moist Deciduous Sal, 4 = Tropical Dry Deciduous
            layers["nrsc_lulc"] = np.where(layers["aridity"] >= 1.15, 3.0, 4.0)
    if "hwsd_soil_class" not in layers:
        layers["hwsd_soil_class"] = np.where(elev > 1500.0, 1.0, np.where(layers["twi"] > 7.5, 2.0, 3.0))

    return {b: layers[b] for b in band_names if b in layers}
