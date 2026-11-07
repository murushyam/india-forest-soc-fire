"""Discovery and ingestion of pre-existing local rasters, shapefiles, and flux CSVs in I:\\tele."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


DEFAULT_TELE_DIR = Path(r"I:\tele")

LOCAL_RASTER_REGISTRY: dict[str, dict[str, str]] = {
    "uttarakhand": {
        "mcd64_burn_count": "Burned_Area_2022_Uttarakhand.tif",
        "firms_hotspot_density": "FIRMS_Fire_Intensity_Median_2022_Uttarakhand.tif",
        "dnbr_2022": "MODIS_NBR_PercentChange_Burned_JanJun2022.tif",
        "esa_worldcover": "Uttarakhand_LandCover_2022.tif",
        "nrsc_lulc": "Uttarakhand_LandCover_2022.tif",
    },
    "odisha_similipal": {
        "mcd64_burn_count": "burned_overlap_Odisha_2021_2025.tif",
        "esa_worldcover": "Odisha_Full_LC_Classes_2021.tif",
        "nrsc_lulc": "Odisha_Burned_LC_Classes_2021.tif",
    },
}

LOCAL_SHAPEFILE_REGISTRY: dict[str, str] = {
    "uttarakhand": r"UTTARAKHAND\UTTARAKHAND.shp",
    "odisha_similipal": r"odisha\odisha.shp",
}


def discover_local_rasters(
    region: str,
    tele_dir: str | Path = DEFAULT_TELE_DIR,
) -> dict[str, Path]:
    """Return mapping of `covariate_name -> Path` for existing GeoTIFFs in `tele_dir`."""
    base = Path(tele_dir)
    reg_map = LOCAL_RASTER_REGISTRY.get(region, {})
    found: dict[str, Path] = {}
    for cov_name, rel_name in reg_map.items():
        candidate = base / rel_name
        if candidate.exists():
            found[cov_name] = candidate
    return found


def discover_local_shapefile(
    region: str,
    tele_dir: str | Path = DEFAULT_TELE_DIR,
) -> Path | None:
    """Return path to regional boundary shapefile in `tele_dir` if present."""
    base = Path(tele_dir)
    rel = LOCAL_SHAPEFILE_REGISTRY.get(region)
    if rel is None:
        return None
    candidate = base / rel
    return candidate if candidate.exists() else None


def load_odisha_flux_timeseries(
    tele_dir: str | Path = DEFAULT_TELE_DIR,
    years: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
) -> pd.DataFrame:
    """Load and concatenate 2021-2025 Odisha burned-area GPP, ET, uWUE, SH, and LH time series."""
    base = Path(tele_dir)
    gpp_frames: list[pd.DataFrame] = []
    for yr in years:
        f_gpp = base / f"Odisha_BurnedArea_GPP_ET_uWUE_8day_{yr}.csv"
        if f_gpp.exists():
            df = pd.read_csv(f_gpp)
            df["year"] = yr
            df["region"] = "odisha_similipal"
            gpp_frames.append(df)
    if not gpp_frames:
        return pd.DataFrame()
    return pd.concat(gpp_frames, ignore_index=True)


def load_uttarakhand_flux_timeseries(
    tele_dir: str | Path = DEFAULT_TELE_DIR,
    years: tuple[int, ...] = (2022, 2023, 2024, 2025),
) -> pd.DataFrame:
    """Load 2022-2025 Uttarakhand land-cover stratified LH and SH turbulent heat flux CSVs."""
    base = Path(tele_dir)
    frames: list[pd.DataFrame] = []
    for yr in years:
        f_lh = base / f"LH_lc_uttarakhand_{yr}.csv"
        f_sh = base / f"SH_lc_uttarakhand_{yr}.csv"
        if f_lh.exists() and f_sh.exists():
            lh = pd.read_csv(f_lh)
            sh = pd.read_csv(f_sh)
            lh["flux_type"] = "LH"
            lh["year"] = yr
            sh["flux_type"] = "SH"
            sh["year"] = yr
            frames.extend([lh, sh])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["region"] = "uttarakhand"
    return out
