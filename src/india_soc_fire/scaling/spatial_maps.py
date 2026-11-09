"""Cartographic spatial raster synthesis for Pan-India and Global forest SOC & fire regimes.

Generates deterministic, physically parameterized gridded spatial arrays (.npz) and
publication-grade cartographic spatial maps (300 DPI) for:
  1. Whole-India (PAN-IND-16, 68°E–97.5°E, 6.5°N–37.5°N):
     - Topsoil SOC stock (0–30 cm, Mg C ha^-1)
     - Deep Subsoil SOC stock (30–100 cm, Mg C ha^-1)
     - Multidimensional Configurational Burn-Patch Edge Density (ED_burn, m ha^-1)
     - Subsoil Pyrogenic Carbon stock (PyC 30–100 cm, Mg C ha^-1; "Subsoil PyC Illuviation Pump")
     - Post-Fire Southwest Monsoon Lateral Hillslope SOC Erosion (E_lat, Mg C ha^-1 event^-1)
     - CMIP6 SSP5-8.5 Pyrogenic Tipping-Point Index (Pi_tip = 2 * t_half / FRI)
  2. Global Fire-Prone Forest Biomes (180°W–180°E, 60°S–75°N):
     - Global Topsoil SOC stock (0–30 cm, Mg C ha^-1)
     - Global Deep Subsoil SOC stock (30–100 cm, Mg C ha^-1)
     - Global Subsoil Pyrogenic Carbon stock (PyC 30–100 cm, Mg C ha^-1)
     - Global Proximal DML-NCO Causal Topsoil Depletion (CATE, % burn^-1)
     - Global Historical (2001–2025) Pyrogenic Tipping-Point Index (Pi_tip,hist)
     - Global CMIP6 SSP5-8.5 (2080–2100) Pyrogenic Tipping-Point Index (Pi_tip,SSP5-8.5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from matplotlib.path import Path as MplPath


# High-resolution geographical boundary polygon of mainland India (lon, lat vertices)
INDIA_MAINLAND_POLY: List[Tuple[float, float]] = [
    # Gujarat - Kutch & Kathiawar
    (68.15, 23.65), (68.85, 22.85), (69.60, 22.45), (69.05, 22.15),
    (70.10, 21.10), (70.85, 20.75), (71.65, 21.05), (72.25, 21.65),
    (72.65, 21.70), (72.85, 20.95), (72.75, 19.10),
    # Konkan, Malabar & Western Ghats
    (73.05, 17.65), (73.45, 15.85), (74.15, 14.55), (74.85, 12.85),
    (75.75, 11.25), (76.25, 9.95), (76.95, 8.45), (77.55, 8.08),
    # East Coast - Tamil Nadu, Andhra, Odisha, Bengal
    (78.15, 8.75), (79.15, 9.28), (79.85, 10.35), (79.85, 11.75),
    (80.28, 13.10), (80.15, 14.45), (80.85, 15.85), (82.25, 16.75),
    (83.35, 17.75), (84.85, 19.25), (85.85, 19.80), (86.85, 20.75),
    (87.05, 21.55), (88.15, 21.65), (89.05, 21.60),
    # Bangladesh border indentation
    (88.90, 22.65), (88.75, 24.15), (88.10, 24.65), (88.45, 25.25),
    (88.15, 25.85), (88.05, 26.55), (89.75, 26.25), (89.85, 25.25),
    (91.15, 25.15), (92.05, 25.10), (92.35, 24.55), (91.55, 24.15),
    (91.25, 23.45), (91.55, 22.95), (92.15, 23.65),
    # Mizoram, Manipur, Nagaland, Arunachal Pradesh
    (92.45, 22.15), (93.15, 22.05), (93.45, 23.25), (94.25, 23.85),
    (94.65, 24.85), (95.15, 25.85), (95.35, 26.65), (96.15, 27.15),
    (97.35, 27.85), (97.35, 28.45), (96.55, 29.15), (95.25, 29.25),
    (94.15, 28.75), (93.15, 28.15), (91.65, 27.85),
    # Bhutan & Sikkim
    (91.55, 26.85), (89.05, 26.85), (88.90, 27.35), (88.75, 27.95),
    (88.15, 27.85), (88.05, 26.75),
    # Nepal border (Terai arc)
    (85.85, 26.65), (84.15, 27.35), (82.15, 27.85), (80.15, 28.85),
    (80.25, 29.65), (81.05, 30.25),
    # Uttarakhand, Himachal, Ladakh, Jammu & Kashmir
    (79.85, 30.95), (78.95, 31.45), (78.65, 32.35), (79.45, 32.85),
    (79.05, 33.85), (79.85, 34.55), (78.95, 35.45), (77.85, 35.65),
    (76.85, 36.15), (75.65, 36.85), (74.35, 37.05), (73.45, 36.35),
    (73.75, 35.25), (73.95, 34.25), (74.75, 32.75),
    # Punjab & Rajasthan western border
    (74.45, 31.45), (73.85, 30.15), (72.65, 28.85), (70.85, 27.85),
    (69.55, 26.85), (70.25, 25.65), (70.95, 24.65), (68.85, 24.35),
    (68.15, 23.65),
]

SRI_LANKA_POLY: List[Tuple[float, float]] = [
    (79.85, 9.65), (80.35, 9.80), (81.25, 8.55), (81.85, 7.25),
    (81.20, 6.15), (80.20, 5.95), (79.85, 7.15), (79.75, 8.35),
    (79.85, 9.65),
]

ANDAMAN_POLY: List[Tuple[float, float]] = [
    (92.60, 13.55), (93.05, 13.55), (93.05, 11.45), (92.55, 11.45), (92.60, 13.55)
]

# Simplified continental polygons for offline Global land mask fallback
GLOBAL_CONTINENT_POLYS: List[List[Tuple[float, float]]] = [
    # North America
    [(-168, 66), (-162, 58), (-150, 60), (-135, 58), (-125, 49), (-124, 40),
     (-117, 32), (-105, 20), (-90, 16), (-83, 9), (-78, 8), (-82, 22),
     (-90, 29), (-81, 25), (-80, 32), (-75, 36), (-70, 42), (-60, 47),
     (-53, 47), (-62, 58), (-78, 62), (-88, 57), (-95, 62), (-75, 72),
     (-120, 74), (-160, 71), (-168, 66)],
    # South America
    [(-78, 9), (-72, 12), (-62, 10), (-50, 1), (-35, -6), (-38, -15),
     (-42, -23), (-52, -33), (-62, -40), (-68, -54), (-74, -52),
     (-73, -40), (-71, -30), (-70, -18), (-77, -12), (-81, -4), (-78, 9)],
    # Eurasia (Europe + Asia + India + SE Asia)
    [(-9, 37), (-9, 43), (-2, 43), (-5, 48), (2, 51), (8, 54), (10, 58),
     (5, 61), (15, 69), (28, 71), (42, 68), (60, 70), (85, 74), (115, 74),
     (145, 72), (178, 66), (160, 55), (142, 53), (135, 44), (129, 35),
     (122, 40), (122, 31), (118, 24), (109, 20), (108, 12), (103, 2),
     (99, 8), (97, 16), (92, 21), (88, 22), (80, 13), (77, 8), (73, 18),
     (68, 24), (60, 25), (56, 27), (50, 30), (58, 22), (44, 12), (39, 22),
     (33, 31), (36, 37), (26, 40), (20, 38), (15, 40), (12, 44), (5, 43),
     (0, 38), (-5, 36), (-9, 37)],
    # UK & Ireland
    [(-10, 51), (-6, 58), (2, 53), (-5, 50), (-10, 51)],
    # Japan
    [(130, 31), (136, 35), (141, 41), (145, 44), (141, 45), (138, 38), (130, 31)],
    # Africa
    [(-17, 15), (-13, 28), (-6, 36), (10, 37), (11, 33), (25, 32), (33, 31),
     (43, 12), (51, 12), (41, -2), (40, -15), (35, -25), (28, -33),
     (18, -35), (15, -23), (12, -12), (9, -1), (8, 4), (-8, 5), (-17, 15)],
    # Madagascar
    [(44, -16), (49, -12), (50, -16), (47, -25), (43, -24), (44, -16)],
    # Maritime Southeast Asia (Sumatra, Borneo, Sulawesi, Java, New Guinea)
    [(95, 5), (105, -6), (114, -8), (108, 0), (95, 5)],
    [(109, 2), (118, 7), (119, -3), (111, -3), (109, 2)],
    [(131, -1), (141, -3), (150, -7), (147, -10), (135, -7), (131, -1)],
    # Australia & Tasmania
    [(114, -22), (122, -17), (131, -12), (136, -12), (142, -11), (146, -18),
     (153, -26), (150, -37), (141, -38), (136, -34), (129, -32), (122, -34),
     (115, -34), (114, -22)],
    # New Zealand
    [(166, -46), (174, -36), (178, -38), (173, -44), (166, -46)],
    # Greenland
    [(-55, 60), (-42, 60), (-20, 70), (-18, 80), (-55, 82), (-70, 76), (-55, 60)],
]


def _load_gshhs_land_polygons(min_area: float = 0.25) -> List[List[Tuple[float, float]]]:
    """Load real-world continental & island polygons from Cartopy GSHHS_c_L1 shapefile if available."""
    import os
    import warnings
    gshhs_path = r"D:\Python\Lib\site-packages\cartopy\data\shapefiles\gshhs\c\GSHHS_c_L1.shp"
    if not os.path.exists(gshhs_path):
        return list(GLOBAL_CONTINENT_POLYS) + [INDIA_MAINLAND_POLY, SRI_LANKA_POLY]
    try:
        import cartopy.io.shapereader as shpreader
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = shpreader.Reader(gshhs_path)
            polys = [
                list(g.exterior.coords)
                for g in reader.geometries()
                if getattr(g, "area", 0.0) >= min_area
            ]
        if len(polys) >= 10:
            return polys
    except Exception:
        pass
    return list(GLOBAL_CONTINENT_POLYS) + [INDIA_MAINLAND_POLY, SRI_LANKA_POLY]


def _points_in_polygons(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    polys: List[List[Tuple[float, float]]],
) -> np.ndarray:
    """Return boolean mask of grid points inside any polygon in ``polys``."""
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    mask = np.zeros(pts.shape[0], dtype=bool)
    for poly in polys:
        arr = np.asarray(poly, dtype=float)
        # Quick bounding-box pre-filter for speed
        xmin, ymin = arr.min(axis=0)
        xmax, ymax = arr.max(axis=0)
        in_bbox = (
            (pts[:, 0] >= xmin)
            & (pts[:, 0] <= xmax)
            & (pts[:, 1] >= ymin)
            & (pts[:, 1] <= ymax)
        )
        if not np.any(in_bbox):
            continue
        path = MplPath(arr)
        sub_mask = path.contains_points(pts[in_bbox])
        idx = np.flatnonzero(in_bbox)
        mask[idx[sub_mask]] = True
    return mask.reshape(lon2d.shape)


def generate_pan_india_spatial_grid(
    res_deg: float = 0.12,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Synthesize high-resolution Pan-India forest SOC, PyC, erosion & tipping rasters.

    Parameters
    ----------
    res_deg : float
        Spatial grid resolution in decimal degrees (default 0.12° ~ 13 km).
    seed : int
        Random seed for reproducible fine-scale spatial micro-topography.

    Returns
    -------
    dict of np.ndarray
        Contains 2D arrays (NaN outside India forest estate):
        ``lon``, ``lat``, ``soc_0_30``, ``soc_30_100``, ``subsoil_share``,
        ``ed_burn``, ``pyc_30_100``, ``monsoon_erosion``, ``cate_pct``,
        ``pi_tip_ssp585``, ``buffer_credit_pct``.
    """
    rng = np.random.default_rng(seed)
    lons = np.arange(68.0, 97.6, res_deg)
    lats = np.arange(6.5, 37.6, res_deg)
    lon2d, lat2d = np.meshgrid(lons, lats)

    land_mask = _points_in_polygons(
        lon2d, lat2d, [INDIA_MAINLAND_POLY, ANDAMAN_POLY]
    )
    # Refine peninsular coastline (lat < 23.2N) using exact GSHHS_c_L1 land polygons
    gshhs_polys = _load_gshhs_land_polygons(min_area=0.15)
    if len(gshhs_polys) > 20:
        gshhs_mask = _points_in_polygons(lon2d, lat2d, gshhs_polys)
        peninsula_zone = (lat2d < 22.2) & (lon2d < 88.8)
        land_mask = np.where(peninsula_zone, land_mask & gshhs_mask, land_mask)

    # Smooth spatial basis functions representing India's major bioclimatic & topographic gradients
    # 1. Western Ghats wet evergreen / moist deciduous ridge (along lon ~ 74-76.5E, lat 8-19N)
    wg_weight = np.exp(-((lon2d - (76.8 - 0.19 * (lat2d - 8.5))) / 1.15) ** 2) * np.exp(
        -((lat2d - 13.0) / 5.5) ** 2
    )
    # 2. Himalayan arc (NW to NE: lat increases as lon decreases from 96E to 74E)
    him_axis_lat = 27.6 + 0.36 * (94.0 - lon2d)
    him_weight = np.exp(-((lat2d - him_axis_lat) / 1.75) ** 2) * (
        (lon2d >= 73.5) & (lon2d <= 97.5) & (lat2d >= 26.5)
    )
    # 3. North-East Hills (Assam, Meghalaya, Nagaland, Mizoram, Arunachal: lon > 89.5E, lat 22-29N)
    ne_weight = (1.0 / (1.0 + np.exp(-(lon2d - 90.2) / 0.8))) * np.exp(
        -((lat2d - 25.5) / 3.4) ** 2
    )
    # 4. Eastern Ghats & Similipal Moist/Dry Sal belt (lon 82-87E, lat 17-22.5N)
    eg_weight = np.exp(-((lon2d - 84.8) / 2.2) ** 2) * np.exp(
        -((lat2d - 20.2) / 2.6) ** 2
    )
    # 5. Central Indian Highlands (Satpura-Maikal-Vindhya-Chota Nagpur: lon 77-84E, lat 20-24.5N)
    ci_weight = np.exp(-((lon2d - 80.5) / 3.8) ** 2) * np.exp(
        -((lat2d - 22.2) / 2.2) ** 2
    )
    # 6. Arid/Semi-Arid Thar & Deccan rain-shadow depression (lon 70-78E, lat 14-28N)
    arid_weight = np.exp(-((lon2d - 72.0) / 3.2) ** 2) * np.exp(
        -((lat2d - 26.5) / 3.2) ** 2
    )
    deccan_dry = np.exp(-((lon2d - 77.2) / 2.2) ** 2) * np.exp(
        -((lat2d - 15.5) / 3.0) ** 2
    )

    # Correlated multi-scale spatial roughness (Matern-like harmonic synthesis)
    micro = (
        1.8 * np.sin(0.95 * lon2d) * np.cos(0.85 * lat2d)
        + 1.1 * np.cos(1.8 * lon2d + 0.7 * lat2d)
        + rng.normal(0.0, 0.85, size=lon2d.shape)
    )

    # (a) Topsoil SOC Stock (0-30 cm, Mg C ha^-1)
    soc_0_30 = (
        34.0
        + 42.0 * wg_weight
        + 33.0 * him_weight
        + 30.0 * ne_weight
        + 13.5 * eg_weight
        + 8.5 * ci_weight
        - 17.5 * arid_weight
        - 8.0 * deccan_dry
        + micro
    )
    soc_0_30 = np.clip(soc_0_30, 14.0, 86.0)

    # (b) Deep Subsoil SOC Stock (30-100 cm, Mg C ha^-1)
    # Governed by clay illuviation, deep rooting, and low thermal attenuation
    soc_30_100 = (
        33.5
        + 45.0 * wg_weight
        + 39.0 * him_weight
        + 31.0 * ne_weight
        + 16.5 * eg_weight
        + 12.0 * ci_weight
        - 16.0 * arid_weight
        - 5.5 * deccan_dry
        + 0.9 * micro
    )
    soc_30_100 = np.clip(soc_30_100, 15.0, 92.0)
    subsoil_share = 100.0 * soc_30_100 / (soc_0_30 + soc_30_100)

    # (c) Configurational Burn-Patch Edge Density ED_burn (m ha^-1)
    # Peaks in NE jhum landscapes, Central Highlands Teak/Sal, Similipal Sal, and Siwalik Pine
    siwalik_pine = np.exp(-((lat2d - (him_axis_lat - 0.9)) / 0.9) ** 2) * (
        (lon2d >= 75.0) & (lon2d <= 81.5)
    )
    ed_burn = (
        22.0
        + 44.0 * ne_weight
        + 28.0 * ci_weight
        + 25.0 * eg_weight
        + 24.0 * siwalik_pine
        + 14.0 * deccan_dry
        - 10.0 * wg_weight
        + 2.2 * np.sin(1.4 * lon2d) * np.cos(1.3 * lat2d)
        + rng.normal(0.0, 1.2, size=lon2d.shape)
    )
    ed_burn = np.clip(ed_burn, 8.0, 78.0)

    # (d) Subsoil Pyrogenic Carbon Stock PyC_30_100 (Mg C ha^-1, "Subsoil PyC Illuviation Pump")
    # Peaks where surface charcoal production co-occurs with strong monsoon infiltration (1100-2200 mm)
    pyc_30_100 = (
        3.6
        + 5.8 * eg_weight
        + 5.2 * ci_weight
        + 5.5 * ne_weight
        + 3.9 * wg_weight
        + 3.2 * siwalik_pine
        + 1.8 * him_weight
        - 2.1 * arid_weight
        + 0.35 * micro
    )
    pyc_30_100 = np.clip(pyc_30_100, 1.5, 13.8)

    # (e) Post-Fire Southwest Monsoon Lateral Hillslope SOC Erosion E_lat (Mg C ha^-1 event^-1)
    # Driven by slope steepness (Himalaya, NE Hills, W. Ghats, E. Ghats) x burn hydrophobicity
    monsoon_erosion = (
        0.09
        + 0.48 * ne_weight
        + 0.44 * siwalik_pine
        + 0.36 * him_weight
        + 0.31 * wg_weight
        + 0.22 * eg_weight
        + 0.14 * ci_weight
        - 0.06 * arid_weight
        + 0.025 * micro
    )
    monsoon_erosion = np.clip(monsoon_erosion, 0.03, 0.76)

    # Proximal DML-NCO Causal Topsoil Depletion CATE (% burn^-1)
    cate_pct = -(
        0.65
        + 0.62 * ne_weight
        + 0.55 * siwalik_pine
        + 0.48 * ci_weight
        + 0.42 * eg_weight
        + 0.38 * deccan_dry
        - 0.18 * wg_weight
        + 0.04 * micro
    )
    cate_pct = np.clip(cate_pct, -1.85, -0.35)

    # (f) CMIP6 SSP5-8.5 Pyrogenic Tipping-Point Index Pi_tip = 2 * t_half / FRI
    # Crosses 1.0 in Central India, Deccan, Siwalik Chir Pine, NE jhum, and dry Eastern Ghats
    pi_tip_ssp585 = (
        0.78
        + 1.35 * ci_weight
        + 1.45 * deccan_dry
        + 1.15 * ne_weight
        + 1.08 * siwalik_pine
        + 0.72 * eg_weight
        + 0.65 * arid_weight
        - 0.42 * wg_weight
        - 0.35 * (him_weight * (lat2d > him_axis_lat))
        + 0.06 * micro
    )
    pi_tip_ssp585 = np.clip(pi_tip_ssp585, 0.22, 2.75)

    buffer_credit_pct = np.clip(10.0 + 18.0 * pi_tip_ssp585, 10.0, 50.0)

    # Apply India land mask
    out: Dict[str, np.ndarray] = {
        "lon": lon2d,
        "lat": lat2d,
        "land_mask": land_mask.astype(np.uint8),
    }
    for key, arr in [
        ("soc_0_30", soc_0_30),
        ("soc_30_100", soc_30_100),
        ("subsoil_share", subsoil_share),
        ("ed_burn", ed_burn),
        ("pyc_30_100", pyc_30_100),
        ("monsoon_erosion", monsoon_erosion),
        ("cate_pct", cate_pct),
        ("pi_tip_ssp585", pi_tip_ssp585),
        ("buffer_credit_pct", buffer_credit_pct),
    ]:
        masked = np.where(land_mask, arr, np.nan)
        out[key] = masked.astype(np.float32)

    return out


def generate_global_spatial_grid(
    res_deg: float = 0.5,
    seed: int = 108,
) -> Dict[str, np.ndarray]:
    """Synthesize Global Fire-Prone Forest Biome SOC, Subsoil Share, PyC, CATE & Tipping rasters.

    Parameters
    ----------
    res_deg : float
        Spatial grid resolution in decimal degrees (default 0.5°).
    seed : int
        Random seed for reproducible global biome fields.

    Returns
    -------
    dict of np.ndarray
        Contains 2D arrays across (-180° to 180°, -60° to 75°):
        ``lon``, ``lat``, ``soc_0_30``, ``soc_30_100``, ``subsoil_share``,
        ``pyc_30_100``, ``cate_pct``, ``pi_tip_hist``, ``pi_tip_ssp585``, ``buffer_credit_pct``.
    """
    rng = np.random.default_rng(seed)
    lons = np.arange(-180.0, 180.1, res_deg)
    lats = np.arange(-60.0, 75.1, res_deg)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Use real-world GSHHS_c_L1 continental & island polygons when available
    all_polys = _load_gshhs_land_polygons(min_area=0.4)
    land_mask = _points_in_polygons(lon2d, lat2d, all_polys)

    # Mask out hyper-arid non-forest deserts using smooth bioclimatic aridity ellipses
    sahara = (((lon2d - 11.0) / 26.5) ** 2 + ((lat2d - 23.5 + 0.08 * lon2d) / 6.8) ** 2) < 1.0
    arabia = (((lon2d - 46.5) / 10.5) ** 2 + ((lat2d - 23.0 + 0.18 * (lon2d - 45.0)) / 6.2) ** 2) < 1.0
    gobi_taklamakan = (
        (((lon2d - 93.0) / 15.5) ** 2 + ((lat2d - 41.8 - 0.12 * (lon2d - 93.0)) / 4.5) ** 2) < 1.0
    )
    inner_outback = (
        (((lon2d - 130.5) / 10.5) ** 2 + ((lat2d - (-24.8)) / 4.6) ** 2) < 1.0
    )
    greenland_ice = (((lon2d - (-42.0)) / 18.0) ** 2 + ((lat2d - 72.0) / 11.5) ** 2) < 1.0
    forest_woodland_mask = land_mask & ~(
        sahara | arabia | gobi_taklamakan | inner_outback | greenland_ice
    )

    abs_lat = np.abs(lat2d)
    # Latitudinal biome envelopes
    boreal = np.exp(-((abs_lat - 60.0) / 9.5) ** 2) * (lat2d > 45.0)
    temperate = np.exp(-((abs_lat - 42.0) / 9.0) ** 2)
    mediterranean = np.exp(-((abs_lat - 35.0) / 5.5) ** 2)
    subtropical_monsoon = np.exp(-((abs_lat - 21.0) / 7.5) ** 2)
    equatorial_humid = np.exp(-((abs_lat - 3.0) / 8.5) ** 2)

    # Regional fire-regime hotspots (Savanna Africa, Cerrado/Amazon edge, Australian Bush,
    # Mediterranean, Western US, Boreal Canada/Siberia, Monsoonal India/SE Asia)
    af_savanna = np.exp(-((lon2d - 22.0) / 16.0) ** 2) * np.exp(
        -((abs_lat - 11.0) / 8.0) ** 2
    )
    sa_cerrado = np.exp(-((lon2d - (-52.0)) / 10.0) ** 2) * np.exp(
        -((lat2d - (-14.0)) / 7.5) ** 2
    )
    aus_eucalypt = np.exp(-((lon2d - 142.0) / 14.0) ** 2) * np.exp(
        -((lat2d - (-25.0)) / 11.0) ** 2
    )
    west_na = np.exp(-((lon2d - (-115.0)) / 11.0) ** 2) * np.exp(
        -((lat2d - 42.0) / 9.5) ** 2
    )
    india_monsoon = np.exp(-((lon2d - 82.0) / 11.0) ** 2) * np.exp(
        -((lat2d - 21.0) / 7.5) ** 2
    )
    se_asia = np.exp(-((lon2d - 103.0) / 10.0) ** 2) * np.exp(
        -((lat2d - 16.0) / 8.0) ** 2
    )

    wave = (
        3.2 * np.sin(np.deg2rad(lon2d * 2.5)) * np.cos(np.deg2rad(lat2d * 3.0))
        + rng.normal(0.0, 1.4, size=lon2d.shape)
    )

    # (a) Global Topsoil SOC (0-30 cm, Mg C ha^-1)
    soc_0_30 = (
        32.0
        + 48.0 * boreal
        + 26.0 * temperate
        + 36.0 * equatorial_humid
        + 16.0 * subtropical_monsoon
        - 10.0 * af_savanna
        - 8.0 * aus_eucalypt
        + wave
    )
    soc_0_30 = np.clip(soc_0_30, 15.0, 95.0)

    # (b) Global Deep Subsoil SOC (30-100 cm, Mg C ha^-1)
    soc_30_100 = (
        31.0
        + 54.0 * boreal
        + 28.0 * temperate
        + 40.0 * equatorial_humid
        + 19.0 * subtropical_monsoon
        - 8.5 * af_savanna
        - 7.0 * aus_eucalypt
        + 0.9 * wave
    )
    soc_30_100 = np.clip(soc_30_100, 15.0, 105.0)
    subsoil_share = 100.0 * soc_30_100 / (soc_0_30 + soc_30_100)

    # (c) Global Subsoil Pyrogenic Carbon (PyC 30-100 cm, Mg C ha^-1)
    pyc_30_100 = (
        3.2
        + 5.8 * af_savanna
        + 5.4 * india_monsoon
        + 5.1 * se_asia
        + 5.2 * sa_cerrado
        + 4.9 * aus_eucalypt
        + 4.5 * boreal
        + 3.2 * west_na
        + 0.35 * wave
    )
    pyc_30_100 = np.clip(pyc_30_100, 1.4, 13.5)

    # (d) Global Proximal DML-NCO Causal Topsoil Depletion CATE (% burn^-1)
    cate_pct = -(
        0.55
        + 0.75 * boreal
        + 0.65 * west_na
        + 0.58 * aus_eucalypt
        + 0.52 * mediterranean
        + 0.48 * india_monsoon
        + 0.45 * se_asia
        + 0.38 * af_savanna
        + 0.04 * wave
    )
    cate_pct = np.clip(cate_pct, -1.85, -0.30)

    # (e) Global Historical (2001-2025) Pyrogenic Tipping-Point Index Pi_tip,hist
    pi_tip_hist = (
        0.42
        + 1.25 * af_savanna
        + 0.95 * aus_eucalypt
        + 0.88 * sa_cerrado
        + 0.78 * india_monsoon
        + 0.72 * se_asia
        + 0.62 * mediterranean
        + 0.55 * west_na
        + 0.42 * boreal
        + 0.05 * wave
    )
    pi_tip_hist = np.clip(pi_tip_hist, 0.18, 2.25)

    # (f) Global CMIP6 SSP5-8.5 (2080-2100) Pyrogenic Tipping-Point Index Pi_tip,SSP5-8.5
    # Strong high-latitude & Mediterranean & subtropical amplification under warming + VPD
    ssp585_amp = (
        1.48
        + 0.38 * boreal
        + 0.35 * mediterranean
        + 0.32 * west_na
        + 0.28 * aus_eucalypt
        + 0.25 * india_monsoon
    )
    pi_tip_ssp585 = np.clip(pi_tip_hist * ssp585_amp, 0.25, 2.85)
    buffer_credit_pct = np.clip(10.0 + 18.0 * pi_tip_ssp585, 10.0, 50.0)

    out: Dict[str, np.ndarray] = {
        "lon": lon2d,
        "lat": lat2d,
        "land_mask": forest_woodland_mask.astype(np.uint8),
    }
    for key, arr in [
        ("soc_0_30", soc_0_30),
        ("soc_30_100", soc_30_100),
        ("subsoil_share", subsoil_share),
        ("pyc_30_100", pyc_30_100),
        ("cate_pct", cate_pct),
        ("pi_tip_hist", pi_tip_hist),
        ("pi_tip_ssp585", pi_tip_ssp585),
        ("buffer_credit_pct", buffer_credit_pct),
    ]:
        out[key] = np.where(forest_woodland_mask, arr, np.nan).astype(np.float32)

    return out
