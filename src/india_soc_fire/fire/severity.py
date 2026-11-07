"""Spectral burn severity indices (dNBR, RdNBR, RBR), phenology offset, and Indian biome calibration."""
from __future__ import annotations

import numpy as np

# Standard USGS Key & Benson (2006) dNBR thresholds (unit scale; x1000 values > 2.0 are auto-scaled)
DNBR_CLASSES: list[tuple[str, float, float]] = [
    ("unburned", -0.50, 0.10),
    ("low", 0.10, 0.27),
    ("moderate_low", 0.27, 0.44),
    ("moderate_high", 0.44, 0.66),
    ("high", 0.66, 1.30),
]

# Biome-calibrated dNBR thresholds (unit scale) accounting for dry-season leaf shedding
# in tropical moist deciduous Sal (Similipal) vs subtropical Chir pine & Banj oak (Uttarakhand)
BIOME_DNBR_CLASSES: dict[str, list[tuple[str, float, float]]] = {
    "tropical_moist_deciduous": [
        ("unburned", -0.50, 0.085),
        ("low", 0.085, 0.220),
        ("moderate_low", 0.220, 0.370),
        ("moderate_high", 0.370, 0.540),
        ("high", 0.540, 1.30),
    ],
    "subtropical_chir_pine": [
        ("unburned", -0.50, 0.095),
        ("low", 0.095, 0.255),
        ("moderate_low", 0.255, 0.420),
        ("moderate_high", 0.420, 0.630),
        ("high", 0.630, 1.30),
    ],
    "temperate_banj_oak": [
        ("unburned", -0.50, 0.105),
        ("low", 0.105, 0.275),
        ("moderate_low", 0.275, 0.450),
        ("moderate_high", 0.450, 0.670),
        ("high", 0.670, 1.30),
    ],
}


def _to_unit_dnbr(val: float | np.ndarray) -> float | np.ndarray:
    """Convert x1000 scaled dNBR values (|v| > 2.0) to unit scale [-2.0, 2.0]."""
    if np.ndim(val) == 0:
        v = float(val)
        return v / 1000.0 if abs(v) > 2.0 else v
    arr = np.asarray(val, dtype=float)
    return np.where(np.abs(arr) > 2.0, arr / 1000.0, arr)


def dnbr_class(value: float, biome: str | None = None) -> str:
    """Classify a scalar dNBR value (unit scale or x1000 scale) into a burn-severity class."""
    v = float(_to_unit_dnbr(value))
    table = BIOME_DNBR_CLASSES.get(biome or "", DNBR_CLASSES)
    for name, lo, hi in table:
        if lo <= v < hi:
            return name
    return "out_of_range"


def classify_burn_severity(
    values: np.ndarray | float,
    biome: str | None = None,
) -> np.ndarray | str:
    """Vectorized or scalar burn-severity classification using biome-calibrated dNBR thresholds."""
    if np.ndim(values) == 0:
        return dnbr_class(float(values), biome=biome)
    arr = np.asarray(_to_unit_dnbr(values), dtype=float)
    table = BIOME_DNBR_CLASSES.get(biome or "", DNBR_CLASSES)
    out = np.full(arr.shape, "out_of_range", dtype=object)
    for name, lo, hi in table:
        mask = (arr >= lo) & (arr < hi)
        out[mask] = name
    return out


def compute_dnbr(
    nbr_prefire: np.ndarray | float,
    nbr_postfire: np.ndarray | float,
) -> np.ndarray | float:
    """Differenced Normalized Burn Ratio (x1000): (NBR_pre - NBR_post) * 1000."""
    return (np.asarray(nbr_prefire, dtype=float) - np.asarray(nbr_postfire, dtype=float)) * 1000.0


def compute_rdnbr(
    nbr_prefire: np.ndarray | float,
    dnbr: np.ndarray | float,
) -> np.ndarray:
    """Relative dNBR (Miller & Thode, 2007): dNBR / sqrt(|NBR_pre|), with safe floor at |NBR_pre| >= 0.05."""
    pre = np.asarray(nbr_prefire, dtype=float)
    pre_unit = pre / 1000.0 if np.nanmax(np.abs(pre)) > 2.0 else pre
    denom = np.sqrt(np.maximum(np.abs(pre_unit), 0.05))
    return np.asarray(dnbr, dtype=float) / denom


def compute_rbr(
    nbr_prefire: np.ndarray | float,
    dnbr: np.ndarray | float,
) -> np.ndarray:
    """Relativized Burn Ratio (Parks et al., 2014): dNBR / (NBR_pre + 1.001)."""
    pre = np.asarray(nbr_prefire, dtype=float)
    pre_unit = pre / 1000.0 if np.nanmax(np.abs(pre)) > 2.0 else pre
    denom = np.maximum(pre_unit + 1.001, 0.1)
    return np.asarray(dnbr, dtype=float) / denom


def apply_phenology_offset(
    dnbr: np.ndarray,
    unburned_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Subtract median dNBR of unburned reference pixels to remove seasonal phenology bias."""
    arr = np.asarray(dnbr, dtype=float)
    mask = np.asarray(unburned_mask, dtype=bool) & np.isfinite(arr)
    offset = float(np.median(arr[mask])) if mask.any() else 0.0
    return arr - offset, offset
