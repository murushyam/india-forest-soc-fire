"""Terrain covariate derivation: Horn (1981) 3x3 slope, aspect (northness/eastness), and TWI."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def compute_slope_aspect_twi(
    dem_m: np.ndarray,
    res_m: float = 30.0,
) -> dict[str, np.ndarray]:
    """Compute slope (deg), aspect (deg), northness, eastness, and Topographic Wetness Index (TWI).

    Parameters
    ----------
    dem_m : np.ndarray
        2D elevation array in metres above sea level, shape (H, W).
    res_m : float, default 30.0
        Grid cell resolution in metres.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``elevation``, ``slope``, ``aspect``,
        ``aspect_northness``, ``aspect_eastness``, and ``twi``.
    """
    z = np.asarray(dem_m, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"dem_m must be 2D, got shape {z.shape}")
    if res_m <= 0:
        raise ValueError(f"res_m must be positive, got {res_m}")

    # Pad with edge values for 3x3 Horn (1981) finite-difference operator
    zp = np.pad(z, pad_width=1, mode="edge")
    dz_dx = (
        (zp[:-2, 2:] + 2.0 * zp[1:-1, 2:] + zp[2:, 2:])
        - (zp[:-2, :-2] + 2.0 * zp[1:-1, :-2] + zp[2:, :-2])
    ) / (8.0 * res_m)
    dz_dy = (
        (zp[:-2, :-2] + 2.0 * zp[:-2, 1:-1] + zp[:-2, 2:])
        - (zp[2:, :-2] + 2.0 * zp[2:, 1:-1] + zp[2:, 2:])
    ) / (8.0 * res_m)

    grad_mag = np.hypot(dz_dx, dz_dy)
    slope_rad = np.arctan(grad_mag)
    slope_deg = np.degrees(slope_rad)

    # Aspect in degrees clockwise from North [0, 360)
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = np.mod(np.degrees(aspect_rad), 360.0)
    northness = np.cos(np.radians(aspect_deg))
    eastness = np.sin(np.radians(aspect_deg))

    # Specific catchment area approximation using local relief convergence & upslope drainage
    local_mean = uniform_filter(z, size=7, mode="nearest")
    concavity = np.maximum(local_mean - z, 0.0)
    specific_area = res_m * (1.0 + 15.0 * (concavity / (np.std(z) + 1.0)))
    tan_beta = np.maximum(np.tan(slope_rad), 0.005)
    twi = np.log(specific_area / tan_beta)

    return {
        "elevation": z,
        "slope": slope_deg,
        "aspect": aspect_deg,
        "aspect_northness": northness,
        "aspect_eastness": eastness,
        "twi": twi,
    }
