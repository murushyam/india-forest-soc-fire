"""Raster co-registration, UTM grid alignment, and focal 3x3 point extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


def align_array_to_shape(
    arr: np.ndarray,
    target_shape: tuple[int, int],
    categorical: bool = False,
) -> np.ndarray:
    """Resample a 2D array to `target_shape` using bilinear (order=1) or nearest (order=0) interpolation."""
    src = np.asarray(arr, dtype=float)
    if src.shape == target_shape:
        return src.copy()
    zoom_factors = (target_shape[0] / src.shape[0], target_shape[1] / src.shape[1])
    order = 0 if categorical else 1
    nan_mask = ~np.isfinite(src)
    if nan_mask.any():
        fill_val = float(np.nanmedian(src)) if (~nan_mask).any() else 0.0
        src_filled = np.where(nan_mask, fill_val, src)
        out = zoom(src_filled, zoom_factors, order=order, mode="nearest")
    else:
        out = zoom(src, zoom_factors, order=order, mode="nearest")
    return out


def read_and_align_geotiff(
    path: str | Path,
    dst_crs: str,
    dst_bounds: tuple[float, float, float, float],
    dst_shape: tuple[int, int],
    categorical: bool = False,
) -> np.ndarray:
    """Read a single-band GeoTIFF and warp/resample it onto `(dst_crs, dst_bounds, dst_shape)`.

    Uses `rasterio.warp.reproject` when `rasterio` is installed; otherwise reads
    GeoTIFF `ModelTiepointTag` (33922), `ModelPixelScaleTag` (33550), and
    `GDAL_NODATA` (42113) via `tifffile` to crop to `dst_bounds` and resample.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raster file not found: {p}")

    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject

        dst_transform = from_bounds(*dst_bounds, width=dst_shape[1], height=dst_shape[0])
        dst_arr = np.full(dst_shape, np.nan, dtype=np.float32)
        resampling = Resampling.nearest if categorical else Resampling.bilinear

        with rasterio.open(p) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
        return dst_arr.astype(float)
    except ImportError:
        import tifffile

        with tifffile.TiffFile(str(p)) as tf:
            page = tf.pages[0]
            arr = page.asarray().astype(float)
            if arr.ndim == 3:
                arr = arr[0] if arr.shape[0] < arr.shape[2] else arr[:, :, 0]

            tags = page.tags
            if 42113 in tags:  # GDAL_NODATA
                try:
                    nd_val = float(str(tags[42113].value).strip().rstrip("\x00"))
                    arr = np.where(np.isclose(arr, nd_val) | (arr <= -9000.0), np.nan, arr)
                except ValueError:
                    arr = np.where(arr <= -9000.0, np.nan, arr)
            else:
                arr = np.where(arr <= -9000.0, np.nan, arr)

            # Crop to dst_bounds if GeoTIFF tiepoint & pixel scale tags exist
            if 33550 in tags and 33922 in tags:
                scale = tags[33550].value
                tie = tags[33922].value
                sx, sy = float(scale[0]), float(scale[1])
                lon0, lat0 = float(tie[3]), float(tie[4])
                h_src, w_src = arr.shape
                min_x, min_y, max_x, max_y = dst_bounds
                # Only crop if source tiepoint is in geographic degrees matching dst_bounds
                if -180.0 <= lon0 <= 180.0 and -90.0 <= lat0 <= 90.0 and sx > 0 and sy > 0:
                    c0 = int(np.clip(np.floor((min_x - lon0) / sx), 0, w_src - 1))
                    c1 = int(np.clip(np.ceil((max_x - lon0) / sx), c0 + 1, w_src))
                    r0 = int(np.clip(np.floor((lat0 - max_y) / sy), 0, h_src - 1))
                    r1 = int(np.clip(np.ceil((lat0 - min_y) / sy), r0 + 1, h_src))
                    arr = arr[r0:r1, c0:c1]

        return align_array_to_shape(arr, dst_shape, categorical=categorical)


def extract_covariates_at_points(
    stack: np.ndarray,
    band_names: Sequence[str],
    bounds: tuple[float, float, float, float],
    xs: Sequence[float],
    ys: Sequence[float],
    window_size: int = 3,
    categorical_bands: set[str] | None = None,
) -> pd.DataFrame:
    """Extract covariate values at point coordinates using a focal window (default 3x3 median/mode)."""
    if stack.ndim != 3 or stack.shape[0] != len(band_names):
        raise ValueError("stack must have shape (len(band_names), H, W)")
    _, h, w = stack.shape
    min_x, min_y, max_x, max_y = bounds
    if max_x <= min_x or max_y <= min_y:
        raise ValueError(f"Invalid bounds: {bounds}")

    cat_set = categorical_bands or {"esa_worldcover", "nrsc_lulc", "hwsd_soil_class"}
    half = max(0, window_size // 2)

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)

    cols = np.clip(((x_arr - min_x) / (max_x - min_x) * (w - 1)).round().astype(int), 0, w - 1)
    rows = np.clip(((max_y - y_arr) / (max_y - min_y) * (h - 1)).round().astype(int), 0, h - 1)

    records: list[dict[str, float]] = []
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - half), min(h, r + half + 1)
        c0, c1 = max(0, c - half), min(w, c + half + 1)
        patch = stack[:, r0:r1, c0:c1]
        row_dict: dict[str, float] = {}
        for b_idx, b_name in enumerate(band_names):
            sub = patch[b_idx]
            finite = sub[np.isfinite(sub)]
            if finite.size == 0:
                row_dict[b_name] = np.nan
            elif b_name in cat_set:
                vals, counts = np.unique(np.round(finite), return_counts=True)
                row_dict[b_name] = float(vals[np.argmax(counts)])
            else:
                row_dict[b_name] = float(np.median(finite))
        records.append(row_dict)

    return pd.DataFrame(records, columns=list(band_names))
