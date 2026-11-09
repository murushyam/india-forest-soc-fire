"""Wall-to-wall windowed raster prediction and GeoTIFF export for SOC stocks and uncertainty."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from ..covariates import CovariateStack
from ..models.interpret import compute_aoa, compute_shap_importance
from ..models import build_model
from . import bootstrap_predict, qrf_interval


def predict_soc_raster_stack(
    stack: CovariateStack,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "rf",
    n_boot: int = 25,
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95),
    block_ids: np.ndarray | None = None,
    chunk_rows: int = 64,
    seed: int = 42,
    **model_params: Any,
) -> dict[str, np.ndarray]:
    """Predict wall-to-wall SOC stock mean, SD, CV, QRF quantiles, and AOA mask across `stack`."""
    n_bands, h, w = stack.shape
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)

    # Fit full model once for SHAP feature weights used in AOA
    base_model = build_model(model_name, random_state=seed, **model_params)
    base_model.fit(X_tr, y_tr)
    shap_summary, _ = compute_shap_importance(base_model, X_tr, feature_names=stack.band_names)
    weight_map = dict(zip(shap_summary["feature"], shap_summary["mean_abs_shap"]))
    weights = np.array([weight_map.get(b, 1.0) for b in stack.band_names], dtype=float)

    # Flatten raster in row chunks to bound memory usage
    out_keys = (
        "mean",
        "sd",
        "cv",
        "q05",
        "q50",
        "q95",
        "piw90",
        "rpiw90",
        "di",
        "aoa_mask",
        "soc_mean",
        "soc_sd",
        "soc_cv",
        "soc_q05",
        "soc_q50",
        "soc_q95",
        "soc_piw90",
        "soc_rpiw90",
    )
    maps = {k: np.empty((h, w), dtype=np.float32) for k in out_keys}

    for r0 in range(0, h, chunk_rows):
        r1 = min(h, r0 + chunk_rows)
        sub_cube = stack.data[:, r0:r1, :]  # (n_bands, r_chunk, w)
        X_chunk = np.moveaxis(sub_cube, 0, -1).reshape(-1, n_bands)

        boot_res = bootstrap_predict(
            X_tr,
            y_tr,
            X_chunk,
            model_name=model_name,
            n_boot=n_boot,
            block_ids=block_ids,
            seed=seed,
            **model_params,
        )
        qrf_res = qrf_interval(
            X_tr,
            y_tr,
            X_chunk,
            quantiles=quantiles,
            seed=seed,
            **model_params,
        )
        aoa_res = compute_aoa(X_tr, X_chunk, weights=weights)

        rc = r1 - r0
        for k in ("mean", "sd", "cv"):
            arr = boot_res[k].reshape(rc, w).astype(np.float32)
            maps[k][r0:r1, :] = arr
            maps[f"soc_{k}"][r0:r1, :] = arr
        for k in ("q05", "q50", "q95", "piw90", "rpiw90"):
            arr = qrf_res[k].reshape(rc, w).astype(np.float32)
            maps[k][r0:r1, :] = arr
            maps[f"soc_{k}"][r0:r1, :] = arr
        maps["di"][r0:r1, :] = np.asarray(aoa_res["di"], dtype=np.float32).reshape(rc, w)
        maps["aoa_mask"][r0:r1, :] = np.asarray(aoa_res["aoa_mask"], dtype=np.float32).reshape(rc, w)

    return maps


def save_prediction_geotiffs(
    pred_maps: dict[str, np.ndarray],
    bounds: tuple[float, float, float, float],
    crs: str,
    out_dir: str | Path,
    prefix: str,
) -> list[Path]:
    """Write predicted SOC stock and uncertainty rasters to GeoTIFFs (or `.tif` + `.npz`)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Save compressed numpy bundle for fast reproducible reloads
    npz_file = out_path / f"{prefix}_maps.npz"
    np.savez_compressed(npz_file, **pred_maps)
    written.append(npz_file)

    try:
        import rasterio
        from rasterio.transform import from_bounds

        for name, arr in pred_maps.items():
            h, w = arr.shape
            transform = from_bounds(*bounds, width=w, height=h)
            tif_file = out_path / f"{prefix}_{name}.tif"
            with rasterio.open(
                tif_file,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                nodata=-9999.0,
                compress="deflate",
            ) as dst:
                clean = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
                dst.write(clean, 1)
            written.append(tif_file)
    except ImportError:
        import tifffile

        for name, arr in pred_maps.items():
            tif_file = out_path / f"{prefix}_{name}.tif"
            tifffile.imwrite(str(tif_file), np.asarray(arr, dtype=np.float32))
            written.append(tif_file)

    return written
