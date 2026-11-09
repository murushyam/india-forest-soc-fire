"""Spatial cross-validation blocks and variogram diagnostics."""
from .spatial_blocks import hex_block_folds, hex_block_ids
from .variogram import (
    compute_empirical_variogram,
    empirical_variogram,
    fit_variogram_range,
    morans_i,
)

__all__ = [
    "compute_empirical_variogram",
    "empirical_variogram",
    "fit_variogram_range",
    "hex_block_folds",
    "hex_block_ids",
    "morans_i",
]
