"""Hexagonal spatial blocks for spatial cross-validation.

Coordinates must be projected metres (e.g. UTM). Hexagons are pointy-top with
the given flat-to-flat diameter (km); points are assigned via axial rounding.
"""
from __future__ import annotations
import numpy as np


def hex_block_ids(x, y, diameter_km: float) -> np.ndarray:
    x = np.asarray(x, float); y = np.asarray(y, float)
    size = diameter_km * 1000.0 / np.sqrt(3.0)  # centre-to-vertex radius
    q = (np.sqrt(3) / 3 * x - y / 3) / size
    r = (2.0 / 3 * y) / size
    # cube rounding
    cx, cz = q, r
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    m1 = (dx > dy) & (dx > dz)
    m2 = ~m1 & (dy > dz)
    rx = np.where(m1, -ry - rz, rx)
    rz = np.where(~m1 & ~m2, -rx - ry, rz)
    rx = rx.astype(np.int64); rz = rz.astype(np.int64)
    # unique integer id per hex
    return rx * 1_000_003 + rz


def hex_block_folds(x, y, diameter_km: float, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Assign whole hex blocks to folds (greedy balancing by point count)."""
    ids = hex_block_ids(x, y, diameter_km)
    blocks, counts = np.unique(ids, return_counts=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(blocks))
    order = perm[np.argsort(-counts[perm], kind="stable")]
    load = np.zeros(n_folds, int)
    block_fold = {}
    for i in order:
        f = int(np.argmin(load))
        block_fold[blocks[i]] = f
        load[f] += counts[i]
    return np.array([block_fold[i] for i in ids])
