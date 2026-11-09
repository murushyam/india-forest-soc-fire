"""Pan-India (16 Champion & Seth forest types / 5 sentinel landscapes) and Global scaling modules."""
from __future__ import annotations

from india_soc_fire.scaling.pan_india_global import (
    PAN_INDIA_CHAMPION_SETH_BIOMES,
    conformalize_spatial_intervals,
    estimate_double_negative_control_dml,
    generate_pan_india_and_global_tiles,
    synthesize_pan_india_sentinel_landscapes,
)
from india_soc_fire.scaling.spatial_maps import (
    generate_global_spatial_grid,
    generate_pan_india_spatial_grid,
)

__all__ = [
    "PAN_INDIA_CHAMPION_SETH_BIOMES",
    "synthesize_pan_india_sentinel_landscapes",
    "estimate_double_negative_control_dml",
    "conformalize_spatial_intervals",
    "generate_pan_india_and_global_tiles",
    "generate_pan_india_spatial_grid",
    "generate_global_spatial_grid",
]
