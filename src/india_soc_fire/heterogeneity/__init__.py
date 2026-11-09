"""Multidimensional landscape heterogeneity, land-cover proportion interactions, PCA, and Piecewise SEM.

Adapted from Bae et al. (2026, Scientific Reports 16) for 3D soil organic carbon (SOC)
and post-wildfire ecosystem recovery across Indian forest landscapes.
"""
from __future__ import annotations

from india_soc_fire.heterogeneity.multidimensional import (
    build_pedon_multidimensional_dataset,
    compute_compositional_heterogeneity,
    compute_configurational_heterogeneity,
    compute_temporal_heterogeneity,
    compute_vertical_heterogeneity,
    shannon_diversity_1d,
    simpson_evenness_1d,
)
from india_soc_fire.heterogeneity.sem_pca_gam import (
    evaluate_2d_interaction_surface,
    fit_piecewise_sem_and_interactions,
    run_multidimensional_pca,
)

__all__ = [
    "shannon_diversity_1d",
    "simpson_evenness_1d",
    "compute_compositional_heterogeneity",
    "compute_configurational_heterogeneity",
    "compute_vertical_heterogeneity",
    "compute_temporal_heterogeneity",
    "build_pedon_multidimensional_dataset",
    "run_multidimensional_pca",
    "fit_piecewise_sem_and_interactions",
    "evaluate_2d_interaction_surface",
]
