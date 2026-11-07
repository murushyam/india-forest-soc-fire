"""Fire-SOC nexus: burn severity calibration, paired combustion loss, and 1-5 yr recovery."""
from .combustion_loss import (
    DEFAULT_MATCH_COLS,
    SEVERITY_ORDER,
    estimate_paired_combustion_loss,
)
from .recovery import (
    fit_postfire_recovery_trajectory,
    recovery_curve_model,
    simulate_biome_recovery_chronosequence,
    summarize_local_postfire_flux_recovery,
)
from .severity import (
    BIOME_DNBR_CLASSES,
    DNBR_CLASSES,
    apply_phenology_offset,
    classify_burn_severity,
    compute_dnbr,
    compute_rbr,
    compute_rdnbr,
    dnbr_class,
)

__all__ = [
    "BIOME_DNBR_CLASSES",
    "DEFAULT_MATCH_COLS",
    "DNBR_CLASSES",
    "SEVERITY_ORDER",
    "apply_phenology_offset",
    "classify_burn_severity",
    "compute_dnbr",
    "compute_rbr",
    "compute_rdnbr",
    "dnbr_class",
    "estimate_paired_combustion_loss",
    "fit_postfire_recovery_trajectory",
    "recovery_curve_model",
    "simulate_biome_recovery_chronosequence",
    "summarize_local_postfire_flux_recovery",
]
