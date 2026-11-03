"""Fire-SOC nexus (STUB): pre-fire SOC stocks, dNBR-based combustion-loss classes,
1-5 yr recovery. Class thresholds will follow standard dNBR severity bins
(Key & Benson) and are NOT yet calibrated for Indian forests.
"""
DNBR_CLASSES = {"unburned": (-0.1, 0.1), "low": (0.1, 0.27), "moderate_low": (0.27, 0.44),
                "moderate_high": (0.44, 0.66), "high": (0.66, 2.0)}

def dnbr_class(value: float) -> str:
    for k, (lo, hi) in DNBR_CLASSES.items():
        if lo <= value < hi:
            return k
    return "enhanced_regrowth" if value < -0.1 else "high"
