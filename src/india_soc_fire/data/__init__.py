"""Profile data loaders (STUB). See docs/DATA_PLAN.md for sources and licences.

Every loader must return a DataFrame with columns:
profile_id, lon, lat, top_cm, bottom_cm, soc_g_kg, bd_g_cm3, coarse_frac, source, licence
"""
def load_wosis(path):
    raise NotImplementedError

def load_published_pedons(path):
    raise NotImplementedError

def soc_stock_mg_ha(soc_g_kg, bd_g_cm3, thickness_cm, coarse_frac=0.0):
    """SOC stock (Mg C/ha) = SOC(g/kg) * BD(g/cm3) * thickness(cm) * (1-CF) / 10."""
    return soc_g_kg * bd_g_cm3 * thickness_cm * (1 - coarse_frac) / 10.0
