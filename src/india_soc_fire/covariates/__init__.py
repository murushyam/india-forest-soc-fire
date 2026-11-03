"""Covariate stack (STUB). Sources planned (extraction via GEE / Planetary Computer):
- climate: ERA5-Land MAT/MAP, IMD gridded rainfall incl. JJAS monsoon fraction, PET, aridity
- vegetation: Sentinel-2 / MODIS NDVI, NDWI, NBR phenology (max/min/dry season)
- terrain: SRTM/FABDEM elevation, slope, aspect, TWI
- land cover: ESA WorldCover, NRSC LULC, tree cover
- soil class: HWSD v2.0
- fire: MODIS MCD64A1 burn counts, FIRMS hotspot density, Sentinel-2 dNBR (2022)
"""
def build_stack(region: str, cfg: dict):
    raise NotImplementedError("covariate extraction not implemented yet")
