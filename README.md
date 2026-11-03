# india-forest-soc-fire

Depth-harmonised soil organic carbon (SOC) mapping for Indian forest landscapes, linked to wildfire (pre-fire SOC, combustion-loss classes, recovery).

> **Status: pre-release scaffold (v0.0.1). No maps, models or results exist yet. Nothing here should be cited as a finding.**

Methodological template: [antsaro/madagascar-soc-harmonized-mapping](https://github.com/antsaro/madagascar-soc-harmonized-mapping) (depth harmonisation → nested CV → interpretation → uncertainty rasters), adapted for Indian pedology, climate and fire regimes.

## Scope (v1)
| Item | Choice |
|---|---|
| Regions | Odisha (Similipal + surroundings); Uttarakhand (2022 fire footprint + unburned controls) |
| Depths | 0–30, 30–100, 0–100 cm via Jobbágy & Jackson (2000) `Y = 1 − β^d`, β recalibrated per Indian biome |
| Models | Random Forest, XGBoost, Quantile Regression Forest |
| Validation | Nested CV: random K-fold (optimistic) + hexagonal spatial blocks, 25/50/100 km (honest) |
| Interpretation | SHAP + partial dependence |
| Uncertainty | Bootstrap mean/SD/CV rasters; QRF 90 % prediction interval (q05–q95) |
| Covariates | ERA5-Land / IMD climate incl. monsoon (JJAS) fraction; Sentinel-2/MODIS NDVI, NDWI, NBR phenology; DEM derivatives; ESA WorldCover / NRSC LULC; tree cover; HWSD soil class; FIRMS, MCD64A1, dNBR fire history |
| Fire angle | Pre-fire SOC stocks, dNBR combustion-loss classes, 1–5 yr recovery |

## Layout
```
config/config.yaml              regions, depths, CV, models, covariates
src/india_soc_fire/
  harmonize/depth.py            J&J fit + harmonisation   [implemented, tested]
  cv/spatial_blocks.py          hex blocks + fold assign  [implemented, tested]
  data/                         profile loaders, SOC stock [stock formula done; loaders stub]
  covariates/                   covariate stack            [stub]
  models/                       RF / XGB / QRF factory     [factory only; CV runner stub]
  uncertainty/                  bootstrap + QRF PI         [summary helper; rest stub]
  fire/                         dNBR classes               [uncalibrated default bins]
docs/DATA_PLAN.md               profile sources, access, licences
tests/                          pytest suite
```

## Quick start
```bash
pip install -e ".[dev]"      # add ".[ml,geo]" for modelling and raster I/O
pytest -q
```

## What is NOT done
- No soil profile data assembled or committed (see `docs/DATA_PLAN.md`; licences must be cleared first).
- No covariate extraction, model training, SHAP, or rasters.
- β values not fitted on any real Indian profile.
- dNBR severity bins are generic defaults, not calibrated for Indian forests.
- No national mosaic, carbon-market calculation, or quantum/hybrid ML — deliberately out of v1 scope.

## Licence
Code: MIT. Data products will carry the most restrictive licence of their inputs (e.g. CC-BY-NC for some WoSIS profiles, CC BY-NC-SA 3.0 IGO for HWSD-derived layers).
