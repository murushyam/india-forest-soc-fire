# India Forest Soil Organic Carbon & Fire — Harmonised Multi-Depth Mapping (v1.0.0)

**Status:** v1.0.0 — complete multi-depth Soil Organic Carbon (SOC) harmonisation, honest spatial block cross-validation, uncertainty quantification, and post-fire SOC combustion-loss & 1–5 year recovery modelling across fire-prone Indian forest biomes.

This repository implements an end-to-end, reproducible Digital Soil Mapping (DSM) and Fire–SOC nexus pipeline adapted to Indian forest types, monsoon climatology, and fire regimes.

## Scientific Scope

1. **Multi-depth SOC stock mapping** (`0–30`, `30–100`, `0–100 cm`, in $\text{Mg C ha}^{-1}$) across two fire-prone Indian forest pilot landscapes:
   - **Similipal Biosphere Reserve, Odisha** (`EPSG:32645`, tropical moist deciduous Sal *Shorea robusta* forest; recurrent Feb–May dry-season surface fires).
   - **Uttarakhand forest divisions** (`EPSG:32644`, Garhwal & Kumaun subtropical *Pinus roxburghii* Chir pine and temperate *Quercus leucotrichophora* Banj oak forests; peak fire season Apr–Jun).
2. **Biome-Calibrated Depth Harmonisation** ([`src/india_soc_fire/harmonize/`](file:///i:/tele/SOC%20India/src/india_soc_fire/harmonize/__init__.py)) via the Jobbágy & Jackson (2000) cumulative exponential model $Y(d) = 1 - \beta^d$, with $\beta$ recalibrated per Indian forest biome on profiles $\ge 50\text{ cm}$ and benchmarked against mass-preserving equal-area splines (Bishop et al., 1999; Malone et al., 2009).
3. **Honest Spatial Validation** ([`src/india_soc_fire/cv/`](file:///i:/tele/SOC%20India/src/india_soc_fire/cv/__init__.py), [`src/india_soc_fire/models/nested_cv.py`](file:///i:/tele/SOC%20India/src/india_soc_fire/models/nested_cv.py)) comparing standard random $K$-fold CV against variogram-guided hexagonal spatial block CV (`25 km`, `50 km`, `100 km`) in a nested $5\times 3$ design across Random Forest (`rf`), XGBoost (`xgb`), and Quantile Regression Forest (`qrf`).
4. **Dual Spatial Uncertainty & Area of Applicability** ([`src/india_soc_fire/uncertainty/`](file:///i:/tele/SOC%20India/src/india_soc_fire/uncertainty/__init__.py)) producing spatial-block bootstrap `mean`, `SD`, and `CV` rasters alongside QRF 90% prediction intervals ($q_{0.05}, q_{0.50}, q_{0.95}$, $\text{PICP}_{90}$) and Meyer & Pebesma (2021) Area of Applicability (`AOA`) masks.
5. **Fire–SOC Nexus** ([`src/india_soc_fire/fire/`](file:///i:/tele/SOC%20India/src/india_soc_fire/fire/__init__.py)) coupling pre-fire SOC stock baselines with biome-calibrated spectral burn severity ($\text{dNBR}$, $\text{RdNBR}$, $\text{RBR}$ with phenology offset correction), covariate-matched paired topsoil (`0–30 cm`) combustion loss estimation (with `30–100 cm` subsoil falsification controls), and 1–5 yr post-fire recovery trajectories linked with 2021–2025 $\text{GPP}$, $\text{uWUE}$, and turbulent heat flux ($\text{SH}/\text{LH}$) time series.

## Repository Layout

```text
india-forest-soc-fire/
├── config/
│   └── config.yaml                  # Regions, depths, covariates, CV & fire parameters
├── data/
│   └── derived/
│       └── published_pedons_india.csv # Curated open-access Indian forest soil profiles
├── docs/
│   └── DATA_PLAN.md                 # Data sources, licensing gate & circularity rules
├── src/india_soc_fire/
│   ├── cli.py                       # End-to-end CLI pipeline runner
│   ├── config.py                    # YAML config loader
│   ├── data/                        # WoSIS & published pedon loaders, QC, Walkley-Black, BD PTF
│   ├── harmonize/                   # Jobbágy-Jackson biome β fit & equal-area spline
│   ├── covariates/                  # Local I:\tele raster/flux ingestion, terrain (TWI), GEE stack
│   ├── cv/                          # Hexagonal spatial blocks, empirical variogram & Moran's I
│   ├── models/                      # RF, XGB, QRF, 5x3 nested CV, TreeSHAP, PDP, and AOA
│   ├── uncertainty/                 # Spatial-block bootstrap, QRF 90% intervals, GeoTIFF writer, benchmarks
│   ├── fire/                        # dNBR/RdNBR/RBR, paired combustion loss, 1-5 yr recovery
│   └── viz/                         # Publication figures (depth curves, CV gap, SHAP, fire recovery)
└── tests/                           # Comprehensive pytest suite across all 6 phases
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest -v
python -m india_soc_fire.cli --config config/config.yaml --output-dir outputs
```

## Licensing & Attribution

Code: MIT. Data sources carry their own licences — see [`docs/DATA_PLAN.md`](file:///i:/tele/SOC%20India/docs/DATA_PLAN.md). Conceptual debt to [`antsaro/madagascar-soc-harmonized-mapping`](https://github.com/antsaro/madagascar-soc-harmonized-mapping) is acknowledged for the depth-harmonisation + spatial-block CV + bootstrap/QRF uncertainty architecture.
