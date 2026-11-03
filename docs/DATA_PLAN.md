# Data plan — SOC profile sources (v1: Odisha, Uttarakhand)

Licences checked on 2026-11-03. **"Unverified"** means the licence could not be confirmed from an official page; confirm before redistributing anything.

| Source | What it gives | Access route | Licence | Redistribute in repo? |
|---|---|---|---|---|
| **WoSIS** (ISRIC) — 2023 snapshot / `wosis_latest` | Standardised profiles: OC, BD, texture, coarse fragments, depths; Indian coverage sparse | Snapshot TSV, DOI [10.17027/isric-wdcsoils-20231130](https://doi.org/10.17027/isric-wdcsoils-20231130); latest via WFS / GraphQL ([docs](https://docs.isric.org/globaldata/wosis/faq-wosis.html)) | **Per profile: CC-BY or CC-BY-NC** (set by provider; check `licence` column) | Derived tables only, keep per-row licence; treat NC rows as non-commercial |
| **WISE** soil profile DB (ISRIC, v3.1) / WISE30sec | Legacy global profiles; harmonised derived properties | ISRIC data hub (data.isric.org) | **Unverified** — confirm on the dataset record | Not until verified |
| **SoilGrids 2.0** (ISRIC) | 250 m SOC, BD, OCS predictions + quantiles | WCS / WebDAV at soilgrids.org; also in GEE community catalog | CC-BY 4.0 | Benchmark only (not training truth); cite |
| **NBSS&LUP** (ICAR) — soil survey reports, state soil maps, pedon descriptions | Densest Indian profile data incl. Odisha, Uttarakhand | Formal request to NBSS&LUP Nagpur / regional centres; some data in published soil series reports | **Unknown / no open licence found** — assume restricted | No — use under written permission only; publish aggregates |
| **Published Indian pedon datasets** (journal articles, theses, Forest Research Institute / ICFRE forest soil studies, ISRO-GBP SOC campaigns) | Site-level SOC & BD for forests, incl. Himalaya | Literature mining; supplementary tables; author request | **Varies per paper** — record in `sources.csv` | Only when supplement licence permits (e.g. CC-BY) |
| **HWSD v2.0** (FAO & IIASA, 2023) | 1 km soil mapping units, WRB class, reference BD — categorical covariate | FAO / IIASA download; GEE community catalog | CC BY-NC-SA 3.0 IGO | Derived layers must stay NC-SA |
| **India Soil Health Card** data (DAC&FW) | Topsoil (0–15 cm) OC from farms | soilhealth.dac.gov.in portal | Unverified (Govt. portal terms) | Not used for forest v1; cropland context only |

## Rules
1. Every profile row carries `source` and `licence`; the pipeline drops rows whose licence is unknown before any public release.
2. Raw data stays out of git (`data/raw/` ignored); commit only checksums and scripts.
3. SoilGrids is a comparison baseline, never training data (circularity: it already uses WoSIS).
4. Minimum profile depth ≥ 50 cm for β fitting; shallower profiles used only for 0–30 cm training.

## Biggest gaps
1. **Deep (≥ 1 m) forest profiles with BD in Odisha and Uttarakhand** — open sources are sparse; NBSS&LUP holds most data but without an open licence.
2. **Bulk density and coarse fragments** — frequently missing in Indian legacy data → pedotransfer + SoilGrids BD with an error flag.
3. **Paired pre/post-fire soil sampling** — essentially absent; the fire–SOC link will rely on space-for-time (burned vs unburned controls) unless field plots are added.

## Next steps
- Pull WoSIS profiles inside the two bboxes and count by depth/licence.
- Literature search for Similipal and Uttarakhand forest SOC/BD tables.
- Draft data-request letter to NBSS&LUP (do not send without approval).
