# Data source audit (Indian government and public sources)

Audit date: 2026-09-26. Study area: the Central-India manganese belt (Balaghat – Nagpur – Bhandara;
20.75–22.75 °N, 78.5–81.0 °E). Status codes:

| Code | Meaning |
|---|---|
| INTEGRATED | Downloaded, processed, and used by an engine or model. |
| INTEGRATED_CONTEXT | Used as documented evidence or constraints, not as a training feature. |
| ACCESS_BARRIER | Needs an account, API key or login. GEO-MN does not create accounts or enter credentials. |
| UNREACHABLE | The host did not answer from this network (timeout / DNS) during the audit. |
| NOT_IN_AOI | Reachable, but the relevant data do not cover the study area. |
| NOT_PUBLISHED | The period needed is not yet released by the provider. |
| NOT_NEEDED | An equivalent source is already integrated. |

## 1. Sources audited

| Source | What was checked | Finding | Status |
|---|---|---|---|
| **IMD gridded rainfall** (imdpune.gov.in, 0.25°) | POST `rainfall.php` per year 2012–2026 | Binary grids 2012–2025 downloaded (135×129 float32, −999 missing). 2026 returns 0 bytes. | INTEGRATED 2012–2025; NOT_PUBLISHED 2026 |
| **IMD gridded Tmax** (1°) | POST `maxtemp.php` per year | 31×31 grids 2012–2025 (99.9 missing). | INTEGRATED |
| IMD MRS portal (mrs.imd.gov.in) | Station data | Host unreachable during audit | UNREACHABLE |
| **MOIL Ltd.** (moil.nic.in investor relations) | Backend list API `get-financial-public-list`, type `quantitative-details` | 52 public "Quantitative Details" PDFs (production, company total). Parsed → 142 stated figures → 54 quarters, 0 cross-check conflicts. | INTEGRATED (REAL_MOIL_PUBLIC) |
| MOIL annual reports | Exploration disclosures | FY2024-25: 107,530 m drilled, 16.07 Mt resources added, reserves / resources 53.47 / 68.50 Mt (company level). | INTEGRATED_CONTEXT |
| MOIL mine-level telemetry (equipment, shifts, delays) | — | Not public. | Not available → verified gap filled by SYNTHETIC operations |
| **NRSC Bhuvan** (bhuvan-vec2 WMS) | GetCapabilities; `geomorphology:{MP,MH}_GM50K_0506`, `lineament:{MP,MH}_LN50K_0506` | WFS disabled. WMS GetFeatureInfo returns JSON polygons (4,109 geomorphology polygons harvested). Lineaments rasterised with GetMap at 0.0005°. | INTEGRATED (REAL_GOVERNMENT) |
| Bhuvan lithology (`LITHGEOM`) | Capabilities | Layers exist only for Andhra Pradesh districts. | NOT_IN_AOI |
| Bhuvan mineral layers (`school:Minerals_*`) | Capabilities | School-atlas iron-ore and bauxite layers only; no manganese in the AOI. | NOT_IN_AOI |
| **NMET / Ministry of Mines** (nmet.gov.in) | Project proposals for Mn blocks in the belt | 5 block documents (Kawalewada-Sakkardara, Rongha, Katori-Jhiriya, Nagardhan, Lanjera-Futala). Block corners, reported findings, surface XRF samples and proposed drilling were transcribed with sha256 of each PDF. | INTEGRATED (REAL_GOVERNMENT) |
| GSI NGDR (geodataindia.gov.in) | Landing page | Redirects to `/login`. | ACCESS_BARRIER |
| GSI Bhukosh | Public map portal | No response (timeout). | UNREACHABLE |
| GSI website (gsi.gov.in) | Reports | Reachable; no downloadable borehole, assay or geophysics tables for the AOI found. | Checked, no usable tables |
| **AIKosh** (aikosh.indiaai.gov.in) | Dataset search (manganese, geological, geochemical) | API returns 403 without a session. The UI lists GSI geochemical datasets, but for regions outside the AOI (Rajasthan / Karnataka / Andhra Pradesh). Download needs login. | ACCESS_BARRIER + NOT_IN_AOI |
| **OGD India** (data.gov.in) | Public backend API | `Authorization field missing`: needs an API key issued after Janparichay login. | ACCESS_BARRIER |
| **Bhoonidhi** (NRSC EO archive) | Landing page | Downloads need login. Sentinel-2 L2A via Microsoft Planetary Computer already covers the need. | ACCESS_BARRIER / NOT_NEEDED |
| **IBM** (ibm.gov.in) | Indian Minerals Yearbook, manganese chapter | The IMY 2022 manganese chapter is public (grade classes, national context). No mine-level production series. | INTEGRATED_CONTEXT (grade classes in geological constraints) |
| Ministry of Mines (mines.gov.in) | Statistics | Reachable. Monthly statistics are state / national aggregates, not at the belt level. | Checked, not used |
| MP DGM (mpdgm.gov.in) | State mining portal | No response. | UNREACHABLE |
| Maharashtra DGM (dgm.maharashtra.gov.in) | State mining portal | No response. The DGM-authored NMET proposals were obtained via nmet.gov.in. | UNREACHABLE (content via NMET) |
| USGS MRDS | Mn occurrences | 73 positive records in the AOI. | INTEGRATED (REAL_PUBLIC; baseline labels) |
| Sentinel-2 L2A / MODIS / NASADEM (Microsoft Planetary Computer) | STAC | Full-AOI extended feature grid. | INTEGRATED (REAL_PUBLIC → REAL_DERIVED features) |
| ERA5 / ERA5-Land (Open-Meteo archive) | Daily | Soil moisture, plus rain and Tmax for dates IMD has not published. | INTEGRATED (REAL_PUBLIC, secondary) |

No account was created and no credential was entered for any source.

## 2. Mandatory real subsurface audit

Question: is there public, georeferenced, observed subsurface data (borehole collars, lithology logs,
assays, geophysics) for the AOI?

* **Borehole collars, logs, assays**: none public. The NMET documents give block polygons and
  narrative outcomes. Nagardhan reports historical drilling (6–7 boreholes, Mn intersected in BH-1..BH-4)
  and pits (7 pits, 3 with Mn boulders), with a grade range of 26–38 % Mn. No hole coordinates or interval
  tables are given.
* **Surface geochemistry**: 5 hand-held XRF samples with coordinates (Katori-Jhiriya, 12.4–26.8 % Mn).
* **Geophysics**: none public for the AOI.
* **Company level**: MOIL reports drilled metres and resource additions, not locations.

Conclusion:

1. Observed subsurface evidence is used only at the level it is published: **REPORTED_BLOCK_LEVEL**,
   and only for targets whose footprint overlaps an official block.
2. Everywhere else, subsurface evidence is **UNAVAILABLE**. No borehole, interval, assay or
   geophysical value is generated. The product shows the next required evidence, and a rule-based
   sensitivity of investigation priority to possible outcomes of that evidence.

## 3. Verified gaps and how each is handled

| Gap | Handling |
|---|---|
| Mine-level daily / shift operations (equipment, delays, trucks) | SYNTHETIC equipment-level simulator driven by REAL IMD weather (`ml/synthetic_ops.py`). |
| Historical recovery-action outcomes | SIMULATED counterfactuals from the same simulator (`recovery_scenario_matrix.csv`). |
| Subsurface drilling / assays / geophysics | **Not filled.** Shown as UNAVAILABLE, with next required evidence (section 2). |
| IMD 2026 rainfall | ERA5 used for those dates, with the source recorded per row. Nothing is imputed. |
| Geomorphology outside MP / MH layers (~15 % of cells) | Left missing (NaN). The model handles missing values natively. |
