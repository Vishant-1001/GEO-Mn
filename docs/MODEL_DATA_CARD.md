# Model and data card

## Exploration prospectivity — `exploration-pu-ensemble-2.0`

| Item | Value |
|---|---|
| Task | Relative ranking (0–100) of 0.01° cells in the Central-India Mn belt (20.75–22.75 N, 78.5–81.0 E) |
| Output semantics | Percentile rank within the study area. **Not** a probability of a deposit, and not a reserve or tonnage. |
| Model | PU-bagging ensemble of 15 HistGradientBoosting classifiers (handles missing geology natively) |
| Features (18) | S2: NDVI, B4/B2, B11/B12, NDRE, B4/B3, B12/B8A, B11/B4, SWIR albedo. LST (MODIS). Elevation, slope, elevation SD (NASADEM). NRSC geomorphology flags and dissection. |
| Labels | 72 MRDS positive cells (REAL_PUBLIC) + 3 NMET cells (REAL_GOVERNMENT: Katori XRF samples, Nagardhan and Kawalewada block centroids). Background is unlabelled cells > 3 km from any positive. |
| Validation | East development region (spatial block CV) → untouched west test, 4-seed average: ROC 0.803 ± 0.030, PR-AUC 0.0128 (prevalence 0.0006), 43 % of positives in the top 10 % of area. Full-area spatial CV ROC 0.719. Mixed PR evidence: see `MODEL_COMPARISON_REPORT.md`. |
| Uncertainty | SD of member ranks (LOW < 5, HIGH > 10 rank points; project thresholds). 78 % of cells are HIGH. |
| Applicability | Geographic envelope; IsolationForest on the EO features; hard training-range envelope. LOW → `REVIEW_REQUIRED`. |
| Known risks | Clustered labels of mixed precision. In-sample ranks where footprints contain labels (flagged). Geology covers MP / MH only. Surface data cannot see subsurface ore. |
| Not for | Reserve or resource statements, drilling decisions without field verification, areas outside the study area. |

## Target engine and evidence levels

Targets are clusters of cells with rank ≥ 95 (40 targets). Evidence levels:

* **L0**: remote sensing only.
* **L1**: plus a Precambrian host domain, or official structural / geological context.
* **L2**: plus a documented MRDS record within 1 km, or official surface geochemistry, grade or pit
  report.
* **L3**: **reported** drilling outcome in an overlapping official block (one target, T07 /
  Nagardhan; no public logs or assays).
* **L4**: never reached.

## Next-evidence sensitivity (rule-based)

For each possible outcome of the recommended next investigation, the project-configured rules
(`config/exploration_config.json → subsurface_fusion`) show how a copy of the target's maturity,
uncertainty and investigation priority would change. No subsurface records are generated, and
observed evidence levels are never changed.

## Target priority

Hard gates come first: study area → applicability ≥ MODERATE → evidence ≥ L1. Then:

* Contingency OFF: prospectivity + evidence/applicability.
* Strategic contingency ON: strategic relevance is added.

No development-readiness proxy is scored.

## Weekly production forecast — `production-qgbm-1.0` (SYNTHETIC operations)

| Item | Value |
|---|---|
| Task | 7-day demo-mine production P10 / P50 / P90, conditional on the trailing state persisting |
| Data | SYNTHETIC equipment-level simulator (seed 42) with REAL IMD weather. **Not MOIL data.** |
| Validation | Selection 2022-07..2023-06, calibration 2023-07..2024-06 (offsets frozen), test 2024-07..2026-09 (116 periods). P50 MAE 1144 t vs 1150 t for the best baseline (a tie). P10–P90 coverage 0.80 (validated). |
| Explanations | SHAP model contributions, which are associations, not causes |
| Not for | Any claim about real MOIL mine accuracy |

## Real quarterly production — `real-moil-quarterly-1.0` (REAL_MOIL_PUBLIC)

| Item | Value |
|---|---|
| Task | Next-quarter MOIL company-total Mn-ore production |
| Data | 54 quarters derived from MOIL public disclosures (2012–2026); IMD rain anomaly as a candidate feature |
| Validation | Selected ridge (no weather): test MAPE 11.7 % vs seasonal naive 8.6 %. **Does not beat baselines.** Synthetic augmentation hurts. |
| Use | Context panel showing model and baseline forecasts side by side. It does not drive decisions. |

## Recovery and contingency (SIMULATED effects)

Scenario and action effects come from simulator counterfactuals. Portfolios are eligible only if
modelled-feasible and inside model applicability. Selection minimises the worst-case residual gap,
then the mean gap, then intervention burden. There are three decision states, with a horizon gate.
Nothing guarantees a recovery tonnage.

## Data

See `DATA_SOURCE_AUDIT.md`, `DATA_PROVENANCE.md`, `DATA_DICTIONARY.md` and
`DATA_QUALITY_REPORT.md`.
