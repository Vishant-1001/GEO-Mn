# Real data implementation report

Baseline before this work: `CURRENT_DATA_MODEL_BASELINE.md` (no Indian government data; weather ERA5
only; no real production; subsurface UNAVAILABLE everywhere).

## 1. What was integrated

| Area | Real source | Pipeline | Used by |
|---|---|---|---|
| Weather | IMD gridded rainfall 0.25° + Tmax 1°, 2012–2025 (REAL_GOVERNMENT) | `scripts/data_acquisition/fetch_imd_gridded.py` → `data/processed/weather/imd_daily_mine_belt.csv` | Simulator weather driver; real quarterly model (rain anomaly candidate); heavy-rain scenario intensity (IMD p95 7-day = 148 mm) |
| Production | MOIL quantitative-details disclosures (REAL_MOIL_PUBLIC) | `fetch_moil_production.py` (list API → 52 PDFs → pdfplumber → cumulative differencing with cross-checks) | `ml/train_real_production.py`; `/api/production/real-quarterly`; Production Risk panel |
| Geomorphology / lineaments | NRSC Bhuvan 1:50k (REAL_GOVERNMENT) | `fetch_bhuvan_geology.py` (WMS GetFeatureInfo polygons, GetMap lineaments) → `build_geology_features.py` | Exploration features (ablation C–F); live-query lookup |
| Official exploration blocks | NMET / DGM Maharashtra / MECL proposals (REAL_GOVERNMENT) | Transcribed → `build_real_subsurface.py` | Target evidence (REPORTED_BLOCK_LEVEL), label set B, cited geological expectations shown with the target's observed evidence |
| Company exploration | MOIL annual report (REAL_MOIL_PUBLIC) | Transcribed | Context only |
| Grade classes | IBM Indian Minerals Yearbook (manganese chapter) | Cited in `geological_constraints.json` | Simulated interval grade categories |
| Extended EO features | Sentinel-2 L2A (red-edge, SWIR ratios), NASADEM relief | `build_exploration_features.py` | Exploration ablation A–F |

## 2. Key engineering decisions

* **MOIL semantics.** Only a column header saying "quarter ended" means quarterly values. Every
  other column is April-to-date, even when the document title says "quarter ended". This was
  verified against magnitudes. Units come from magnitude (lakh t < 100) rather than labels, because
  one PDF labels lakh values "MT". Result: 54 quarters, 0 conflicts, one gap left missing.
* **IMD reader.** Binary size is checked against days × grid, so leap years can't be misread.
  Missing sentinels become NaN. Climatology was sanity-checked (monsoon dominance, annual totals).
* **No imputation.** IMD 2026 is unpublished, so ERA5 is used for 2026 dates and the source is
  recorded per row.
* **Bhuvan.** WFS is disabled, so polygons were harvested with JSON GetFeatureInfo over a 0.1° tiling
  (0 failed requests). Lineaments were rasterised from GetMap at ~55 m.
* **Evidence honesty.** NMET proposals (PROPOSAL_ONLY) never raise evidence level. Only a reported
  drilling intersection gives L3, and only at block level. Katori "Mansar terrain" was reclassified
  to GEOLOGICAL_CONTEXT (L1).

## 3. Real production results (REAL_MOIL_PUBLIC)

The model is selected on FY2019-20..FY2022-23 and tested on 12 later real quarters.

* Selected ridge (no weather): test MAPE 11.7 %.
* Seasonal naive × YoY: **8.6 %**.
* Last quarter: 9.2 %.

The model does **not** beat the baselines, and the product says so. Weather adds nothing at
company-quarter level. Synthetic-operations augmentation made real error worse and is excluded.
Next quarter (FY2026-27 Q2):

* Model: 440,921 t (indicative band 389,003–521,669 t).
* Seasonal naive: 481,751 t.
* Last quarter: 507,000 t.

## 4. Real exploration results

The extended real feature stack was tested against the six baseline features, and NMET labels
against MRDS-only labels. On the untouched western region (4-seed mean), the deployed model
(S2 red-edge / SWIR ratios + relief + NRSC geomorphology, MRDS + NMET labels) scores:

* ROC **0.803** vs 0.697 for the baseline;
* PR-AUC **0.0128** vs 0.0021 (prevalence 0.0006);
* **43 %** of positives in the top 10 % of area, vs 25 %.

The three NMET label cells did not by themselves improve the test region. Lineament features did not
help. On the full area the evidence is mixed, as the baseline has the higher PR-AUC. See
`MODEL_COMPARISON_REPORT.md`.

## 5. Access barriers (not bypassed)

AIKosh (login; GSI geochemistry outside the AOI), OGD India (API key via Janparichay login), GSI
NGDR (login), Bhoonidhi (login), Bhukosh / MP DGM / Maharashtra DGM / IMD MRS (unreachable). No
accounts were created and no credentials were used. Details: `DATA_SOURCE_AUDIT.md`.
