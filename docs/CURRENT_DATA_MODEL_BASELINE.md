# GEO-MN — data & model baseline (before the real-data upgrade)

Recorded 2026-09-26 from commit `a0256aa` (before any change in this upgrade). All values below were
read from the committed reports / data files, not re-estimated.

## Test status

`pytest -q` → **118 passed**, 0 failed (process exits cleanly).

## Datasets in use

| File | Rows | Provenance | Consumer |
|---|---|---|---|
| `data/mrds_mn_occurrences.csv` | 80 (73 used as positives) | REAL_PUBLIC — USGS MRDS (international) | exploration labels |
| `data/exploration_grid_features.csv.gz` | 50,000 cells (0.01°) | REAL_PUBLIC — Sentinel-2 L2A, MODIS MOD11A2, NASADEM (2024 window) | exploration features |
| `data/exploration_grid.csv` | 50,000 | derived model output | map / targets |
| `data/geology_lattice.csv` | 500 (0.1° lattice) | REAL_PUBLIC — Macrostrat / GSC world geology (1:35M) | evidence level L1 |
| `data/weather_daily.csv` | 2,093 days (2021-01-01 → 2026-09-24) | REAL_PUBLIC — ERA5 / ERA5-Land reanalysis (Open-Meteo), international | production features |
| `data/production_history.csv` | 2,093 days | **SYNTHETIC** operations & production for one demo mine (seed 42) + real ERA5 weather | production model |
| `data/production_backtest_forecasts.csv` | 168 weekly periods | derived | reconciliation |
| `data/subsurface_evidence.csv` | 0 (schema only) | — | none |

No Indian government dataset was integrated at baseline. All production labels were synthetic.

## Exploration model (`exploration-pu-ensemble-1.0`)

* 15-member PU-bagging HistGradientBoosting ensemble; 6 features (NDVI, B4/B2, B11/B12, LST, elevation, slope).
* Spatial block CV (0.25° blocks, 5 folds): **ROC-AUC 0.688**, PR-AUC 0.0118 (prevalence 0.0015),
  top-10 %-area capture 0.222.
* East/west region holdout: ROC-AUC 0.676.
* 40 targets; max evidence level L2; subsurface UNAVAILABLE.

## Production model (`production-qgbm-1.0`)

* Weekly (7-day) conditional quantile GBM on the **synthetic** demo mine.
* Untouched test window 2024-07-05 → 2026-09-24 (116 weeks): P50 MAE **843.6 t**, naive 891.8 t,
  4-week MA 953.4 t; P10–P90 coverage 0.776 (nominal 0.80); hit rates 0.13 / 0.40 / 0.91 →
  `quantiles_validated = false`.
* These metrics were computed on synthetic labels and say nothing about real accuracy.

## Recovery / contingency

5 disruption scenarios × 7 action portfolios + no action; applicability gate; intervention-burden
tie-break (equal weights); three decision states; deterministic demo states DEMO_A…F all produce
their expected decisions.
