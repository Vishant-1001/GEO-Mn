# GEO-MN backend — supply-continuity decision support

SIH 2026 · Problem Statement 26009 (text in `docs/26009ps.txt`).

GEO-MN is an uncertainty-aware manganese supply-continuity decision-support system. It forecasts the
next production period, estimates how far robust operational recovery can close a supply gap, and —
only when a residual gap persists over a **strategic** horizon — activates an exploration contingency
and ranks exploration targets by their relevance to that gap.

```
forecast (P10/P50/P90) -> model contributions -> robust recovery (5 scenarios x 7 portfolios)
 -> residual gap -> horizon gate -> [strategic] exploration target priority -> why this target now
 -> decision flip -> human review -> reconciliation
```

## What is real, what is synthetic

| Component | Mode | Source |
|---|---|---|
| Exploration labels | REAL_PUBLIC | USGS MRDS manganese records (73 locations used) |
| Spectral / thermal / terrain features | REAL_PUBLIC | Sentinel-2 L2A, MODIS MOD11A2, NASADEM (Microsoft Planetary Computer), fixed 2024 window |
| Geology context | REAL_PUBLIC | Macrostrat → GSC *Generalized geology of the world* (world scale, context only) |
| Weather in production data | REAL_PUBLIC | ERA5 / ERA5-Land reanalysis via Open-Meteo, demo-mine coordinate |
| Mine operations, production, targets | **SYNTHETIC** | `ml/generate_operations.py`, seed 42 — **not MOIL data** |
| Disruptions, actions, demo states | SIMULATED | `config/recovery_config.json`, `data/demo_scenarios.json` |
| Subsurface (drilling / assay / geophysics) | UNAVAILABLE | none legitimately available; schema only |
| Reserve / resource tonnage | UNAVAILABLE | never produced or implied |

Satellite composites use the **2024-01-01/2024-12-31** observation window. Nothing is real-time.

## Layout

```
main.py                     FastAPI routes only (thin)
services/                   runtime logic
  production_service.py     history, P10/P50/P90 forecast, risk policy, SHAP contributions, reconciliation
  recovery_service.py       scenario x portfolio simulation, feasibility, robust (min-max) selection
  contingency_service.py    horizon gate, 3 decision states, target selection, decision flip
  exploration_service.py    live/cached point query, cache-distance guard, target priority, why-now
  trust_service.py          validation metrics (read from reports), provenance catalogue
  decision_service.py       append-only human review log
  health_service.py, demo_service.py, common.py
ml/                         offline pipeline (python -m ml.run_pipeline [--fetch])
  eo_features.py            the ONE feature recipe used for training, grid and live queries
  build_exploration_dataset.py, train_exploration.py, target_engine.py, ood.py, uncertainty.py
  fetch_weather.py, generate_operations.py, production_features.py,
  train_production.py, evaluate_production.py
data/                       committed inputs/outputs (see data/data_dictionary.csv)
models/                     models, model_manifest.json, reports/*.json, legacy/ (previous prototype)
config/                     risk_policy.json, exploration_config.json, recovery_config.json, demo_config.json
tests/                      pytest suite (network-free)
```

## Exploration engine

* **Study area:** 20.75–22.75 N, 78.5–81.0 E (Balaghat – Nagpur – Bhandara Mn belt), 50,000 cells of
  0.01° (~1.1 km × 1.0 km). The effective resolution is ~1 km, not the 10 m Sentinel-2 pixel.
* **Features (unchanged family from the prototype, rebuilt reproducibly):** NDVI, B4/B2 iron-oxide
  ratio, B11/B12 clay/hydroxyl ratio (2024 SCL-masked median of the 12 least-cloudy scenes per tile,
  offset-corrected), MODIS daytime LST in K (QC-filtered, ×0.02), NASADEM elevation and slope.
  Cells missing any feature are `INSUFFICIENT_DATA`, never mean-filled.
* **Labels:** positive-unlabelled design. Background cells are > 3 km from every MRDS record and are
  treated as *unlabelled*, not barren.
* **Model:** PU-bagging ensemble of 15 `HistGradientBoostingClassifier`s. The output is a **relative
  percentile rank (0–100)** within the study area — never a deposit probability.
* **Uncertainty:** SD of the member ranks (LOW < 5, HIGH > 10 rank points; project thresholds).
* **Applicability:** geographic envelope + IsolationForest on the feature space (MODERATE below the
  5 % and LOW below the 1 % training score quantile). High rank + LOW applicability → `REVIEW_REQUIRED`.
* **Targets:** cells with rank ≥ 95 → 8-connected clusters (≥ 4 cells; clusters > 50 cells split by
  seeded k-means) → 40 targets with footprint polygons, evidence and priority inputs.
* **Evidence levels:** L0 remote sensing; L1 + Precambrian host domain on the world geology map (all
  73 MRDS training records fall in Precambrian units vs 62 % of the area); L2 + documented MRDS record
  within ~1 km; L3/L4 require drilling/resource data, which do not exist here, so no target exceeds L2.
  Targets overlapping training labels are flagged `CAUTION_TRAINING_LABEL_OVERLAP` (in-sample rank).
* **Priority (project defaults):** 45 % prospectivity, 25 % evidence/applicability/certainty, 20 %
  strategic relevance (gap severity × distance decay to the supply point, **0 when no strategic gap**),
  10 % development-readiness proxy (distance to a documented producer). Missing components are dropped
  and weights re-normalised.
* **Live query:** `POST /api/exploration/predict` recomputes the same recipe for the query cell from
  Planetary Computer (~20–40 s). A live result inside the study area reproduces the cached grid value
  exactly. On failure/timeout the nearest cached cell is used **only within 2 km**; otherwise
  `503 PREDICTION_UNAVAILABLE`. Earth Engine is no longer used (its recipe differed from training and
  its debug endpoint leaked credential details).

### Exploration validation (held-out data only)

| Check | ROC-AUC | PR-AUC (prevalence 0.0015) | Positives in top 10 % of area |
|---|---|---|---|
| Spatial block CV, 0.25° blocks, 5 folds — **selected** (HGB, all six features) | 0.688 | 0.0118 | 22 % |
| Same, spectral features only | 0.557 | 0.0020 | 15 % |
| Same, thermal + terrain only | 0.682 | 0.0069 | 24 % |
| Same, random forest, all six | 0.661 | 0.0059 | 31 % |
| East/west region holdout (pooled) | 0.676 | — | 29 % |

The signal is modest. Most of it comes from thermal/terrain context; the Sentinel-2 ratios add
little on their own. The random forest captures more positives in the top 10 % but ranks worse
overall; selection used ROC-AUC. The legacy classifier scores 0.722 ROC-AUC on the full area, but its
training data have no coordinates, so this is **not** a clean holdout and is likely optimistic.
Success-rate AUC (spatial CV): 0.654.

## Production engine

* **Unit:** total tonnes in the 7-day period starting at the forecast origin (`horizon_days = 7`
  only; other horizons → 400 `UNSUPPORTED_HORIZON`). History is daily; every feature is aggregated to
  the period.
* **Conditional-forecast design:** three quantile GBMs (q = 0.1 / 0.5 / 0.9, monotonic constraints on
  availability, downtime, delays, trucks and rainfall) learn period production from the **period's**
  operating conditions plus origin-known lags (1–4 periods, 4/12-period means, trend, attainment) and
  seasonality. At forecast time the period conditions are unknown, so the trailing 7-day state is
  **assumed to persist**. Backtests use exactly these persistence inputs, so no future weather or
  operations leak into evaluation. Scenarios override the assumed state and are labelled SIMULATED.
* **Recalibration:** additive quantile offsets estimated only from **earlier** out-of-sample errors
  (split-conformal style).
* **Contributions:** SHAP on the P50 model, in tonnes;
  `base + Σ contributions + calibration = P50`. These are associations, not causal effects.
* **Risk:** `config/risk_policy.json` (P50 gap-% bands, escalated one band if the P10 gap exceeds 25 %).
  These are demonstration thresholds.

### Production validation (synthetic operations)

Configuration was selected on origins 2023-07 → 2024-06. Metrics come from the untouched window
2024-07-05 → 2026-09-24 (116 weekly periods, expanding window, 13-period folds, purged).

| | MAE (t) | RMSE (t) | R² | MAPE |
|---|---|---|---|---|
| GEO-MN P50 | **839** | 1142 | 0.566 | 9.9 % |
| Previous period (naive) | 892 | 1201 | 0.520 | 10.4 % |
| 4-period moving average | 953 | 1296 | 0.441 | 11.0 % |
| *Diagnostic: true period conditions (not forecast skill)* | *418* | *516* | *0.911* | *4.9 %* |

* Pinball loss P10 / P50 / P90: 237 / 419 / 230 t.
* P10–P90 coverage: **0.75** (nominal 0.80); it was 0.28 before recalibration. Hit rates are
  0.14 / 0.52 / 0.89, which passes the ±0.08 rule, so `quantiles_validated = true`.
* Shortfall events (actual < 95 % of target): precision 0.85, recall 0.81, F1 0.83.
* **Honest caveats:**
  * In the selection window no candidate beat the naive baseline (best 867 vs 825 t).
  * The most recent 26 periods (monsoon-heavy, `/api/production/reconciliation`) show MAE 1141 t and
    coverage 0.54.
  * All of these figures come from synthetic data and say nothing about real MOIL accuracy.

## Recovery and contingency

* **Scenarios:** NORMAL, HEAVY_RAIN, EQUIPMENT_DEGRADATION, BLAST_DELAY, COMBINED_DISRUPTION.
* **Actions:** A1 equipment recovery, A2 schedule adjustment, A3 blast/drill delay reduction. All
  7 combinations are evaluated, plus no-action.
* **Feasibility:** availability ≤ mine cap and within [0, 1], trucks ≤ fleet, delays ≥ 0, lost hours
  consistent with availability. States are FEASIBLE, FEASIBLE_CONSTRAINED, NOT_FEASIBLE or
  UNKNOWN — HUMAN REVIEW. Model applicability is reported per scenario, separately from feasibility.
* **Selection:** minimise the worst-case residual gap; ties go to the lower mean residual, then to
  fewer interventions (tolerance 0.5 % of target). This is a lightweight robust-recourse
  approximation, not a mine scheduler. "Expected" means the nominal scenario; "worst case" means the
  maximum over the tested scenarios.
* **Horizon gate:**
  * NEAR_TERM: exploration is never recovery. Residual ≤ 2 % of target → `OPERATIONAL_RESPONSE`;
    otherwise → `REVIEW_REQUIRED`.
  * STRATEGIC: if the expected residual is > 2 % of target, or the worst case is > 8 %, targets are
    ranked under the current gap state. The best target with applicability ≥ MODERATE gives
    `OPERATIONAL_AND_EXPLORATION_CONTINGENCY`; if no target qualifies → `REVIEW_REQUIRED`.
  * An out-of-distribution production state, or no feasible action → `REVIEW_REQUIRED`.

### Demo states (inputs only; outcomes computed and asserted by tests)

| mine_id | demo_state | computed decision |
|---|---|---|
| DEMO_A | ON_TRACK | OPERATIONAL_RESPONSE (no action required) |
| DEMO_B | OPERATIONALLY_RECOVERABLE | OPERATIONAL_RESPONSE |
| DEMO_C | STRATEGIC_CONTINGENCY | OPERATIONAL_AND_EXPLORATION_CONTINGENCY |
| DEMO_D | REVIEW_REQUIRED | REVIEW_REQUIRED (inputs out of distribution) |
| DEMO_E | SATELLITE_FALLBACK | contingency, plus live failure → cached ≤ 2 km / unavailable > 2 km |
| DEMO_F | DECISION_FLIP | OPERATIONAL_RESPONSE → OPERATIONAL_AND_EXPLORATION_CONTINGENCY under rain / equipment / blast stress |
| DEMO_MINE | (current state) | STRATEGIC: contingency; NEAR_TERM: operational response |

## API (all JSON; errors are `{"error": CODE, "message": ...}`)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | component flags only, no secrets |
| GET | `/api/supply-command?mine_id=&horizon=` | whole loop in one response; DEMO_E/F add fallback / flip blocks |
| GET | `/api/exploration/targets?mine_id=&horizon=` | priority computed under that mine's strategic state |
| GET | `/api/exploration/targets/{id}` | evidence, why-this-target, why-now, next evidence |
| POST | `/api/exploration/predict` | `{lat, lon, mode?: AUTO\|LIVE_ONLY\|CACHED_ONLY\|SIMULATE_LIVE_FAILURE}` |
| GET | `/api/exploration/grid?stride=` | sampled ~1 km rank grid for maps (`/reserve_grid` = legacy alias) |
| POST | `/api/production/forecast` | `{mine_id, forecast_origin?, horizon_days?=7, target_tonnes?, conditions?}` |
| GET | `/api/production/history?mine_id=&days=` | daily rows + 7-day periods |
| GET | `/api/production/reconciliation?mine_id=` | out-of-sample forecast vs actual, error = forecast − actual |
| POST | `/api/recovery/evaluate` | contract form or frontend form `{scenario, actions, conditions}` |
| POST | `/api/contingency/evaluate` | recomputes server-side; client forecast/recovery objects are not trusted |
| POST | `/api/decision/flip` | `{mine_id, baseline_conditions, perturbed_conditions, horizon?}` |
| POST | `/api/decision/review`, GET `/api/decision/history` | human review log |
| GET | `/api/trust/exploration`, `/api/trust/production`, `/api/trust/provenance` | computed metrics only |
| GET | `/api/model/manifest`, `/api/demo/scenarios` | |

Accepted condition aliases: `rainfall_mm` → `rainfall_7d_mm` (7-day total),
`blast_delay_hours` → `blast_delay_h`, `drilling_delay_hours`, `haulage_delay_hours`, `soil_moisture`,
`temperature`. Values outside physical ranges → 400.

## Running

```bash
pip install -r requirements-dev.txt
python -m ml.run_pipeline          # deterministic rebuild of all models/artifacts (add --fetch to re-download)
pytest -q
uvicorn main:app --port 8000       # serves the API and index.html / app.js / style.css
```

Environment variables: `GEOMN_CORS_ORIGINS` (default: none, same-origin only), `GEOMN_LIVE_EO`
(1/0), `GEOMN_DECISION_LOG`, `PORT`. No credentials are required or stored.

## Known limitations

* Operational data are synthetic. Every production, recovery and decision number demonstrates the
  method; none is a statement about a real MOIL mine.
* Prospectivity is a weak-to-moderate relative signal (spatial ROC-AUC ≈ 0.69) trained on 73
  clustered MRDS points of mixed location precision. Subsurface evidence is unavailable.
* Intervention effects are learned associations through a monotonic model, not causal estimates.
* The strategic horizon is represented as persistence of the per-period residual gap (13 periods),
  not a multi-period mine plan.
* Priority weights, risk bands and materiality thresholds are project defaults.
