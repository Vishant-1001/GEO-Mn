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
  exactly. On failure/timeout the nearest cached cell is used **only within 2 km** (response fields
  `fallback_used`, `fallback_distance_km`, `max_supported_fallback_distance_km`; exactly at the limit is
  allowed); otherwise `503 PREDICTION_UNAVAILABLE`. `mode=SIMULATE_LIVE_FAILURE` demonstrates the
  fallback (DEMO_E). Earth Engine is not used.
* **Map surface:** `GET /api/exploration/grid` returns the cached ~1 km `prospectivity_rank` grid. The
  legacy `/reserve_grid` endpoint (old 0.25° cache with a `probability` column) has been **removed**;
  `data/legacy/` keeps that file for reference only.

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
  only; other horizons → 400 `UNSUPPORTED_HORIZON`). History is daily; features are aggregated to
  the period. The plan target is **not** a model input (the forecast is tonnes, the gap is computed
  afterwards).
* **7-day conditional forecast:** three quantile GBMs (q = 0.1 / 0.5 / 0.9, monotonic constraints on
  availability, downtime, delays, trucks and rainfall) learn period production from the **period's**
  operating conditions plus origin-known lags (1–4 periods, 4/12-period means, trend, attainment) and
  seasonality. At forecast time the trailing 7-day state is **assumed to persist**
  (`forecast_method = conditional_production_forecast`, `persistence_assumption = true`,
  `scenario_override_applied` when a scenario overrides it). Backtests use exactly these persistence
  inputs, so no future weather or operations leak into evaluation. This is not a weather forecast.
* **Contributions:** SHAP on the P50 model, in tonnes;
  `base + Σ contributions + calibration = P50`. They are model contributions (associations), not
  causal effects. If SHAP is unavailable the UI says "Model contributions unavailable".
* **Risk:** `config/risk_policy.json` (P50 gap-% bands, escalated one band if the P10 gap exceeds 25 %).
  These are demonstration thresholds, not an industry standard.

### Validation protocol (windows fixed before evaluation)

| Window | Origins | Use |
|---|---|---|
| Train | from 2021-03-26, expanding | each fold trains only on origins whose outcome ends before the fold |
| Selection | 2022-07-01 → 2023-06-30 | choose the configuration (lowest raw P50 MAE) |
| Calibration | 2023-07-01 → 2024-06-30 (52 periods) | estimate the P10/P50/P90 offsets once, then **freeze** them |
| Test | 2024-07-05 → 2026-09-24 (116 periods) | untouched evaluation with the frozen offsets |

The deployed artifact is a **post-evaluation refit**: the selected configuration is refit on all
complete origins and carries the frozen calibration offsets. The reported metrics describe the
backtest procedure, not the refit. All of this is recorded in `models/reports/production_validation.json`
(`protocol`) and in `models/model_manifest.json` (`production.validation_protocol`). Tests check that
the calibration periods end before the test window and that test forecasts equal
raw + frozen offsets.

### Production results (test window, synthetic operations)

| | MAE (t) | RMSE (t) | R² | MAPE |
|---|---|---|---|---|
| GEO-MN P50 | **844** | 1145 | 0.563 | 9.8 % |
| Previous period (naive) | 892 | 1201 | 0.520 | 10.4 % |
| 4-period moving average | 953 | 1296 | 0.441 | 11.0 % |
| *Diagnostic: true period conditions (not forecast skill)* | *418* | *516* | *0.911* | *4.9 %* |

* The P50 beats the best baseline by 5.4 %. In the selection window it also beat the baselines
  (1029 vs 1130–1157 t).
* Pinball loss P10 / P50 / P90: 234 / 422 / 230 t.
* **P10–P90 observed coverage 0.78** vs nominal 0.80 (0.28 before calibration). Hit rates are
  0.13 / **0.40** / 0.91. The P50 hit rate falls outside the ±0.08 rule, so
  **`quantiles_validated = false`**.
* The API returns `quantile_validation` {status, nominal_coverage, observed_coverage,
  evaluation_window}. The UI shows "P10–P90 interval not validated — indicative only", with the
  observed and nominal values. It never claims "80 % of outcomes fall inside".
* Shortfall events (actual < 95 % of target): precision 0.82, recall 0.87, F1 0.84.
* Reconciliation of the last 26 test periods: MAE ≈ 1.1 kt, 13 over- and 13 under-forecasts. This is
  a historical diagnostic, not proof of future accuracy.
* All figures come from synthetic operations and say nothing about real mine accuracy, which requires
  mine-level operational data.

## Recovery and contingency

* **Scenarios:** NORMAL, HEAVY_RAIN, EQUIPMENT_DEGRADATION, BLAST_DELAY, COMBINED_DISRUPTION.
* **Actions:** A1 equipment recovery (+8 points availability; lost hours re-derived), A2 schedule
  adjustment (+3 trucks, ×0.7 haulage delay), A3 blast/drill delay reduction (×0.5 blast, ×0.6 drill).
  All 7 combinations are evaluated, plus no-action.
* **Modelled feasibility (`modelled_feasibility`, alias `physical_feasibility`):** an input-constraint
  check only. It checks availability within [0, mine cap], trucks ≤ fleet, delays ≥ 0 and lost hours
  consistent with availability. Results are FEASIBLE, FEASIBLE_CONSTRAINED, NOT_FEASIBLE or
  UNKNOWN — HUMAN REVIEW. It is not a mine-plan, geotechnical, blasting or crew assessment, so
  planner/site confirmation is required.
* **Applicability gate:** a portfolio is **eligible for automated selection** only if it is
  modelled-feasible **and** none of its tested scenarios has LOW model applicability
  (`selection_applicability` = worst over its scenarios).
  * HIGH is eligible.
  * MODERATE is eligible with reduced confidence (reason code `REDUCED_CONFIDENCE_MODERATE_APPLICABILITY`).
  * LOW is **not eligible**. It is still returned per scenario as a diagnostic, with `selection_blocked_reason`.

  If nothing is eligible, `selection_status = REVIEW_REQUIRED`, `selected_portfolio = null` and the
  contingency engine returns `REVIEW_REQUIRED` (`NO_ELIGIBLE_OPERATIONAL_PORTFOLIO`). The one
  exception is a near-term case already on track, where no action is needed.
  The gate matters in practice. In healthy states near the top of the training range (e.g. DEMO_A),
  portfolios containing equipment recovery push availability past the highest value in training
  (0.94), so they are blocked rather than optimised over.
* **Selection among eligible portfolios** (a lightweight robust-recourse approximation, not a mine
  scheduler). Tolerance for "equivalent" is 0.5 % of target.
  1. Minimise the worst-case residual P50 gap over scenarios.
  2. If equivalent, minimise the mean residual gap.
  3. If equivalent, minimise **intervention burden**. Weights are in
     `config/recovery_config.json → intervention_burden`: equal project weights of 1 per action, not
     cost data.
  4. Then prefer fewer actions, then portfolio id.

  `selection_explanation` states which rule decided. A higher-burden portfolio is chosen only when it
  materially lowers the residual gap.
* **Horizon gate:**
  * NEAR_TERM: exploration is never recovery. Residual ≤ 2 % of target → `OPERATIONAL_RESPONSE`;
    otherwise → `REVIEW_REQUIRED`.
  * STRATEGIC: if the expected residual is > 2 % of target, or the worst case is > 8 %, targets are
    ranked under the current gap state. The best target with applicability ≥ MODERATE gives
    `OPERATIONAL_AND_EXPLORATION_CONTINGENCY`. `why_target_now` and `next_evidence` are returned
    only in this case. If no target qualifies → `REVIEW_REQUIRED`.
  * An out-of-distribution production state → `REVIEW_REQUIRED`.

  There are exactly three decision states. `ON_TRACK` / `OPERATIONALLY_RECOVERABLE` /
  `RESIDUAL_GAP` are supply statuses, not decision states.
* **Decision flip:** `POST /api/decision/flip` recomputes the full decision twice (baseline = mine
  state, perturbed = user conditions). It returns both results, the changed inputs, the backend
  `flipped` boolean and the transition. An out-of-distribution perturbation returns `REVIEW_REQUIRED`
  with no portfolio and no target.

### Demo states (inputs only; outcomes computed and asserted by tests)

| mine_id | demo_state | computed decision |
|---|---|---|
| DEMO_A | ON_TRACK | OPERATIONAL_RESPONSE (no action required) |
| DEMO_B | OPERATIONALLY_RECOVERABLE | OPERATIONAL_RESPONSE |
| DEMO_C | STRATEGIC_CONTINGENCY | OPERATIONAL_AND_EXPLORATION_CONTINGENCY |
| DEMO_D | REVIEW_REQUIRED | REVIEW_REQUIRED (inputs out of distribution) |
| DEMO_E | SATELLITE_FALLBACK | contingency, plus live failure → cached ≤ 2 km / unavailable > 2 km |
| DEMO_F | DECISION_FLIP | OPERATIONAL_RESPONSE (portfolio A2+A3) → OPERATIONAL_AND_EXPLORATION_CONTINGENCY (A1+A3, target T14) when rainfall 5→140 mm, availability 0.85→0.74, blast delay 0.4→3 h/day |
| DEMO_MINE | (current state) | STRATEGIC: contingency; NEAR_TERM: operational response |

## API (all JSON; errors are `{"error": CODE, "message": ...}`)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | component flags only, no secrets |
| GET | `/api/supply-command?mine_id=&horizon=` | whole loop in one response; DEMO_E/F add fallback / flip blocks |
| GET | `/api/exploration/targets?mine_id=&horizon=` | priority computed under that mine's strategic state |
| GET | `/api/exploration/targets/{id}` | evidence, why-this-target, why-now, next evidence |
| POST | `/api/exploration/predict` | `{lat, lon, mode?: AUTO\|LIVE_ONLY\|CACHED_ONLY\|SIMULATE_LIVE_FAILURE}` |
| GET | `/api/exploration/grid?stride=` | sampled ~1 km `prospectivity_rank` grid for maps |
| POST | `/api/production/forecast` | `{mine_id, forecast_origin?, horizon_days?=7, target_tonnes?, conditions?}` |
| GET | `/api/production/history?mine_id=&days=` | daily rows + 7-day periods |
| GET | `/api/production/reconciliation?mine_id=` | out-of-sample forecast vs actual, error = forecast − actual |
| POST | `/api/recovery/evaluate` | contract form or frontend form `{scenario, actions, conditions}` |
| POST | `/api/contingency/evaluate` | recomputes server-side; client forecast/recovery objects are not trusted |
| POST | `/api/decision/flip` | `{mine_id, baseline_conditions, perturbed_conditions, horizon?}` |
| POST | `/api/decision/review`, GET `/api/decision/history` | **local demo review log** (JSONL file on the server; not a durable audit trail — production needs managed storage) |
| GET | `/api/trust/exploration`, `/api/trust/production`, `/api/trust/provenance` | computed metrics only |
| GET | `/api/model/manifest`, `/api/demo/scenarios` | |

Accepted condition aliases: `rainfall_mm` → `rainfall_7d_mm` (7-day total),
`blast_delay_hours` → `blast_delay_h`, `drilling_delay_hours`, `haulage_delay_hours`, `soil_moisture`,
`temperature`. Values outside physical ranges → 400.

## Frontend (index.html / app.js / style.css)

* The five screens are Supply Command, Exploration, Production Risk, Recovery & Contingency and Model
  Trust. No other top-level modules.
* **Demo-state selector** (top bar): the list comes from `GET /api/demo/scenarios`. Choosing a state
  only changes `mine_id` and reloads backend results; there is no decision logic in JavaScript. It is
  labelled DEMO / SYNTHETIC.
* Every number, decision, portfolio ranking, target priority and interval statement comes from the
  backend. Interval wording is built from `quantile_validation`.
* The map heat layer uses `prospectivity_rank / 100`. The MODIS LST tile is labelled a *context layer*
  (a 2024-05-01 NASA GIBS image, not the model feature). Tile or Leaflet failures show a non-blocking
  notice and never affect decisions.
* Frontend files are served with `Cache-Control: no-cache`, so a redeploy never leaves a stale UI.

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
* P10/P90 did not pass the quantile validation rule on the untouched test window (P50 hit rate 0.40).
  They are shown as indicative only.
* The applicability gate can block useful actions when a state sits at the edge of the training
  range. This is deliberate: the model cannot speak for states it has not seen.
* The decision review log is a local file (lost on ephemeral hosts).
* Prospectivity is a weak-to-moderate relative signal (spatial ROC-AUC ≈ 0.69) trained on 73
  clustered MRDS points of mixed location precision. Subsurface evidence is unavailable.
* Intervention effects are learned associations through a monotonic model, not causal estimates.
* The strategic horizon is represented as persistence of the per-period residual gap (13 periods),
  not a multi-period mine plan.
* Priority weights, risk bands and materiality thresholds are project defaults.
