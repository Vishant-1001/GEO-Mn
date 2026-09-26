# GEO-MN backend — supply-continuity decision support

SIH 2026 · Problem Statement 26009 (text in `docs/26009ps.txt`).

GEO-MN is an uncertainty-aware manganese supply-continuity decision-support system. It forecasts the
next production period, estimates how far robust operational recovery can close a supply gap, and —
only when a residual gap persists over a **strategic** horizon — activates an exploration contingency
and ranks exploration targets by their relevance to that gap.

```
forecast (P10/P50/P90) -> model contributions -> robust recovery (7 scenarios x 8 portfolios, simulator-defined)
 -> residual gap -> horizon gate -> [strategic] exploration target priority -> why this target now
 -> decision flip -> human review -> reconciliation
```

## What is real, what is synthetic

Full audit: `docs/DATA_SOURCE_AUDIT.md`. Mode definitions: `docs/DATA_PROVENANCE.md`.

| Component | Mode | Source |
|---|---|---|
| Exploration labels | REAL_PUBLIC (+ REAL_GOVERNMENT in label set B) | USGS MRDS manganese records (73); NMET block / sample evidence |
| Spectral / thermal / terrain features | REAL_DERIVED | Sentinel-2 L2A, MODIS MOD11A2, NASADEM (Microsoft Planetary Computer), fixed 2024 window |
| Geomorphology / lineaments | REAL_GOVERNMENT | NRSC Bhuvan 1:50k (MP + MH layers) |
| Geology context | REAL_PUBLIC | Macrostrat → GSC *Generalized geology of the world* (world scale, context only) |
| Observed subsurface | REAL_GOVERNMENT | NMET / DGM / MECL block records: REPORTED_BLOCK_LEVEL where a target overlaps a block |
| Real production | REAL_MOIL_PUBLIC | MOIL quantitative-details disclosures → 54 company-total quarters |
| Weather in production data | REAL_GOVERNMENT | IMD gridded rainfall / Tmax (2021–2025); ERA5-Land soil moisture; ERA5 for 2026 |
| Demo-mine operations | **SYNTHETIC** | equipment-level simulator `ml/synthetic_ops.py`, seed 42 — **not MOIL data** |
| Disruptions, actions, demo states | SIMULATED | simulator counterfactuals (`recovery_scenario_matrix.csv`), `data/demo_scenarios.json` |
| Subsurface where no official record exists | UNAVAILABLE | nothing is fabricated; next-evidence sensitivity is rule-based only |
| Reserve / resource tonnage | UNAVAILABLE | never produced or implied |

Satellite composites use the **2024-01-01/2024-12-31** observation window. Nothing is real-time.

## Layout

```
main.py                     FastAPI routes only (thin)
services/                   runtime logic
  production_service.py     history, P10/P50/P90 forecast, risk policy, SHAP contributions, reconciliation
  recovery_service.py       scenario x portfolio evaluation from the simulator matrix, feasibility, robust selection
  subsurface_service.py     observed subsurface/ground evidence + rule-based next-evidence priority sensitivity
  real_production_service.py  REAL MOIL quarterly series, validated forecast and baselines
  contingency_service.py    horizon gate, 3 decision states, target selection, decision flip
  exploration_service.py    live/cached point query, cache-distance guard, target priority, why-now
  trust_service.py          validation metrics (read from reports), provenance catalogue
  decision_service.py       append-only human review log
  health_service.py, demo_service.py, common.py
ml/                         offline pipeline (python -m ml.run_pipeline [--fetch])
  eo_features.py            the ONE feature recipe used for training, grid and live queries
  build_exploration_dataset.py, train_exploration.py, target_engine.py, ood.py, uncertainty.py
  fetch_weather.py, generate_operations.py, synthetic_ops.py,
  production_features.py, train_production.py, evaluate_production.py, train_real_production.py,
  exploration_experiments.py
scripts/                    data_acquisition/ (IMD, MOIL, Bhuvan), data_processing/, synthetic/ (generate_all.py)
data/                       raw/ processed/ synthetic/ manifests/ + engine inputs (docs/DATA_DICTIONARY.md)
models/                     models, model_manifest.json, reports/*.json, legacy/ (previous prototype)
config/                     risk_policy.json, exploration_config.json, recovery_config.json, demo_config.json
tests/                      pytest suite (network-free)
```

## Exploration engine

* **Study area:** 20.75–22.75 N, 78.5–81.0 E (Balaghat – Nagpur – Bhandara Mn belt), 50,000 cells of
  0.01° (~1.1 km × 1.0 km). The effective resolution is ~1 km, not the 10 m Sentinel-2 pixel.
* **Features (18, selected by `ml/exploration_experiments.py`):** Sentinel-2 NDVI, B4/B2 iron oxide,
  B11/B12 clay/hydroxyl, NDRE, B4/B3 ferric, B12/B8A ferrous silicate, B11/B4 gossan, SWIR albedo
  (2024 SCL-masked median of the 12 least-cloudy scenes per tile, offset-corrected). MODIS daytime LST.
  NASADEM elevation, slope and elevation SD. NRSC 1:50k geomorphology flags and dissection
  (REAL_GOVERNMENT; NaN outside the MP/MH layers, handled natively). Cells missing EO features are
  `INSUFFICIENT_DATA` (78 of 50,000), never mean-filled.
* **Labels:** positive-unlabelled design. 72 MRDS cells + 3 NMET cells (Katori XRF samples,
  Nagardhan and Kawalewada block centroids). Background cells are > 3 km from every positive and are
  treated as *unlabelled*, not barren.
* **Model:** PU-bagging ensemble of 15 `HistGradientBoostingClassifier`s (`exploration-pu-ensemble-2.0`).
  The output is a **relative percentile rank (0–100)** within the study area, never a deposit
  probability.
* **Uncertainty:** SD of the member ranks (LOW < 5, HIGH > 10 rank points; project thresholds).
* **Applicability:** geographic envelope + IsolationForest on the EO features (MODERATE below the
  5 % and LOW below the 1 % training score quantile) + a hard training-range envelope (any EO feature
  outside the study-area range → LOW). High rank + LOW applicability → `REVIEW_REQUIRED`.
* **Targets:** cells with rank ≥ 95 → 8-connected clusters (≥ 4 cells; clusters > 50 cells split by
  seeded k-means) → 40 targets with footprint polygons, evidence and priority inputs.
* **Evidence levels:**
  * L0: remote sensing.
  * L1: plus a Precambrian host domain, or official structural / geological context.
  * L2: plus a documented MRDS record within ~1 km, or official mapping / grade / pit / surface
    geochemistry.
  * L3: an overlapping official block **reports** a drilling intersection. Only T07 (Nagardhan)
    qualifies; its `subsurface_status` is `REPORTED_BLOCK_LEVEL`, with no public logs or assays.
  * L4: never reached.

  Targets overlapping any training label (MRDS or NMET) are flagged `CAUTION_TRAINING_LABEL_OVERLAP`
  (in-sample rank).
* **Subsurface / ground evidence:** `/api/exploration/targets/{id}/subsurface-scenarios` returns the
  observed record (REAL_GOVERNMENT block evidence, or `UNAVAILABLE` with "requires additional
  ground/subsurface validation"), the next required evidence, and a rule-based **next-evidence
  sensitivity**: how the investigation priority *would* change for each possible outcome of the next
  investigation (`config/exploration_config.json → subsurface_fusion`). No boreholes, assays or
  geophysical values are generated, and observed records are never changed.
* **Priority (project defaults):**
  * Hard gates first, applied before scoring: inside the study area → applicability ≥ MODERATE →
    evidence ≥ L1. Gated targets rank after eligible ones, and only gated-in targets can be selected
    for contingency.
  * Contingency OFF: 50 % prospectivity + 30 % evidence/applicability/certainty, renormalised.
  * Strategic contingency ON: adds 20 % strategic relevance (gap severity × distance decay to the
    supply point).
  * No development-readiness proxy is scored, because no defensible development data exist.
  * Prospectivity itself never changes with the supply state; only the priority to investigate does.
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

Full details: `docs/MODEL_COMPARISON_REPORT.md`. Selection was made on the eastern development
region only. The western region (15 MRDS cells) was scored once, as a confirmation gate.

| Model (final test, west; 4-seed mean) | ROC-AUC | PR-AUC (prevalence 0.0006) | Positives in top 10 % of area |
|---|---|---|---|
| A — MRDS labels, six baseline features | 0.697 | 0.0021 | 25 % |
| B — + NMET labels | 0.701 | 0.0022 | 30 % |
| **C — + S2 red-edge/SWIR, relief, geomorphology (deployed)** | **0.803** | **0.0128** | **43 %** |

The deployed model on the full area (5-fold spatial CV) scores ROC 0.719, PR-AUC 0.0044 (prevalence
0.0015), and captures 31 % of positives in the top 10 %. East/west holdout pooled ROC is 0.695.

The evidence is **mixed**. Under the same protocol, the six-feature baseline has the higher
full-area PR-AUC (0.021), and training on the west does not improve performance in the east. The
gain comes from the features, not the three NMET labels. Model D (real + synthetic) was not run: the
no synthetic exploration data exist (fabricated subsurface evidence is not generated).

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

### Production results (test window 2024-07-05 → 2026-09-24, 116 periods, synthetic operations)

| | MAE (t) | RMSE (t) | R² | WAPE |
|---|---|---|---|---|
| GEO-MN P50 | **1144** | 1589 | 0.213 | 13.2 % |
| Previous period (naive) | 1197 | 1698 | 0.101 | — |
| 4-period moving average | 1150 | 1569 | 0.232 | — |
| *Diagnostic: true period conditions (not forecast skill)* | *520* | *976* | *0.703* | — |

* The P50 is **0.5 % better** than the best baseline on the test window, which is effectively a tie.
  In the selection window the raw P50 (1139 t) lost to the previous-period baseline (1118 t). The
  equipment-level simulator produces bursty regimes (breakdown clusters, explosive-supply
  stoppages, one zero-output week), so persistence is hard to beat. The trust panel says so, and
  the model's value here is its conditional scenario response and calibrated interval.
* MAPE is not reported because the zero-output week makes it undefined. WAPE is 13.2 %, and MAPE
  excluding that one near-zero period is 14.9 %.
* Pinball loss P10 / P50 / P90: 299 / 572 / 273 t.
* **P10–P90 observed coverage 0.80** vs nominal 0.80 (0.36 before calibration). Hit rates are
  0.12 / 0.48 / 0.92, all within ±0.08, so **`quantiles_validated = true`**.
* The API returns `quantile_validation` {status, nominal_coverage, observed_coverage,
  evaluation_window}. Interval wording in the UI is built from it.
* Shortfall events (actual < 95 % of target): precision 0.86, recall 0.90, F1 0.88.
* All figures come from synthetic operations and say nothing about real mine accuracy.

### Real MOIL quarterly production (REAL_MOIL_PUBLIC)

`ml/train_real_production.py` works at company level, quarterly, one step ahead. Candidates are
chosen on FY2019-20..FY2022-23 and tested on FY2023-24..FY2025-26 (12 real quarters, untouched).

| Method (test) | MAE (t) | MAPE |
|---|---|---|
| ridge, no weather (selected on the selection window) | 52,703 | 11.7 % |
| ridge + IMD rainfall anomaly | 52,841 | 11.7 % |
| GBM + weather | 61,478 | 14.2 % |
| seasonal naive × YoY growth (baseline) | **38,837** | **8.6 %** |
| last quarter (baseline) | 40,500 | 9.2 % |

The selected model **does not beat** the baselines on the untouched test. The API
(`/api/production/real-quarterly`) and the UI say this, and show the baseline forecasts next to the
model forecast. Adding synthetic-operations features (model B) made real test error worse
(60,486 vs 44,772 t MAE on the 8 common quarters), so synthetic augmentation is **not** used for
real forecasting. Weather adds nothing at company-quarter level.

## Recovery and contingency

* **Scenarios (7):** NORMAL, HEAVY_RAIN, EQUIPMENT_DEGRADATION, BLAST_DELAY, DRILL_DELAY,
  HAULAGE_DISRUPTION, COMBINED_DISRUPTION.
* **Actions:** A1 equipment recovery, A2 schedule adjustment, A3 blast/drill delay reduction. All 7
  combinations are evaluated, plus no-action.
* **Where the effects come from:** `data/synthetic/recovery/recovery_scenario_matrix.csv`. Every
  scenario and portfolio is the median change in each model input, measured by re-running the
  equipment-level simulator from 143 historical states with common random numbers
  (`docs/SYNTHETIC_GENERATION_METHOD.md`). The heavy-rain intensity is the IMD 95th-percentile
  7-day total (148 mm). The production model converts the changed state into tonnes. Deltas below
  a materiality threshold (0.005 availability, 0.1 trucks, 0.05 h) count as "no material effect". A
  portfolio whose material levers are all capped is NOT_FEASIBLE ("no headroom").
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
  The gate matters in practice. When the current state is already degraded (e.g. the latest
  DEMO_MINE week, availability 0.73), stacking EQUIPMENT_DEGRADATION or HAULAGE_DISRUPTION pushes
  inputs below anything seen in training. Only portfolios that restore availability or trucks stay
  eligible.
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
  state, perturbed = user conditions). It returns:
  * both results (forecast P10/P50/P90, target, gap, residual gaps, portfolio, contingency on/off,
    why-this-target-now);
  * the changed inputs and `deltas`;
  * `exploration_priority_changes`, i.e. rank and priority under both supply states (prospectivity
    unchanged);
  * the backend `flipped` boolean and the transition. An out-of-distribution perturbation returns `REVIEW_REQUIRED`
  with no portfolio and no target.

### Demo states (inputs only; outcomes computed and asserted by tests)

| mine_id | demo_state | computed decision |
|---|---|---|
Demo inputs were re-set in `demo-scenarios-1.2` after the simulator changed, so every state lies
inside the new training range (truck count 10.9–16.9). No model was tuned to the demos.

| mine_id | demo_state | computed decision |
|---|---|---|
| DEMO_A | ON_TRACK | OPERATIONAL_RESPONSE (no action required) |
| DEMO_B | OPERATIONALLY_RECOVERABLE | NEAR_TERM: OPERATIONAL_RESPONSE; the same state under STRATEGIC → contingency (horizon-gate test) |
| DEMO_C | STRATEGIC_CONTINGENCY | OPERATIONAL_AND_EXPLORATION_CONTINGENCY |
| DEMO_D | REVIEW_REQUIRED | REVIEW_REQUIRED (inputs out of distribution) |
| DEMO_E | SATELLITE_FALLBACK | contingency, plus live failure → cached ≤ 2 km / unavailable > 2 km |
| DEMO_F | DECISION_FLIP | OPERATIONAL_RESPONSE → OPERATIONAL_AND_EXPLORATION_CONTINGENCY when rainfall 5→140 mm, availability 0.88→0.74, blast delay 0.4→3 h/day |
| DEMO_MINE | (current state) | STRATEGIC: contingency; NEAR_TERM: REVIEW_REQUIRED (latest synthetic week is degraded, residual not closable) |

## API (all JSON; errors are `{"error": CODE, "message": ...}`)

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | component flags only, no secrets |
| GET | `/api/supply-command?mine_id=&horizon=` | whole loop in one response; DEMO_E/F add fallback / flip blocks |
| GET | `/api/exploration/targets?mine_id=&horizon=` | priority computed under that mine's strategic state |
| GET | `/api/exploration/targets/{id}` | evidence, why-this-target, why-now, next evidence |
| GET | `/api/exploration/targets/{id}/subsurface-scenarios?scenario=` | observed (REAL_GOVERNMENT) vs SIMULATED scenarios, priority before/after |
| POST | `/api/exploration/predict` | `{lat, lon, mode?: AUTO\|LIVE_ONLY\|CACHED_ONLY\|SIMULATE_LIVE_FAILURE}` |
| GET | `/api/exploration/grid?stride=` | sampled ~1 km `prospectivity_rank` grid for maps |
| POST | `/api/production/forecast` | `{mine_id, forecast_origin?, horizon_days?=7, target_tonnes?, conditions?}` |
| GET | `/api/production/history?mine_id=&days=` | daily rows + 7-day periods |
| GET | `/api/production/real-quarterly?last_n=` | REAL MOIL company quarters, held-out validation, model vs baseline forecasts |
| GET | `/api/production/reconciliation?mine_id=` | out-of-sample forecast vs actual, error = forecast − actual |
| POST | `/api/recovery/evaluate` | contract form or frontend form `{scenario, actions, conditions}` |
| POST | `/api/contingency/evaluate` | recomputes server-side; client forecast/recovery objects are not trusted |
| POST | `/api/decision/flip` | `{mine_id, baseline_conditions, perturbed_conditions, horizon?}` |
| POST | `/api/decision/review`, GET `/api/decision/history` | **local demo review log** (JSONL file on the server; not a durable audit trail — production needs managed storage) |
| GET | `/api/trust/exploration`, `/api/trust/production`, `/api/trust/provenance` | computed metrics only |
| GET | `/api/trust/recovery?mine_id=` | scenarios tested, eligible / applicability-blocked portfolios, no-action vs selected worst-case residual, burden, constraint notes (no "accuracy") |
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

* Demo-mine operational data are synthetic. Every production, recovery and decision number
  demonstrates the method; none is a statement about a real MOIL mine.
* The weekly P50 is only marginally better than persistence baselines on synthetic data. The real
  MOIL quarterly model does not beat seasonal naive on real held-out quarters.
* The applicability gate can block useful actions when a state sits at the edge of the training
  range. This is deliberate: the model cannot speak for states it has not seen.
* The decision review log is a local file (lost on ephemeral hosts).
* Prospectivity is a weak-to-moderate relative signal trained on 73 clustered MRDS points of mixed
  location precision. Observed subsurface evidence exists only at block level for one target.
* Intervention effects are simulator counterfactuals scored by an associative model. They are not
  historical MOIL interventions or causal estimates.
* The strategic horizon is represented as persistence of the per-period residual gap (13 periods),
  not a multi-period mine plan.
* Priority weights, risk bands and materiality thresholds are project defaults.
