# Synthetic data implementation report

## Gap verification (why each dataset exists)

| Dataset | Real alternative searched | Result | Decision |
|---|---|---|---|
| Mine-level operations | MOIL disclosures, Ministry of Mines statistics, OGD, AIKosh | Only company totals (quarterly) and state or national aggregates exist | Build a SYNTHETIC equipment-level simulator driven by REAL IMD weather |
| Recovery-action outcomes | MOIL reports | No intervention records | SIMULATED counterfactuals from the same simulator |
| Subsurface drilling / assays / geophysics | NMET, GSI NGDR (login), Bhukosh (unreachable), AIKosh (outside AOI) | Block-level narratives only | SIMULATED scenarios for what-if analysis. Observed evidence is used only at block level. |
| Weather | IMD | Real series available for 2012–2025 | **No synthetic weather.** ERA5 fills 2026 and is labelled per row. |
| Exploration labels | NMET / MRDS | Real | **No synthetic labels.** |

## What changed from the previous generator

The previous `generate_operations.py` drew daily values from parametric noise around a capacity
curve. The new generator (`ops-sim-2.0`) is a process simulator: unit-level failures with ageing
and preventive maintenance, a drill → blast → inventory → load → haul chain, fleet regimes, and
shared shocks. Every daily number can be traced to unit-shift records and events.

The same simulator produces the recovery counterfactuals, so disruption and action effects are
internally consistent with the training history. They are no longer hand-typed constants in
`config/recovery_config.json`, which now keeps only labels, floors and burden weights.

## Does synthetic data help?

| Question | Test | Result |
|---|---|---|
| Do synthetic operations features improve **real** MOIL quarterly forecasts? | Model A (real only) vs model B (+ simulated availability / blast / haulage), on the same 8 real quarters | **No.** MAE 44,772 → 60,486 t. Excluded from real forecasting. |
| Does synthetic subsurface belong in exploration training? | Design review | **No.** It is generated from model targets (circular), so it is used for what-if analysis only. |
| Is the weekly engine useful on synthetic operations? | Untouched test window vs persistence baselines | P50 MAE only 0.5 % better than MA4 (a tie); calibrated P10–P90 coverage 0.80 (validated). The engine's value is conditional scenario response, not point skill. |

## Consumption map

| Dataset | Read by | Visible in |
|---|---|---|
| `production_history.csv` | `services/production_service.py` (training, current state); `ml/train_production.py` | Production Risk history and forecast |
| `recovery_scenario_matrix.csv` | `services/recovery_service.py` (`scenario_matrix()`, `apply_disruption`, `apply_actions`) | Recovery portfolios table, Supply Command |
| `synthetic_action_outcomes.csv`, `synthetic_disruption_scenarios.csv` | Matrix derivation; tests | — |
| `synthetic_subsurface/*` | `services/subsurface_service.py` | Exploration → Subsurface scenario simulator |

## Reproducibility

`python scripts/synthetic/generate_all.py` regenerates everything from the YAML configs (seed 42).
Manifests store sha256 and row counts, and tests check them against the committed files. The
simulator and the subsurface generator are also checked for seed determinism in isolation.
