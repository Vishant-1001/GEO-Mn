# Synthetic and simulated data in GEO-MN

GEO-MN uses real data wherever a usable public source exists (see `DATA_SOURCE_AUDIT.md`).
Synthetic data are generated **only for verified gaps**. Every synthetic dataset is:

* **Labelled**: `provenance` is `SYNTHETIC` (generated history) or `SIMULATED` (scenario or
  counterfactual output). IDs start with `SYN_` / `SIM_`.
* **Reproducible**: fixed seed (42) in a YAML config. Re-running the generator reproduces the files
  byte for byte, and the manifests store their sha256.
* **Domain-constrained**: parameters are either cited real values (IMD thresholds, NMET-reported
  thicknesses and grades, IBM grade classes) or marked `assumption` in the config.
* **Consumed**: each dataset is read by a named engine (table below). Nothing is generated just
  to fill space.

**What synthetic data is never used for:**

* It is never presented as MOIL telemetry.
* Simulated drilling or assays are never presented as observed.
* It is not a training label for exploration: the subsurface scenarios are generated from the
  model's own targets, so training on them would be circular.
* It is not in any headline real-data metric. Synthetic augmentation of the real MOIL quarterly
  model was tested and made the real held-out error worse, so it is excluded (see
  `MODEL_COMPARISON_REPORT.md`).

## Datasets

| Dataset | Rows | Mode | Generator | Consumer |
|---|---|---|---|---|
| `data/production_history.csv` | 2,093 days | SYNTHETIC ops + REAL weather | `ml/generate_operations.py` | Weekly production engine (training and forecast state), recovery, contingency |
| `data/synthetic/production/synthetic_mine_operational_history.csv.gz` (+ `.parquet`, `_sample.csv`) | 115,530 unit-shifts | SYNTHETIC | same | Aggregation source of the daily history; inspection |
| `data/synthetic/production/synthetic_equipment_events.csv` | 9,428 events | SYNTHETIC | same | Inspection; event statistics in DATA_QUALITY_REPORT |
| `data/synthetic/production/synthetic_disruption_scenarios.csv` | 1,001 | SIMULATED | `scripts/synthetic/generate_recovery.py` | Scenario library; severity statistics |
| `data/synthetic/recovery/synthetic_action_outcomes.csv` | 8,008 | SIMULATED | same | Source of the recovery matrix; trust comparison |
| `data/synthetic/recovery/recovery_scenario_matrix.csv` | 56 | SIMULATED | same | **`services/recovery_service.py`**: every disruption and action effect in feature space |
| `data/synthetic/subsurface/synthetic_boreholes.csv` | 360 | SIMULATED | `scripts/synthetic/generate_subsurface.py` | **`services/subsurface_service.py`** (what-if scenarios) |
| `data/synthetic/subsurface/synthetic_borehole_intervals.csv` | 1,316 | SIMULATED | same | same |
| `data/synthetic/subsurface/synthetic_geochemistry.csv` | 1,636 | SIMULATED | same | same |
| `data/synthetic/subsurface/synthetic_geophysics.csv` | 160 | SIMULATED | same | same |
| `data/synthetic/subsurface/synthetic_subsurface_targets.csv` | 240 | SIMULATED | same | same (per-target summary) |

## Regenerating

```bash
python scripts/synthetic/generate_all.py            # operations -> recovery -> subsurface
python -m ml.run_pipeline                           # full offline rebuild incl. models
```

Method details: `SYNTHETIC_GENERATION_METHOD.md` (operations and recovery),
`SYNTHETIC_SUBSURFACE_METHOD.md` (subsurface). Column definitions: `DATA_DICTIONARY.md`.
