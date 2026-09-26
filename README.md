# GEO-MN — manganese supply-continuity decision support

SIH 2026 · Problem Statement 26009. GEO-MN answers four questions:

1. Are we likely to miss the next production target?
2. Can operations recover the gap?
3. If a gap persists over a strategic horizon, which exploration target should be investigated
   next, and why?
4. How far should each of these answers be trusted?

It has five screens: **Supply Command, Exploration, Production Risk, Recovery & Contingency, Model
Trust**. Every number shown comes from the backend, with a provenance badge.

## What is real and what is not

| Real (Government of India / public) | Synthetic / simulated (labelled as such) |
|---|---|
| IMD gridded rainfall and Tmax (2012–2025) | Demo-mine operations: equipment-level simulator, seed 42. **Not MOIL telemetry.** |
| MOIL public production disclosures → 54 company quarters | Disruption and recovery-action effects (simulator counterfactuals) |
| NRSC Bhuvan 1:50k geomorphology and lineaments | Next-evidence priority sensitivity (rule-based; no records generated) |
| NMET / DGM / MECL exploration-block records (block level) | Demo states DEMO_A–F (documented input overrides) |
| USGS MRDS Mn records; Sentinel-2 / MODIS / NASADEM features; ERA5 | — |

GEO-MN never produces a reserve, resource or tonnage estimate, and never fabricates subsurface
evidence (boreholes, assays, geophysics). Where no official record exists, subsurface evidence is
UNAVAILABLE. Prospectivity is a relative rank, not a probability.

## Headline results (honest version)

* **Exploration (untouched western test region, 4 seeds):** ROC 0.80, PR-AUC 0.013 (prevalence
  0.0006), 43 % of known occurrences in the top 10 % of area. The six-feature baseline scores 0.70 /
  0.002 / 25 %. The evidence on the full area is mixed; see `docs/MODEL_COMPARISON_REPORT.md`.
* **Real MOIL quarterly forecast:** the selected model (MAPE 11.7 %) **does not beat** seasonal naive
  (8.6 %) on 12 held-out real quarters. Synthetic augmentation made it worse, so it is not used.
* **Weekly demo-mine forecast (synthetic):** P50 is effectively tied with persistence baselines.
  The calibrated P10–P90 interval is validated (coverage 0.80).

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                   # network-free test suite
uvicorn main:app --port 8000                # API + UI at http://localhost:8000
python -m ml.run_pipeline                   # deterministic offline rebuild (add --fetch to re-download)
python scripts/synthetic/generate_all.py    # regenerate synthetic / simulated datasets only
```

Environment variables: `GEOMN_LIVE_EO` (1 = live Planetary Computer coordinate queries),
`GEOMN_CORS_ORIGINS`, `GEOMN_DECISION_LOG`, `PORT`. No credentials are required or stored.

## Documentation

| Topic | File |
|---|---|
| Architecture, engines, API | `docs/BACKEND.md` |
| Baseline before this work | `docs/CURRENT_DATA_MODEL_BASELINE.md` |
| Source audit (Indian government portals, access barriers) | `docs/DATA_SOURCE_AUDIT.md` |
| Real-data integration | `docs/REAL_DATA_IMPLEMENTATION_REPORT.md` |
| Synthetic data (what, why, how) | `docs/SYNTHETIC_DATA_README.md`, `docs/SYNTHETIC_GENERATION_METHOD.md`, `docs/SYNTHETIC_DATA_IMPLEMENTATION_REPORT.md` |
| Provenance modes | `docs/DATA_PROVENANCE.md` |
| Columns | `docs/DATA_DICTIONARY.md` |
| Model comparison and experiments | `docs/MODEL_COMPARISON_REPORT.md` |
| Model / data card | `docs/MODEL_DATA_CARD.md` |
| Data quality | `docs/DATA_QUALITY_REPORT.md` |

Large raw downloads (IMD binary grids, Bhuvan GeoJSON) are not committed. The
`scripts/data_acquisition/*` scripts re-fetch them, and the processed outputs they produce are
committed.
