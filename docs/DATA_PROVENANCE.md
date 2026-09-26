# Data provenance

Every major API response carries a `provenance` block with `data_mode`, `model_version`,
`observation_window`, `source_timestamp` and `fallback_used`, plus per-input modes
(e.g. `operations_mode`, `weather_mode`, `geology_features_mode`). The UI shows these as badges.
`GET /api/trust/provenance` returns the full catalogue.

## Modes

| Mode | Badge | Meaning | Examples |
|---|---|---|---|
| REAL_GOVERNMENT | ■ REAL / GOVERNMENT OF INDIA | Official Government of India data | IMD rainfall and Tmax; NRSC Bhuvan geomorphology and lineaments; NMET / DGM / MECL block records |
| REAL_PUBLIC | ● REAL / PUBLIC | Public international data | USGS MRDS; Sentinel-2 L2A; MODIS; NASADEM; ERA5 |
| REAL_MOIL_PUBLIC | ▲ REAL / MOIL PUBLIC DISCLOSURE | MOIL Ltd. investor disclosures, company level | Quarterly production; reported exploration |
| REAL_DERIVED | ○ REAL-DERIVED FEATURES | Features computed from real data with a documented recipe | EO indices; lineament distance |
| SYNTHETIC | SYNTHETIC DEMONSTRATION DATA | Seeded simulator output | Demo-mine operations, equipment events |
| SIMULATED | ◇ SIMULATED SCENARIO | Scenario or counterfactual output | Disruptions, action effects, subsurface scenarios, demo states |
| CACHED | CACHED / REPLAY | Precomputed from the above, replayed | Prospectivity grid; cached coordinate fallback |
| LIVE_COORDINATE_QUERY | LIVE COORDINATE QUERY | Features computed on request from MPC | Exploration predict (when enabled) |
| UNAVAILABLE | UNAVAILABLE | No data. Shown instead of filling the gap. | Reserves; observed subsurface for most targets |

## Per product area

| Area | Inputs and modes |
|---|---|
| Exploration | Labels: MRDS (REAL_PUBLIC). Model B / C labels may add NMET cells (REAL_GOVERNMENT). Features: EO (REAL_DERIVED from REAL_PUBLIC); geomorphology and lineaments (REAL_GOVERNMENT). Grid: CACHED. Observed subsurface: REAL_GOVERNMENT at REPORTED_BLOCK_LEVEL. What-if subsurface: SIMULATED. |
| Production Risk (weekly) | Operations: SYNTHETIC. Weather: REAL_GOVERNMENT (IMD) with REAL_PUBLIC ERA5 soil moisture. Forecast: model estimate. |
| Production Risk (real quarterly panel) | REAL_MOIL_PUBLIC company totals; forecast validated on real held-out quarters. |
| Recovery | Scenario and action effects: SIMULATED (simulator counterfactuals). Tonnes: production-model estimate. |
| Supply Command / contingency | Combination of the above. Demo states DEMO_A–F: SIMULATED overrides. |
| Model Trust | Metrics computed from held-out evaluations; nothing is typed in by hand. |

## Traceability

* **Raw real downloads**: `data/raw/…`. The large files (IMD grids, Bhuvan GeoJSON) are gitignored
  and re-fetched by `scripts/data_acquisition/*`.
* **Manifests** (`data/manifests/real_*.json`): provider, official URL, retrieval timestamp, licence
  note, counts, sha256.
* **NMET PDFs**: sha256 recorded in `data/raw/subsurface/nmet_documents.csv`. Transcribed values keep
  the quoted text.
* **Synthetic / simulated manifests**: generator, version, seed, timestamp, assumptions, sha256 and
  row counts. Tests check the sha256 values against the committed files.
