# Data dictionary

Every table carries a `provenance` column (or the whole file has a single stated mode). Modes:
REAL_GOVERNMENT, REAL_PUBLIC, REAL_MOIL_PUBLIC, REAL_DERIVED, SYNTHETIC, SIMULATED, CACHED.

## Real

### `data/processed/production/moil_quarterly_production.csv` (REAL_MOIL_PUBLIC)
| Column | Unit | Meaning |
|---|---|---|
| fy, fy_start_year, fy_quarter | — | Indian fiscal year (Apr–Mar) and quarter 1–4 |
| quarter_start | date | First day of the quarter |
| production_t, non_fines_t, fines_t | t | MOIL company-total manganese-ore production |
| basis | — | `stated quarter`, `stated cumulative` (Q1) or `derived: cumulative difference` |
| source_documents | — | Disclosure title(s) and column used, e.g. `[stated April-to-date (previous year)]` |

`moil_production_disclosures.csv` has one row per stated figure (`period_kind`
CUMULATIVE / QUARTER / FY, `stated_unit`, `column_header`, `source_url`).

### `data/processed/weather/imd_daily_mine_belt.csv` (REAL_GOVERNMENT)
| Column | Unit | Meaning |
|---|---|---|
| belt_rainfall_mm | mm/day | Mean of the 40 IMD 0.25° cells in 21.25–22.25 N, 79.0–80.75 E |
| belt_tmax_c | °C | Mean of the 6 IMD 1° Tmax cells in the belt |
| mine_cell_rainfall_mm, mine_cell_tmax_c | mm, °C | IMD cell containing the demo mine (21.75 N 80.0 E; Tmax 21.5 N 79.5 E) |

`demo_mine_daily_weather.csv`: the rain / Tmax / soil moisture series that drives the simulator,
with `rainfall_source` recorded per row (IMD or ERA5).

### `data/processed/exploration/geology_features_grid.csv` (REAL_GOVERNMENT, NRSC 1:50k)
| Column | Unit | Meaning |
|---|---|---|
| geomorphology_class | — | NRSC class text at the cell centre (NaN = unmapped) |
| geom_structural_origin, geom_hills_valleys, geom_plateau, geom_pediplain, geom_water_fluvial | 0/1 | Class flags |
| geom_dissection | 0–3 | none / low / moderate / high dissection |
| lineament_dist_km | km | Distance to the nearest mapped structural lineament |
| lineament_density | fraction | Share of lineament pixels within ±2.5 km |

### `data/processed/exploration/eo_features_grid.csv.gz` (REAL_DERIVED)
0.01° cells. Sentinel-2 median composite for 2024: NDVI, Iron_Oxide_Index (B4/B2),
Clay_Hydroxyl_Index (B11/B12), NDRE, Ferric_Iron_B4B3, Ferrous_Silicate_B12B8A, Gossan_B11B4,
SWIR_Albedo. MODIS LST_Day_K. NASADEM elevation, slope and elevation_sd. QC counts: s2_valid_scenes,
s2_pixels, lst_valid_composites, dem_pixels. Formulas are in `ml/eo_features.py`
(`FEATURE_RATIONALE`).

### `data/processed/subsurface/*` (REAL_GOVERNMENT / REAL_MOIL_PUBLIC; transcribed)
* `exploration_blocks.csv`: block_id, centroid lat / lon, area_km2 (computed from the stated
  corners), GeoJSON geometry, stage (G4 / G3 proposal), agency, source_doc, official_url,
  evidence_classes, implied_evidence_level.
* `reported_findings.csv`: block_id, finding_type, evidence_class (PROPOSAL_ONLY, STRUCTURAL,
  GEOLOGICAL_CONTEXT, GEOLOGICAL_MAPPING, GRADE_REPORTED, GEOCHEMICAL_SURFACE,
  PIT_INTERSECTION_REPORTED, DRILLING_INTERSECTION_REPORTED), value_text, mn_pct_min / max,
  count_tested / count_positive, depth_m, dip range and direction, strike, source_doc, quote_or_summary.
* `surface_geochem_samples.csv`: NMET Katori hand-held XRF samples (lat, lon, mn/ca/si/p/fe/al %).
* `geological_constraints.json`: stratigraphy, horizons, dips, reef widths, grades, proposed
  depths and IBM grade classes, each with its source.

## Synthetic (SYNTHETIC)

### `data/production_history.csv` (daily, DEMO_MINE = SYN_MINE_01)
| Column | Unit | Meaning |
|---|---|---|
| target_tonnes | t/day | Planned target (demo plan, monsoon factor, growth) |
| actual_tonnes | t/day | Simulated production |
| equipment_availability | 0–1 | Available ÷ scheduled excavator hours |
| equipment_downtime_h, maintenance_hours | h/day per excavator | Breakdown and preventive-maintenance hours |
| drilling_delay_h | h | Lost hours per drill, averaged per shift |
| blast_delay_h | h/day | Excavator hours waiting on blasted ore |
| truck_count | trucks | Time-averaged available dumpers |
| haulage_delay_h | h/day | Excavator hours waiting on trucks + ½ site stoppage |
| utilization | 0–1 | Operating ÷ available excavator hours |
| rainfall_mm, soil_moisture_m3m3, temperature_max_c | — | REAL weather inputs |
| rainfall_source, weather_data_mode | — | IMD (REAL_GOVERNMENT) or ERA5 (REAL_PUBLIC) |
| ops_data_mode | — | Always SYNTHETIC |

### `synthetic_mine_operational_history.csv.gz` (one row per unit per shift)
date, shift (A / B / C), equipment_id (`SYN_EXC_*`, `SYN_TRUCK_*`, `SYN_DRILL_*`), equipment_type,
capacity_per_h + capacity_unit, scheduled / available / operating / idle / maintenance /
breakdown hours, availability, utilization, failure_event, maintenance_event, weather_state
(DRY / WET / HEAVY_RAIN), haulage_state (NORMAL / CONSTRAINED), production_contribution_t.

### `synthetic_equipment_events.csv`
date, shift, equipment_id, equipment_type, event_type (BREAKDOWN, PREVENTIVE_MAINTENANCE,
WEATHER_SITE_STOPPAGE, POWER_OUTAGE, EXPLOSIVE_SUPPLY_EPISODE_START), hours_in_shift.

## Simulated (SIMULATED)

### `synthetic_disruption_scenarios.csv`
scenario_id, start_date, duration_days (7), disruption_type, severity_production_loss_pct,
affected_equipment, availability_change, haulage / blast / drill delay changes (h), truck_change,
rainfall_7d_mm, weather_state, production_impact_t.

### `synthetic_action_outcomes.csv`
One row per origin × scenario × portfolio. Seven operating-state features `*_before` / `*_after`.
production_normal_t / before_t / after_t, target_t, gap_before_t, gap_after_t, residual_gap_t,
recovery_t, action_cost_proxy (number of actions), uncertainty_recovery_sd_t.

### `recovery_scenario_matrix.csv` (read by the recovery engine)
Per scenario × portfolio: `scenario_delta_<feature>` (median effect of the disruption),
`action_delta_<feature>` with `_p10` / `_p90` (effect of the portfolio given the disruption),
scenario_rainfall_7d_mm_min, and simulated_recovery_t_median / p10 / p90.

### Subsurface (`data/synthetic/subsurface/`)
* `synthetic_boreholes.csv`: borehole_id `SIM_*`, target_id, scenario, latitude, longitude, azimuth,
  dip (−60), total_depth_m, bed_dip_deg, formation_context, generation_basis.
* `synthetic_borehole_intervals.csv`: depth_from_m, depth_to_m, interval_thickness_m, formation,
  host_lithology, weathering_zone, manganese_presence, mn_grade_pct, grade_category (IBM classes),
  confidence, evidence_level.
* `synthetic_geochemistry.csv`: sample_id, borehole_id (empty for surface samples), depth, sample_type,
  mn / fe / sio2 %, sample_quality (SIMULATED_ASSAY / SIMULATED_SCREENING).
* `synthetic_geophysics.csv`: method (IP chargeability), response, background, anomaly_strength,
  depth_or_scale, noise, confidence.
* `synthetic_subsurface_targets.csv`: per target × scenario counts, and observed_evidence_level for
  comparison.
