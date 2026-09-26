"""Build the SYNTHETIC demonstration-mine operations history (equipment-level simulator).

Inputs
  * REAL weather for the demo-mine location:
      rainfall, Tmax  — IMD gridded (REAL_GOVERNMENT) 2021-2025; ERA5 (REAL_PUBLIC) for 2026, because
                        IMD has not yet published 2026 grids (verified 2026-09-26)
      soil moisture   — ERA5-Land (REAL_PUBLIC); IMD publishes no soil-moisture grid
  * simulator parameters: data/synthetic/production/synthetic_generation_config.yaml
Mechanism: ml/synthetic_ops.py (event-based failures, maintenance, shared shocks, drill-blast-load-haul
chain, capacity-constrained production). Seeded -> identical output for identical config.

Outputs
  data/processed/weather/demo_mine_daily_weather.csv                       REAL (source-tagged per row)
  data/synthetic/production/synthetic_mine_operational_history.csv.gz      equipment x shift (SYNTHETIC)
  data/synthetic/production/synthetic_mine_operational_history_sample.csv  first 2,000 rows, uncompressed
  data/synthetic/production/synthetic_mine_operational_history.parquet
  data/synthetic/production/synthetic_equipment_events.csv                 failures / PM / shocks / episodes
  data/production_history.csv                                              daily mine aggregate -> production engine
  data/synthetic/production/synthetic_generation_manifest.json

Run:  python -m ml.generate_operations      (or python scripts/synthetic/generate_all.py)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from ml.common import CONFIG_DIR, DATA_DIR, load_json
from ml.synthetic_ops import simulate

SYN_DIR = DATA_DIR / "synthetic" / "production"
CFG_PATH = SYN_DIR / "synthetic_generation_config.yaml"


def load_cfg():
    with open(CFG_PATH) as fh:
        return yaml.safe_load(fh)


def planned_daily_target(dates, demo=None):
    """Synthetic production PLAN (t/day): fiscal-year growth, monsoon and Sunday factors.

    Used both to generate history and to supply the planned target for future
    periods, so a forecast's target always follows the same documented rule.
    """
    demo = demo or load_json(CONFIG_DIR / "demo_config.json")
    gen = demo["synthetic_generator"]
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    start_year = pd.Timestamp(demo["history_start"]).year
    fy_index = (dates.dt.year - start_year + (dates.dt.month >= 4).astype(int) - 1).clip(lower=0).to_numpy()
    month = dates.dt.month.to_numpy()
    sunday = (dates.dt.dayofweek == 6).to_numpy()
    return (gen["base_target_tpd"] * (1 + gen["target_growth_per_year"]) ** fy_index
            * np.where(np.isin(month, [7, 8, 9]), gen["monsoon_target_factor"], 1.0)
            * np.where(sunday, 0.85, 1.0))


def build_weather(cfg) -> pd.DataFrame:
    """Daily REAL weather for the demo-mine cell, with per-row source tags."""
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    dates = pd.date_range(start, end, freq="D")
    era5 = pd.read_csv(DATA_DIR / "raw" / "weather" / "era5_open_meteo_demo_mine.csv", parse_dates=["date"]).set_index("date")
    w = pd.DataFrame(index=dates)
    w.index.name = "date"
    imd_path = DATA_DIR / "processed" / "weather" / "imd_daily_mine_belt.csv"
    imd = pd.read_csv(imd_path, parse_dates=["date"]).set_index("date") if imd_path.exists() else None
    w["rainfall_mm"] = np.nan
    w["temperature_max_c"] = np.nan
    w["rainfall_source"] = ""
    w["tmax_source"] = ""
    if imd is not None:
        j = imd.reindex(dates)
        ok_r = j["mine_cell_rainfall_mm"].notna()
        ok_t = j["mine_cell_tmax_c"].notna()
        w.loc[ok_r, "rainfall_mm"] = j.loc[ok_r, "mine_cell_rainfall_mm"]
        w.loc[ok_r, "rainfall_source"] = "IMD gridded 0.25deg (REAL_GOVERNMENT)"
        w.loc[ok_t, "temperature_max_c"] = j.loc[ok_t, "mine_cell_tmax_c"]
        w.loc[ok_t, "tmax_source"] = "IMD gridded 1deg (REAL_GOVERNMENT)"
    e = era5.reindex(dates)
    miss_r, miss_t = w["rainfall_mm"].isna(), w["temperature_max_c"].isna()
    w.loc[miss_r, "rainfall_mm"] = e.loc[miss_r, "rainfall_mm"]
    w.loc[miss_r, "rainfall_source"] = "ERA5 via Open-Meteo (REAL_PUBLIC; IMD not yet published)"
    w.loc[miss_t, "temperature_max_c"] = e.loc[miss_t, "temperature_max_c"]
    w.loc[miss_t, "tmax_source"] = "ERA5 via Open-Meteo (REAL_PUBLIC; IMD not yet published)"
    w["soil_moisture_m3m3"] = e["soil_moisture_m3m3"].to_numpy()
    w["soil_moisture_source"] = "ERA5-Land via Open-Meteo (REAL_PUBLIC; no IMD soil-moisture grid)"
    if w[["rainfall_mm", "temperature_max_c", "soil_moisture_m3m3"]].isna().any().any():
        raise ValueError("weather gaps remain; refusing to impute")
    out = w.reset_index()
    out.to_csv(DATA_DIR / "processed" / "weather" / "demo_mine_daily_weather.csv", index=False)
    return out


def generate(cfg=None, seed=None):
    """Run the simulator; returns (daily history df, unit-shift df, events df, weather df)."""
    cfg = cfg or load_cfg()
    weather = build_weather(cfg)
    daily, units, _, _ = simulate(cfg, weather[["date", "rainfall_mm", "soil_moisture_m3m3", "temperature_max_c"]],
                                  cfg["seed"] if seed is None else seed, record_units=True)
    daily["date"] = weather["date"].dt.strftime("%Y-%m-%d").to_numpy()
    target = planned_daily_target(weather["date"])
    hist = pd.DataFrame({
        "date": daily["date"],
        "mine_id": "DEMO_MINE",
        "synthetic_mine_id": cfg["mine"]["mine_id"],
        "target_tonnes": np.round(target, 1),
        "actual_tonnes": daily["actual_tonnes"].round(1),
        "equipment_availability": daily["equipment_availability"].round(4),
        "equipment_downtime_h": daily["equipment_downtime_h"].round(3),
        "maintenance_hours": daily["maintenance_hours"].round(3),
        "drilling_delay_h": daily["drilling_delay_h"].round(3),
        "blast_delay_h": daily["blast_delay_h"].round(3),
        "truck_count": daily["truck_count"].round(2),
        "haulage_delay_h": daily["haulage_delay_h"].round(3),
        "utilization": daily["utilization"].round(4),
        "rainfall_mm": weather["rainfall_mm"].round(2),
        "soil_moisture_m3m3": weather["soil_moisture_m3m3"].round(4),
        "temperature_max_c": weather["temperature_max_c"].round(2),
        "rainfall_source": weather["rainfall_source"],
        "ops_data_mode": "SYNTHETIC",
        "weather_data_mode": np.where(weather["rainfall_source"].str.startswith("IMD"), "REAL_GOVERNMENT", "REAL_PUBLIC"),
    })
    units.insert(0, "mine_id", cfg["mine"]["mine_id"])
    units["provenance"] = "SYNTHETIC"
    ev = []
    for r in units[units["failure_event"] | units["maintenance_event"]].itertuples():
        ev.append({"date": r.date, "shift": r.shift, "mine_id": r.mine_id, "equipment_id": r.equipment_id,
                   "equipment_type": r.equipment_type,
                   "event_type": "BREAKDOWN" if r.failure_event else "PREVENTIVE_MAINTENANCE",
                   "hours_in_shift": r.breakdown_hours if r.failure_event else r.maintenance_hours})
    for r in daily.itertuples():
        if r.site_stoppage_h > 0:
            ev.append({"date": r.date, "shift": "ALL", "mine_id": cfg["mine"]["mine_id"], "equipment_id": "SITE",
                       "equipment_type": "site", "event_type": "WEATHER_SITE_STOPPAGE", "hours_in_shift": round(r.site_stoppage_h, 2)})
        if r.power_outage_h > 0:
            ev.append({"date": r.date, "shift": "ALL", "mine_id": cfg["mine"]["mine_id"], "equipment_id": "SITE",
                       "equipment_type": "site", "event_type": "POWER_OUTAGE", "hours_in_shift": round(r.power_outage_h, 2)})
    ep = daily["explosive_episode"].astype(int).diff().fillna(daily["explosive_episode"].astype(int))
    for d in daily.loc[ep == 1, "date"]:
        ev.append({"date": d, "shift": "ALL", "mine_id": cfg["mine"]["mine_id"], "equipment_id": "SITE", "equipment_type": "site",
                   "event_type": "EXPLOSIVE_SUPPLY_EPISODE_START", "hours_in_shift": 0.0})
    events = pd.DataFrame(ev).sort_values(["date", "equipment_id"]).reset_index(drop=True)
    events["provenance"] = "SYNTHETIC"
    return hist, units, events, weather


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cfg = load_cfg()
    hist, units, events, weather = generate(cfg)
    SYN_DIR.mkdir(parents=True, exist_ok=True)
    hist.to_csv(DATA_DIR / "production_history.csv", index=False)
    units.to_csv(SYN_DIR / "synthetic_mine_operational_history.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    units.head(2000).to_csv(SYN_DIR / "synthetic_mine_operational_history_sample.csv", index=False)
    units.to_parquet(SYN_DIR / "synthetic_mine_operational_history.parquet", index=False)
    events.to_csv(SYN_DIR / "synthetic_equipment_events.csv", index=False)
    files = {p.name: {"rows": n, "sha256": _sha(p)} for p, n in (
        (DATA_DIR / "production_history.csv", len(hist)),
        (SYN_DIR / "synthetic_mine_operational_history.csv.gz", len(units)),
        (SYN_DIR / "synthetic_equipment_events.csv", len(events)))}
    man = {
        "dataset_name": "Synthetic demonstration-mine operations (SYN_MINE_01)",
        "generator_name": "ml/synthetic_ops.py via ml/generate_operations.py",
        "generator_version": cfg["generator_version"], "schema_version": cfg["schema_version"], "seed": cfg["seed"],
        "generation_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_constraints": [
            "REAL daily rainfall & Tmax: IMD gridded (2021-2025), ERA5 (2026 only) — per-row source columns",
            "REAL soil moisture: ERA5-Land",
            "IMD 'heavy rain' threshold 64.5 mm/day used for weather-state labels",
            "Production magnitude sized as a larger single mine relative to MOIL's company total "
            "(~18-19 lakh t/yr across its mines, REAL_MOIL_PUBLIC)",
        ],
        "assumptions": "all fleet, failure, repair, maintenance, blasting and utilisation parameters in "
                       "synthetic_generation_config.yaml are DOMAIN-CALIBRATED SYNTHETIC ASSUMPTIONS (not MOIL data)",
        "files": files,
        "consumers": {
            "production_history.csv": "production engine (weekly conditional quantile model), recovery & contingency",
            "synthetic_mine_operational_history": "judge inspection; aggregation source of production_history.csv",
            "synthetic_equipment_events.csv": "judge inspection; disruption-scenario library derivation",
        },
        "provenance": "SYNTHETIC (operations) + REAL (weather inputs)",
    }
    (SYN_DIR / "synthetic_generation_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    wk = hist.assign(date=pd.to_datetime(hist["date"])).set_index("date")[["actual_tonnes", "target_tonnes"]].resample("YE").sum()
    wk["attainment"] = (wk["actual_tonnes"] / wk["target_tonnes"]).round(3)
    print(wk)
    print(hist.describe().T[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
