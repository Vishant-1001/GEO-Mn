"""Generate SIMULATED disruption scenarios and action outcomes from simulator counterfactuals.

For ~140 weekly origins taken from the synthetic history (state snapshots of the SAME seeded run),
every disruption scenario x action portfolio is simulated for the next 7 days with common random
numbers and the REAL weather of that week. Heavy-rain intensity is set from REAL IMD data (95th
percentile of 7-day mine-cell rainfall, 2021-2025).

Outputs (data/synthetic/production/ and data/synthetic/recovery/):
  synthetic_disruption_scenarios.csv   one row per origin x scenario (no action)       -> recovery engine
  synthetic_action_outcomes.csv        one row per origin x scenario x portfolio          -> recovery engine
  recovery_scenario_matrix.csv         median / p10 / p90 feature deltas per scenario x portfolio
                                       (CONSUMED by services/recovery_service.py to define scenarios
                                       and actions in model-feature space)
  recovery_generation_manifest.json
Every row is SIMULATED — not historical MOIL interventions.
"""
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.generate_operations import build_weather, load_cfg, planned_daily_target  # noqa: E402
from ml.synthetic_ops import ACTIONS, SCENARIOS, counterfactual_week, simulate  # noqa: E402

REC = ROOT / "data" / "synthetic" / "recovery"
PROD = ROOT / "data" / "synthetic" / "production"
CODES = {"EQUIPMENT_RECOVERY": "A1", "SCHEDULE_ADJUSTMENT": "A2", "BLAST_DRILL_DELAY_REDUCTION": "A3"}
FEATS = ["equipment_availability", "equipment_downtime_h", "maintenance_hours", "drilling_delay_h", "blast_delay_h",
         "truck_count", "haulage_delay_h"]


def portfolios():
    out = [()]
    for n in (1, 2, 3):
        out += list(itertools.combinations(ACTIONS, n))
    return out


def pid(p):
    return "+".join(CODES[a] for a in p) if p else "NO_ACTION"


def week_features(days: pd.DataFrame, weather: pd.DataFrame, rain_override=None):
    f = {k: float(days[k].mean()) for k in FEATS}
    rain = weather["rainfall_mm"].to_numpy()
    if rain_override is not None:
        rain = np.maximum(rain, rain_override)
    f["rainfall_7d_mm"] = float(rain.sum())
    f["soil_moisture_m3m3"] = float(max(weather["soil_moisture_m3m3"].mean(), 0.46 if rain_override is not None else 0))
    return f


def main():
    cfg = load_cfg()
    weather = build_weather(cfg)
    w = weather[["date", "rainfall_mm", "soil_moisture_m3m3", "temperature_max_c"]].copy()
    imd = weather[weather["rainfall_source"].str.startswith("IMD")]
    heavy_7d = float(imd["rainfall_mm"].rolling(7).sum().quantile(0.95))
    scen_rain = heavy_7d / 7.0
    w["scenario_rain_mm"] = scen_rain
    _, _, snaps, sim = simulate(cfg, w, cfg["seed"], snapshot_every=7)
    dates = sorted(snaps)[13::2]                    # skip the first quarter, then every 2nd week
    dates = [d for d in dates if pd.Timestamp(d) + pd.Timedelta(days=6) <= pd.Timestamp(cfg["period"]["end"])]
    widx = w.set_index(w["date"].dt.strftime("%Y-%m-%d"))
    rows, dis = [], []
    ports = portfolios()
    for d in dates:
        state, seed = snaps[d]
        wk = widx.loc[pd.date_range(d, periods=7).strftime("%Y-%m-%d")].reset_index(drop=True)
        target = float(planned_daily_target(pd.date_range(d, periods=7)).sum())
        base_t, base_days = counterfactual_week(sim, state, wk, seed, "NORMAL", ())
        base_f = week_features(base_days, wk)
        for s in SCENARIOS:
            heavy = s in ("HEAVY_RAIN", "COMBINED_DISRUPTION")
            s_t, s_days = counterfactual_week(sim, state, wk, seed, s, ())
            s_f = week_features(s_days, wk, scen_rain if heavy else None)
            dis.append({"scenario_id": f"SIM_{d}_{s}", "mine_id": cfg["mine"]["mine_id"], "start_date": d, "duration_days": 7,
                        "disruption_type": s, "severity_production_loss_pct": round(100 * (base_t - s_t) / max(base_t, 1), 2),
                        "affected_equipment": {"HEAVY_RAIN": "site / dumpers", "EQUIPMENT_DEGRADATION": "excavators, dumpers",
                                               "BLAST_DELAY": "blasting / excavators", "DRILL_DELAY": "drills",
                                               "HAULAGE_DISRUPTION": "dumpers / haul road",
                                               "COMBINED_DISRUPTION": "site-wide", "NORMAL": "none"}[s],
                        "availability_change": round(s_f["equipment_availability"] - base_f["equipment_availability"], 4),
                        "haulage_delay_change_h": round(s_f["haulage_delay_h"] - base_f["haulage_delay_h"], 3),
                        "blast_delay_change_h": round(s_f["blast_delay_h"] - base_f["blast_delay_h"], 3),
                        "drill_delay_change_h": round(s_f["drilling_delay_h"] - base_f["drilling_delay_h"], 3),
                        "truck_change": round(s_f["truck_count"] - base_f["truck_count"], 2),
                        "rainfall_7d_mm": round(s_f["rainfall_7d_mm"], 1),
                        "weather_state": "HEAVY_RAIN (IMD p95 7-day)" if heavy else "OBSERVED WEEK (IMD/ERA5)",
                        "production_impact_t": round(s_t - base_t, 1), "provenance": "SIMULATED"})
            for p in ports:
                a_t, a_days = counterfactual_week(sim, state, wk, seed, s, p)
                a_f = week_features(a_days, wk, scen_rain if heavy else None)
                rows.append({"scenario_id": f"SIM_{d}_{s}", "origin": d, "baseline_state": "SIMULATED_HISTORY_STATE",
                             "scenario": s, "action_portfolio": pid(p), "n_actions": len(p),
                             **{f"{k}_before": round(s_f[k], 4) for k in FEATS},
                             **{f"{k}_after": round(a_f[k], 4) for k in FEATS},
                             "rainfall_7d_mm": round(a_f["rainfall_7d_mm"], 1), "soil_moisture_m3m3": round(a_f["soil_moisture_m3m3"], 3),
                             "production_normal_t": round(base_t, 1), "production_before_t": round(s_t, 1),
                             "production_after_t": round(a_t, 1), "target_t": round(target, 1),
                             "gap_before_t": round(max(0.0, target - s_t), 1), "gap_after_t": round(max(0.0, target - a_t), 1),
                             "residual_gap_t": round(max(0.0, target - a_t), 1), "action_cost_proxy": len(p),
                             "provenance": "SIMULATED"})
    ao = pd.DataFrame(rows)
    ds = pd.DataFrame(dis)
    # uncertainty: spread of simulated recovery across origins
    ao["recovery_t"] = ao["production_after_t"] - ao["production_before_t"]
    grp = ao.groupby(["scenario", "action_portfolio"])
    ao["uncertainty_recovery_sd_t"] = grp["recovery_t"].transform("std").round(1)
    # feature-space deltas the recovery engine consumes
    base_norm = ao[(ao["scenario"] == "NORMAL") & (ao["action_portfolio"] == "NO_ACTION")].set_index("origin")
    mat = []
    for (s, p), g in grp:
        g = g.set_index("origin")
        rec = {"scenario": s, "action_portfolio": p, "n_origins": len(g), "provenance": "SIMULATED"}
        for k in FEATS:
            dscen = g[f"{k}_before"] - base_norm.loc[g.index, f"{k}_before"]     # disruption effect
            dact = g[f"{k}_after"] - g[f"{k}_before"]                             # action effect given disruption
            rec[f"scenario_delta_{k}"] = round(float(dscen.median()), 4)
            rec[f"action_delta_{k}"] = round(float(dact.median()), 4)
            rec[f"action_delta_{k}_p10"] = round(float(dact.quantile(0.1)), 4)
            rec[f"action_delta_{k}_p90"] = round(float(dact.quantile(0.9)), 4)
        rec["scenario_rainfall_7d_mm_min"] = round(heavy_7d, 1) if s in ("HEAVY_RAIN", "COMBINED_DISRUPTION") else None
        rec["simulated_recovery_t_median"] = round(float(g["recovery_t"].median()), 1)
        rec["simulated_recovery_t_p10"] = round(float(g["recovery_t"].quantile(0.1)), 1)
        rec["simulated_recovery_t_p90"] = round(float(g["recovery_t"].quantile(0.9)), 1)
        mat.append(rec)
    mdf = pd.DataFrame(mat)
    REC.mkdir(parents=True, exist_ok=True)
    ds.to_csv(PROD / "synthetic_disruption_scenarios.csv", index=False)
    ao.to_csv(REC / "synthetic_action_outcomes.csv", index=False)
    mdf.to_csv(REC / "recovery_scenario_matrix.csv", index=False)
    man = {"dataset_name": "Simulated disruption scenarios and action outcomes (simulator counterfactuals)",
           "generator_name": "scripts/synthetic/generate_recovery.py (ml/synthetic_ops.py)",
           "generator_version": cfg["generator_version"], "seed": cfg["seed"],
           "generation_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "origins": len(dates), "scenarios": list(SCENARIOS), "portfolios": [pid(p) for p in ports],
           "heavy_rain_7d_mm": round(heavy_7d, 1),
           "heavy_rain_basis": "REAL: 95th percentile of 7-day IMD mine-cell rainfall (2021-2025)",
           "files": {n: {"rows": int(len(df)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for n, df, p in (
               ("synthetic_disruption_scenarios.csv", ds, PROD / "synthetic_disruption_scenarios.csv"),
               ("synthetic_action_outcomes.csv", ao, REC / "synthetic_action_outcomes.csv"),
               ("recovery_scenario_matrix.csv", mdf, REC / "recovery_scenario_matrix.csv"))},
           "consumer": "services/recovery_service.py reads recovery_scenario_matrix.csv (scenario and action deltas); "
                       "trust layer compares engine estimates with simulated recovery",
           "provenance": "SIMULATED — not historical MOIL interventions; recovery is not guaranteed"}
    (REC / "recovery_generation_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(len(dates), "origins;", len(ds), "disruption rows;", len(ao), "action rows; heavy 7d", round(heavy_7d, 1))
    show = mdf[mdf["action_portfolio"] == "NO_ACTION"][["scenario", "scenario_delta_equipment_availability", "scenario_delta_blast_delay_h",
                                                         "scenario_delta_drilling_delay_h", "scenario_delta_haulage_delay_h", "scenario_delta_truck_count"]]
    print(show.to_string())
    print(mdf[mdf["scenario"] == "NORMAL"][["action_portfolio", "action_delta_equipment_availability", "action_delta_truck_count",
                                            "action_delta_blast_delay_h", "action_delta_haulage_delay_h", "simulated_recovery_t_median"]].to_string())


if __name__ == "__main__":
    main()
