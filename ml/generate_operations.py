"""Generate the deterministic SYNTHETIC daily operations history for DEMO_MINE.

No real MOIL operational data are available to this project, so equipment,
fleet, drilling/blasting/haulage and production records are SIMULATED from a
documented data-generating process (DGP) with a fixed seed. Weather columns are
the REAL public reanalysis values from data/weather_daily.csv.

The DGP exists only to exercise and validate the forecasting/decision pipeline
end to end. Validation metrics computed on it demonstrate that the pipeline
works; they are NOT evidence of real-world accuracy at any MOIL mine.

Run:  python -m ml.generate_operations
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.common import CONFIG_DIR, DATA_DIR, load_json

AVAIL_MEAN = 0.87
TRUCKS_REF = 18


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


def generate(seed=None):
    demo = load_json(CONFIG_DIR / "demo_config.json")
    gen = demo["synthetic_generator"]
    mine = demo["mines"]["DEMO_MINE"]
    rng = np.random.default_rng(demo["seed"] if seed is None else seed)

    w = pd.read_csv(DATA_DIR / "weather_daily.csv", parse_dates=["date"])
    n = len(w)
    rain = w["rainfall_mm"].to_numpy()
    sm = w["soil_moisture_m3m3"].to_numpy()
    tmax = w["temperature_max_c"].to_numpy()
    dates = w["date"]
    years = ((dates - dates.iloc[0]).dt.days / 365.25).to_numpy()
    sunday = (dates.dt.dayofweek == 6).to_numpy()
    month = dates.dt.month.to_numpy()
    rain3 = pd.Series(rain).rolling(3, min_periods=1).sum().to_numpy()
    sched = mine["scheduled_hours_per_day"]

    # --- regime episodes -------------------------------------------------
    def episodes(rate, dur_lo, dur_hi, sev_lo, sev_hi):
        out = np.zeros(n)
        t = 0
        while t < n:
            t += int(rng.exponential(1 / rate)) + 1
            if t >= n:
                break
            d = int(rng.integers(dur_lo, dur_hi + 1))
            out[t:t + d] = rng.uniform(sev_lo, sev_hi)
            t += d
        return out

    breakdown = episodes(1 / 55, 4, 16, 0.08, 0.25)       # availability loss
    explosive = episodes(1 / 110, 5, 21, 1.5, 4.0)        # blast delay hours
    rig_issue = episodes(1 / 90, 3, 12, 0.8, 2.5)         # drilling delay hours

    fleet = np.zeros(n)
    level = TRUCKS_REF
    t = 0
    while t < n:
        d = int(rng.integers(60, 150))
        fleet[t:t + d] = level
        level = int(np.clip(level + rng.choice([-3, -2, -1, 1, 2, 3]), 13, 22))
        t += d

    # --- operating state -------------------------------------------------
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.85 * ar[i - 1] + rng.normal(0, 0.018)
    wet = np.where(rain3 > 40, 0.03, 0.0)
    avail = np.clip(AVAIL_MEAN + ar - breakdown - wet, 0.35, 0.97)
    lost = sched * (1 - avail)
    planned_pm = np.clip(1.4 + np.where(sunday, 2.2, 0.0) + rng.normal(0, 0.3, n), 0, None)
    maintenance = np.minimum(planned_pm + np.where(breakdown > 0, 1.2, 0.0), 0.6 * lost)
    downtime = lost - maintenance

    trucks = np.clip(fleet - rng.poisson(0.7 + 12 * breakdown), 8, None).astype(int)
    drilling = rng.gamma(1.5, 0.35, n) + 0.015 * rain3 + rig_issue
    blast = rng.gamma(1.2, 0.3, n) + 0.012 * rain + explosive
    haulage = rng.gamma(1.5, 0.2, n) + 0.02 * rain + 6.0 * np.clip(sm - 0.38, 0, None)

    # --- production ------------------------------------------------------
    cap = gen["base_capacity_tpd"] * (1 + gen["capacity_trend_per_year"]) ** years
    f_equip = avail / AVAIL_MEAN
    f_truck = np.minimum(1.0, trucks / TRUCKS_REF) ** 0.7
    lost_h = 0.35 * drilling + 0.6 * blast + 0.5 * haulage
    f_delay = np.clip(1 - lost_h / sched, 0.3, 1.0)
    f_rain = np.exp(-0.004 * rain) * np.where(rain > 80, 0.35, 1.0)
    f_soil = 1 - 1.2 * np.clip(sm - 0.40, 0, None)
    f_heat = 1 - 0.01 * np.clip(tmax - 42, 0, None)
    f_sun = np.where(sunday, 0.85, 1.0)
    noise = rng.lognormal(0, gen["noise_sd"], n)
    actual = cap * f_equip * f_truck * f_delay * f_rain * f_soil * f_heat * f_sun * noise

    target = planned_daily_target(dates, demo)

    df = pd.DataFrame({
        "date": dates.dt.strftime("%Y-%m-%d"),
        "mine_id": "DEMO_MINE",
        "target_tonnes": np.round(target, 1),
        "actual_tonnes": np.round(actual, 1),
        "equipment_availability": np.round(avail, 4),
        "equipment_downtime_h": np.round(downtime, 2),
        "maintenance_hours": np.round(maintenance, 2),
        "drilling_delay_h": np.round(drilling, 2),
        "blast_delay_h": np.round(blast, 2),
        "truck_count": trucks,
        "haulage_delay_h": np.round(haulage, 2),
        "rainfall_mm": rain,
        "soil_moisture_m3m3": sm,
        "temperature_max_c": tmax,
        "ops_data_mode": "SYNTHETIC",
        "weather_data_mode": "REAL_PUBLIC",
    })
    return df


def main():
    df = generate()
    df.to_csv(DATA_DIR / "production_history.csv", index=False)
    wk = df.assign(date=pd.to_datetime(df["date"])).set_index("date")[["actual_tonnes", "target_tonnes"]].resample("YE").sum()
    wk["attainment"] = (wk["actual_tonnes"] / wk["target_tonnes"]).round(3)
    print(wk)


if __name__ == "__main__":
    main()
