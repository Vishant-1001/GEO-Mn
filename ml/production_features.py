"""Production forecasting feature schema — shared by training, backtest and the API.

Temporal design
---------------
* Source granularity: daily records (production, operations, weather).
* Forecast unit: one production PERIOD of 7 days starting at the forecast origin
  (origin day included). The forecast target is the period's total tonnes.
* Conditional-forecast design:
    - The model learns period production as a function of the operating
      CONDITIONS DURING THE PERIOD (equipment, fleet, delays, weather) plus
      origin-known production lags and seasonality. Historically both are
      observed, so this is a legitimate response function.
    - At forecast time the period conditions are unknown. The forecast uses
      only origin-known information: the trailing 7-day state is ASSUMED to
      persist through the period (persistence assumption), unless a scenario
      explicitly overrides it (SIMULATION mode, labelled as such).
    - Backtests evaluate exactly this procedure (persistence inputs), so no
      future observed weather or operations leak into evaluation.
* Lags: weekly totals of the 1-4 periods before the origin, 4- and 12-period
  rolling means, 4-period trend, recent target attainment.
* The planned target is NOT a model input, so the forecast is independent of the
  plan and the gap is computed afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIOD_DAYS = 7

STATE_FEATURES = [
    "equipment_availability",
    "equipment_downtime_h",
    "maintenance_hours",
    "drilling_delay_h",
    "blast_delay_h",
    "truck_count",
    "haulage_delay_h",
]
WEATHER_FEATURES = ["rainfall_7d_mm", "soil_moisture_m3m3", "temperature_max_c"]
LAG_FEATURES = [
    "prod_lag1",
    "prod_lag2",
    "prod_lag3",
    "prod_lag4",
    "prod_roll_mean_4",
    "prod_roll_mean_12",
    "prod_trend_4",
    "attainment_lag1",
    "attainment_roll_4",
]
SEASON_FEATURES = ["woy_sin", "woy_cos"]

FEATURE_COLUMNS = LAG_FEATURES + STATE_FEATURES + WEATHER_FEATURES + SEASON_FEATURES
SCENARIO_FEATURES = STATE_FEATURES + WEATHER_FEATURES

# Direction of the expected physical effect, used as monotonic constraints so
# that simulated interventions cannot move production in a physically
# implausible direction (+1 increasing, -1 decreasing, 0 unconstrained).
MONOTONIC = {
    "equipment_availability": 1,
    "equipment_downtime_h": -1,
    "drilling_delay_h": -1,
    "blast_delay_h": -1,
    "truck_count": 1,
    "haulage_delay_h": -1,
    "rainfall_7d_mm": -1,
}

FEATURE_DOCS = {
    "prod_lag1": ("t/period", "actual tonnes in the period immediately before the origin"),
    "prod_lag2": ("t/period", "actual tonnes 2 periods before the origin"),
    "prod_lag3": ("t/period", "actual tonnes 3 periods before the origin"),
    "prod_lag4": ("t/period", "actual tonnes 4 periods before the origin"),
    "prod_roll_mean_4": ("t/period", "mean of the last 4 period totals"),
    "prod_roll_mean_12": ("t/period", "mean of the last 12 period totals"),
    "prod_trend_4": ("t/period per period", "(lag1 - lag4) / 3"),
    "attainment_lag1": ("ratio", "actual / target in the previous period"),
    "attainment_roll_4": ("ratio", "sum actual / sum target over the last 4 periods"),
    "equipment_availability": ("fraction 0-1", "mean fleet availability in the period (forecast: trailing 7-day value, persistence)"),
    "equipment_downtime_h": ("h/day", "mean unplanned downtime in the period (forecast: trailing 7-day, persistence)"),
    "maintenance_hours": ("h/day", "mean maintenance hours in the period (forecast: trailing 7-day, persistence)"),
    "drilling_delay_h": ("h/day", "mean drilling delay in the period (forecast: trailing 7-day, persistence)"),
    "blast_delay_h": ("h/day", "mean blasting delay in the period (forecast: trailing 7-day, persistence)"),
    "truck_count": ("trucks", "mean trucks deployed in the period (forecast: trailing 7-day, persistence)"),
    "haulage_delay_h": ("h/day", "mean haulage delay in the period (forecast: trailing 7-day, persistence)"),
    "rainfall_7d_mm": ("mm/7 days", "rainfall total in the period, ERA5 (forecast: trailing 7-day total, persistence)"),
    "soil_moisture_m3m3": ("m3/m3", "mean 0-7 cm soil moisture in the period, ERA5-Land (forecast: trailing 7-day, persistence)"),
    "temperature_max_c": ("deg C", "mean daily max 2 m temperature in the period, ERA5 (forecast: trailing 7-day, persistence)"),
    "woy_sin": ("-", "sin(2*pi*day_of_year/365.25) at the origin"),
    "woy_cos": ("-", "cos(2*pi*day_of_year/365.25) at the origin"),
}

HUMAN = {
    "prod_lag1": "Last period production",
    "prod_lag2": "Production 2 periods ago",
    "prod_lag3": "Production 3 periods ago",
    "prod_lag4": "Production 4 periods ago",
    "prod_roll_mean_4": "4-period production mean",
    "prod_roll_mean_12": "12-period production mean",
    "prod_trend_4": "Production trend",
    "attainment_lag1": "Last period target attainment",
    "attainment_roll_4": "4-period target attainment",
    "equipment_availability": "Equipment availability",
    "equipment_downtime_h": "Equipment downtime",
    "maintenance_hours": "Maintenance hours",
    "drilling_delay_h": "Drilling delay",
    "blast_delay_h": "Blast delay",
    "truck_count": "Truck count",
    "haulage_delay_h": "Haulage delay",
    "rainfall_7d_mm": "Rainfall (7-day)",
    "soil_moisture_m3m3": "Soil moisture",
    "temperature_max_c": "Max temperature",
    "woy_sin": "Seasonality (sin)",
    "woy_cos": "Seasonality (cos)",
}


def load_daily(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").set_index("date")


def features_at(daily: pd.DataFrame, origin) -> dict:
    """Feature dict for a forecast origin using only rows strictly before it."""
    origin = pd.Timestamp(origin)
    hist = daily.loc[: origin - pd.Timedelta(days=1)]
    need = PERIOD_DAYS * 12
    if len(hist) < need:
        raise ValueError(f"need at least {need} days of history before {origin.date()}")
    tail = hist.iloc[-need:]
    act = tail["actual_tonnes"].to_numpy().reshape(12, PERIOD_DAYS).sum(axis=1)
    tgt = tail["target_tonnes"].to_numpy().reshape(12, PERIOD_DAYS).sum(axis=1)
    lag = act[::-1]  # lag[0] = most recent period
    tlag = tgt[::-1]
    last7 = hist.iloc[-PERIOD_DAYS:]
    doy = origin.dayofyear
    f = {
        "prod_lag1": lag[0],
        "prod_lag2": lag[1],
        "prod_lag3": lag[2],
        "prod_lag4": lag[3],
        "prod_roll_mean_4": lag[:4].mean(),
        "prod_roll_mean_12": lag.mean(),
        "prod_trend_4": (lag[0] - lag[3]) / 3.0,
        "attainment_lag1": lag[0] / tlag[0] if tlag[0] > 0 else np.nan,
        "attainment_roll_4": lag[:4].sum() / tlag[:4].sum() if tlag[:4].sum() > 0 else np.nan,
        "rainfall_7d_mm": last7["rainfall_mm"].sum(),
        "soil_moisture_m3m3": last7["soil_moisture_m3m3"].mean(),
        "temperature_max_c": last7["temperature_max_c"].mean(),
        "woy_sin": np.sin(2 * np.pi * doy / 365.25),
        "woy_cos": np.cos(2 * np.pi * doy / 365.25),
    }
    for c in STATE_FEATURES:
        f[c] = last7[c].mean()
    return f


def period_conditions(daily: pd.DataFrame, origin) -> dict:
    """Mean operating state / weather DURING the period starting at origin (training only)."""
    origin = pd.Timestamp(origin)
    win = daily.loc[origin: origin + pd.Timedelta(days=PERIOD_DAYS - 1)]
    c = {k: win[k].mean() for k in STATE_FEATURES}
    c["rainfall_7d_mm"] = win["rainfall_mm"].sum()
    c["soil_moisture_m3m3"] = win["soil_moisture_m3m3"].mean()
    c["temperature_max_c"] = win["temperature_max_c"].mean()
    return c


def training_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Frame with the condition features replaced by the concurrent (period) values."""
    X = frame.copy()
    for k in SCENARIO_FEATURES:
        X[k] = frame["cond_" + k]
    return X


def period_outcome(daily: pd.DataFrame, origin):
    """(actual, target) totals for the period starting at origin, or (None, target) if incomplete."""
    origin = pd.Timestamp(origin)
    win = daily.loc[origin: origin + pd.Timedelta(days=PERIOD_DAYS - 1)]
    tgt = win["target_tonnes"].sum() if len(win) else None
    if len(win) < PERIOD_DAYS:
        return None, tgt
    return win["actual_tonnes"].sum(), tgt


def build_training_frame(daily: pd.DataFrame, stride_days=1) -> pd.DataFrame:
    """One row per origin: persistence features (origin-known), cond_* period conditions, y, target."""
    first = daily.index[0] + pd.Timedelta(days=PERIOD_DAYS * 12)
    last = daily.index[-1] - pd.Timedelta(days=PERIOD_DAYS - 1)
    rows = []
    for origin in pd.date_range(first, last, freq=f"{stride_days}D"):
        y, tgt = period_outcome(daily, origin)
        if y is None:
            continue
        f = features_at(daily, origin)
        f.update({"cond_" + k: v for k, v in period_conditions(daily, origin).items()})
        f.update({"origin": origin, "y": y, "period_target": tgt})
        rows.append(f)
    return pd.DataFrame(rows)


def weekly_origins(daily: pd.DataFrame, anchor) -> pd.DatetimeIndex:
    """Non-overlapping origins aligned (in 7-day steps) to the anchor forecast origin."""
    anchor = pd.Timestamp(anchor)
    first = daily.index[0] + pd.Timedelta(days=PERIOD_DAYS * 12)
    k = int(np.ceil((anchor - first).days / PERIOD_DAYS))
    return pd.DatetimeIndex([anchor - pd.Timedelta(days=PERIOD_DAYS * i) for i in range(k, -1, -1)])
