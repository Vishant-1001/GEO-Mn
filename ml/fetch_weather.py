"""Fetch real daily weather for the demo mine location (ERA5 reanalysis via Open-Meteo).

Output: data/raw/weather/era5_open_meteo_demo_mine.csv with date, rainfall_mm, temperature_max_c, soil_moisture_m3m3.

These values are REAL_PUBLIC reanalysis (ECMWF ERA5 / ERA5-Land via the Open-Meteo
archive API), not direct satellite observations and not mine-site gauges.

Run:  python -m ml.fetch_weather
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from ml.common import CONFIG_DIR, DATA_DIR, load_json

URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=precipitation_sum,temperature_2m_max,soil_moisture_0_to_7cm_mean&timezone=Asia%2FKolkata"
)


def main():
    demo = load_json(CONFIG_DIR / "demo_config.json")
    mine = demo["mines"]["DEMO_MINE"]
    start, end = demo["history_start"], demo["history_end"]
    url = URL.format(lat=mine["lat"], lon=mine["lon"], start=start, end=end)
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.loads(r.read().decode())["daily"]
    df = pd.DataFrame({
        "date": d["time"],
        "rainfall_mm": d["precipitation_sum"],
        "temperature_max_c": d["temperature_2m_max"],
        "soil_moisture_m3m3": d["soil_moisture_0_to_7cm_mean"],
    })
    missing = int(df.isna().any(axis=1).sum())
    df.to_csv(DATA_DIR / "raw" / "weather" / "era5_open_meteo_demo_mine.csv", index=False)
    meta = {
        "source": "Open-Meteo historical weather API (ERA5 / ERA5-Land reanalysis, ECMWF Copernicus)",
        "url": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "location": {"lat": mine["lat"], "lon": mine["lon"]},
        "rows": int(len(df)),
        "rows_with_missing_values": missing,
        "data_mode": "REAL_PUBLIC",
        "note": "Reanalysis values for a ~10 km grid cell; not a mine-site rain gauge.",
    }
    (DATA_DIR / "raw" / "weather" / "era5_open_meteo_sources.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta)


if __name__ == "__main__":
    main()
