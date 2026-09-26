"""Acquire IMD gridded daily rainfall (0.25 deg) and maximum temperature (1 deg) and extract series.

Source: India Meteorological Department, Pune — gridded data archive
  rainfall  POST https://www.imdpune.gov.in/cmpg/Griddata/rainfall.php  rain=<year>
            135 x 129 grid, 66.5-100.0E, 6.5-38.5N, 0.25 deg, float32 mm/day, -999 missing
            (Pai et al. 2014, Mausam 65(1))
  max temp  POST https://www.imdpune.gov.in/cmpg/Griddata/maxtemp.php   maxtemp=<year>
            31 x 31 grid, 67.5-97.5E, 7.5-37.5N, 1.0 deg, float32 deg C, 99.9 missing
            (Srivastava et al. 2009)
Provenance: REAL_GOVERNMENT. Verified 2026-09-26: years up to 2025 are published; 2026 returns
an empty file (not yet released) and is therefore NOT fabricated or interpolated.

Outputs
  data/raw/weather/imd/*.grd                       raw binaries (git-ignored, re-downloadable)
  data/processed/weather/imd_daily_mine_belt.csv    daily belt-mean rainfall / Tmax + demo-mine cell
  data/manifests/real_imd_gridded.json

Run: python scripts/data_acquisition/fetch_imd_gridded.py [--years 2012-2025]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.common import CONFIG_DIR, load_json  # noqa: E402

RAW = ROOT / "data" / "raw" / "weather" / "imd"
PROC = ROOT / "data" / "processed" / "weather"
RF = {"url": "https://www.imdpune.gov.in/cmpg/Griddata/rainfall.php", "field": "rain", "nx": 135, "ny": 129,
      "lon0": 66.5, "lat0": 6.5, "res": 0.25, "missing": -999.0}
TX = {"url": "https://www.imdpune.gov.in/cmpg/Griddata/maxtemp.php", "field": "maxtemp", "nx": 31, "ny": 31,
      "lon0": 67.5, "lat0": 7.5, "res": 1.0, "missing": 99.9}
# MOIL's Balaghat-Nagpur-Bhandara mining belt (all MOIL Central-India mines lie inside)
BELT = {"lat_min": 21.25, "lat_max": 22.25, "lon_min": 79.0, "lon_max": 80.75}


def fetch(spec, year):
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"{spec['field']}_{year}.grd"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    data = urllib.parse.urlencode({spec["field"]: str(year)}).encode()
    req = urllib.request.Request(spec["url"], data=data, headers={"User-Agent": "GEO-MN research prototype"})
    with urllib.request.urlopen(req, timeout=600) as r:
        blob = r.read()
    if not blob:
        raise FileNotFoundError(f"IMD returned no data for {spec['field']} {year} (not published)")
    dest.write_bytes(blob)
    return dest


def read(spec, path, year):
    ndays = (date(year + 1, 1, 1) - date(year, 1, 1)).days
    a = np.fromfile(path, dtype="<f4")
    if a.size != ndays * spec["ny"] * spec["nx"]:
        raise ValueError(f"{path.name}: unexpected size {a.size}")
    a = a.reshape(ndays, spec["ny"], spec["nx"]).astype("float64")
    a[np.isclose(a, spec["missing"]) | (a < -900)] = np.nan
    return a


def cell_index(spec, lat, lon):
    return int(round((lat - spec["lat0"]) / spec["res"])), int(round((lon - spec["lon0"]) / spec["res"]))


def belt_mask(spec):
    lats = spec["lat0"] + np.arange(spec["ny"]) * spec["res"]
    lons = spec["lon0"] + np.arange(spec["nx"]) * spec["res"]
    mi = (lats >= BELT["lat_min"] - spec["res"] / 2) & (lats <= BELT["lat_max"] + spec["res"] / 2)
    mj = (lons >= BELT["lon_min"] - spec["res"] / 2) & (lons <= BELT["lon_max"] + spec["res"] / 2)
    return np.ix_(mi, mj), int(mi.sum() * mj.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2012-2025")
    a, b = (int(x) for x in ap.parse_args().years.split("-"))
    mine = load_json(CONFIG_DIR / "demo_config.json")["mines"]["DEMO_MINE"]
    rows, files, unavailable = [], {}, []
    for year in range(a, b + 1):
        try:
            rf = read(RF, fetch(RF, year), year)
            tx = read(TX, fetch(TX, year), year)
        except FileNotFoundError as e:
            unavailable.append(str(e))
            continue
        files[f"rain_{year}.grd"] = hashlib.sha256((RAW / f"rain_{year}.grd").read_bytes()).hexdigest()
        files[f"maxtemp_{year}.grd"] = hashlib.sha256((RAW / f"maxtemp_{year}.grd").read_bytes()).hexdigest()
        (rsel, n_rf), (tsel, n_tx) = belt_mask(RF), belt_mask(TX)
        ri, rj = cell_index(RF, mine["lat"], mine["lon"])
        ti, tj = cell_index(TX, mine["lat"], mine["lon"])
        for d in range(rf.shape[0]):
            with np.errstate(all="ignore"):
                rows.append({
                    "date": (date(year, 1, 1) + timedelta(days=d)).isoformat(),
                    "belt_rainfall_mm": round(float(np.nanmean(rf[d][rsel])), 2),
                    "belt_tmax_c": round(float(np.nanmean(tx[d][tsel])), 2),
                    "mine_cell_rainfall_mm": round(float(rf[d, ri, rj]), 2),
                    "mine_cell_tmax_c": round(float(tx[d, ti, tj]), 2),
                })
        print(f"  {year}: {rf.shape[0]} days", flush=True)
    df = pd.DataFrame(rows)
    df["source"] = "IMD gridded (rainfall 0.25 deg, Tmax 1 deg)"
    df["provenance"] = "REAL_GOVERNMENT"
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "imd_daily_mine_belt.csv", index=False)
    man = {
        "dataset_name": "IMD gridded daily rainfall (0.25 deg) and maximum temperature (1 deg)",
        "provider": "India Meteorological Department (IMD), Pune, Government of India",
        "official_url": "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "IMD public gridded data (research use; cite Pai et al. 2014 / Srivastava et al. 2009)",
        "years": [a, b], "rows": int(len(df)), "unavailable": unavailable,
        "belt": BELT, "belt_cells": {"rainfall": belt_mask(RF)[1], "tmax": belt_mask(TX)[1]},
        "demo_mine_cell": {"rainfall": [RF["lat0"] + ri * RF["res"], RF["lon0"] + rj * RF["res"]],
                           "tmax": [TX["lat0"] + ti * TX["res"], TX["lon0"] + tj * TX["res"]]},
        "native_resolution": {"rainfall": "0.25 deg daily", "tmax": "1.0 deg daily"},
        "units": {"rainfall": "mm/day", "tmax": "deg C"},
        "missing_days": {"belt_rainfall": int(df["belt_rainfall_mm"].isna().sum()), "belt_tmax": int(df["belt_tmax_c"].isna().sum())},
        "provenance": "REAL_GOVERNMENT",
        "sha256": files,
    }
    (ROOT / "data" / "manifests" / "real_imd_gridded.json").write_text(json.dumps(man, indent=2) + "\n")
    print({k: v for k, v in man.items() if k != "sha256"})


if __name__ == "__main__":
    main()
