"""Build the exploration input datasets from public sources.

Outputs (all REAL_PUBLIC-derived):
  data/mrds_mn_occurrences.csv        MRDS manganese records inside the study area
  data/exploration_grid_features.csv.gz  0.01 degree cell features (see ml/eo_features.py)
  data/geology_lattice.csv            Macrostrat / GSC world-geology units on a 0.1 degree lattice

Run:  python -m ml.build_exploration_dataset [--skip-grid] [--skip-geology]
Network access is required; outputs are committed so the API never needs it.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.common import CONFIG_DIR, DATA_DIR, load_json
from ml.eo_features import extract_cell_features

MRDS_WFS = (
    "https://mrdata.usgs.gov/services/wfs/mrds?service=WFS&version=1.1.0&request=GetFeature"
    "&typeName=mrds&bbox={lon_min},{lat_min},{lon_max},{lat_max}&maxFeatures=5000"
)
MACROSTRAT = "https://macrostrat.org/api/v2/geologic_units/map?lat={lat}&lng={lon}&scale=tiny"


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "GEO-MN research prototype"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def fetch_mrds(aoi, cfg):
    xml = _get(MRDS_WFS.format(**aoi))
    pat = re.compile(
        r"<gml:pos>([\d.\-]+) ([\d.\-]+)</gml:pos>.*?<ms:dep_id>(\d+)</ms:dep_id>\s*"
        r"<ms:site_name>(.*?)</ms:site_name>\s*<ms:dev_stat>(.*?)</ms:dev_stat>.*?"
        r"<ms:url>(.*?)</ms:url>\s*<ms:code_list>(.*?)</ms:code_list>",
        re.S,
    )
    rows = []
    for lat, lon, dep_id, name, stat, url, codes in pat.findall(xml):
        rows.append({
            "dep_id": int(dep_id), "site_name": name.strip(), "dev_stat": stat.strip(),
            "lat": float(lat), "lon": float(lon), "commodities": codes.strip(), "url": url.strip(),
        })
    df = pd.DataFrame(rows)
    n_all = len(df)
    df = df[df["commodities"].str.split().apply(lambda c: "MN" in c)]
    n_mn = len(df)
    inside = df["lat"].between(aoi["lat_min"], aoi["lat_max"]) & df["lon"].between(aoi["lon_min"], aoi["lon_max"])
    df = df[inside].copy()
    pats = cfg["positive_labels"]["exclude_name_patterns"]
    vague = df["site_name"].apply(lambda s: any(re.search(rf"\b{p}\b", s) for p in pats))
    df["label_use"] = np.where(vague, "EXCLUDED_REGIONAL_RECORD", "POSITIVE")
    prec = cfg["positive_labels"]["dedupe_precision_deg"]
    key = (df["lat"] / prec).round().astype(int).astype(str) + "_" + (df["lon"] / prec).round().astype(int).astype(str)
    dup = key.duplicated() & (df["label_use"] == "POSITIVE")
    df.loc[dup, "label_use"] = "EXCLUDED_DUPLICATE_LOCATION"
    df = df.sort_values("dep_id").reset_index(drop=True)
    summary = {
        "mrds_records_in_query_bbox": n_all,
        "mn_records": n_mn,
        "mn_records_in_study_area": int(len(df)),
        "positives_used": int((df["label_use"] == "POSITIVE").sum()),
        "excluded": df["label_use"].value_counts().to_dict(),
    }
    return df, summary


def fetch_geology(aoi, step=0.1, pause=0.15):
    rows = []
    lats = np.round(np.arange(aoi["lat_min"] + step / 2, aoi["lat_max"], step), 3)
    lons = np.round(np.arange(aoi["lon_min"] + step / 2, aoi["lon_max"], step), 3)
    for la in lats:
        for lo in lons:
            rec = {"lat": la, "lon": lo, "unit_name": None, "lith": None, "age_top_ma": None,
                   "age_bottom_ma": None, "interval": None, "source_id": None}
            for attempt in range(3):
                try:
                    d = json.loads(_get(MACROSTRAT.format(lat=la, lon=lo), timeout=30))["success"]["data"]
                    if d:
                        u = d[0]
                        rec.update(unit_name=u.get("name"), lith=u.get("lith"), age_top_ma=u.get("t_age"),
                                   age_bottom_ma=u.get("b_age"), interval=u.get("best_int_name"),
                                   source_id=u.get("source_id"))
                    break
                except Exception:
                    time.sleep(1 + attempt)
            rows.append(rec)
            time.sleep(pause)
    df = pd.DataFrame(rows)
    df["precambrian"] = df["age_bottom_ma"].astype(float) > 541.0
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--skip-geology", action="store_true")
    ap.add_argument("--skip-mrds", action="store_true")
    args = ap.parse_args()

    cfg = load_json(CONFIG_DIR / "exploration_config.json")
    aoi = cfg["study_area"]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_path = DATA_DIR / "exploration_sources.json"
    meta = load_json(meta_path) if meta_path.exists() else {}

    if not args.skip_mrds:
        print("Fetching MRDS manganese records ...")
        mrds, summary = fetch_mrds(aoi, cfg)
        mrds.to_csv(DATA_DIR / "mrds_mn_occurrences.csv", index=False)
        meta["mrds"] = {"retrieved_utc": stamp, "url": MRDS_WFS.format(**aoi), **summary}
        print(json.dumps(summary, indent=2))

    if not args.skip_geology:
        print("Fetching geology lattice (Macrostrat, scale=tiny) ...")
        geo = fetch_geology(aoi)
        geo.to_csv(DATA_DIR / "geology_lattice.csv", index=False)
        meta["geology"] = {
            "retrieved_utc": stamp,
            "api": "https://macrostrat.org/api/v2/geologic_units/map (scale=tiny)",
            "underlying_map": "Chorlton, L.B. (2007) Generalized geology of the world, Geological Survey of Canada Open File 5529, doi:10.4095/223767",
            "license": "CC-BY 4.0",
            "lattice_step_deg": 0.1,
            "cells": int(len(geo)),
            "cells_with_unit": int(geo["unit_name"].notna().sum()),
            "note": "Small-scale (world) map: unit boundaries are uncertain by kilometres to tens of kilometres. Used only as geological context for evidence levels, not as a model feature.",
        }

    if not args.skip_grid:
        print("Extracting grid features (this reads Sentinel-2, MODIS and NASADEM) ...")
        t0 = time.time()
        bbox = (aoi["lon_min"], aoi["lat_min"], aoi["lon_max"], aoi["lat_max"])
        grid = extract_cell_features(bbox, threads=24)
        grid.to_csv(DATA_DIR / "exploration_grid_features.csv.gz", index=False, float_format="%.6g")
        meta["grid"] = {
            "retrieved_utc": stamp,
            "cells": int(len(grid)),
            "cells_complete": int(grid[["NDVI", "Iron_Oxide_Index", "Clay_Hydroxyl_Index", "LST_Day_K", "elevation", "slope"]].notna().all(axis=1).sum()),
            "extraction_seconds": round(time.time() - t0, 1),
        }
        print(meta["grid"])

    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
