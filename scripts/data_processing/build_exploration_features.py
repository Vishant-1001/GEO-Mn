"""Extract the full (baseline + extended) real EO feature grid over the study area.

Uses the single shared recipe in ml/eo_features.py (Sentinel-2 L2A, MODIS MOD11A2, NASADEM via
Microsoft Planetary Computer, 2024 window). Output:
  data/processed/exploration/eo_features_grid.csv.gz   (0.01 degree cells; REAL_DERIVED)
Run: python scripts/data_processing/build_exploration_features.py   (network; ~5-10 min)
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.common import CONFIG_DIR, load_json  # noqa: E402
from ml.eo_features import ALL_EO_FEATURES, FEATURE_RATIONALE, RECIPE, extract_cell_features  # noqa: E402

if __name__ == "__main__":
    a = load_json(CONFIG_DIR / "exploration_config.json")["study_area"]
    t0 = time.time()
    df = extract_cell_features((a["lon_min"], a["lat_min"], a["lon_max"], a["lat_max"]), threads=12,
                               cache_dir=ROOT / "data" / "runtime" / "eo_cache")
    out = ROOT / "data" / "processed" / "exploration" / "eo_features_grid.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")
    man = {"dataset_name": "Real EO feature grid (Sentinel-2 L2A, MODIS MOD11A2, NASADEM)",
           "provider": "ESA Copernicus / NASA / USGS via Microsoft Planetary Computer",
           "official_url": "https://planetarycomputer.microsoft.com/api/stac/v1",
           "retrieval_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "cells": int(len(df)), "complete_cells": int(df[ALL_EO_FEATURES].notna().all(axis=1).sum()),
           "features": ALL_EO_FEATURES, "rationale": FEATURE_RATIONALE, "recipe": RECIPE,
           "native_resolution": "S2 read at 200 m, NASADEM 30 m, MODIS 1 km; aggregated to 0.01 degree cells",
           "provenance": "REAL_DERIVED", "seconds": round(time.time() - t0, 1)}
    (ROOT / "data" / "manifests" / "real_eo_features.json").write_text(json.dumps(man, indent=2, default=str) + "\n")
    print({k: man[k] for k in ("cells", "complete_cells", "seconds")})
