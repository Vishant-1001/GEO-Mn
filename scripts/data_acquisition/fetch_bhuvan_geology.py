"""Acquire NRSC (ISRO) 1:50k geomorphology polygons and structural lineaments from Bhuvan.

Source: Bhuvan GeoServer (https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms), layers
  geomorphology:{MP,MH}_GM50K_0506   (Geomorphology 1:50,000, 2005-06 mapping)
  lineament:{MP,MH}_LN50K_0506       (Structural lineaments 1:50,000, 2005-06 mapping)
Provenance: REAL_GOVERNMENT (NRSC / ISRO, Government of India).

Access notes (verified 2026-09-26): WFS returns ServiceUnavailable, so vectors are obtained with
WMS GetFeatureInfo (JSON, full geometry + attributes) on 0.1 degree tiles; lineaments are thin
lines that GetFeatureInfo only returns on exact hits, so they are rasterised from WMS GetMap
renderings (0.0005 degree/pixel, ~55 m) instead. Output (raw, git-ignored except manifest):
  data/raw/geology/bhuvan_geomorphology.geojson
  data/raw/geology/bhuvan_lineament_mask.npz

Run: python scripts/data_acquisition/fetch_bhuvan_geology.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.common import CONFIG_DIR, load_json  # noqa: E402

WMS = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
STATES = ("MP", "MH")
OUT = ROOT / "data" / "raw" / "geology"


def _get(params, tries=4):
    url = WMS + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GEO-MN research prototype"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception:
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"failed: {url}")


def feature_info(layer, bbox):
    p = {"service": "WMS", "version": "1.1.1", "request": "GetFeatureInfo", "layers": layer, "query_layers": layer,
         "styles": "", "srs": "EPSG:4326", "bbox": ",".join(map(str, bbox)), "width": 101, "height": 101,
         "x": 50, "y": 50, "info_format": "application/json", "feature_count": 1000, "buffer": 75}
    return json.loads(_get(p))["features"]


def get_map(layer, bbox, w, h):
    from PIL import Image

    p = {"service": "WMS", "version": "1.1.1", "request": "GetMap", "layers": layer, "styles": "",
         "srs": "EPSG:4326", "bbox": ",".join(map(str, bbox)), "width": w, "height": h,
         "format": "image/png", "transparent": "true"}
    return np.array(Image.open(io.BytesIO(_get(p))).convert("RGBA"))


def main():
    cfg = load_json(CONFIG_DIR / "exploration_config.json")["study_area"]
    OUT.mkdir(parents=True, exist_ok=True)
    step = 0.1
    tiles = [(round(x, 3), round(y, 3), round(x + step, 3), round(y + step, 3))
             for x in np.arange(cfg["lon_min"], cfg["lon_max"] - 1e-9, step)
             for y in np.arange(cfg["lat_min"], cfg["lat_max"] - 1e-9, step)]

    # ---- geomorphology polygons ---------------------------------------------------
    feats = {}
    jobs = [(f"geomorphology:{s}_GM50K_0506", t) for s in STATES for t in tiles]

    def run(job):
        layer, t = job
        try:
            return layer, feature_info(layer, t)
        except Exception as e:  # recorded, not silently dropped
            return layer, e

    failures = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (layer, res) in enumerate(pool.map(run, jobs)):
            if isinstance(res, Exception):
                failures += 1
                continue
            for f in res:
                f["properties"]["_layer"] = layer
                feats[f["id"]] = f
            if i % 100 == 0:
                print(f"  geomorphology {i}/{len(jobs)} requests, {len(feats)} polygons", flush=True)
    gj = {"type": "FeatureCollection", "features": list(feats.values())}
    (OUT / "bhuvan_geomorphology.geojson").write_text(json.dumps(gj))

    # ---- lineament rasters (GetMap renderings) -----------------------------------
    res_deg = 0.0005
    nx = int(round((cfg["lon_max"] - cfg["lon_min"]) / res_deg))
    ny = int(round((cfg["lat_max"] - cfg["lat_min"]) / res_deg))
    mask = np.zeros((ny, nx), dtype=bool)
    tile_px = 1000
    for s in STATES:
        for j0 in range(0, ny, tile_px):
            for i0 in range(0, nx, tile_px):
                w, h = min(tile_px, nx - i0), min(tile_px, ny - j0)
                bbox = (cfg["lon_min"] + i0 * res_deg, cfg["lat_max"] - (j0 + h) * res_deg,
                        cfg["lon_min"] + (i0 + w) * res_deg, cfg["lat_max"] - j0 * res_deg)
                img = get_map(f"lineament:{s}_LN50K_0506", bbox, w, h)
                mask[j0:j0 + h, i0:i0 + w] |= img[..., 3] > 0
        print(f"  lineaments {s}: {int(mask.sum())} line pixels so far", flush=True)
    np.savez_compressed(OUT / "bhuvan_lineament_mask.npz", mask=mask, lon_min=cfg["lon_min"], lat_max=cfg["lat_max"],
                        res_deg=res_deg)

    man = {
        "dataset_name": "NRSC 1:50k geomorphology + structural lineaments (Bhuvan)",
        "provider": "National Remote Sensing Centre (NRSC), ISRO, Government of India",
        "official_url": WMS,
        "layers": [f"geomorphology:{s}_GM50K_0506" for s in STATES] + [f"lineament:{s}_LN50K_0506" for s in STATES],
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "Bhuvan terms of use (public viewing/analysis); redistribution of raw vectors not asserted",
        "access_method": "WMS GetFeatureInfo (polygons, JSON) and WMS GetMap (lineament rendering)",
        "geomorphology_polygons": len(feats),
        "geomorphology_failed_requests": failures,
        "lineament_raster_res_deg": res_deg,
        "lineament_pixels": int(mask.sum()),
        "native_scale": "1:50,000",
        "provenance": "REAL_GOVERNMENT",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (OUT / "bhuvan_geomorphology.geojson", OUT / "bhuvan_lineament_mask.npz")},
    }
    (ROOT / "data" / "manifests" / "real_bhuvan_geology.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
