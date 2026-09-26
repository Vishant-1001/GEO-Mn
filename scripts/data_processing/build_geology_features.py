"""Per-cell REAL geology / structural features from NRSC (Bhuvan) 1:50k layers.

Inputs (from scripts/data_acquisition/fetch_bhuvan_geology.py; REAL_GOVERNMENT):
  data/raw/geology/bhuvan_geomorphology.geojson    geomorphology polygons (attribute 'Des')
  data/raw/geology/bhuvan_lineament_mask.npz       lineament raster (0.0005 deg ~ 55 m)
Output:
  data/processed/exploration/geology_features_grid.csv   one row per 0.01 degree cell centre

Features (documented encodings; source class text kept for audit):
  geom_structural_origin  1 if the class starts with "Structural Origin"
  geom_hills_valleys      1 if landform is "Hills and Valleys"
  geom_plateau            1 if landform is a plateau (upper / lower)
  geom_pediplain          1 if "Pediment-PediPlain Complex"
  geom_dissection         0 none, 1 low, 2 moderate, 3 highly dissected
  geom_water_fluvial      1 for water bodies / fluvial landforms
  lineament_dist_km       distance from the cell centre to the nearest mapped lineament pixel
  lineament_density       share of lineament pixels within +-2.5 km of the cell centre
Alignment: polygons sampled at the cell centre (no areal averaging); the lineament raster is
rendered at 55 m, so positional precision is ~ +-1 pixel plus the 1:50k mapping precision.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.common import CONFIG_DIR, load_json  # noqa: E402

GEO_FEATURES = ["geom_structural_origin", "geom_hills_valleys", "geom_plateau", "geom_pediplain", "geom_dissection",
                "geom_water_fluvial", "lineament_dist_km", "lineament_density"]


def encode(des):
    if not isinstance(des, str):
        return {k: np.nan for k in GEO_FEATURES[:6]}
    d = des.lower()
    diss = 3 if "highly dissected" in d else 2 if "moderately dissected" in d else 1 if "low dissected" in d else 0
    return {"geom_structural_origin": float(d.startswith("structural origin")),
            "geom_hills_valleys": float("hills and valleys" in d), "geom_plateau": float("plateau" in d),
            "geom_pediplain": float("pediplain" in d), "geom_dissection": float(diss),
            "geom_water_fluvial": float(d.startswith("water bodies") or d.startswith("fluvial"))}


def lineament_features(lats, lons, npz):
    m = npz["mask"]
    res = float(npz["res_deg"])
    lon0, lat_max = float(npz["lon_min"]), float(npz["lat_max"])
    lat_mid = np.radians(np.mean(lats))
    dy_km, dx_km = res * 111.32, res * 111.32 * np.cos(lat_mid)
    dist = ndimage.distance_transform_edt(~m, sampling=(dy_km, dx_km))
    win = int(round(2.5 / dy_km))
    dens = ndimage.uniform_filter(m.astype("float32"), size=2 * win + 1, mode="constant")
    rows = np.clip(np.floor((lat_max - lats) / res).astype(int), 0, m.shape[0] - 1)
    cols = np.clip(np.floor((lons - lon0) / res).astype(int), 0, m.shape[1] - 1)
    return dist[rows, cols], dens[rows, cols]


def main():
    from shapely import STRtree
    from shapely.geometry import Point, shape

    cfg = load_json(CONFIG_DIR / "exploration_config.json")
    a, res = cfg["study_area"], cfg["grid_res_deg"]
    lats = np.round(np.arange(a["lat_min"] + res / 2, a["lat_max"], res), 4)
    lons = np.round(np.arange(a["lon_min"] + res / 2, a["lon_max"], res), 4)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    cells = pd.DataFrame({"lat": LA.ravel(), "lon": LO.ravel()})

    gj = json.loads((ROOT / "data" / "raw" / "geology" / "bhuvan_geomorphology.geojson").read_text())
    shapes = [shape(f["geometry"]) for f in gj["features"]]
    des = [f["properties"].get("Des") for f in gj["features"]]
    tree = STRtree(shapes)
    pts = [Point(x, y) for x, y in zip(cells["lon"], cells["lat"])]
    hit = tree.query(pts, predicate="within")          # (point_idx, poly_idx) pairs
    cls = np.full(len(cells), None, dtype=object)
    for pi, gi in zip(hit[0], hit[1]):
        if cls[pi] is None:
            cls[pi] = des[gi]
    cells["geomorphology_class"] = cls
    enc = pd.DataFrame([encode(d) for d in cls])
    cells = pd.concat([cells, enc], axis=1)
    d_km, dens = lineament_features(cells["lat"].to_numpy(), cells["lon"].to_numpy(),
                                    np.load(ROOT / "data" / "raw" / "geology" / "bhuvan_lineament_mask.npz"))
    cells["lineament_dist_km"] = np.round(d_km, 3)
    cells["lineament_density"] = np.round(dens, 5)
    cells["provenance"] = "REAL_GOVERNMENT (NRSC Bhuvan 1:50k)"
    out = ROOT / "data" / "processed" / "exploration" / "geology_features_grid.csv"
    cells.to_csv(out, index=False)
    print(len(cells), "cells; geomorphology coverage", round(float(cells["geomorphology_class"].notna().mean()), 4))
    print(cells[GEO_FEATURES].describe().T[["mean", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
