"""Exploration target generation: grid -> clusters -> target-level evidence.

Pipeline
  1. keep scored cells with prospectivity_rank >= targets.min_rank
  2. 8-connected clustering on the 0.01 degree lattice; drop clusters < min_cells
  3. aggregate target-level evidence:
       prospectivity (mean / peak rank), uncertainty (mean member rank SD),
       applicability (share of HIGH / LOW cells), surface feature summary,
       geological context (Macrostrat / GSC world map), documented MRDS records
       inside or within 1 km of the footprint, subsurface evidence (if any)
  4. evidence level (0-4, never auto-promoted beyond what the data show)
  5. static priority inputs (strategic relevance is added at runtime because it
     depends on the current supply-gap state)

Output: data/exploration_targets.json

Run:  python -m ml.target_engine
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import KMeans

from ml.common import CONFIG_DIR, DATA_DIR, haversine_km, load_json, save_json
from ml.eo_features import FEATURES

EVIDENCE_LEVELS = {
    0: "LEVEL 0 — remote sensing indication",
    1: "LEVEL 1 — remote sensing + geological support",
    2: "LEVEL 2 — ground / geophysical / geochemical support",
    3: "LEVEL 3 — drilling / assay support",
    4: "LEVEL 4 — resource / reserve work",
}
SUBSURFACE_COLUMNS = ["target_id", "source_type", "source_id", "depth_m", "evidence_strength", "evidence_status", "source_mode"]
DRILL_TYPES = {"drill_intersection", "assay"}
GROUND_TYPES = {"geophysical_anomaly", "geochemical_evidence", "geological_section"}


def load_subsurface():
    p = DATA_DIR / "subsurface_evidence.csv"
    if not p.exists():
        return pd.DataFrame(columns=SUBSURFACE_COLUMNS)
    return pd.read_csv(p)


def applicability_of(levels):
    s = pd.Series(levels)
    if (s == "LOW").mean() >= 0.2:
        return "LOW"
    if (s == "HIGH").mean() >= 0.8:
        return "HIGH"
    return "MODERATE"


def uncertainty_of(sd, cfg):
    t = cfg["uncertainty_thresholds_rank_sd"]
    return "LOW" if sd < t["low_below"] else ("HIGH" if sd > t["high_above"] else "MODERATE")


def footprint_geojson(cells, res):
    """Dissolved outline of the target's cell squares (exact footprint, no smoothing)."""
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union

    shape = unary_union([box(lon - res / 2, lat - res / 2, lon + res / 2, lat + res / 2)
                         for lat, lon in zip(cells["lat"], cells["lon"])])
    gj = mapping(shape.simplify(0))

    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 4), round(c[1], 4)]

    return {"type": gj["type"], "coordinates": rnd(gj["coordinates"])}


def main():
    cfg = load_json(CONFIG_DIR / "exploration_config.json")
    demo = load_json(CONFIG_DIR / "demo_config.json")
    res = cfg["grid_res_deg"]
    tcfg = cfg["targets"]
    grid = pd.read_csv(DATA_DIR / "exploration_grid.csv")
    feats = pd.read_csv(DATA_DIR / "exploration_grid_features.csv.gz")
    grid = grid.merge(feats[["lat", "lon"] + FEATURES], on=["lat", "lon"], how="left")
    geo = pd.read_csv(DATA_DIR / "geology_lattice.csv")
    mrds = pd.read_csv(DATA_DIR / "mrds_mn_occurrences.csv")
    mrds = mrds[mrds["label_use"] != "EXCLUDED_REGIONAL_RECORD"]
    producers = mrds[mrds["dev_stat"].isin(["Producer", "Past Producer"])]
    subsurface = load_subsurface()

    area = cfg["study_area"]
    grid["i"] = np.round((grid["lat"] - area["lat_min"]) / res - 0.5).astype(int)
    grid["j"] = np.round((grid["lon"] - area["lon_min"]) / res - 0.5).astype(int)
    ni, nj = grid["i"].max() + 1, grid["j"].max() + 1
    mask = np.zeros((ni, nj), dtype=bool)
    hot = grid[(grid["data_status"] == "OK") & (grid["prospectivity_rank"] >= tcfg["min_rank"])]
    mask[hot["i"], hot["j"]] = True
    structure = np.ones((3, 3)) if tcfg["connectivity"] == 8 else None
    lab, n = ndimage.label(mask, structure=structure)
    grid["cluster"] = lab[grid["i"], grid["j"]]

    pct = {f: grid[f].rank(pct=True) for f in FEATURES}
    scored = grid[grid["data_status"] == "OK"]
    clusters = []
    max_cells = tcfg["max_cells_per_target"]
    for cid in range(1, n + 1):
        cells = grid[grid["cluster"] == cid]
        if len(cells) < tcfg["min_cells"]:
            continue
        # split valley-scale components into compact sub-targets (deterministic k-means on coordinates)
        parts = [cells]
        if len(cells) > max_cells:
            k = int(np.ceil(len(cells) / max_cells))
            xy = np.c_[cells["lat"], cells["lon"] * np.cos(np.radians(cells["lat"]))]
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(xy)
            parts = [cells[labels == c] for c in range(k)]
        for part in parts:
            if len(part) >= tcfg["min_cells"]:
                clusters.append((cid, part, float(part["prospectivity_rank"].mean())))
    clusters.sort(key=lambda c: (-c[2], -len(c[1])))
    clusters = clusters[: tcfg["max_targets"]]

    mine = demo["mines"]["DEMO_MINE"]
    targets = []
    for k, (cid, cells, mean_rank) in enumerate(clusters, start=1):
        tid = f"T{k:02d}"
        w = cells["prospectivity_rank"].to_numpy()
        clat = float(np.average(cells["lat"], weights=w))
        clon = float(np.average(cells["lon"], weights=w))
        cell_km2 = (res * 111.32) * (res * 111.32 * np.cos(np.radians(clat)))
        sd = float(cells["rank_sd"].mean())

        # documented MRDS records inside the footprint or within 1 km of a footprint cell
        dm = haversine_km(mrds["lat"].to_numpy()[:, None], mrds["lon"].to_numpy()[:, None],
                          cells["lat"].to_numpy()[None, :], cells["lon"].to_numpy()[None, :]).min(axis=1)
        near = mrds[dm <= 1.0 + res * 111.32 / 2].copy()
        near["distance_km"] = dm[dm <= 1.0 + res * 111.32 / 2]
        records = [{"dep_id": int(r.dep_id), "site_name": r.site_name, "dev_stat": r.dev_stat,
                    "distance_km": round(float(r.distance_km), 2), "url": r.url,
                    "used_as_training_label": r.label_use == "POSITIVE"} for r in near.itertuples()]

        dg = haversine_km(geo["lat"].to_numpy(), geo["lon"].to_numpy(), clat, clon)
        gi = int(np.argmin(dg))
        grow = geo.iloc[gi]
        geo_support = bool(grow["precambrian"]) if pd.notna(grow["unit_name"]) else None

        sub = subsurface[subsurface["target_id"] == tid]
        sub_types = set(sub["source_type"]) if len(sub) else set()
        if len(sub):
            subsurface_status = "AVAILABLE"
        else:
            subsurface_status = "UNAVAILABLE"

        level = 0
        basis = ["Model prospectivity from Sentinel-2 / MODIS / NASADEM features (remote sensing indication)."]
        if geo_support:
            level = 1
            basis.append(f"Centroid lies in a Precambrian domain ('{grow['unit_name']}') on a small-scale world geology map; "
                         "all MRDS Mn records used for training in this study area lie in Precambrian domains.")
        if records or (sub_types & GROUND_TYPES):
            level = 2
            if records:
                basis.append(f"{len(records)} documented MRDS manganese record(s) within ~1 km of the footprint (reported context).")
        if sub_types & DRILL_TYPES:
            level = 3
            basis.append("Drilling / assay evidence recorded in subsurface_evidence.csv.")

        dp = haversine_km(producers["lat"].to_numpy(), producers["lon"].to_numpy(), clat, clon)
        d_prod = float(dp.min()) if len(dp) else None
        readiness = float(np.exp(-d_prod / cfg["development_readiness"]["distance_decay_km"])) if d_prod is not None else None

        surface = {}
        for f in FEATURES:
            surface[f] = {"mean": round(float(cells[f].mean()), 4),
                          "study_area_percentile": round(float(pct[f].loc[cells.index].mean() * 100), 1)}

        targets.append({
            "target_id": tid,
            "lat": round(clat, 4),
            "lon": round(clon, 4),
            "n_cells": int(len(cells)),
            "parent_component": int(cid),
            "area_km2": round(float(len(cells) * cell_km2), 1),
            "bbox": [round(float(cells["lon"].min() - res / 2), 4), round(float(cells["lat"].min() - res / 2), 4),
                     round(float(cells["lon"].max() + res / 2), 4), round(float(cells["lat"].max() + res / 2), 4)],
            "geometry": footprint_geojson(cells, res),
            "prospectivity_rank": round(mean_rank, 1),
            "peak_prospectivity_rank": round(float(w.max()), 1),
            "rank_sd": round(sd, 1),
            "uncertainty": uncertainty_of(sd, cfg),
            "applicability": applicability_of(cells["applicability"]),
            "applicability_cell_shares": {k: round(float(v), 3) for k, v in cells["applicability"].value_counts(normalize=True).items()},
            "evidence_level": level,
            "evidence_level_label": EVIDENCE_LEVELS[level],
            "evidence_basis": basis,
            "surface_evidence": surface,
            "geological_evidence": {
                "status": "AVAILABLE" if pd.notna(grow["unit_name"]) else "UNAVAILABLE",
                "unit_name": None if pd.isna(grow["unit_name"]) else grow["unit_name"],
                "interval": None if pd.isna(grow["interval"]) else grow["interval"],
                "age_bottom_ma": None if pd.isna(grow["age_bottom_ma"]) else float(grow["age_bottom_ma"]),
                "precambrian_host_domain": geo_support,
                "lattice_distance_km": round(float(dg[gi]), 1),
                "source": "Macrostrat API (scale=tiny) — Chorlton (2007) Generalized geology of the world, GSC OF 5529; CC-BY 4.0",
                "caveat": "World-scale map; unit boundaries uncertain by kilometres. Context only, not a model input.",
            },
            "documented_occurrences": records,
            "target_context": "BROWNFIELD" if records else "GREENFIELD",
            "contains_training_labels": any(r["used_as_training_label"] for r in records),
            "subsurface_status": subsurface_status,
            "subsurface_evidence": sub.to_dict(orient="records"),
            "nearest_documented_producer_km": round(d_prod, 1) if d_prod is not None else None,
            "development_readiness": round(readiness, 3) if readiness is not None else None,
            "distance_to_demo_mine_km": round(float(haversine_km(mine["lat"], mine["lon"], clat, clon)), 1),
        })

    meta = {
        "generated_from": "data/exploration_grid.csv",
        "method": (f"cells with prospectivity_rank >= {tcfg['min_rank']} clustered with {tcfg['connectivity']}-connectivity "
                   f"on the {res} degree lattice; components > {tcfg['max_cells_per_target']} cells split by k-means on "
                   f"coordinates; clusters with < {tcfg['min_cells']} cells dropped; "
                   f"top {tcfg['max_targets']} by mean rank kept"),
        "n_hot_cells": int(len(hot)),
        "n_clusters": int(n),
        "n_targets": len(targets),
        "scored_cells": int(len(scored)),
        "evidence_levels": EVIDENCE_LEVELS,
        "geology_support_basis": "All MRDS Mn training records in the study area fall on Precambrian units of the geology lattice (vs ~62% of the area).",
        "subsurface_note": ("No legitimate drilling, assay or geophysical data are available to this project; "
                            "subsurface_status is UNAVAILABLE for every target. data/subsurface_evidence.csv defines the schema."),
    }
    save_json(DATA_DIR / "exploration_targets.json", {"meta": meta, "targets": targets})
    print(f"{len(targets)} targets from {n} clusters ({len(hot)} hot cells)")
    for t in targets[:12]:
        print(t["target_id"], t["lat"], t["lon"], t["n_cells"], t["prospectivity_rank"], t["uncertainty"], t["applicability"],
              "L", t["evidence_level"], t["target_context"], t["distance_to_demo_mine_km"])


if __name__ == "__main__":
    main()
