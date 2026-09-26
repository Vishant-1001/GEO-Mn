"""Normalise the REAL subsurface / ground-evidence records transcribed from official documents.

Inputs (data/raw/subsurface/, transcribed verbatim from the cited PDFs; see nmet_documents.csv for
URLs and SHA-256 of each source document):
  nmet_block_corners.csv       official NMET exploration-block corner coordinates (WGS84)
  nmet_reported_findings.csv   reported outcomes / grades / structure / proposals, with quotes
  nmet_surface_samples.csv     reported surface-sample chemistry (hand-held XRF)
  moil_reported_exploration.csv MOIL company-level exploration & resource statements

Outputs (data/processed/subsurface/):
  exploration_blocks.csv        block polygons (GeoJSON), centroid, area, stage, evidence summary
  surface_geochem_samples.csv   decimal coordinates, validated ranges
  reported_findings.csv         findings joined to blocks
  moil_reported_exploration.csv company-level context (not linked to any target)
  geological_constraints.json   real, cited expectations shown with a target's observed block evidence

Nothing here is a borehole log: the only drilling outcome available (Nagardhan) is a REPORTED
block-level count without coordinates or assays, and is kept exactly that way.

Run: python scripts/data_processing/build_real_subsurface.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.common import CONFIG_DIR, load_json, save_json  # noqa: E402

RAW = ROOT / "data" / "raw" / "subsurface"
OUT = ROOT / "data" / "processed" / "subsurface"

EVIDENCE_RANK = {"PROPOSAL_ONLY": 0, "STRUCTURAL": 1, "GEOLOGICAL_CONTEXT": 1, "GEOLOGICAL_MAPPING": 2, "GRADE_REPORTED": 2,
                 "GEOCHEMICAL_SURFACE": 2, "PIT_INTERSECTION_REPORTED": 2, "DRILLING_INTERSECTION_REPORTED": 3}


def dms(s: str) -> float:
    d, m, sec = (float(x) for x in re.findall(r"[\d.]+", s)[:3])
    if not (0 <= m < 60 and 0 <= sec < 60):
        raise ValueError(f"bad DMS {s}")
    return d + m / 60 + sec / 3600


def main():
    from shapely.geometry import MultiPoint, Polygon, mapping

    area = load_json(CONFIG_DIR / "exploration_config.json")["study_area"]
    docs = pd.read_csv(RAW / "nmet_documents.csv")
    corners = pd.read_csv(RAW / "nmet_block_corners.csv")
    corners["lat"] = corners["lat_dms"].map(dms)
    corners["lon"] = corners["lon_dms"].map(dms)
    findings = pd.read_csv(RAW / "nmet_reported_findings.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    blocks = []
    for bid, g in corners.groupby("block_id", sort=False):
        pts = list(zip(g["lon"], g["lat"]))
        poly = Polygon(pts)
        if not poly.is_valid:                       # corner order not a simple ring -> convex hull
            poly = MultiPoint(pts).convex_hull
        c = poly.centroid
        km2 = poly.area * (111.32 ** 2) * abs(__import__("math").cos(__import__("math").radians(c.y)))
        doc = docs[docs["doc_id"] == g["source_doc"].iloc[0]].iloc[0]
        f = findings[findings["block_id"] == bid]
        top = max((EVIDENCE_RANK[e] for e in f["evidence_class"]), default=0)
        blocks.append({
            "block_id": bid, "lat": round(c.y, 5), "lon": round(c.x, 5), "area_km2": round(km2, 2),
            "in_study_area": bool(area["lat_min"] <= c.y <= area["lat_max"] and area["lon_min"] <= c.x <= area["lon_max"]),
            "geometry": json.dumps(mapping(poly)), "stage": doc["stage"], "agency": doc["agency"], "commodity": doc["commodity"],
            "district": doc["district"], "state": doc["state"], "source_doc": doc["doc_id"], "official_url": doc["official_url"],
            "evidence_classes": ";".join(sorted(set(f["evidence_class"]))),
            "implied_evidence_level": top, "provenance": "REAL_GOVERNMENT",
        })
    bdf = pd.DataFrame(blocks)
    bdf.to_csv(OUT / "exploration_blocks.csv", index=False)

    s = pd.read_csv(RAW / "nmet_surface_samples.csv")
    s["lat"], s["lon"] = s["lat_dms"].map(dms).round(6), s["lon_dms"].map(dms).round(6)
    for col in ("mn_pct", "ca_pct", "si_pct", "p_pct", "fe_pct", "al_pct"):
        bad = s[col].dropna()
        if ((bad < 0) | (bad > 100)).any():
            raise ValueError(f"{col} outside 0-100 %")
    s["provenance"] = "REAL_GOVERNMENT"
    s.drop(columns=["lat_dms", "lon_dms"]).to_csv(OUT / "surface_geochem_samples.csv", index=False)

    f = findings.merge(bdf[["block_id", "lat", "lon"]], on="block_id", how="left")
    f["provenance"] = "REAL_GOVERNMENT"
    f.to_csv(OUT / "reported_findings.csv", index=False)
    m = pd.read_csv(RAW / "moil_reported_exploration.csv")
    m["provenance"] = "REAL_MOIL_PUBLIC"
    m.to_csv(OUT / "moil_reported_exploration.csv", index=False)

    struct = findings[findings["evidence_class"] == "STRUCTURAL"]
    grades = findings[findings["evidence_class"].isin(["GRADE_REPORTED", "GEOCHEMICAL_SURFACE"])]
    drill = findings[findings["evidence_class"] == "DRILLING_INTERSECTION_REPORTED"].iloc[0]
    constraints = {
        "_note": "REAL, CITED geological relationships reported by the official block documents; shown as expectations, never as measurements at a target.",
        "stratigraphy": {
            "group": "Sausar Group (Mesoproterozoic), Central Indian manganese belt",
            "formations_top_down": ["Bichua", "Junewani", "Chorbaoli", "Mansar", "Lohangi", "Sitasaongi"],
            "mn_horizons": {
                "H1": "contact of Mansar Formation with overlying Chorbaoli Formation",
                "H2": "within the Mansar Formation",
                "H3": "contact of Mansar Formation with underlying Lohangi/Sitasaongi Formation",
            },
            "host_lithologies": ["muscovite / quartz-mica schist", "gondite (Mn-silicate rock)", "calc-silicate / calc-gneiss",
                                 "marble", "quartzite", "biotite gneiss"],
            "source": "NMET_KAWALEWADA (DGM Maharashtra), section 3",
        },
        "structure": {
            "strike": "ENE-WSW (E-W in the central belt)",
            "dip_direction": "S",
            "dip_deg_ranges_reported": [[int(r.dip_min_deg), int(r.dip_max_deg), r.block_id] for r in struct.itertuples()],
            "source": "NMET_KAWALEWADA, NMET_RONGHA, NMET_LANJERA",
        },
        "ore_body": {
            "reef_width_m_reported": [1.5, 2.0],
            "note": "reefs often intercalated with gondite; supergene oxide enrichment near surface (e.g. Dongri Buzurg)",
            "source": "NMET_KAWALEWADA",
        },
        # a bound that is not stated (e.g. "< 38 % Mn") is null, never NaN (invalid JSON)
        "grades_reported_pct_mn": [[r.block_id, None if pd.isna(r.mn_pct_min) else r.mn_pct_min,
                                    None if pd.isna(r.mn_pct_max) else r.mn_pct_max, r.finding_type] for r in grades.itertuples()],
        "drilling_outcome_reported": {"block": drill.block_id, "boreholes": int(drill.count_tested),
                                      "intersected": int(drill.count_positive), "source": drill.source_doc},
        "proposed_borehole_depths_m": sorted(set(int(x) for x in findings["depth_m"].dropna())),
        "ibm_grade_classes_pct_mn": {
            "high": ">= 46", "medium": "35 to < 46", "low": "25 to < 35", "very_low": "< 25",
            "source": "Indian Bureau of Mines, Indian Minerals Yearbook 2022 — Manganese Ore (grade-wise production classes)",
            "official_url": "https://ibm.gov.in/writereaddata/files/17125770456613da1576db0Manganese_Ore_2022.pdf",
        },
    }
    save_json(OUT / "geological_constraints.json", constraints)
    man = {
        "dataset_name": "Official manganese exploration-block records (NMET) + MOIL reported exploration",
        "provider": "National Mineral Exploration Trust (Ministry of Mines) / DGM Maharashtra / MECL; MOIL Ltd.",
        "documents": docs.to_dict(orient="records"),
        "blocks": int(len(bdf)), "blocks_in_study_area": int(bdf["in_study_area"].sum()),
        "surface_samples": int(len(s)), "findings": int(len(findings)),
        "transcription": "values transcribed verbatim from the cited PDFs (quotes kept in nmet_reported_findings.csv)",
        "provenance": "REAL_GOVERNMENT / REAL_MOIL_PUBLIC",
        "limitations": [
            "No borehole collar coordinates, depths, logs or assays are published in these documents.",
            "Nagardhan drilling outcome is a reported block-level count; analytical data are stated as not available.",
            "Katori-Jhiriya samples are hand-held XRF surface readings (screening quality).",
            "Proposed boreholes are plans, never treated as observations.",
        ],
    }
    (ROOT / "data" / "manifests" / "real_subsurface_nmet.json").write_text(json.dumps(man, indent=2) + "\n")
    print(bdf[["block_id", "lat", "lon", "area_km2", "in_study_area", "stage", "implied_evidence_level"]].to_string())


if __name__ == "__main__":
    main()
