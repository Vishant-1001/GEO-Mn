"""Geologically constrained SYNTHETIC subsurface-scenario generator.

Real subsurface audit (docs/DATA_SOURCE_AUDIT.md): no borehole collars, logs or assays are publicly
available for the study area. What IS available (data/processed/subsurface/) are official NMET block
records, one reported block-level drilling outcome (Nagardhan: 4 of 7 holes intersected Mn, no
assays), five hand-held-XRF surface samples, and cited geological constraints. This generator uses
those REAL constraints to simulate what future evidence at a target COULD look like.

Every output row is SIMULATED — never an observed borehole, assay or survey reading.

Hierarchy per target and scenario:
  1. real geological context  (Macrostrat unit; NMET block constraints if the target overlaps a block)
  2. real exploration-model prospectivity / uncertainty of the target
  3. host stratigraphy: Sausar Group, Mn horizons H1 (Mansar/Chorbaoli contact), H2 (within Mansar),
     H3 (Mansar/Lohangi-Sitasaongi contact)                                     [NMET_KAWALEWADA]
  4. structure: strike ENE-WSW, dip S, dip sampled from reported ranges 45-80 deg [NMET docs]
  5. hole trajectory: azimuth drilled against dip (NNW), inclination -60, depth from reported
     proposed depths (50-100 m)
  6. intervals: weathered zone / overburden (2-3 m reported at Nagardhan), host rock, ore horizons
  7. mineralised / barren intersections sampled conditionally on scenario
  8. grades: true reef thickness 1.5-2 m [reported]; Mn % from reported ranges (26-45 %), classified
     with IBM grade classes (>=46, 35-46, 25-35, <25 % Mn)
  9. assay noise (relative), Fe / SiO2 with Mn-SiO2 anti-correlation (oxide vs gondite)
 10. geophysical (resistivity / IP chargeability) response with noise
 11. scenario outcome: positive / negative / ambiguous drilling, geophysical or geochemical support
 12. uncertainty / confidence label
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SCENARIOS = ("NO_SUBSURFACE_EVIDENCE", "POSITIVE_GEOPHYSICAL_SUPPORT", "POSITIVE_GEOCHEMICAL_SUPPORT",
             "POSITIVE_DRILLING_INTERSECTION", "NEGATIVE_DRILLING_RESULT", "AMBIGUOUS_DRILLING_RESULT")
HORIZONS = {"H1": "Mansar/Chorbaoli contact", "H2": "within Mansar Formation", "H3": "Mansar/Lohangi-Sitasaongi contact"}


def ibm_grade_class(mn):
    if mn >= 46:
        return "HIGH (>=46% Mn)"
    if mn >= 35:
        return "MEDIUM (35-46% Mn)"
    if mn >= 25:
        return "LOW (25-35% Mn)"
    return "BELOW 25% Mn"


def apparent_thickness(true_m, bed_dip_deg, hole_incl_deg=60.0):
    """Down-hole length for a hole drilled against dip: angle to bed = bed dip + hole inclination."""
    angle = math.radians(min(89.0, bed_dip_deg + hole_incl_deg))
    return true_m / max(math.sin(angle), 0.2)


def _dip_range(target, constraints):
    ranges = constraints["structure"]["dip_deg_ranges_reported"]
    blk = target.get("nmet_block_id")
    own = [r for r in ranges if r[2] == blk]
    lo = min(r[0] for r in (own or ranges))
    hi = max(r[1] for r in (own or ranges))
    return lo, hi


def generate_target(t, scenario, constraints, cfg, rng):
    """Simulated evidence for one target under one scenario -> dict of record lists."""
    holes, intervals, geochem, geophys = [], [], [], []
    rank = float(t["prospectivity_rank"])
    dip_lo, dip_hi = _dip_range(t, constraints)
    lat0, lon0 = float(t["lat"]), float(t["lon"])
    drilling = scenario in ("POSITIVE_DRILLING_INTERSECTION", "NEGATIVE_DRILLING_RESULT", "AMBIGUOUS_DRILLING_RESULT")
    depths = constraints["proposed_borehole_depths_m"]
    g = cfg["grade"]

    if drilling:
        for h in range(cfg["holes_per_target"]):
            bid = f"SIM_{t['target_id']}_{scenario[:3]}_BH{h + 1:02d}"
            dip = float(rng.uniform(dip_lo, dip_hi))
            total = round(float(rng.uniform(min(depths), max(depths))), 1)
            weather_zone = float(rng.uniform(*cfg["overburden_m"]))
            holes.append({"borehole_id": bid, "target_id": t["target_id"], "scenario": scenario,
                          "latitude": round(lat0 + rng.normal(0, 0.002), 6), "longitude": round(lon0 + rng.normal(0, 0.002), 6),
                          "azimuth": round(float((340 + rng.normal(0, 10)) % 360), 1), "dip": -60.0,
                          "total_depth_m": round(total, 1), "bed_dip_deg": round(dip, 1),
                          "formation_context": "Sausar Group (Mansar Formation host)",
                          "generation_basis": "NMET-reported structure/depths + target prospectivity",
                          "provenance": "SIMULATED"})
            # ---- stratigraphic intervals down the hole
            cursor = 0.0
            seq = [("weathered zone / laterite cover", "OVERBURDEN", weather_zone)]
            n_hor = {"POSITIVE_DRILLING_INTERSECTION": 1 + int(rng.random() < 0.5 * rank / 100),
                     "AMBIGUOUS_DRILLING_RESULT": 1, "NEGATIVE_DRILLING_RESULT": 0}[scenario]
            hz = list(rng.choice(list(HORIZONS), size=n_hor, replace=False)) if n_hor else []
            host_len = total - weather_zone
            # place horizons at ordered positions down-hole
            positions = sorted(rng.uniform(0.15, 0.85, size=len(hz)))
            placed = []
            for hzn, p in zip(hz, positions):
                true_t = float(rng.uniform(*cfg["reef_true_thickness_m"]))
                if scenario == "AMBIGUOUS_DRILLING_RESULT":
                    true_t *= float(rng.uniform(0.2, 0.5))          # thin / pinched
                app = apparent_thickness(true_t, dip)
                placed.append((weather_zone + p * host_len, app, hzn, true_t))
            prev = weather_zone
            for start, app, hzn, true_t in placed:
                if start > prev:
                    seq.append(("quartz-mica schist / calc-silicate (host)", "HOST", start - prev))
                seq.append((f"Mn ore / gondite horizon {hzn} ({HORIZONS[hzn]})", f"ORE_{hzn}", app))
                prev = start + app
            if total > prev:
                seq.append(("quartz-mica schist / calc-silicate (host)", "HOST", total - prev))
            for lith, kind, length in seq:
                if length <= 0.01:
                    continue
                d0, d1 = cursor, min(total, cursor + length)
                cursor = d1
                ore = kind.startswith("ORE")
                if ore:
                    if scenario == "AMBIGUOUS_DRILLING_RESULT":
                        mn = float(np.clip(rng.normal(g["ambiguous_mean"], g["ambiguous_sd"]), 8, 30))
                    else:
                        mn = float(np.clip(rng.normal(g["ore_mean"], g["ore_sd"]), g["ore_min"], g["ore_max"]))
                    if d0 < cfg["supergene_depth_m"]:
                        mn = min(g["ore_max"], mn + g["supergene_uplift"])   # near-surface oxide enrichment
                else:
                    mn = float(np.clip(rng.lognormal(math.log(g["background_median"]), 0.5), 0.05, 6))
                mn_obs = max(0.0, mn * (1 + rng.normal(0, cfg["assay_rel_sd"])))
                sio2 = float(np.clip((g["sio2_ore_base"] - 0.8 * mn) if ore else rng.normal(58, 6), 2, 80))
                fe = float(np.clip(rng.normal(6 if ore else 5, 1.5), 0.5, 20))
                intervals.append({"borehole_id": holes[-1]["borehole_id"], "target_id": t["target_id"], "scenario": scenario,
                                  "depth_from_m": round(d0, 2), "depth_to_m": round(d1, 2),
                                  "interval_thickness_m": round(d1 - d0, 2), "formation": "Sausar Group",
                                  "host_lithology": lith, "weathering_zone": d0 < cfg["supergene_depth_m"],
                                  "manganese_presence": ore and mn_obs >= g["mineralised_cutoff"],
                                  "mn_grade_pct": round(mn_obs, 2), "grade_category": ibm_grade_class(mn_obs),
                                  "confidence": "LOW" if scenario == "AMBIGUOUS_DRILLING_RESULT" else "MODERATE",
                                  "evidence_level": 3, "provenance": "SIMULATED"})
                geochem.append({"sample_id": f"{holes[-1]['borehole_id']}_{len(geochem) + 1:03d}", "target_id": t["target_id"],
                                "scenario": scenario, "borehole_id": holes[-1]["borehole_id"], "depth_from_m": round(d0, 2),
                                "depth_to_m": round(d1, 2), "sample_type": "drill core (simulated)",
                                "mn_pct": round(mn_obs, 2), "fe_pct": round(fe, 2), "sio2_pct": round(sio2, 2),
                                "sample_quality": "SIMULATED_ASSAY", "provenance": "SIMULATED"})
    if scenario == "POSITIVE_GEOCHEMICAL_SUPPORT":
        for k in range(cfg["surface_samples"]):
            mn = float(np.clip(rng.lognormal(math.log(g["surface_anomaly_median"]), 0.35), 3, 40))
            geochem.append({"sample_id": f"SIM_{t['target_id']}_SS{k + 1:02d}", "target_id": t["target_id"], "scenario": scenario,
                            "borehole_id": None, "depth_from_m": 0.0, "depth_to_m": 0.2, "sample_type": "soil / rock chip (simulated)",
                            "mn_pct": round(mn, 2), "fe_pct": round(float(rng.normal(5, 1.2)), 2),
                            "sio2_pct": round(float(np.clip(60 - 0.7 * mn + rng.normal(0, 4), 5, 80)), 2),
                            "sample_quality": "SIMULATED_SCREENING", "provenance": "SIMULATED"})
    if scenario in ("POSITIVE_GEOPHYSICAL_SUPPORT", "POSITIVE_DRILLING_INTERSECTION", "NEGATIVE_DRILLING_RESULT",
                    "AMBIGUOUS_DRILLING_RESULT"):
        strong = {"POSITIVE_GEOPHYSICAL_SUPPORT": 1.0, "POSITIVE_DRILLING_INTERSECTION": 1.0,
                  "AMBIGUOUS_DRILLING_RESULT": 0.5, "NEGATIVE_DRILLING_RESULT": 0.1}[scenario]
        bg = cfg["ip_background_ms"]
        resp = bg * (1 + strong * rng.uniform(*cfg["ip_anomaly_factor"])) + rng.normal(0, cfg["ip_noise_ms"])
        geophys.append({"target_id": t["target_id"], "scenario": scenario, "method": "IP chargeability (simulated)",
                        "response": round(float(resp), 2), "background": bg,
                        "anomaly_strength": round(float((resp - bg) / bg), 3),
                        "depth_or_scale": "30-80 m (dipole spacing)", "noise": cfg["ip_noise_ms"],
                        "confidence": "MODERATE" if strong >= 1 else "LOW", "provenance": "SIMULATED"})
    return holes, intervals, geochem, geophys


def check_subsurface(holes: pd.DataFrame, intervals: pd.DataFrame):
    """Geometric / geological sanity checks (raises on violation)."""
    assert (intervals["depth_to_m"] > intervals["depth_from_m"]).all()
    assert np.allclose(intervals["interval_thickness_m"], intervals["depth_to_m"] - intervals["depth_from_m"], atol=0.02)
    assert (intervals["mn_grade_pct"] >= 0).all() and (intervals["mn_grade_pct"] <= 60).all()
    tot = intervals.groupby("borehole_id")["depth_to_m"].max()
    depth = holes.set_index("borehole_id")["total_depth_m"]
    assert (tot <= depth.reindex(tot.index) + 0.05).all()
    assert holes["latitude"].between(-90, 90).all() and holes["longitude"].between(-180, 180).all()
    for bid, g in intervals.sort_values("depth_from_m").groupby("borehole_id"):
        assert (g["depth_from_m"].iloc[1:].to_numpy() >= g["depth_to_m"].iloc[:-1].to_numpy() - 0.02).all(), bid
