"""Exploration service: point prospectivity, cached grid, targets and target priority.

* prospectivity_rank (0-100) is a RELATIVE rank within the study area, never a
  probability of a deposit / reserve / tonnage.
* Live coordinate queries recompute the documented feature recipe (ml/eo_features.py)
  from the fixed 2024 observation window — this is NOT real-time imagery.
* If the live query fails / is disabled, the nearest cached grid cell is used only
  if it lies within `cache_max_distance_km`; otherwise PREDICTION_UNAVAILABLE.
* Exploration priority = transparent weighted score of normalised components
  (project defaults in config/exploration_config.json; not industry standards).
  Components that are unavailable are dropped and the weights re-normalised.
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from ml.common import haversine_km
from ml.ood import LEVEL_SCORE, in_envelope
from ml.uncertainty import percentile_rank
from services.common import (
    CACHED, DATA_DIR, LIVE_COORDINATE_QUERY, MODELS_DIR, REAL_PUBLIC, ApiError, env_flag, file_timestamp, load_config,
    load_manifest, log, provenance,
)

UNCERTAINTY_SCORE = {"LOW": 1.0, "MODERATE": 0.6, "HIGH": 0.2}
NEXT_EVIDENCE = [
    "Field geological mapping and outcrop verification of the target footprint",
    "Stream-sediment / soil geochemistry (Mn, Fe, Ba) over the footprint",
    "Ground magnetic / gravity or IP traverse across the strongest cells",
    "Check existing state / GSI records for prior drilling before planning new holes",
]


def _obs_window():
    return load_manifest().get("exploration", {}).get("observation_window", "2024-01-01/2024-12-31")


class ExplorationService:
    def __init__(self):
        self.cfg = load_config("exploration_config.json")
        self.error = None
        self._live_cache = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2)
        try:
            self.grid = pd.read_csv(DATA_DIR / "exploration_grid.csv")
            self.grid_ok = self.grid[self.grid["data_status"] == "OK"].reset_index(drop=True)
        except Exception as e:
            self.grid = self.grid_ok = None
            self.error = f"grid unavailable: {type(e).__name__}"
        try:
            doc = json.loads((DATA_DIR / "exploration_targets.json").read_text())
            self.targets = doc["targets"]
            self.targets_meta = doc["meta"]
        except Exception:
            self.targets, self.targets_meta = [], {}
        try:
            self.members = joblib.load(MODELS_DIR / "exploration_model.pkl")
            self.features = joblib.load(MODELS_DIR / "exploration_feature_columns.pkl")
            self.reference = joblib.load(MODELS_DIR / "exploration_train_reference.pkl")
        except Exception as e:
            self.members = None
            self.error = (self.error or "") + f" model unavailable: {type(e).__name__}"
        try:
            self.geology = pd.read_csv(DATA_DIR / "geology_lattice.csv")
        except Exception:
            self.geology = None
        try:
            m = pd.read_csv(DATA_DIR / "mrds_mn_occurrences.csv")
            self.mrds = m[m["label_use"] != "EXCLUDED_REGIONAL_RECORD"].reset_index(drop=True)
        except Exception:
            self.mrds = None

    # ------------------------------------------------------------------ status
    @property
    def model_loaded(self):
        return self.members is not None

    @property
    def cache_loaded(self):
        return self.grid_ok is not None and len(self.grid_ok) > 0

    @property
    def model_version(self):
        return load_manifest().get("exploration", {}).get("model_version")

    def live_enabled(self) -> bool:
        if not env_flag("GEOMN_LIVE_EO", True) or not self.model_loaded:
            return False
        try:
            import planetary_computer  # noqa: F401
            import pystac_client  # noqa: F401
            import rasterio  # noqa: F401
        except Exception:
            return False
        return True

    def _prov(self, mode, fallback=False, **extra):
        return provenance(mode, self.model_version, _obs_window(),
                          file_timestamp(DATA_DIR / "exploration_grid.csv") if mode == CACHED else None,
                          fallback, labels_mode=REAL_PUBLIC, features_mode=REAL_PUBLIC, **extra)

    # ------------------------------------------------------------- scoring
    def score_features(self, feats: dict) -> dict:
        """Rank / uncertainty / applicability for one feature vector (identical to training)."""
        X = pd.DataFrame([feats])[self.features]
        if X.isna().any(axis=None):
            raise ValueError("incomplete features")
        S = np.array([m.predict_proba(X)[:, 1][0] for m in self.members])
        ref = self.reference
        rank = float(percentile_rank(ref["ensemble_ref"], [S.mean()])[0])
        mranks = [float(percentile_rank(ref["member_refs"][k], [S[k]])[0]) for k in range(len(S))]
        sd = float(np.std(mranks))
        iso = float(ref["isolation_forest"].score_samples(X)[0])
        appl = "HIGH" if iso >= ref["iso_q_moderate"] else ("MODERATE" if iso >= ref["iso_q_low"] else "LOW")
        t = self.cfg["uncertainty_thresholds_rank_sd"]
        unc = "LOW" if sd < t["low_below"] else ("HIGH" if sd > t["high_above"] else "MODERATE")
        return {"prospectivity_rank": round(rank, 1), "rank_sd": round(sd, 1), "uncertainty": unc,
                "feature_applicability": appl, "isolation_score": round(iso, 4)}

    def _live_features(self, lat, lon):
        from ml.eo_features import cell_centre, extract_point_features

        key = cell_centre(lat, lon)
        with self._lock:
            if key in self._live_cache:
                return self._live_cache[key]
        fut = self._pool.submit(extract_point_features, lat, lon)
        feats = fut.result(timeout=self.cfg["live_query_timeout_s"])
        with self._lock:
            self._live_cache[key] = feats
        return feats

    def nearest_cell(self, lat, lon):
        g = self.grid_ok
        d = haversine_km(g["lat"].to_numpy(), g["lon"].to_numpy(), lat, lon)
        i = int(np.argmin(d))
        return g.iloc[i], float(d[i])

    def evidence_at(self, lat, lon):
        level, basis = 0, ["Remote-sensing prospectivity only."]
        geo = None
        if self.geology is not None and in_envelope(lat, lon, self.cfg["study_area"]):
            d = haversine_km(self.geology["lat"].to_numpy(), self.geology["lon"].to_numpy(), lat, lon)
            g = self.geology.iloc[int(np.argmin(d))]
            geo = {"unit_name": None if pd.isna(g["unit_name"]) else g["unit_name"],
                   "precambrian_host_domain": bool(g["precambrian"]) if pd.notna(g["unit_name"]) else None}
            if geo["precambrian_host_domain"]:
                level, basis = 1, basis + [f"Precambrian domain on world-scale geology map ('{g['unit_name']}')."]
        near = []
        if self.mrds is not None:
            d = haversine_km(self.mrds["lat"].to_numpy(), self.mrds["lon"].to_numpy(), lat, lon)
            for i in np.where(d <= 1.0)[0]:
                r = self.mrds.iloc[i]
                near.append({"site_name": r["site_name"], "dev_stat": r["dev_stat"], "distance_km": round(float(d[i]), 2)})
            if near:
                level, basis = 2, basis + [f"{len(near)} documented MRDS Mn record(s) within 1 km (reported context)."]
        return level, basis, geo, near

    def predict(self, lat, lon, mode="AUTO") -> dict:
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            raise ApiError(400, "INVALID_INPUT", "lat and lon must be numbers")
        if not (math.isfinite(lat) and math.isfinite(lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ApiError(400, "INVALID_INPUT", "lat must be in [-90, 90] and lon in [-180, 180]")
        mode = (mode or "AUTO").upper()
        if mode not in ("AUTO", "LIVE_ONLY", "CACHED_ONLY", "SIMULATE_LIVE_FAILURE"):
            raise ApiError(400, "INVALID_INPUT", "mode must be AUTO, LIVE_ONLY, CACHED_ONLY or SIMULATE_LIVE_FAILURE")
        area = self.cfg["study_area"]
        inside = in_envelope(lat, lon, area)
        coverage_warning = None if inside else (
            f"Outside the study area ({area['lat_min']}-{area['lat_max']}N, {area['lon_min']}-{area['lon_max']}E). "
            "Rank is an extrapolation relative to that area and must not be relied on.")

        result, source, cache_km, live_error = None, None, None, None
        if mode == "SIMULATE_LIVE_FAILURE":
            live_error = "LIVE_QUERY_FAILED (simulated for demonstration)"
        elif mode != "CACHED_ONLY" and self.live_enabled():
            try:
                feats = self._live_features(lat, lon)
                result = self.score_features(feats)
                result["features"] = {k: round(float(feats[k]), 4) for k in self.features}
                result["quality"] = {k: (None if pd.isna(feats.get(k)) else round(float(feats[k]), 2))
                                     for k in ("s2_valid_scenes", "s2_pixels", "lst_valid_composites", "dem_pixels")}
                source = LIVE_COORDINATE_QUERY
            except (FutureTimeout, TimeoutError):
                live_error = "LIVE_QUERY_TIMEOUT"
            except Exception as e:
                live_error = f"LIVE_QUERY_FAILED: {type(e).__name__}"
                log.info("live exploration query failed at (%s, %s): %s", lat, lon, e)
        elif mode != "CACHED_ONLY":
            live_error = "LIVE_QUERY_DISABLED"

        if result is None:
            if mode == "LIVE_ONLY":
                raise ApiError(503, "PREDICTION_UNAVAILABLE", f"Live query unavailable ({live_error}).")
            if not self.cache_loaded:
                raise ApiError(503, "PREDICTION_UNAVAILABLE", "Live query unavailable and the cached grid is not loaded.")
            cell, cache_km = self.nearest_cell(lat, lon)
            max_km = self.cfg["cache_max_distance_km"]
            if cache_km > max_km:
                raise ApiError(503, "PREDICTION_UNAVAILABLE",
                               f"Live query unavailable ({live_error or 'not requested'}) and the nearest cached cell is "
                               f"{cache_km:.1f} km away (maximum supported {max_km} km).",
                               {"cache_distance_km": round(cache_km, 2), "max_cache_distance_km": max_km,
                                "live_status": live_error})
            result = {"prospectivity_rank": float(cell["prospectivity_rank"]), "rank_sd": float(cell["rank_sd"]),
                      "uncertainty": cell["uncertainty"], "feature_applicability": cell["applicability"]}
            source = CACHED
            result["cached_cell"] = {"lat": float(cell["lat"]), "lon": float(cell["lon"])}

        applicability = result["feature_applicability"] if inside else "LOW"
        level, basis, geo, near = self.evidence_at(lat, lon)
        rank = result["prospectivity_rank"]
        min_rank = self.cfg["targets"]["min_rank"]
        if rank >= min_rank and applicability == "LOW":
            status = "REVIEW_REQUIRED"
        elif rank >= min_rank:
            status = "EXPLORATION_TARGET"
        else:
            status = "NOT_PRIORITISED"
        return {
            "query_lat": lat,
            "query_lon": lon,
            "prospectivity_rank": rank,
            "prospectivity_semantics": "Relative rank (0-100) within the study area — not a probability of a deposit, reserve or tonnage.",
            "rank_sd": result["rank_sd"],
            "uncertainty": result["uncertainty"],
            "applicability": applicability,
            "feature_applicability": result["feature_applicability"],
            "in_study_area": inside,
            "evidence_level": level,
            "evidence_basis": basis,
            "geological_context": geo,
            "documented_occurrences_within_1km": near,
            "subsurface_status": "UNAVAILABLE",
            "source_mode": source,
            "observation_window": _obs_window(),
            "fallback_used": source == CACHED,
            "cache_distance_km": round(cache_km, 3) if cache_km is not None else None,
            "live_status": "OK" if source == LIVE_COORDINATE_QUERY else live_error,
            "coverage_warning": coverage_warning,
            "status": status,
            "features": result.get("features"),
            "quality": result.get("quality"),
            "cached_cell": result.get("cached_cell"),
            "provenance": self._prov(source, source == CACHED, effective_resolution="~1 km (0.01 degree cell)"),
        }

    # ------------------------------------------------------------- targets
    def strategic_relevance(self, t, strategic):
        """Relevance of a target to the CURRENT strategic supply-gap state (not tonnage)."""
        sr = self.cfg["strategic_relevance"]
        if not strategic or not strategic.get("active"):
            return 0.0, "LOW", "No active strategic supply gap for this mine; relevance is not elevated."
        severity = strategic["severity"]
        proximity = math.exp(-t["distance_to_demo_mine_km"] / sr["distance_decay_km"])
        score = severity * proximity
        level = "HIGH" if score >= sr["high_at_or_above"] else ("MODERATE" if score >= sr["moderate_at_or_above"] else "LOW")
        return score, level, (f"Strategic gap severity {severity:.2f} x proximity {proximity:.2f} "
                              f"({t['distance_to_demo_mine_km']} km from the supply point).")

    def prioritise(self, strategic=None) -> list[dict]:
        w = self.cfg["priority_weights"]
        out = []
        for t in self.targets:
            ev = (t["evidence_level"] / 4.0 + LEVEL_SCORE[t["applicability"]] + UNCERTAINTY_SCORE[t["uncertainty"]]) / 3.0
            srel, slevel, snote = self.strategic_relevance(t, strategic)
            comps = {
                "prospectivity": t["prospectivity_rank"] / 100.0,
                "evidence_applicability": ev,
                "strategic_relevance": srel,
                "development_readiness": t.get("development_readiness"),
            }
            avail = {k: v for k, v in comps.items() if v is not None}
            wsum = sum(w[k] for k in avail)
            priority = 100.0 * sum(w[k] * avail[k] for k in avail) / wsum
            out.append({**t,
                        "strategic_relevance": slevel,
                        "strategic_relevance_score": round(srel, 3),
                        "strategic_relevance_note": snote,
                        "exploration_priority": round(priority, 1),
                        "priority_components": {k: (round(v, 3) if v is not None else None) for k, v in comps.items()},
                        "priority_weights_used": {k: round(w[k] / wsum, 3) for k in avail},
                        "priority_renormalised": len(avail) < len(comps),
                        "status": "REVIEW_REQUIRED" if t["applicability"] == "LOW" else "EXPLORATION_TARGET",
                        "source_mode": CACHED})
        out.sort(key=lambda x: (-x["exploration_priority"], x["target_id"]))
        for i, x in enumerate(out, start=1):
            x["priority_rank"] = i
        return out

    def why_this_target(self, t) -> list[dict]:
        r = [{"code": "HIGH_PROSPECTIVITY" if t["prospectivity_rank"] >= 97 else "ELEVATED_PROSPECTIVITY",
              "text": f"Mean relative prospectivity rank {t['prospectivity_rank']} (top {100 - t['prospectivity_rank']:.1f}% of the study area)."}]
        if t["applicability"] in ("HIGH", "MODERATE"):
            r.append({"code": "ACCEPTABLE_APPLICABILITY", "text": f"Feature-space applicability {t['applicability']}."})
        else:
            r.append({"code": "LOW_APPLICABILITY", "text": "Features are atypical for the training data; rank may be unreliable."})
        r.append({"code": f"{t['uncertainty']}_UNCERTAINTY",
                  "text": f"Ensemble rank spread {t['rank_sd']} points ({t['uncertainty']})."})
        if t["evidence_level"] >= 1:
            r.append({"code": "SUPPORTING_EVIDENCE", "text": t["evidence_level_label"]})
        if t.get("target_context") == "BROWNFIELD":
            r.append({"code": "BROWNFIELD_CONTEXT", "text": "Documented Mn records nearby (extension of known mineralisation)."})
        if t.get("contains_training_labels"):
            r.append({"code": "CAUTION_TRAINING_LABEL_OVERLAP",
                      "text": "Footprint contains MRDS records used as training labels; its rank is partly in-sample."})
        r.append({"code": "NO_SUBSURFACE_DATA", "text": "No drilling, assay or geophysical data available (subsurface UNAVAILABLE)."})
        return r

    def why_now(self, t, strategic) -> list[dict]:
        if not strategic or not strategic.get("active"):
            return [{"code": "NO_ACTIVE_STRATEGIC_GAP",
                     "text": "Operational recovery covers the current requirement; no exploration contingency is active."}]
        r = [{"code": "RESIDUAL_STRATEGIC_GAP",
              "text": (f"After the best robust operational portfolio, a residual of {strategic['expected_residual_gap_tonnes']:.0f} t "
                       f"(worst case {strategic['worst_case_residual_gap_tonnes']:.0f} t) per period remains over the strategic horizon.")}]
        if t["strategic_relevance"] in ("HIGH", "MODERATE"):
            r.append({"code": "PROXIMITY_TO_SUPPLY_NEED",
                      "text": f"{t['distance_to_demo_mine_km']} km from the supply point with the gap ({t['strategic_relevance']} strategic relevance)."})
        if (t.get("development_readiness") or 0) >= 0.5:
            r.append({"code": "DEVELOPMENT_READINESS_PROXY",
                      "text": f"{t['nearest_documented_producer_km']} km from a documented producing / past-producing site (access proxy)."})
        r.append({"code": "HIGHEST_FEASIBLE_PRIORITY",
                  "text": f"Highest exploration priority ({t['exploration_priority']}) among targets with acceptable applicability."})
        r.append({"code": "LEAD_TIME_CAVEAT",
                  "text": "Exploration is a strategic contingency: it cannot supply ore in the next period."})
        return r

    def list_targets(self, strategic=None) -> dict:
        if not self.targets:
            raise ApiError(503, "DATA_UNAVAILABLE", "Exploration targets are not available.")
        ranked = self.prioritise(strategic)
        keep = ("target_id", "lat", "lon", "geometry", "bbox", "n_cells", "area_km2", "prospectivity_rank", "peak_prospectivity_rank",
                "rank_sd", "uncertainty", "applicability", "evidence_level", "evidence_level_label", "target_context",
                "contains_training_labels", "subsurface_status", "strategic_relevance", "strategic_relevance_score",
                "development_readiness", "distance_to_demo_mine_km", "exploration_priority", "priority_rank", "status", "source_mode")
        return {
            "targets": [{k: t.get(k) for k in keep} for t in ranked],
            "count": len(ranked),
            "strategic_context": strategic,
            "priority_formula": {
                "weights": self.cfg["priority_weights"],
                "components": {
                    "prospectivity": "prospectivity_rank / 100",
                    "evidence_applicability": "mean(evidence_level/4, applicability score, 1 - uncertainty penalty)",
                    "strategic_relevance": "gap severity x exp(-distance / decay) under the current strategic state",
                    "development_readiness": "exp(-distance to nearest documented producer / decay) — access proxy",
                },
                "note": "PROJECT DEFAULT weights, not mining-industry standards. Unavailable components are dropped and weights re-normalised.",
            },
            "method": self.targets_meta.get("method"),
            "provenance": self._prov(CACHED, False, effective_resolution="~1 km (0.01 degree cell)"),
        }

    def target_detail(self, target_id: str, strategic=None) -> dict:
        ranked = self.prioritise(strategic)
        t = next((x for x in ranked if x["target_id"].upper() == str(target_id).upper()), None)
        if t is None:
            raise ApiError(404, "TARGET_NOT_FOUND", f"Unknown target_id '{target_id}'")
        return {
            **{k: t[k] for k in ("target_id", "lat", "lon", "geometry", "bbox", "n_cells", "area_km2", "prospectivity_rank",
                                 "peak_prospectivity_rank", "rank_sd", "uncertainty", "applicability", "applicability_cell_shares",
                                 "evidence_level", "evidence_level_label", "evidence_basis", "target_context",
                                 "contains_training_labels", "strategic_relevance", "strategic_relevance_score",
                                 "strategic_relevance_note", "development_readiness", "nearest_documented_producer_km",
                                 "distance_to_demo_mine_km", "exploration_priority", "priority_rank", "priority_components",
                                 "priority_weights_used", "status", "source_mode")},
            "surface_evidence": {"status": "AVAILABLE", "observation_window": _obs_window(), "features": t["surface_evidence"]},
            "geological_evidence": t["geological_evidence"],
            "subsurface_evidence": {"status": t["subsurface_status"], "records": t["subsurface_evidence"],
                                    "note": self.targets_meta.get("subsurface_note")},
            "documented_occurrences": t["documented_occurrences"],
            "why_this_target": self.why_this_target(t),
            "why_this_target_now": self.why_now(t, strategic),
            "next_evidence": NEXT_EVIDENCE,
            "provenance": self._prov(CACHED, False, effective_resolution="~1 km (0.01 degree cell)"),
        }

    def grid_points(self, stride=5) -> list[dict]:
        if not self.cache_loaded:
            raise ApiError(503, "DATA_UNAVAILABLE", "Cached grid not loaded.")
        stride = max(1, min(int(stride), 50))
        res = self.cfg["grid_res_deg"]
        g = self.grid_ok
        i = np.round(g["lat"] / res - 0.5).astype(int)
        j = np.round(g["lon"] / res - 0.5).astype(int)
        sub = g[(i % stride == 0) & (j % stride == 0)]
        return [{"lat": float(r.lat), "lon": float(r.lon), "prospectivity_rank": float(r.prospectivity_rank),
                 "rank": float(r.prospectivity_rank), "uncertainty": r.uncertainty, "applicability": r.applicability}
                for r in sub.itertuples()]


@lru_cache(maxsize=1)
def get_service() -> ExplorationService:
    return ExplorationService()
