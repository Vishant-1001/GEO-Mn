"""Subsurface evidence: OBSERVED (real, reported) vs SIMULATED scenario evidence fusion.

Observed evidence = official NMET / DGM / MECL records attached to targets by ml/target_engine.py
(REAL_GOVERNMENT; block-level, no published collars or assays).
Simulated evidence = data/synthetic/subsurface/* produced by ml/synthetic_subsurface.py, conditioned
on real geological constraints. The fusion engine shows how each SIMULATED scenario WOULD change
evidence maturity, uncertainty and exploration priority. It never modifies the observed target
record and never produces a reserve, resource or tonnage.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from services.common import DATA_DIR, SIMULATED, ApiError, load_config, provenance
from services.exploration_service import get_service as exploration

SYN = DATA_DIR / "synthetic" / "subsurface"
UNC = ["LOW", "MODERATE", "HIGH"]


@lru_cache(maxsize=1)
def synthetic():
    try:
        return {n: pd.read_csv(SYN / f"synthetic_{n}.csv") for n in
                ("boreholes", "borehole_intervals", "geochemistry", "geophysics", "subsurface_targets")}
    except FileNotFoundError:
        return None


def _shift(unc, k):
    return UNC[max(0, min(2, UNC.index(unc) + k))]


def scenarios(target_id: str, strategic=None, scenario: str | None = None) -> dict:
    ex = exploration()
    t = next((x for x in ex.targets if x["target_id"].upper() == str(target_id).upper()), None)
    if t is None:
        raise ApiError(404, "TARGET_NOT_FOUND", f"Unknown target_id '{target_id}'")
    syn = synthetic()
    if syn is None:
        raise ApiError(503, "DATA_UNAVAILABLE", "Synthetic subsurface scenarios have not been generated.")
    rules = load_config("exploration_config.json")["subsurface_fusion"]
    base = ex.score_target(t, strategic)
    names = [scenario.upper()] if scenario else [k for k in rules if not k.startswith("_")]
    for n in names:
        if n not in rules or n.startswith("_"):
            raise ApiError(400, "INVALID_INPUT", f"Unknown scenario '{scenario}'")
    out = []
    for n in names:
        r = rules[n]
        sim_t = dict(t, evidence_level=max(t["evidence_level"], r["min_level"]),
                     uncertainty=_shift(t["uncertainty"], r["uncertainty_shift"]))
        scored = ex.score_target(sim_t, strategic, r["prospectivity_factor"])
        iv = syn["borehole_intervals"]
        iv = iv[(iv["target_id"] == t["target_id"]) & (iv["scenario"] == n)]
        ore = iv[iv["manganese_presence"]]
        gc = syn["geochemistry"]
        gc = gc[(gc["target_id"] == t["target_id"]) & (gc["scenario"] == n)]
        gp = syn["geophysics"]
        gp = gp[(gp["target_id"] == t["target_id"]) & (gp["scenario"] == n)]
        bh = syn["boreholes"]
        bh = bh[(bh["target_id"] == t["target_id"]) & (bh["scenario"] == n)]
        out.append({
            "scenario": n,
            "label": "SIMULATED — NOT OBSERVED",
            "simulated_evidence_level": sim_t["evidence_level"],
            "simulated_uncertainty": sim_t["uncertainty"],
            "exploration_priority_before": base["exploration_priority"],
            "exploration_priority_after": scored["exploration_priority"],
            "priority_change": round(scored["exploration_priority"] - base["exploration_priority"], 1),
            "recommended_next_investigation": r["next"],
            "summary": {
                "boreholes": int(len(bh)), "intervals": int(len(iv)), "mn_bearing_intervals": int(len(ore)),
                "max_mn_pct": round(float(ore["mn_grade_pct"].max()), 2) if len(ore) else None,
                "grade_classes": ore["grade_category"].value_counts().to_dict(),
                "geochem_samples": int(len(gc)),
                "surface_max_mn_pct": round(float(gc[gc["borehole_id"].isna()]["mn_pct"].max()), 2)
                if len(gc) and gc["borehole_id"].isna().any() else None,
                "geophysics": gp[["method", "response", "background", "anomaly_strength", "confidence"]].to_dict(orient="records"),
            },
            "boreholes": bh.to_dict(orient="records"),
            "intervals": iv.to_dict(orient="records"),
            "not_claimed": "No reserve, resource or tonnage is derived from simulated drilling.",
        })
    return {
        "target_id": t["target_id"],
        "observed_evidence": {
            "evidence_level": t["evidence_level"], "evidence_level_label": t["evidence_level_label"],
            "subsurface_status": t["subsurface_status"], "records": t.get("observed_ground_evidence", []),
            "label": "OBSERVED / REPORTED (REAL_GOVERNMENT)" if t.get("observed_ground_evidence") else "NO OBSERVED SUBSURFACE EVIDENCE",
        },
        "simulated_scenarios": out,
        "fusion_rules": {k: v for k, v in rules.items()},
        "provenance": provenance(SIMULATED, ex.model_version, None, None, False, observed_mode="REAL_GOVERNMENT",
                                 scenario_mode="SIMULATED"),
    }
