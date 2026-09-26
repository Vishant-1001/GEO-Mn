"""Subsurface / ground evidence for a target: OBSERVED record + next-evidence sensitivity.

Observed: official NMET / DGM / MECL block records attached to targets by ml/target_engine.py
(REAL_GOVERNMENT; block level, no public collars, logs or assays). Everything else is UNAVAILABLE.

Next-evidence sensitivity: for each possible OUTCOME of the recommended next investigation
(geochemistry, geophysics, drilling), the project-configured rules in
config/exploration_config.json -> subsurface_fusion show how the target's evidence maturity,
uncertainty and investigation priority WOULD change. No boreholes, assays or geophysical values are
generated: these are hypothetical outcomes, not observations, and never a reserve or tonnage.
"""
from __future__ import annotations

import json

from services.common import DATA_DIR, SIMULATED, ApiError, load_config, provenance
from services.exploration_service import get_service as exploration

UNC = ["LOW", "MODERATE", "HIGH"]
OUTCOME_TEXT = {
    "NO_SUBSURFACE_EVIDENCE": "No further investigation (current state)",
    "POSITIVE_GEOCHEMICAL_SUPPORT": "If soil / stream-sediment geochemistry returned a Mn anomaly",
    "POSITIVE_GEOPHYSICAL_SUPPORT": "If an IP / magnetic traverse returned a coherent anomaly",
    "POSITIVE_DRILLING_INTERSECTION": "If scout drilling intersected Mn mineralisation",
    "AMBIGUOUS_DRILLING_RESULT": "If scout drilling returned thin / low-grade intersections",
    "NEGATIVE_DRILLING_RESULT": "If scout drilling found no Mn horizon",
}


def _shift(unc, k):
    return UNC[max(0, min(2, UNC.index(unc) + k))]


def _constraints(block_id):
    """REAL, cited expectations from the official records (host, dips, widths, grades, proposed depths)."""
    p = DATA_DIR / "processed" / "subsurface" / "geological_constraints.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    keep = {k: v for k, v in c.items() if not k.startswith("_") and k not in ("ibm_grade_classes",)}
    return {"block": block_id, "official_expectations": keep,
            "note": "Reported by official block documents; expectations, not measurements at this target."}


def scenarios(target_id: str, strategic=None, scenario: str | None = None) -> dict:
    ex = exploration()
    t = next((x for x in ex.targets if x["target_id"].upper() == str(target_id).upper()), None)
    if t is None:
        raise ApiError(404, "TARGET_NOT_FOUND", f"Unknown target_id '{target_id}'")
    rules = {k: v for k, v in load_config("exploration_config.json")["subsurface_fusion"].items() if not k.startswith("_")}
    names = [scenario.upper()] if scenario else list(rules)
    for n in names:
        if n not in rules:
            raise ApiError(400, "INVALID_INPUT", f"Unknown scenario '{scenario}'")
    base = ex.score_target(t, strategic)
    out = []
    for n in names:
        r = rules[n]
        hyp = dict(t, evidence_level=max(t["evidence_level"], r["min_level"]),
                   uncertainty=_shift(t["uncertainty"], r["uncertainty_shift"]))
        scored = ex.score_target(hyp, strategic, r["prospectivity_factor"])
        out.append({
            "scenario": n,
            "outcome": OUTCOME_TEXT.get(n, n),
            "label": "HYPOTHETICAL OUTCOME — NOT OBSERVED",
            "hypothetical_evidence_level": hyp["evidence_level"],
            "hypothetical_uncertainty": hyp["uncertainty"],
            "exploration_priority_before": base["exploration_priority"],
            "exploration_priority_after": scored["exploration_priority"],
            "priority_change": round(scored["exploration_priority"] - base["exploration_priority"], 1),
            "recommended_next_investigation": r["next"],
            "not_claimed": "No borehole, assay, geophysical value, reserve or tonnage is generated or implied.",
        })
    obs = t.get("observed_ground_evidence") or []
    reported = t.get("subsurface_status") == "REPORTED_BLOCK_LEVEL"
    return {
        "target_id": t["target_id"],
        "observed_evidence": {
            "evidence_level": t["evidence_level"], "evidence_level_label": t["evidence_level_label"],
            "subsurface_status": t["subsurface_status"], "records": obs,
            "label": "OBSERVED / REPORTED (REAL_GOVERNMENT)" if obs else "UNAVAILABLE",
            "statement": (None if reported else
                          "SUBSURFACE EVIDENCE UNAVAILABLE. This target is supported by available surface/geological "
                          "evidence and requires additional ground/subsurface validation."),
            "official_expectations": _constraints(t.get("nmet_block_id")) if obs else None,
        },
        "next_required_evidence": ex.next_evidence(t),
        "next_evidence_sensitivity": out,
        "reserve_confirmed": False,
        "fusion_rules": rules,
        "provenance": provenance(SIMULATED, ex.model_version, None, None, False, observed_mode="REAL_GOVERNMENT",
                                 sensitivity_mode="SIMULATED (rule-based, no generated records)"),
    }
