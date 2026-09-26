"""Supply-contingency engine: forecast -> robust recovery -> residual gap -> horizon gate -> decision.

Decision states (exactly three):
  OPERATIONAL_RESPONSE
  OPERATIONAL_AND_EXPLORATION_CONTINGENCY
  REVIEW_REQUIRED

Logic
  1. baseline gap (nominal scenario, no action) and robust recovery (recovery_service)
  2. confidence checks -> REVIEW_REQUIRED if the model is outside its experience in the
     nominal scenario, or no action portfolio passes the modelled-feasibility and
     applicability gates (automated selection withheld)
  3. horizon gate
       NEAR_TERM : exploration can never be near-term recovery.
                   expected residual <= materiality -> OPERATIONAL_RESPONSE
                   otherwise                         -> REVIEW_REQUIRED (escalate)
       STRATEGIC : residual persists if expected residual > materiality or the
                   worst-case residual > strategic tolerance.
                   not persistent -> OPERATIONAL_RESPONSE
                   persistent     -> rank exploration targets under the current gap
                                     state; best applicable target ->
                                     OPERATIONAL_AND_EXPLORATION_CONTINGENCY,
                                     none applicable -> REVIEW_REQUIRED
Thresholds live in config/risk_policy.json (demonstration defaults, not standards).
The target is chosen by computation over the ranked list — never hard-coded.
"""

from __future__ import annotations

from services import demo_service, recovery_service
from services.common import SIMULATED, load_config, provenance
from services.exploration_service import ExplorationService
from services.exploration_service import get_service as exploration
from services.production_service import get_service as production

STATES = ("OPERATIONAL_RESPONSE", "OPERATIONAL_AND_EXPLORATION_CONTINGENCY", "REVIEW_REQUIRED")


def evaluate(mine_id="DEMO_MINE", horizon=None, target_tonnes=None, base_state=None, conditions=None,
             forecast_origin=None, disruption_scenarios=None, action_portfolios=None, actions=None, scenario=None,
             client_supplied=False) -> dict:
    sc = demo_service.resolve(mine_id)
    horizon = demo_service.normalise_horizon(horizon, sc.get("horizon", "STRATEGIC"))
    pol = load_config("risk_policy.json")["decision"]
    ecfg = load_config("exploration_config.json")
    prod = production()

    overrides = {**(base_state or {}), **(conditions or {})}
    fc = prod.forecast(mine_id, forecast_origin, None, target_tonnes, overrides, explain=True)
    rec = recovery_service.evaluate(mine_id, target_tonnes, base_state, conditions, forecast_origin,
                                    disruption_scenarios, action_portfolios, actions, scenario)
    target = rec["target_tonnes"]
    materiality = pol["materiality_gap_pct"] / 100 * target
    worst_tol = pol["strategic_worst_case_tolerance_pct"] / 100 * target
    baseline_gap = rec["baseline"]["gap_p50_tonnes"]
    selected = rec["best_operational_action"]
    withheld = rec["selection_status"] != "SELECTED"
    # With no eligible portfolio the residual is unknown for decision purposes; fall back to the
    # no-action figures only to describe the situation, never to recommend actions.
    exp_res = rec["expected_residual_gap_tonnes"] if not withheld else rec["no_action_expected_gap_tonnes"]
    worst_res = rec["worst_case_residual_gap_tonnes"] if not withheld else rec["no_action_worst_case_gap_tonnes"]

    reason_codes, review_reasons = [], []
    if rec["nominal_applicability"] == "LOW":
        review_reasons.append({"code": "PRODUCTION_INPUTS_OUT_OF_DISTRIBUTION",
                               "text": "The current operating state is outside the production model's training experience "
                                       f"(range violations: {[v['feature'] for v in fc['applicability']['range_violations']] or 'multivariate'})."})
    on_track = baseline_gap <= materiality
    if withheld and not (on_track and horizon == "NEAR_TERM"):
        blocked = sorted({s for p in rec["evaluated_portfolios"] for s in p["low_applicability_scenarios"]})
        review_reasons.append({"code": "NO_ELIGIBLE_OPERATIONAL_PORTFOLIO",
                               "text": ("No action portfolio passed the modelled-feasibility and applicability gates"
                                        + (f" (scenarios outside model experience: {', '.join(blocked)})" if blocked else "")
                                        + "; automated portfolio selection withheld.")})
    if selected is not None and selected["feasibility"] == recovery_service.UNKNOWN:
        review_reasons.append({"code": "FEASIBILITY_UNKNOWN", "text": "Feasibility of the selected actions could not be judged."})
    if not prod.quantiles_validated:
        reason_codes.append("INTERVALS_NOT_VALIDATED")
    if selected is not None and selected["selection_applicability"] == "MODERATE":
        reason_codes.append("REDUCED_CONFIDENCE_MODERATE_APPLICABILITY")

    if on_track:
        supply_status = "ON_TRACK"
    elif not withheld and exp_res <= materiality:
        supply_status = "OPERATIONALLY_RECOVERABLE"
    else:
        supply_status = "RESIDUAL_GAP"

    persists = exp_res > materiality or worst_res > worst_tol
    strategic = {
        "active": False,
        "horizon": horizon,
        "expected_residual_gap_tonnes": exp_res,
        "worst_case_residual_gap_tonnes": worst_res,
        "materiality_tonnes": round(materiality, 1),
        "worst_case_tolerance_tonnes": round(worst_tol, 1),
        "residual_persists": persists,
    }
    selected_target, ranked = None, []
    if review_reasons:
        state = "REVIEW_REQUIRED"
        summary = "Automated recommendation withheld: " + "; ".join(r["text"] for r in review_reasons)
    elif horizon == "NEAR_TERM":
        if exp_res <= materiality:
            state = "OPERATIONAL_RESPONSE"
            reason_codes.append("ON_TRACK" if supply_status == "ON_TRACK" else "OPERATIONAL_RECOVERY_SUFFICIENT")
            summary = ("Forecast meets the target within materiality; maintain the plan." if supply_status == "ON_TRACK" else
                       f"Portfolio {rec['selected_portfolio']} is estimated to close the near-term gap "
                       f"({baseline_gap:.0f} t -> {exp_res:.0f} t).")
        else:
            state = "REVIEW_REQUIRED"
            review_reasons.append({"code": "NEAR_TERM_RESIDUAL_GAP",
                                   "text": (f"Best eligible operational portfolio leaves {exp_res:.0f} t unrecovered in the next period. "
                                            "Exploration is not treated as an immediate production-recovery action. "
                                            "Operational escalation / supply-plan adjustment is required (stockpile draw, "
                                            "inter-mine transfer or plan revision are human decisions).")})
            summary = review_reasons[-1]["text"]
    else:  # STRATEGIC
        if not persists:
            state = "OPERATIONAL_RESPONSE"
            reason_codes.append("ON_TRACK" if supply_status == "ON_TRACK" else "OPERATIONAL_RECOVERY_SUFFICIENT")
            summary = (f"Operational response is sufficient over the strategic horizon: expected residual {exp_res:.0f} t, "
                       f"worst tested residual {worst_res:.0f} t (tolerance {worst_tol:.0f} t).")
        else:
            worst_pct = worst_res / target * 100 if target else 0.0
            sev = min(1.0, max(exp_res, worst_res) / target * 100 / ecfg["strategic_relevance"]["severity_full_scale_pct"]) if target else 0.0
            strategic.update({
                "active": True,
                "severity": round(sev, 3),
                "worst_case_residual_pct": round(worst_pct, 2),
                "strategic_periods": pol["strategic_periods"],
                "cumulative_requirement_expected_tonnes": round(exp_res * pol["strategic_periods"], 0),
                "cumulative_requirement_worst_case_tonnes": round(worst_res * pol["strategic_periods"], 0),
                "requirement_note": ("Scenario estimate assuming the residual persists for the strategic horizon; "
                                     "not a demand forecast and not an ore-tonnage estimate for any target."),
            })
            reason_codes.append("RESIDUAL_STRATEGIC_GAP")
            ex = exploration()
            ranked = ex.prioritise(strategic)
            eligible = [t for t in ranked if t["eligible_for_contingency"]]
            if eligible:
                selected_target = eligible[0]
                state = "OPERATIONAL_AND_EXPLORATION_CONTINGENCY"
                summary = ("Operational recovery does not fully resolve the projected strategic supply requirement; exploration "
                           "contingency is therefore relevant to future supply continuity. "
                           f"Operational recovery ({rec['selected_portfolio']}) does not keep the residual within tolerance over the strategic horizon "
                           f"(expected residual {exp_res:.0f} t, worst case {worst_res:.0f} t per period). "
                           f"Exploration contingency activated; next target {selected_target['target_id']} "
                           f"(priority {selected_target['exploration_priority']}).")
            else:
                state = "REVIEW_REQUIRED"
                review_reasons.append({"code": "NO_APPLICABLE_TARGET",
                                       "text": "Strategic gap persists but no exploration target passes the study-area / applicability / minimum-evidence gates."})
                summary = review_reasons[-1]["text"]

    why_now, why_codes = [], []
    if selected_target is not None:
        why_now = exploration().why_now(selected_target, strategic)
        why_codes = [r["code"] for r in why_now]
        why_codes += [r["code"] for r in exploration().why_this_target(selected_target)
                      if r["code"] in ("HIGH_PROSPECTIVITY", "ELEVATED_PROSPECTIVITY", "ACCEPTABLE_APPLICABILITY", "SUPPORTING_EVIDENCE")]
    reason_codes = list(dict.fromkeys(reason_codes + why_codes + [r["code"] for r in review_reasons]))

    return {
        "mine_id": sc["mine_id"],
        "demo_state": sc.get("demo_state"),
        "decision_state": state,
        "decision_horizon": horizon,
        "horizon": horizon,
        "supply_status": supply_status,
        "decision_summary": summary,
        "target_tonnes": target,
        "baseline_gap_tonnes": baseline_gap,
        "best_operational_recovery_tonnes": rec["expected_recovery_tonnes"],
        "expected_residual_gap_tonnes": rec["expected_residual_gap_tonnes"],
        "worst_case_residual_gap_tonnes": rec["worst_case_residual_gap_tonnes"],
        "selection_status": rec["selection_status"],
        "selection_explanation": rec["selection_explanation"],
        "worst_case_scenario": rec["worst_case_scenario"],
        "selected_portfolio": rec["selected_portfolio"],
        "best_operational_action": selected,
        "action_required": supply_status != "ON_TRACK",
        "action_note": ("Forecast meets the target: no corrective action is required now; the selected portfolio is the "
                        "robust hedge if a tested disruption materialises.") if supply_status == "ON_TRACK" else None,
        "can_operations_close": rec["can_operations_close"],
        "selected_target": selected_target["target_id"] if selected_target else None,
        "next_target": selected_target["target_id"] if selected_target else None,
        "target_priority": selected_target["exploration_priority"] if selected_target else None,
        "selected_target_detail": ({k: selected_target[k] for k in (
            "target_id", "lat", "lon", "prospectivity_rank", "uncertainty", "applicability", "evidence_level",
            "strategic_relevance", "exploration_priority", "priority_components", "target_context",
            "contains_training_labels", "distance_to_demo_mine_km", "subsurface_status")} if selected_target else None),
        "ranked_targets": [{k: t[k] for k in ("target_id", "exploration_priority", "applicability", "uncertainty",
                                              "strategic_relevance", "evidence_level", "distance_to_demo_mine_km")}
                           for t in ranked[:5]],
        "why_target_now": why_now,
        "next_evidence": ExplorationService.next_evidence(selected_target) if selected_target is not None else [],
        "reason_codes": reason_codes,
        "review_reasons": review_reasons,
        "strategic_requirement": strategic,
        "forecast": {k: fc[k] for k in ("forecast_origin", "forecast_horizon", "p10_tonnes", "p50_tonnes", "p90_tonnes",
                                        "gap_p50_tonnes", "risk_state", "quantiles_validated")}
                    | {"applicability": fc["applicability"]["level"]},
        "nominal_scenario": rec["nominal_scenario"],
        "overrides_applied": rec["overrides_applied"],
        "inputs_recomputed_server_side": True,
        "client_payload_note": ("Client-supplied forecast/recovery objects are not trusted; the decision is recomputed "
                                "from the same inputs on the server.") if client_supplied else None,
        "status": "SIMULATED_SCENARIO" if rec["overrides_applied"] or rec["nominal_scenario"] != "NORMAL" else "MODEL_ESTIMATE",
        "human_review_required": True,
        "provenance": provenance(SIMULATED if rec["overrides_applied"] else "SYNTHETIC", prod.model_version,
                                 fc["provenance"]["observation_window"], None, False,
                                 operations_mode="SYNTHETIC", weather_mode="REAL_GOVERNMENT",
                                 exploration_mode="CACHED", exploration_model_version=exploration().model_version,
                                 risk_policy_version=load_config("risk_policy.json")["version"]),
    }


def _delta(a, b):
    """Change between two backend values; None when either side is withheld (e.g. REVIEW_REQUIRED)."""
    return None if a is None or b is None else round(a - b, 1)


def decision_flip(mine_id="DEMO_MINE", baseline_conditions=None, perturbed_conditions=None, horizon=None,
                  scenario=None, perturbed_scenario=None, actions=None, target_tonnes=None) -> dict:
    """Run the full decision twice — baseline vs user-perturbed conditions — and report the change."""
    base = evaluate(mine_id, horizon, target_tonnes, None, baseline_conditions or {}, scenario=scenario, actions=actions)
    pert = evaluate(mine_id, horizon, target_tonnes, None, perturbed_conditions or {},
                    scenario=perturbed_scenario or scenario, actions=actions)

    def brief(d):
        out = {k: d[k] for k in ("decision_state", "supply_status", "target_tonnes", "baseline_gap_tonnes",
                                 "best_operational_recovery_tonnes", "expected_residual_gap_tonnes",
                                 "worst_case_residual_gap_tonnes", "selected_portfolio", "next_target", "target_priority",
                                 "reason_codes", "decision_summary", "overrides_applied", "nominal_scenario",
                                 "selection_status", "review_reasons", "why_target_now")}
        out["forecast"] = {k: d["forecast"][k] for k in ("p10_tonnes", "p50_tonnes", "p90_tonnes", "risk_state")}
        out["exploration_contingency"] = bool(d["strategic_requirement"].get("active"))
        return out

    # Investigation priority of every target under each supply state (geological prospectivity is unchanged;
    # only the strategic-relevance component moves with the residual gap).
    ex = exploration()
    rank_b = {t["target_id"]: t for t in ex.prioritise(base["strategic_requirement"])}
    rank_p = {t["target_id"]: t for t in ex.prioritise(pert["strategic_requirement"])}
    pos_b = {tid: i + 1 for i, tid in enumerate(rank_b)}
    pos_p = {tid: i + 1 for i, tid in enumerate(rank_p)}
    top = list(dict.fromkeys(list(rank_p)[:5] + list(rank_b)[:5]))
    priority_changes = [{"target_id": tid,
                         "priority_baseline": rank_b[tid]["exploration_priority"],
                         "priority_perturbed": rank_p[tid]["exploration_priority"],
                         "rank_baseline": pos_b[tid], "rank_perturbed": pos_p[tid],
                         "strategic_relevance_baseline": rank_b[tid]["strategic_relevance"],
                         "strategic_relevance_perturbed": rank_p[tid]["strategic_relevance"],
                         "prospectivity_rank": rank_p[tid]["prospectivity_rank"]} for tid in top]

    b_in, p_in = base["overrides_applied"], pert["overrides_applied"]
    keys = sorted(set(b_in) | set(p_in))
    changed = [{"input": k, "baseline": b_in.get(k), "perturbed": p_in.get(k)} for k in keys if b_in.get(k) != p_in.get(k)]
    if base["nominal_scenario"] != pert["nominal_scenario"]:
        changed.append({"input": "scenario", "baseline": base["nominal_scenario"], "perturbed": pert["nominal_scenario"]})
    return {
        "mine_id": base["mine_id"],
        "horizon": base["decision_horizon"],
        "baseline": brief(base),
        "perturbed": brief(pert),
        "flipped": base["decision_state"] != pert["decision_state"],
        "transition": f"{base['decision_state']} -> {pert['decision_state']}",
        "changed_inputs": changed,
        "deltas": {k: _delta(pert.get(k) if k != "forecast_p50_tonnes" else pert["forecast"]["p50_tonnes"],
                             base.get(k) if k != "forecast_p50_tonnes" else base["forecast"]["p50_tonnes"])
                   for k in ("forecast_p50_tonnes", "baseline_gap_tonnes", "expected_residual_gap_tonnes",
                             "worst_case_residual_gap_tonnes")},
        "exploration_priority_changes": priority_changes,
        "priority_note": "Prospectivity is unchanged by supply conditions; only the priority to investigate changes.",
        "status": "SIMULATED_SCENARIO",
        "provenance": pert["provenance"],
    }
