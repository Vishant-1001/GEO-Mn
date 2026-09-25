"""Multi-scenario robust recovery engine.

For every disruption scenario s and action portfolio a (plus "no action"):
  * build the operating state = current state (+ user conditions) -> disruption -> actions
  * check PHYSICAL feasibility against mine constraints (FEASIBLE / FEASIBLE_CONSTRAINED /
    NOT_FEASIBLE / UNKNOWN — HUMAN REVIEW); model applicability of each simulated
    state is reported separately (low applicability = low-confidence estimate)
  * forecast P10/P50/P90 with the production model
  * ResidualGap(a, s) = max(0, Target - P50(a, s))

Selection among feasible portfolios (a lightweight robust-recourse approximation,
NOT an industrial stochastic mine scheduler):
  1. minimise max_s ResidualGap(a, s)
  2. tie -> minimise mean_s ResidualGap(a, s)
  3. tie -> fewer interventions
Ties use a tolerance of `tie_tolerance_pct_of_target` (config/risk_policy.json).

"Expected" figures refer to the nominal (currently assumed) scenario; "worst case"
is the maximum over the evaluated scenarios. All results are SIMULATED SCENARIO
model estimates.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from services.common import SIMULATED, ApiError, load_config, provenance
from services.production_service import get_service as production

SCENARIO_ALIASES = {
    "NORMAL": "NORMAL", "BASELINE": "NORMAL", "NONE": "NORMAL",
    "HEAVY_RAIN": "HEAVY_RAIN", "RAIN": "HEAVY_RAIN",
    "EQUIPMENT_DEGRADATION": "EQUIPMENT_DEGRADATION", "EQUIPMENT": "EQUIPMENT_DEGRADATION",
    "BLAST_DELAY": "BLAST_DELAY", "BLAST": "BLAST_DELAY",
    "COMBINED_DISRUPTION": "COMBINED_DISRUPTION", "COMBINED": "COMBINED_DISRUPTION",
}
ACTION_ALIASES = {
    "EQUIPMENT_RECOVERY": "EQUIPMENT_RECOVERY", "A1": "EQUIPMENT_RECOVERY",
    "SCHEDULE_ADJUSTMENT": "SCHEDULE_ADJUSTMENT", "A2": "SCHEDULE_ADJUSTMENT",
    "BLAST_DRILL_DELAY_REDUCTION": "BLAST_DRILL_DELAY_REDUCTION", "BLAST_DELAY_REDUCTION": "BLAST_DRILL_DELAY_REDUCTION",
    "DRILL_BLAST_DELAY_REDUCTION": "BLAST_DRILL_DELAY_REDUCTION", "A3": "BLAST_DRILL_DELAY_REDUCTION",
}
ACTION_ORDER = ["EQUIPMENT_RECOVERY", "SCHEDULE_ADJUSTMENT", "BLAST_DRILL_DELAY_REDUCTION"]
ALL_SCENARIOS = ["NORMAL", "HEAVY_RAIN", "EQUIPMENT_DEGRADATION", "BLAST_DELAY", "COMBINED_DISRUPTION"]

FEASIBLE = "FEASIBLE"
CONSTRAINED = "FEASIBLE_CONSTRAINED"
NOT_FEASIBLE = "NOT_FEASIBLE"
UNKNOWN = "UNKNOWN — HUMAN REVIEW"
_SEVERITY = {FEASIBLE: 0, CONSTRAINED: 1, UNKNOWN: 2, NOT_FEASIBLE: 3}


def _key(v: str) -> str:
    return str(v).strip().upper().replace("-", "_").replace(" ", "_").replace("/", "_")


def norm_scenario(v) -> str:
    k = SCENARIO_ALIASES.get(_key(v))
    if k is None:
        raise ApiError(400, "INVALID_INPUT", f"Unknown disruption scenario '{v}'. Use one of {ALL_SCENARIOS}")
    return k


def norm_action(v) -> str:
    k = ACTION_ALIASES.get(_key(v))
    if k is None:
        raise ApiError(400, "INVALID_INPUT", f"Unknown action '{v}'. Use one of {ACTION_ORDER}")
    return k


def portfolio_id(actions) -> str:
    cfg = load_config("recovery_config.json")["actions"]
    return "+".join(cfg[a]["code"] for a in actions) if actions else "NO_ACTION"


def portfolio_label(actions) -> str:
    if not actions:
        return "No action (current plan)"
    return " + ".join(a.replace("_", " ").title().replace("Blast Drill", "Blast/Drill") for a in actions)


def build_portfolios(action_portfolios=None, actions=None):
    if action_portfolios:
        out = []
        for p in action_portfolios:
            if isinstance(p, str):
                p = p.split("+")
            acts = sorted({norm_action(a) for a in p}, key=ACTION_ORDER.index)
            if acts and acts not in out:
                out.append(acts)
        if not out:
            raise ApiError(400, "INVALID_INPUT", "action_portfolios contained no valid portfolio")
        return out
    enabled = sorted({norm_action(a) for a in actions}, key=ACTION_ORDER.index) if actions else ACTION_ORDER
    return [list(c) for n in range(1, len(enabled) + 1) for c in combinations(enabled, n)]


def derive_downtime(s: dict, sched: float) -> dict:
    """Keep lost hours consistent with availability (lost = sched x (1 - availability)).

    The new lost hours are split between unplanned downtime and maintenance in the
    same proportion as before the change, preserving the structure of the records.
    """
    lost_new = max(0.0, sched * (1.0 - s["equipment_availability"]))
    lost_old = s["equipment_downtime_h"] + s["maintenance_hours"]
    share = s["equipment_downtime_h"] / lost_old if lost_old > 1e-9 else 0.5
    s["equipment_downtime_h"] = lost_new * share
    s["maintenance_hours"] = lost_new * (1 - share)
    return s


def apply_disruption(state: dict, scenario: str, sched: float) -> dict:
    d = load_config("recovery_config.json")["disruptions"][scenario]
    s = dict(state)
    for sub in d.get("combine", []):
        s = apply_disruption(s, sub, sched)
    for k, v in d.get("set_min", {}).items():
        s[k] = max(s[k], v)
    for k, v in d.get("add", {}).items():
        s[k] = s[k] + v
    s["equipment_availability"] = min(max(s["equipment_availability"], 0.0), 1.0)
    for k in ("blast_delay_h", "drilling_delay_h", "haulage_delay_h"):
        s[k] = max(s[k], 0.0)
    if "equipment_availability" in d.get("add", {}):
        derive_downtime(s, sched)
    return s


def apply_actions(state: dict, actions, mine: dict):
    """Apply actions with physical constraints. Returns (new_state, feasibility, notes)."""
    cfg = load_config("recovery_config.json")
    s = dict(state)
    notes = []
    status = FEASIBLE
    sched = mine["scheduled_hours_per_day"]
    # baseline physical consistency: lost hours cannot exceed the scheduled day
    if s["equipment_downtime_h"] + s["maintenance_hours"] > sched + 1e-6:
        return s, UNKNOWN, ["Downtime + maintenance exceed scheduled hours in the input state; feasibility cannot be judged."]
    for a in actions:
        spec = cfg["actions"][a]
        before = dict(s)
        for k, v in spec.get("add", {}).items():
            s[k] = s[k] + v
        for k, v in spec.get("scale", {}).items():
            s[k] = s[k] * v
        for k, lo in cfg["floors"].items():
            if s[k] < lo:
                s[k] = lo
        cap_a = mine["max_equipment_availability"]
        if s["equipment_availability"] > cap_a:
            s["equipment_availability"] = max(cap_a, before["equipment_availability"])
            notes.append(f"{a}: availability capped at {cap_a:.2f} (mine maximum).")
            status = CONSTRAINED
        cap_t = mine["fleet_max_trucks"]
        if s["truck_count"] > cap_t:
            s["truck_count"] = max(float(cap_t), before["truck_count"])
            notes.append(f"{a}: truck count capped at fleet size {cap_t}.")
            status = CONSTRAINED
        if "equipment_availability" in spec.get("add", {}):
            derive_downtime(s, sched)
        levers = set(spec.get("add", {})) | set(spec.get("scale", {}))
        if all(abs(s[k] - before[k]) < 1e-9 for k in levers):
            notes.append(f"{a}: no headroom — the action cannot change the operating state.")
            status = NOT_FEASIBLE
    # post-action physical validity
    if not (0.0 <= s["equipment_availability"] <= 1.0) or s["truck_count"] < 0:
        status = NOT_FEASIBLE
        notes.append("Resulting state is physically invalid.")
    return s, status, notes


def evaluate(mine_id="DEMO_MINE", target_tonnes=None, base_state=None, conditions=None, forecast_origin=None,
             disruption_scenarios=None, action_portfolios=None, actions=None, scenario=None) -> dict:
    prod = production()
    policy = load_config("risk_policy.json")["decision"]
    overrides = {**(base_state or {}), **(conditions or {})}
    ctx = prod.context(mine_id, forecast_origin, target_tonnes, overrides)
    nominal = norm_scenario(scenario) if scenario else "NORMAL"
    scenarios = [norm_scenario(s) for s in (disruption_scenarios or ALL_SCENARIOS)]
    scenarios = list(dict.fromkeys(scenarios))
    if nominal not in scenarios:
        scenarios.insert(0, nominal)
    portfolios = [[]] + build_portfolios(action_portfolios, actions)
    target = ctx["target"]
    mine = ctx["scenario"]["mine"]
    base = ctx["row"]

    rows, meta = [], []
    sched = mine["scheduled_hours_per_day"]
    for pi, acts in enumerate(portfolios):
        for s in scenarios:
            st, feas, notes = apply_actions(apply_disruption(base, s, sched), acts, mine)
            rows.append(st)
            meta.append((pi, s, feas, notes))
    P = prod.predict(rows)

    appl = [prod.applicability(r)["level"] for r in rows]
    results = {}
    for (pi, s, feas, notes), p, ap, st in zip(meta, P, appl, rows):
        r = results.setdefault(pi, {"per_scenario": [], "feas": FEASIBLE, "notes": [], "extrapolated": []})
        if ap == "LOW":
            r["extrapolated"].append(s)
        r["per_scenario"].append({
            "scenario": s,
            "p10_tonnes": round(float(p[0]), 1), "p50_tonnes": round(float(p[1]), 1), "p90_tonnes": round(float(p[2]), 1),
            "residual_gap_tonnes": round(max(0.0, target - float(p[1])), 1),
            "feasibility": feas, "applicability": ap,
            "state": {k: round(float(v), 3) for k, v in st.items() if k in (
                "equipment_availability", "equipment_downtime_h", "maintenance_hours", "drilling_delay_h",
                "blast_delay_h", "truck_count", "haulage_delay_h", "rainfall_7d_mm", "soil_moisture_m3m3")},
        })
        if _SEVERITY[feas] > _SEVERITY[r["feas"]]:
            r["feas"] = feas
        r["notes"].extend(n for n in notes if n not in r["notes"])

    no_action = {x["scenario"]: x for x in results[0]["per_scenario"]}
    evaluated = []
    for pi, acts in enumerate(portfolios):
        r = results[pi]
        ps = r["per_scenario"]
        for x in ps:
            x["recovery_vs_no_action_tonnes"] = round(x["p50_tonnes"] - no_action[x["scenario"]]["p50_tonnes"], 1)
        nom = next(x for x in ps if x["scenario"] == nominal)
        worst = max(ps, key=lambda x: x["residual_gap_tonnes"])
        evaluated.append({
            "id": portfolio_id(acts),
            "portfolio_id": portfolio_id(acts),
            "name": portfolio_label(acts),
            "actions": acts,
            "n_actions": len(acts),
            "feasibility": r["feas"],
            "eligible": r["feas"] in (FEASIBLE, CONSTRAINED),
            "low_applicability_scenarios": r["extrapolated"],
            "estimate_confidence": ("LOW — nominal scenario outside model experience" if nominal in r["extrapolated"]
                                    else "REDUCED — some stress scenarios outside model experience" if r["extrapolated"]
                                    else "NORMAL"),
            "constraint_notes": r["notes"],
            "expected_p50_tonnes": nom["p50_tonnes"],
            "expected_recovery_tonnes": nom["recovery_vs_no_action_tonnes"],
            "expected_residual_gap_tonnes": nom["residual_gap_tonnes"],
            "worst_case_residual_gap_tonnes": worst["residual_gap_tonnes"],
            "worst_case_scenario": worst["scenario"],
            "mean_residual_gap_tonnes": round(float(np.mean([x["residual_gap_tonnes"] for x in ps])), 1),
            "per_scenario": ps,
            "status": "SIMULATED_SCENARIO",
        })

    tol = policy["tie_tolerance_pct_of_target"] / 100 * max(target, 1.0)
    eligible = [e for e in evaluated if e["eligible"]]
    selected = None
    if eligible:
        best_max = min(e["worst_case_residual_gap_tonnes"] for e in eligible)
        c1 = [e for e in eligible if e["worst_case_residual_gap_tonnes"] <= best_max + tol]
        best_mean = min(e["mean_residual_gap_tonnes"] for e in c1)
        c2 = [e for e in c1 if e["mean_residual_gap_tonnes"] <= best_mean + tol]
        selected = sorted(c2, key=lambda e: (e["n_actions"], e["mean_residual_gap_tonnes"], e["id"]))[0]
    order = sorted(eligible, key=lambda e: (e["worst_case_residual_gap_tonnes"], e["mean_residual_gap_tonnes"], e["n_actions"]))
    for i, e in enumerate(order, start=1):
        e["robust_rank"] = i
    for e in evaluated:
        e["selected"] = selected is not None and e["id"] == selected["id"]

    base_nom = no_action[nominal]
    materiality = policy["materiality_gap_pct"] / 100 * target
    out = {
        "mine_id": ctx["scenario"]["mine_id"],
        "forecast_origin": str(ctx["origin"].date()),
        "forecast_horizon": "next_period_7_days",
        "target_tonnes": round(target, 1),
        "nominal_scenario": nominal,
        "disruption_scenarios": scenarios,
        "baseline": {
            "portfolio": "NO_ACTION",
            "p10_tonnes": base_nom["p10_tonnes"], "p50_tonnes": base_nom["p50_tonnes"], "p90_tonnes": base_nom["p90_tonnes"],
            "gap_p50_tonnes": base_nom["residual_gap_tonnes"],
            "worst_case_gap_tonnes": evaluated[0]["worst_case_residual_gap_tonnes"],
            "per_scenario": results[0]["per_scenario"],
        },
        "evaluated_portfolios": evaluated,
        "portfolios": evaluated,
        "selected_portfolio": selected["id"] if selected else None,
        "selected_actions": selected["actions"] if selected else [],
        "best_operational_action": ({k: selected[k] for k in ("id", "name", "actions", "feasibility", "expected_recovery_tonnes",
                                                              "expected_residual_gap_tonnes", "worst_case_residual_gap_tonnes",
                                                              "constraint_notes")} if selected else None),
        "expected_recovery_tonnes": selected["expected_recovery_tonnes"] if selected else 0.0,
        "expected_residual_gap_tonnes": selected["expected_residual_gap_tonnes"] if selected else base_nom["residual_gap_tonnes"],
        "worst_case_residual_gap_tonnes": (selected["worst_case_residual_gap_tonnes"] if selected
                                           else evaluated[0]["worst_case_residual_gap_tonnes"]),
        "worst_case_scenario": selected["worst_case_scenario"] if selected else None,
        "nominal_applicability": no_action[nominal]["applicability"],
        "low_applicability_scenarios": results[0]["extrapolated"],
        "mean_residual_gap_tonnes": selected["mean_residual_gap_tonnes"] if selected else None,
        "can_operations_close": bool(selected and selected["expected_residual_gap_tonnes"] <= materiality),
        "materiality_threshold_tonnes": round(materiality, 1),
        "selection_rule": ("minimise worst-case residual gap over scenarios; tie -> minimise mean residual gap; "
                           f"tie -> fewer interventions (tie tolerance {policy['tie_tolerance_pct_of_target']}% of target)"),
        "status": "SIMULATED_SCENARIO",
        "human_review_required": True,
        "review_note": "Simulated model estimates; recovery is not guaranteed. A planner must confirm feasibility on site.",
        "inputs": {k: round(float(v), 4) for k, v in base.items() if k in (
            "equipment_availability", "equipment_downtime_h", "maintenance_hours", "drilling_delay_h", "blast_delay_h",
            "truck_count", "haulage_delay_h", "rainfall_7d_mm", "soil_moisture_m3m3", "temperature_max_c")},
        "overrides_applied": ctx["overrides"],
        "provenance": provenance(SIMULATED, prod.model_version, f"{prod.daily.index[0].date()}/{prod.history_end().date()}",
                                 None, False, operations_mode="SYNTHETIC", weather_mode="REAL_PUBLIC",
                                 config_version=load_config("recovery_config.json")["version"]),
    }
    return out
