"""Multi-scenario robust recovery engine.

For every disruption scenario s and action portfolio a (plus "no action"):
  * build the operating state = current state (+ user conditions) -> disruption -> actions
  * check MODELLED feasibility (input-constraint check: FEASIBLE / FEASIBLE_CONSTRAINED /
    NOT_FEASIBLE / UNKNOWN — HUMAN REVIEW) and model applicability of each simulated state
  * forecast P10/P50/P90 with the production model
  * ResidualGap(a, s) = max(0, Target - P50(a, s))

Applicability gate: a portfolio is ELIGIBLE for automated selection only if it is
modelled-feasible AND none of its tested scenarios has LOW model applicability.
LOW-applicability estimates are still returned as diagnostics but never optimised over.
If no portfolio is eligible, selection is withheld (selection_status REVIEW_REQUIRED).

Selection among eligible portfolios (a lightweight robust-recourse approximation,
NOT an industrial stochastic mine scheduler):
  1. minimise max_s ResidualGap(a, s)
  2. tie -> minimise mean_s ResidualGap(a, s)
  3. tie -> minimise intervention burden (config/recovery_config.json project weights)
  4. tie -> fewer actions, then portfolio id
Ties use a tolerance of `tie_tolerance_pct_of_target` (config/risk_policy.json).

"Expected" figures refer to the nominal (currently assumed) scenario; "worst case"
is the maximum over the evaluated scenarios. All results are SIMULATED SCENARIO
model estimates.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd

from services.common import DATA_DIR, SIMULATED, ApiError, load_config, provenance
from services.production_service import get_service as production

SCENARIO_ALIASES = {
    "NORMAL": "NORMAL", "BASELINE": "NORMAL", "NONE": "NORMAL",
    "HEAVY_RAIN": "HEAVY_RAIN", "RAIN": "HEAVY_RAIN",
    "EQUIPMENT_DEGRADATION": "EQUIPMENT_DEGRADATION", "EQUIPMENT": "EQUIPMENT_DEGRADATION",
    "BLAST_DELAY": "BLAST_DELAY", "BLAST": "BLAST_DELAY",
    "DRILL_DELAY": "DRILL_DELAY", "DRILL": "DRILL_DELAY",
    "HAULAGE_DISRUPTION": "HAULAGE_DISRUPTION", "HAULAGE": "HAULAGE_DISRUPTION",
    "COMBINED_DISRUPTION": "COMBINED_DISRUPTION", "COMBINED": "COMBINED_DISRUPTION",
}
ACTION_ALIASES = {
    "EQUIPMENT_RECOVERY": "EQUIPMENT_RECOVERY", "A1": "EQUIPMENT_RECOVERY",
    "SCHEDULE_ADJUSTMENT": "SCHEDULE_ADJUSTMENT", "A2": "SCHEDULE_ADJUSTMENT",
    "BLAST_DRILL_DELAY_REDUCTION": "BLAST_DRILL_DELAY_REDUCTION", "BLAST_DELAY_REDUCTION": "BLAST_DRILL_DELAY_REDUCTION",
    "DRILL_BLAST_DELAY_REDUCTION": "BLAST_DRILL_DELAY_REDUCTION", "A3": "BLAST_DRILL_DELAY_REDUCTION",
}
ACTION_ORDER = ["EQUIPMENT_RECOVERY", "SCHEDULE_ADJUSTMENT", "BLAST_DRILL_DELAY_REDUCTION"]
ALL_SCENARIOS = ["NORMAL", "HEAVY_RAIN", "EQUIPMENT_DEGRADATION", "BLAST_DELAY", "DRILL_DELAY", "HAULAGE_DISRUPTION",
                 "COMBINED_DISRUPTION"]

FEASIBLE = "FEASIBLE"
CONSTRAINED = "FEASIBLE_CONSTRAINED"
NOT_FEASIBLE = "NOT_FEASIBLE"
UNKNOWN = "UNKNOWN — HUMAN REVIEW"

FEASIBILITY_BASIS = ("MODELLED FEASIBILITY — input-constraint check only (availability within [0, mine cap], trucks <= fleet, "
                     "delays >= 0, lost hours consistent with availability). Not a mine-plan, geotechnical, blasting or "
                     "crew feasibility assessment; planner/site confirmation required.")
APPLICABILITY_POLICY = {
    "HIGH": "eligible for automated selection",
    "MODERATE": "eligible for automated selection; confidence reduced",
    "LOW": "NOT eligible for automated selection; shown as sensitivity diagnostics only",
    "portfolio_level": "worst applicability over every scenario used by the robust objective",
}
SELECTION_RULE = ("eligible portfolios only (modelled-feasible and no LOW-applicability scenario); "
                  "minimise worst-case residual P50 gap over scenarios; tie -> minimise mean residual gap; "
                  "tie -> minimise intervention burden; tie -> fewer actions, then portfolio id "
                  "(tie tolerance {tol}% of target)")
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


MATRIX_PATH = DATA_DIR / "synthetic" / "recovery" / "recovery_scenario_matrix.csv"
MATRIX_FEATS = ["equipment_availability", "equipment_downtime_h", "maintenance_hours", "drilling_delay_h", "blast_delay_h",
                "truck_count", "haulage_delay_h"]


@lru_cache(maxsize=1)
def scenario_matrix() -> pd.DataFrame:
    """Scenario / action effects in model-feature space, derived from SIMULATED simulator counterfactuals
    (scripts/synthetic/generate_recovery.py). Missing -> 503 rather than silently using constants."""
    if not MATRIX_PATH.exists():
        raise ApiError(503, "DATA_UNAVAILABLE", "Recovery scenario matrix not generated (run scripts/synthetic/generate_all.py).")
    return pd.read_csv(MATRIX_PATH).set_index(["scenario", "action_portfolio"])


def _row(scenario, pid_):
    m = scenario_matrix()
    if (scenario, pid_) not in m.index:
        raise ApiError(400, "INVALID_INPUT", f"No simulated effects for {scenario} / {pid_}")
    return m.loc[(scenario, pid_)]


def _clean(s, sched):
    s["equipment_availability"] = min(max(s["equipment_availability"], 0.0), 1.0)
    for k in ("blast_delay_h", "drilling_delay_h", "haulage_delay_h", "truck_count", "equipment_downtime_h", "maintenance_hours"):
        s[k] = max(s[k], 0.0)
    return derive_downtime(s, sched)


def apply_disruption(state: dict, scenario: str, sched: float) -> dict:
    """Median simulated disruption effect (feature deltas) added to the operating state."""
    r = _row(scenario, "NO_ACTION")
    s = dict(state)
    for k in MATRIX_FEATS:
        s[k] = s[k] + float(r[f"scenario_delta_{k}"])
    if pd.notna(r.get("scenario_rainfall_7d_mm_min")):
        s["rainfall_7d_mm"] = max(s["rainfall_7d_mm"], float(r["scenario_rainfall_7d_mm_min"]))
        s["soil_moisture_m3m3"] = max(s["soil_moisture_m3m3"], 0.46)
    return _clean(s, sched)


# Smallest simulated median change treated as a real effect (smaller values are common-random-number noise).
MATERIAL_DELTA = {"equipment_availability": 0.005, "truck_count": 0.1}


def apply_actions(state: dict, actions, mine: dict, scenario: str = "NORMAL"):
    """Apply a portfolio's SIMULATED joint effect under the scenario, then input-constraint checks.

    Returns (new_state, modelled_feasibility, notes)."""
    cfg = load_config("recovery_config.json")
    s = dict(state)
    notes = []
    status = FEASIBLE
    sched = mine["scheduled_hours_per_day"]
    if s["equipment_downtime_h"] + s["maintenance_hours"] > sched + 1e-6:
        return s, UNKNOWN, ["Downtime + maintenance exceed scheduled hours in the input state; feasibility cannot be judged."]
    if not actions:
        return s, status, notes
    r = _row(scenario, portfolio_id(actions))
    before = dict(s)
    deltas = {k: float(r[f"action_delta_{k}"]) for k in MATRIX_FEATS}
    for k, v in deltas.items():
        s[k] = s[k] + v
    for k, lo in cfg["floors"].items():
        s[k] = max(s[k], lo)
    cap_a, cap_t = mine["max_equipment_availability"], mine["fleet_max_trucks"]
    if s["equipment_availability"] > cap_a:
        s["equipment_availability"] = max(cap_a, before["equipment_availability"])
        notes.append(f"availability capped at {cap_a:.2f} (mine maximum).")
        status = CONSTRAINED
    if s["truck_count"] > cap_t:
        s["truck_count"] = max(float(cap_t), before["truck_count"])
        notes.append(f"truck count capped at fleet size {cap_t}.")
        status = CONSTRAINED
    _clean(s, sched)
    # downtime / maintenance hours are re-derived from availability by _clean, so they are not independent levers
    material = {k: v for k, v in deltas.items()
                if k not in ("equipment_downtime_h", "maintenance_hours") and abs(v) > MATERIAL_DELTA.get(k, 0.05)}
    if not material:
        notes.append("simulator shows no material effect of this portfolio under this scenario.")
    elif all(abs(s[k] - before[k]) < 1e-9 for k in material):
        notes.append("no headroom — caps prevent the portfolio from changing the operating state.")
        status = NOT_FEASIBLE
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
            st, feas, notes = apply_actions(apply_disruption(base, s, sched), acts, mine, s)
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
    burden_cfg = load_config("recovery_config.json")["intervention_burden"]["weights"]
    evaluated = []
    for pi, acts in enumerate(portfolios):
        r = results[pi]
        ps = r["per_scenario"]
        for x in ps:
            x["recovery_vs_no_action_tonnes"] = round(x["p50_tonnes"] - no_action[x["scenario"]]["p50_tonnes"], 1)
        nom = next(x for x in ps if x["scenario"] == nominal)
        worst = max(ps, key=lambda x: x["residual_gap_tonnes"])
        sel_appl = worst_applicability(x["applicability"] for x in ps)
        feasible = r["feas"] in (FEASIBLE, CONSTRAINED)
        blocked = None
        if not feasible:
            blocked = f"Automated selection blocked: modelled feasibility is {r['feas']}."
        elif sel_appl == "LOW":
            blocked = ("Automated selection blocked because one or more tested scenarios are outside model "
                       f"applicability ({', '.join(r['extrapolated'])}).")
        evaluated.append({
            "id": portfolio_id(acts),
            "portfolio_id": portfolio_id(acts),
            "name": portfolio_label(acts),
            "actions": acts,
            "n_actions": len(acts),
            "intervention_burden": round(sum(burden_cfg[a] for a in acts), 3),
            "feasibility": r["feas"],
            "modelled_feasibility": r["feas"],
            "physical_feasibility": r["feas"],
            "feasibility_basis": FEASIBILITY_BASIS,
            "selection_applicability": sel_appl,
            "low_applicability_scenarios": r["extrapolated"],
            "eligible": blocked is None,
            "selection_blocked_reason": blocked,
            "estimate_confidence": ("LOW — outside model experience" if sel_appl == "LOW"
                                    else "REDUCED — moderate applicability" if sel_appl == "MODERATE" else "NORMAL"),
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
    selected, explanation = select_portfolio(evaluated, tol)
    order = sorted([e for e in evaluated if e["eligible"]],
                   key=lambda e: (e["worst_case_residual_gap_tonnes"], e["mean_residual_gap_tonnes"],
                                  e["intervention_burden"], e["id"]))
    for i, e in enumerate(order, start=1):
        e["robust_rank"] = i
    for e in evaluated:
        e["selected"] = selected is not None and e["id"] == selected["id"]

    base_nom = no_action[nominal]
    materiality = policy["materiality_gap_pct"] / 100 * target
    selection_status = "SELECTED" if selected else "REVIEW_REQUIRED"
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
        "selection_status": selection_status,
        "selected_portfolio": selected["id"] if selected else None,
        "selected_actions": selected["actions"] if selected else [],
        "selection_explanation": explanation,
        "best_operational_action": ({k: selected[k] for k in (
            "id", "name", "actions", "feasibility", "modelled_feasibility", "selection_applicability",
            "intervention_burden", "expected_recovery_tonnes", "expected_residual_gap_tonnes",
            "worst_case_residual_gap_tonnes", "constraint_notes")} if selected else None),
        "expected_recovery_tonnes": selected["expected_recovery_tonnes"] if selected else None,
        "expected_residual_gap_tonnes": selected["expected_residual_gap_tonnes"] if selected else None,
        "worst_case_residual_gap_tonnes": selected["worst_case_residual_gap_tonnes"] if selected else None,
        "worst_case_scenario": selected["worst_case_scenario"] if selected else None,
        "no_action_expected_gap_tonnes": base_nom["residual_gap_tonnes"],
        "no_action_worst_case_gap_tonnes": evaluated[0]["worst_case_residual_gap_tonnes"],
        "nominal_applicability": no_action[nominal]["applicability"],
        "low_applicability_scenarios": results[0]["extrapolated"],
        "mean_residual_gap_tonnes": selected["mean_residual_gap_tonnes"] if selected else None,
        "can_operations_close": bool(selected and selected["expected_residual_gap_tonnes"] <= materiality),
        "materiality_threshold_tonnes": round(materiality, 1),
        "selection_rule": SELECTION_RULE.format(tol=policy["tie_tolerance_pct_of_target"]),
        "applicability_policy": APPLICABILITY_POLICY,
        "status": "SIMULATED_SCENARIO",
        "human_review_required": True,
        "review_note": ("Simulated model estimates; recovery is not guaranteed. Feasibility is an input-constraint check "
                        "only — planner/site confirmation required."
                        + ("" if selected else " Automated portfolio selection was withheld: no portfolio passed the "
                                               "feasibility and applicability gates.")),
        "inputs": {k: round(float(v), 4) for k, v in base.items() if k in (
            "equipment_availability", "equipment_downtime_h", "maintenance_hours", "drilling_delay_h", "blast_delay_h",
            "truck_count", "haulage_delay_h", "rainfall_7d_mm", "soil_moisture_m3m3", "temperature_max_c")},
        "overrides_applied": ctx["overrides"],
        "provenance": provenance(SIMULATED, prod.model_version, f"{prod.daily.index[0].date()}/{prod.history_end().date()}",
                                 None, False, operations_mode="SYNTHETIC", weather_mode="REAL_GOVERNMENT",
                                 config_version=load_config("recovery_config.json")["version"]),
    }
    return out


def worst_applicability(levels) -> str:
    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    return max(levels, key=lambda x: order[x])


def select_portfolio(evaluated: list[dict], tol: float):
    """Robust selection among ELIGIBLE portfolios (feasible and no LOW-applicability scenario).

    1. minimise worst-case residual gap        (within tol of the best counts as equivalent)
    2. then minimise mean residual gap          (within tol)
    3. then minimise intervention burden
    4. then fewer actions, then portfolio id    (deterministic)
    Returns (selected or None, explanation text).
    """
    eligible = [e for e in evaluated if e["eligible"]]
    if not eligible:
        return None, ("No portfolio is eligible for automated selection (all are outside model applicability or fail "
                      "the modelled feasibility check). Human review required.")
    best_max = min(e["worst_case_residual_gap_tonnes"] for e in eligible)
    c1 = [e for e in eligible if e["worst_case_residual_gap_tonnes"] <= best_max + tol]
    best_mean = min(e["mean_residual_gap_tonnes"] for e in c1)
    c2 = [e for e in c1 if e["mean_residual_gap_tonnes"] <= best_mean + tol]
    c2.sort(key=lambda e: (e["intervention_burden"], e["n_actions"], e["id"]))
    sel = c2[0]
    heavier = [e for e in c2[1:] if e["intervention_burden"] > sel["intervention_burden"]]
    if heavier:
        why = (f"Selected because it achieves the lowest tested worst-case residual gap "
               f"({sel['worst_case_residual_gap_tonnes']:.0f} t) with the lowest intervention burden "
               f"({sel['intervention_burden']:g}) among {len(c2)} near-equivalent portfolios.")
    elif len(c1) == 1:
        why = (f"Selected because it has the lowest tested worst-case residual gap "
               f"({sel['worst_case_residual_gap_tonnes']:.0f} t); no other eligible portfolio is within tolerance.")
    else:
        why = (f"Selected because it has the lowest tested worst-case residual gap "
               f"({sel['worst_case_residual_gap_tonnes']:.0f} t) and the lowest mean residual gap among "
               f"near-equivalent portfolios.")
    skipped = [e["id"] for e in evaluated if not e["eligible"]]
    if skipped:
        why += f" Excluded from selection: {', '.join(skipped)}."
    return sel, why
