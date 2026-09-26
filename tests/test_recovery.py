import pytest

from services import recovery_service as rs
from services.common import load_config

ALL = ["NORMAL", "HEAVY_RAIN", "EQUIPMENT_DEGRADATION", "BLAST_DELAY", "COMBINED_DISRUPTION"]


def test_evaluates_all_scenarios_and_portfolios(client):
    d = client.post("/api/recovery/evaluate", json={"mine_id": "DEMO_MINE", "disruption_scenarios": ALL}).json()
    ids = [p["id"] for p in d["evaluated_portfolios"]]
    assert ids == ["NO_ACTION", "A1", "A2", "A3", "A1+A2", "A1+A3", "A2+A3", "A1+A2+A3"]
    for p in d["evaluated_portfolios"]:
        assert [s["scenario"] for s in p["per_scenario"]] == ALL
        for s in p["per_scenario"]:
            assert s["p10_tonnes"] <= s["p50_tonnes"] <= s["p90_tonnes"]
            assert s["residual_gap_tonnes"] == pytest.approx(max(0.0, d["target_tonnes"] - s["p50_tonnes"]), abs=0.2)
    assert d["status"] == "SIMULATED_SCENARIO" and d["human_review_required"] is True


def test_selection_is_robust_minimax(client):
    d = client.post("/api/recovery/evaluate", json={"mine_id": "DEMO_MINE"}).json()
    tol = load_config("risk_policy.json")["decision"]["tie_tolerance_pct_of_target"] / 100 * d["target_tonnes"]
    eligible = [p for p in d["evaluated_portfolios"] if p["eligible"]]
    sel = next(p for p in eligible if p["selected"])
    assert sel["worst_case_residual_gap_tonnes"] <= min(p["worst_case_residual_gap_tonnes"] for p in eligible) + tol
    assert d["selected_portfolio"] == sel["id"]
    assert d["worst_case_residual_gap_tonnes"] == max(s["residual_gap_tonnes"] for s in sel["per_scenario"])


def test_lowest_burden_wins_when_every_portfolio_closes_the_gap():
    # With a tiny target every portfolio has zero residual: the lowest-burden ELIGIBLE one must win.
    d = rs.evaluate("DEMO_MINE", target_tonnes=100.0)
    eligible = [p for p in d["evaluated_portfolios"] if p["eligible"]]
    sel = next(p for p in eligible if p["selected"])
    assert sel["intervention_burden"] == min(p["intervention_burden"] for p in eligible)
    assert d["worst_case_residual_gap_tonnes"] == 0
    no_action = d["evaluated_portfolios"][0]
    if no_action["eligible"]:
        assert d["selected_portfolio"] == "NO_ACTION"
    else:  # "do nothing" cannot be auto-selected when its stress estimate is outside model experience
        assert no_action["low_applicability_scenarios"]


def test_disruptions_reduce_output():
    d = rs.evaluate("DEMO_MINE")
    base = {s["scenario"]: s["p50_tonnes"] for s in d["baseline"]["per_scenario"]}
    assert base["COMBINED_DISRUPTION"] < base["NORMAL"]
    assert base["EQUIPMENT_DEGRADATION"] < base["NORMAL"]


def test_feasibility_constraints():
    mine = {"scheduled_hours_per_day": 20.0, "max_equipment_availability": 0.95, "fleet_max_trucks": 24}
    state = {"equipment_availability": 0.95, "equipment_downtime_h": 0.5, "maintenance_hours": 0.5, "drilling_delay_h": 1,
             "blast_delay_h": 1, "truck_count": 23, "haulage_delay_h": 0.5, "rainfall_7d_mm": 0,
             "soil_moisture_m3m3": 0.3, "temperature_max_c": 30}
    _, status, notes = rs.apply_actions(dict(state, truck_count=24), ["EQUIPMENT_RECOVERY"], mine)
    assert status == rs.NOT_FEASIBLE and any("headroom" in n for n in notes)
    s2, status, _ = rs.apply_actions(state, ["SCHEDULE_ADJUSTMENT"], mine)
    assert status == rs.CONSTRAINED and s2["truck_count"] == 24
    bad = dict(state, equipment_downtime_h=15, maintenance_hours=10)
    assert rs.apply_actions(bad, ["BLAST_DRILL_DELAY_REDUCTION"], mine)[1] == rs.UNKNOWN


def test_actions_keep_state_physically_valid():
    mine = {"scheduled_hours_per_day": 20.0, "max_equipment_availability": 0.95, "fleet_max_trucks": 24}
    state = {"equipment_availability": 0.8, "equipment_downtime_h": 2.4, "maintenance_hours": 1.6, "drilling_delay_h": 1,
             "blast_delay_h": 1, "truck_count": 16, "haulage_delay_h": 0.5, "rainfall_7d_mm": 0,
             "soil_moisture_m3m3": 0.3, "temperature_max_c": 30}
    s, status, _ = rs.apply_actions(state, rs.ACTION_ORDER, mine)
    assert status == rs.FEASIBLE
    assert 0 <= s["equipment_availability"] <= 1
    assert s["equipment_downtime_h"] + s["maintenance_hours"] == pytest.approx(20 * (1 - s["equipment_availability"]))
    assert s["blast_delay_h"] >= 0 and s["truck_count"] >= 0


def test_frontend_request_form(client):
    d = client.post("/api/recovery/evaluate", json={
        "mine_id": "DEMO_MINE", "scenario": "heavy_rain", "actions": ["equipment_recovery", "blast_delay_reduction"],
        "conditions": {"rainfall_mm": 60, "equipment_availability": 0.8, "blast_delay_hours": 2}}).json()
    assert d["nominal_scenario"] == "HEAVY_RAIN"
    assert {p["id"] for p in d["evaluated_portfolios"]} == {"NO_ACTION", "A1", "A3", "A1+A3"}
    assert d["overrides_applied"]["rainfall_7d_mm"] == 60


def test_invalid_inputs(client):
    assert client.post("/api/recovery/evaluate", json={"disruption_scenarios": ["METEOR"]}).status_code == 400
    assert client.post("/api/recovery/evaluate", json={"actions": ["TELEPORT"]}).status_code == 400
    assert client.post("/api/recovery/evaluate", json={"conditions": {"blast_delay_h": -1}}).status_code == 400


# ---------------------------------------------------------------------------
# Applicability gate
# ---------------------------------------------------------------------------

def test_low_applicability_portfolios_are_not_selectable():
    # DEMO_A sits near the top of the training range, so equipment-recovery / truck portfolios
    # push inputs outside model experience for some scenarios.
    d = rs.evaluate("DEMO_A")
    blocked = [p for p in d["evaluated_portfolios"] if p["low_applicability_scenarios"]]
    assert blocked, "expected at least one low-applicability portfolio in DEMO_A"
    for p in blocked:
        assert p["eligible"] is False
        assert p["selection_applicability"] == "LOW"
        assert "outside model applicability" in p["selection_blocked_reason"]
        assert p["selected"] is False
        # still shown as diagnostics
        assert len(p["per_scenario"]) == len(rs.ALL_SCENARIOS) and any(s["applicability"] == "LOW" for s in p["per_scenario"])
    sel = next(p for p in d["evaluated_portfolios"] if p["selected"])
    assert sel["low_applicability_scenarios"] == [] and sel["selection_applicability"] != "LOW"


def _fake_applicability(level_for):
    def f(row):
        return {"level": level_for(row), "score": 0.0, "range_violations": [], "method": "test"}
    return f


def test_moderate_applicability_remains_selectable(production_svc, monkeypatch):
    monkeypatch.setattr(production_svc, "applicability", _fake_applicability(lambda r: "MODERATE"))
    d = rs.evaluate("DEMO_MINE")
    assert d["selection_status"] == "SELECTED"
    sel = next(p for p in d["evaluated_portfolios"] if p["selected"])
    assert sel["selection_applicability"] == "MODERATE" and sel["eligible"] is True


def test_all_portfolios_blocked_withholds_selection(production_svc, monkeypatch):
    monkeypatch.setattr(production_svc, "applicability", _fake_applicability(lambda r: "LOW"))
    d = rs.evaluate("DEMO_MINE")
    assert d["selection_status"] == "REVIEW_REQUIRED"
    assert d["selected_portfolio"] is None and d["best_operational_action"] is None
    assert d["expected_residual_gap_tonnes"] is None
    assert d["human_review_required"] is True
    assert all(not p["eligible"] for p in d["evaluated_portfolios"])
    assert all(p["per_scenario"] for p in d["evaluated_portfolios"])  # diagnostics still present


def test_gate_is_applied_before_optimisation(production_svc, monkeypatch):
    # Block exactly the all-actions portfolio (the unconstrained optimum): it must not be chosen.
    real = production_svc.applicability

    def appl(row):
        out = real(row)
        if row["truck_count"] >= 18 and row["equipment_availability"] >= 0.9:
            out = dict(out, level="LOW")
        return out

    monkeypatch.setattr(production_svc, "applicability", appl)
    d = rs.evaluate("DEMO_MINE")
    sel = next((p for p in d["evaluated_portfolios"] if p["selected"]), None)
    assert sel is None or sel["low_applicability_scenarios"] == []


# ---------------------------------------------------------------------------
# Intervention-burden tie-break (pure selection function)
# ---------------------------------------------------------------------------

def _p(pid, worst, mean, burden, eligible=True):
    return {"id": pid, "worst_case_residual_gap_tonnes": worst, "mean_residual_gap_tonnes": mean,
            "intervention_burden": burden, "n_actions": int(burden), "eligible": eligible}


def test_equivalent_residuals_prefer_lower_burden():
    sel, why = rs.select_portfolio([_p("A1+A2+A3", 100.0, 50.0, 3), _p("A2+A3", 120.0, 60.0, 2)], tol=50.0)
    assert sel["id"] == "A2+A3"
    assert "lowest intervention burden" in why


def test_material_improvement_justifies_higher_burden():
    sel, _ = rs.select_portfolio([_p("A1+A2+A3", 100.0, 50.0, 3), _p("A2+A3", 400.0, 200.0, 2)], tol=50.0)
    assert sel["id"] == "A1+A2+A3"


def test_all_actions_selected_when_necessary():
    ps = [_p("A1", 900, 500, 1), _p("A2", 800, 450, 1), _p("A3", 950, 520, 1),
          _p("A1+A2", 500, 300, 2), _p("A1+A3", 600, 320, 2), _p("A2+A3", 550, 310, 2), _p("A1+A2+A3", 100, 40, 3)]
    assert rs.select_portfolio(ps, tol=50.0)[0]["id"] == "A1+A2+A3"


def test_ineligible_excluded_before_tie_break():
    sel, why = rs.select_portfolio([_p("A1", 0.0, 0.0, 1, eligible=False), _p("A2+A3", 300.0, 100.0, 2)], tol=50.0)
    assert sel["id"] == "A2+A3" and "Excluded from selection: A1" in why
    assert rs.select_portfolio([_p("A1", 0.0, 0.0, 1, eligible=False)], tol=50.0)[0] is None


def test_burden_exposed_and_configured():
    d = rs.evaluate("DEMO_MINE")
    w = load_config("recovery_config.json")["intervention_burden"]["weights"]
    for p in d["evaluated_portfolios"]:
        assert p["intervention_burden"] == pytest.approx(sum(w[a] for a in p["actions"]))
        assert "input-constraint" in p["feasibility_basis"]
    assert "intervention burden" in d["selection_rule"]
