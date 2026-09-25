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


def test_fewer_interventions_break_ties():
    # when nothing is needed, "no action" (0 interventions) must win the tie
    d = rs.evaluate("DEMO_MINE", target_tonnes=100.0)
    assert d["selected_portfolio"] == "NO_ACTION"
    assert d["worst_case_residual_gap_tonnes"] == 0


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
    _, status, notes = rs.apply_actions(state, ["EQUIPMENT_RECOVERY"], mine)
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
