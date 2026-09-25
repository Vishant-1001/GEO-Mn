import pytest

from services import contingency_service as cs
from services import demo_service

STATES = {"OPERATIONAL_RESPONSE", "OPERATIONAL_AND_EXPLORATION_CONTINGENCY", "REVIEW_REQUIRED"}


@pytest.mark.parametrize("mine_id", [m for m in demo_service.list_mines()
                                     if demo_service.resolve(m).get("expected_state")])
def test_demo_states_are_computed(mine_id):
    d = cs.evaluate(mine_id)
    assert d["decision_state"] == demo_service.resolve(mine_id)["expected_state"]


def test_demo_labels_match(client):
    expect = {"DEMO_A": "ON_TRACK", "DEMO_B": "OPERATIONALLY_RECOVERABLE", "DEMO_C": "STRATEGIC_CONTINGENCY",
              "DEMO_D": "REVIEW_REQUIRED", "DEMO_E": "SATELLITE_FALLBACK", "DEMO_F": "DECISION_FLIP"}
    for m, label in expect.items():
        assert client.get(f"/api/supply-command?mine_id={m}").json()["demo_state"] == label
    a = cs.evaluate("DEMO_A")
    assert a["supply_status"] == "ON_TRACK" and a["action_required"] is False
    b = cs.evaluate("DEMO_B")
    assert b["supply_status"] == "OPERATIONALLY_RECOVERABLE"


def test_only_three_decision_states():
    for m in demo_service.list_mines():
        for h in ("NEAR_TERM", "STRATEGIC"):
            assert cs.evaluate(m, h)["decision_state"] in STATES


def test_near_term_never_uses_exploration():
    for m in demo_service.list_mines():
        d = cs.evaluate(m, "NEAR_TERM")
        assert d["decision_state"] != "OPERATIONAL_AND_EXPLORATION_CONTINGENCY"
        assert d["selected_target"] is None


def test_horizon_gate_changes_decision_for_same_state():
    assert cs.evaluate("DEMO_MINE", "NEAR_TERM")["decision_state"] == "OPERATIONAL_RESPONSE"
    assert cs.evaluate("DEMO_MINE", "STRATEGIC")["decision_state"] == "OPERATIONAL_AND_EXPLORATION_CONTINGENCY"


def test_selected_target_is_computed_from_ranking():
    d = cs.evaluate("DEMO_C", "STRATEGIC")
    assert d["decision_state"] == "OPERATIONAL_AND_EXPLORATION_CONTINGENCY"
    top = d["ranked_targets"][0]
    eligible = [t for t in d["ranked_targets"] if t["applicability"] != "LOW"]
    assert d["selected_target"] == eligible[0]["target_id"]
    assert d["target_priority"] == eligible[0]["exploration_priority"] <= top["exploration_priority"]
    codes = set(d["reason_codes"])
    assert {"RESIDUAL_STRATEGIC_GAP", "ACCEPTABLE_APPLICABILITY"} <= codes
    assert d["why_target_now"] and all("code" in r and "text" in r for r in d["why_target_now"])


def test_residual_arithmetic():
    d = cs.evaluate("DEMO_C")
    assert d["expected_residual_gap_tonnes"] <= d["baseline_gap_tonnes"]
    assert d["worst_case_residual_gap_tonnes"] >= d["expected_residual_gap_tonnes"]


def test_ood_state_requires_review():
    d = cs.evaluate("DEMO_D")
    assert d["decision_state"] == "REVIEW_REQUIRED"
    assert "PRODUCTION_INPUTS_OUT_OF_DISTRIBUTION" in {r["code"] for r in d["review_reasons"]}
    assert d["selected_target"] is None


def test_no_applicable_target_requires_review(exploration_svc, monkeypatch):
    low = [dict(t, applicability="LOW") for t in exploration_svc.targets]
    monkeypatch.setattr(exploration_svc, "targets", low)
    d = cs.evaluate("DEMO_C", "STRATEGIC")
    assert d["decision_state"] == "REVIEW_REQUIRED"
    assert "NO_APPLICABLE_TARGET" in d["reason_codes"]


def test_contingency_api_contract(client):
    d = client.post("/api/contingency/evaluate", json={"mine_id": "DEMO_MINE", "forecast": {}, "recovery": {},
                                                       "horizon": "STRATEGIC"}).json()
    for k in ("decision_state", "baseline_gap_tonnes", "best_operational_recovery_tonnes", "expected_residual_gap_tonnes",
              "worst_case_residual_gap_tonnes", "selected_target", "target_priority", "why_target_now", "reason_codes",
              "provenance"):
        assert k in d
    assert d["inputs_recomputed_server_side"] is True
    assert client.post("/api/contingency/evaluate", json={"horizon": "SOMEDAY"}).status_code == 400


def test_supply_command_contract(client):
    d = client.get("/api/supply-command?mine_id=DEMO_MINE").json()
    for k in ("mine_id", "forecast_horizon", "target_tonnes", "p10_tonnes", "p50_tonnes", "p90_tonnes", "gap_p50_tonnes",
              "risk_state", "primary_drivers", "best_operational_action", "expected_recovery_tonnes",
              "expected_residual_gap_tonnes", "worst_case_residual_gap_tonnes", "decision_state", "next_target",
              "why_target_now", "provenance"):
        assert k in d
    assert client.get("/api/supply-command?mine_id=UNKNOWN").status_code == 404


def test_no_eligible_portfolio_requires_review(production_svc, monkeypatch):
    real = production_svc.applicability
    # nominal state stays applicable; every simulated action/stress state is outside experience
    calls = {"n": 0}

    def appl(row):
        calls["n"] += 1
        return real(row) if calls["n"] == 1 else {"level": "LOW", "score": 0, "range_violations": [], "method": "t"}

    monkeypatch.setattr(production_svc, "applicability", appl)
    d = cs.evaluate("DEMO_C", "STRATEGIC")
    assert d["decision_state"] == "REVIEW_REQUIRED"
    assert "NO_ELIGIBLE_OPERATIONAL_PORTFOLIO" in d["reason_codes"]
    assert d["selected_portfolio"] is None and d["selected_target"] is None


def test_why_now_only_with_strategic_gap():
    near = cs.evaluate("DEMO_MINE", "NEAR_TERM")
    assert near["why_target_now"] == [] and near["next_target"] is None
    strat = cs.evaluate("DEMO_MINE", "STRATEGIC")
    assert strat["why_target_now"] and strat["strategic_requirement"]["active"] is True
