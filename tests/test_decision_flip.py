from services import contingency_service as cs

STRESS = {"rainfall_7d_mm": 140.0, "equipment_availability": 0.74, "blast_delay_h": 3.0}


def test_demo_f_flips():
    f = cs.decision_flip("DEMO_F", {}, STRESS)
    assert f["baseline"]["decision_state"] == "OPERATIONAL_RESPONSE"
    assert f["perturbed"]["decision_state"] == "OPERATIONAL_AND_EXPLORATION_CONTINGENCY"
    assert f["flipped"] is True
    assert f["perturbed"]["next_target"] is not None and f["baseline"]["next_target"] is None
    assert {c["input"] for c in f["changed_inputs"]} == set(STRESS)
    assert f["status"] == "SIMULATED_SCENARIO"


def test_no_change_no_flip():
    f = cs.decision_flip("DEMO_F", {}, {})
    assert f["flipped"] is False and f["changed_inputs"] == []


def test_flip_endpoint_and_supply_command(client):
    r = client.post("/api/decision/flip", json={"mine_id": "DEMO_F", "perturbed_conditions": STRESS})
    assert r.status_code == 200 and r.json()["flipped"] is True
    d = client.get("/api/supply-command?mine_id=DEMO_F").json()
    assert d["decision_flip"]["transition"] == "OPERATIONAL_RESPONSE -> OPERATIONAL_AND_EXPLORATION_CONTINGENCY"


def test_frontend_style_flip_via_contingency(client):
    base = client.post("/api/contingency/evaluate", json={"mine_id": "DEMO_F", "scenario": "normal"}).json()
    stressed = client.post("/api/contingency/evaluate", json={
        "mine_id": "DEMO_F", "scenario": "normal",
        "conditions": {"rainfall_mm": 140, "equipment_availability": 0.74, "blast_delay_hours": 3}}).json()
    assert base["decision_state"] != stressed["decision_state"]


def test_decision_review_and_history(client):
    r = client.post("/api/decision/review", json={"mine_id": "DEMO_C", "decision_state": "OPERATIONAL_AND_EXPLORATION_CONTINGENCY",
                                                   "action": "ACCEPT", "target_id": "T14", "assumptions": {"horizon": "STRATEGIC"}})
    assert r.status_code == 200
    rec = r.json()
    assert rec["review_status"] == "ACCEPTED" and rec["decision_id"]
    h = client.get("/api/decision/history?mine_id=DEMO_C").json()
    assert h["history"][0]["decision_id"] == rec["decision_id"]


def test_decision_review_validation(client):
    assert client.post("/api/decision/review", json={"decision_state": "YOLO"}).status_code == 400
    assert client.post("/api/decision/review", json={"decision_state": "REVIEW_REQUIRED", "action": "NUKE"}).status_code == 400
    assert client.post("/api/decision/review", json={}).status_code == 422


def test_ood_perturbation_does_not_produce_confident_recommendation():
    f = cs.decision_flip("DEMO_F", {}, {"equipment_availability": 0.2, "rainfall_7d_mm": 900.0, "blast_delay_h": 15.0})
    p = f["perturbed"]
    assert p["decision_state"] == "REVIEW_REQUIRED"
    assert p["next_target"] is None
    assert p["selected_portfolio"] is None or p["selection_status"] == "REVIEW_REQUIRED"
    assert any(r["code"] == "PRODUCTION_INPUTS_OUT_OF_DISTRIBUTION" for r in p["review_reasons"])


def test_flip_uses_frontend_condition_aliases(client):
    r = client.post("/api/decision/flip", json={
        "mine_id": "DEMO_F", "scenario": "normal", "actions": ["equipment_recovery", "schedule_adjustment", "blast_delay_reduction"],
        "baseline_conditions": {}, "perturbed_conditions": {"rainfall_mm": 140, "equipment_availability": 0.74, "blast_delay_hours": 3}})
    d = r.json()
    assert r.status_code == 200 and d["flipped"] is True
    assert {c["input"] for c in d["changed_inputs"]} == {"rainfall_7d_mm", "equipment_availability", "blast_delay_h"}
