import json

from services.common import DATA_DIR

TARGET_FIELDS = {"target_id", "lat", "lon", "prospectivity_rank", "uncertainty", "applicability", "evidence_level",
                 "strategic_relevance", "exploration_priority", "status", "source_mode"}


def test_targets_list_schema(client):
    r = client.get("/api/exploration/targets")
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == len(d["targets"]) > 0
    for t in d["targets"]:
        assert TARGET_FIELDS <= set(t)
        assert 0 <= t["prospectivity_rank"] <= 100
        assert t["applicability"] in ("HIGH", "MODERATE", "LOW")
        assert t["uncertainty"] in ("LOW", "MODERATE", "HIGH")
        assert t["subsurface_status"] in ("UNAVAILABLE", "REPORTED_BLOCK_LEVEL")
        assert t["evidence_level"] <= 3  # no resource/reserve work exists, so L4 is never reached
        if t["evidence_level"] == 3:     # L3 only from an official REPORTED drilling outcome, never from simulation
            assert t["subsurface_status"] == "REPORTED_BLOCK_LEVEL"
            assert any(o["evidence_class"] == "DRILLING_INTERSECTION_REPORTED" and o["source_mode"] == "REAL_GOVERNMENT"
                       for o in t["observed_ground_evidence"])
        assert t["status"] in ("EXPLORATION_TARGET", "REVIEW_REQUIRED")
    elig = [t["eligible_for_contingency"] for t in d["targets"]]
    assert elig == sorted(elig, reverse=True)                      # gated targets after eligible ones
    for group in (True, False):
        pr = [t["exploration_priority"] for t in d["targets"] if t["eligible_for_contingency"] is group]
        assert pr == sorted(pr, reverse=True)
    assert "provenance" in d


def test_no_reserve_or_probability_claims(client):
    text = client.get("/api/exploration/targets").text.lower()
    assert "probability_of_reserve" not in text
    assert "reserve_tonnes" not in text and "ore_tonnes" not in text


def test_target_detail(client):
    tid = client.get("/api/exploration/targets").json()["targets"][0]["target_id"]
    d = client.get(f"/api/exploration/targets/{tid}").json()
    for k in ("surface_evidence", "geological_evidence", "subsurface_evidence", "why_this_target",
              "why_this_target_now", "next_evidence", "provenance", "evidence_level"):
        assert k in d
    assert d["subsurface_evidence"]["status"] == "UNAVAILABLE"
    codes = {r["code"] for r in d["why_this_target"]}
    assert "NO_SUBSURFACE_DATA" in codes


def test_target_not_found(client):
    r = client.get("/api/exploration/targets/T999")
    assert r.status_code == 404 and r.json()["error"] == "TARGET_NOT_FOUND"


def test_targets_are_clusters_not_pixels():
    doc = json.loads((DATA_DIR / "exploration_targets.json").read_text())
    for t in doc["targets"]:
        assert t["n_cells"] >= 4
        assert t["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_priority_depends_on_strategic_state(exploration_svc):
    none = exploration_svc.prioritise(None)
    assert all(t["strategic_relevance"] == "LOW" and t["strategic_relevance_score"] == 0 for t in none)
    active = exploration_svc.prioritise({"active": True, "severity": 1.0, "expected_residual_gap_tonnes": 1000,
                                         "worst_case_residual_gap_tonnes": 2000})
    assert any(t["strategic_relevance"] == "HIGH" for t in active)
    # closer targets gain relative priority when a strategic gap is active
    assert [t["target_id"] for t in none] != [t["target_id"] for t in active]


def test_priority_without_contingency_uses_geology_evidence_only(exploration_svc):
    out = exploration_svc.prioritise(None)
    for t in out:
        assert set(t["priority_components"]) == {"prospectivity", "evidence_applicability"}
        assert t["priority_mode"] == "GEOLOGICAL_EVIDENCE_ONLY"
        assert "development_readiness" not in t["priority_components"]      # no proxy is scored
    act = exploration_svc.prioritise({"active": True, "severity": 1.0, "expected_residual_gap_tonnes": 1,
                                      "worst_case_residual_gap_tonnes": 1})
    assert all("strategic_relevance" in t["priority_components"] for t in act)
    # geological prospectivity itself does not change with the supply state
    p0 = {t["target_id"]: t["prospectivity_rank"] for t in out}
    assert all(p0[t["target_id"]] == t["prospectivity_rank"] for t in act)


def test_hard_gates_precede_priority(exploration_svc, monkeypatch):
    base = exploration_svc.targets
    low = dict(base[0], target_id="TLOW", applicability="LOW", prospectivity_rank=100.0)
    l0 = dict(base[0], target_id="TL0", evidence_level=0, prospectivity_rank=100.0)
    monkeypatch.setattr(exploration_svc, "targets", base + [low, l0])
    ranked = exploration_svc.prioritise(None)
    blocked = {t["target_id"]: t for t in ranked if not t["eligible_for_contingency"]}
    assert "TLOW" in blocked and "TL0" in blocked
    assert any("applicability" in r for r in blocked["TLOW"]["gate_blocked_reasons"])
    assert any("evidence level" in r for r in blocked["TL0"]["gate_blocked_reasons"])
    first_blocked = min(i for i, t in enumerate(ranked) if not t["eligible_for_contingency"])
    assert all(not t["eligible_for_contingency"] for t in ranked[first_blocked:])   # eligible targets rank first


def test_grid_endpoint(client):
    d = client.get("/api/exploration/grid?stride=10").json()
    assert d["count"] > 0
    assert all(0 <= p["prospectivity_rank"] <= 100 for p in d["points"])


def test_predict_cached_inside_area(client):
    d = client.post("/api/exploration/predict", json={"lat": 21.8161, "lon": 80.1837}).json()
    for k in ("query_lat", "query_lon", "prospectivity_rank", "applicability", "uncertainty", "evidence_level",
              "source_mode", "observation_window", "fallback_used", "coverage_warning", "status"):
        assert k in d
    assert d["source_mode"] == "CACHED" and d["fallback_used"] is True
    assert d["observation_window"] == "2024-01-01/2024-12-31"
    assert d["subsurface_status"] == "UNAVAILABLE"


def test_predict_invalid_inputs(client):
    assert client.post("/api/exploration/predict", json={"lat": 95, "lon": 80}).status_code == 400
    assert client.post("/api/exploration/predict", json={"lat": 21, "lon": 80, "mode": "MAGIC"}).status_code == 400
    r = client.post("/api/exploration/predict", json={"lat": "north"})
    assert r.status_code == 422 and r.json()["error"] == "VALIDATION_ERROR"


def test_no_label_derived_or_proximity_features(exploration_svc):
    """Leakage guard: the model must not learn 'near known positives = positive'."""
    leaky = ("dist", "distance", "proximity", "near", "mrds", "occurrence", "mine", "label", "known")
    feats = [f.lower() for f in exploration_svc.features]
    # lineament distance is a structural geology feature (NRSC map), not label-derived; it is allowed if present
    assert not [f for f in feats if any(k in f for k in leaky) and not f.startswith("lineament_")]
