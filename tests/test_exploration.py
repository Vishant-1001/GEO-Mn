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
        assert t["subsurface_status"] == "UNAVAILABLE"
        assert t["evidence_level"] <= 2  # no drilling/assay/reserve data exist, so never promoted beyond 2
        assert t["status"] in ("EXPLORATION_TARGET", "REVIEW_REQUIRED")
    pr = [t["exploration_priority"] for t in d["targets"]]
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


def test_priority_weights_renormalise_when_component_missing(exploration_svc, monkeypatch):
    t0 = dict(exploration_svc.targets[0])
    t0["development_readiness"] = None
    monkeypatch.setattr(exploration_svc, "targets", [t0])
    out = exploration_svc.prioritise(None)[0]
    assert out["priority_renormalised"] is True
    assert abs(sum(out["priority_weights_used"].values()) - 1.0) < 1e-6
    assert "development_readiness" not in out["priority_weights_used"]


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
