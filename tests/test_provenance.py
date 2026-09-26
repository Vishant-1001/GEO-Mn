import json

from services.common import MODELS_DIR

from tests.conftest import PROVENANCE_KEYS


def _has(p):
    assert PROVENANCE_KEYS <= set(p), set(p)
    assert "real-time" not in json.dumps(p).lower() and "realtime" not in json.dumps(p).lower()


def test_major_responses_carry_provenance(client):
    _has(client.get("/api/supply-command?mine_id=DEMO_MINE").json()["provenance"])
    _has(client.get("/api/exploration/targets").json()["provenance"])
    tid = client.get("/api/exploration/targets").json()["targets"][0]["target_id"]
    _has(client.get(f"/api/exploration/targets/{tid}").json()["provenance"])
    _has(client.post("/api/exploration/predict", json={"lat": 21.8, "lon": 80.2}).json()["provenance"])
    _has(client.post("/api/production/forecast", json={"mine_id": "DEMO_MINE"}).json()["provenance"])
    _has(client.get("/api/production/history").json()["provenance"])
    _has(client.get("/api/production/reconciliation").json()["provenance"])
    _has(client.post("/api/recovery/evaluate", json={}).json()["provenance"])
    _has(client.post("/api/contingency/evaluate", json={}).json()["provenance"])
    _has(client.get("/api/trust/exploration").json()["provenance"])
    _has(client.get("/api/trust/production").json()["provenance"])


def test_data_modes(client):
    assert client.get("/api/production/history").json()["provenance"]["data_mode"] == "SYNTHETIC"
    assert client.post("/api/recovery/evaluate", json={}).json()["provenance"]["data_mode"] == "SIMULATED"
    p = client.post("/api/exploration/predict", json={"lat": 21.8, "lon": 80.2}).json()
    assert p["source_mode"] == "CACHED" and p["provenance"]["fallback_used"] is True
    assert client.get("/api/exploration/targets").json()["provenance"]["data_mode"] == "CACHED"


def test_provenance_catalogue(client):
    d = client.get("/api/trust/provenance").json()
    modes = {k: v["mode"] for k, v in d["sources"].items()}
    assert modes["operations"] == "SYNTHETIC"
    assert modes["reserves"] == "UNAVAILABLE"
    assert modes["subsurface_observed"] == "REAL_GOVERNMENT" and modes["next_evidence_sensitivity"] == "SIMULATED"
    assert modes["recovery_scenarios"] == "SIMULATED" and modes["demo_states"] == "SIMULATED"
    assert modes["weather_imd"] == "REAL_GOVERNMENT" and modes["production_moil"] == "REAL_MOIL_PUBLIC"
    assert modes["geomorphology_lineaments"] == "REAL_GOVERNMENT" and modes["exploration_blocks"] == "REAL_GOVERNMENT"
    assert modes["sentinel2"] == "REAL_PUBLIC" and modes["exploration_labels"] == "REAL_PUBLIC"
    assert set(d["modes"]) >= {"REAL_GOVERNMENT", "REAL_PUBLIC", "REAL_MOIL_PUBLIC", "REAL_DERIVED", "SYNTHETIC",
                               "SIMULATED", "CACHED"}
    assert d["model_versions"]["exploration"] and d["model_versions"]["production"]


def test_trust_exploration_reports_only_computed_metrics(client):
    d = client.get("/api/trust/exploration").json()
    rep = json.loads(open(MODELS_DIR / "reports" / "exploration_validation.json").read())
    assert d["roc_auc"] == round(rep["spatial_block_cv"]["roc_auc"], 3)
    assert d["pr_auc_prevalence_baseline"] < d["pr_auc"]
    assert "NOT CALIBRATED" in d["calibration"]
    assert d["spatial_validation"]


def test_manifest_sections(client):
    m = client.get("/api/model/manifest").json()
    for sec in ("exploration", "production"):
        for k in ("model_version", "model_type", "features", "validation_method", "validation_metrics",
                  "data_provenance", "uncertainty_method", "applicability_method", "limitations"):
            assert k in m[sec], (sec, k)
    assert m["exploration"]["observation_window"] == "2024-01-01/2024-12-31"
