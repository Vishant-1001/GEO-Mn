import pytest


def _fc(client, **body):
    return client.post("/api/production/forecast", json={"mine_id": "DEMO_MINE", **body})


def test_forecast_schema(client):
    r = _fc(client, forecast_origin="2026-09-25", horizon_days=7)
    assert r.status_code == 200
    d = r.json()
    for k in ("mine_id", "forecast_horizon", "target_tonnes", "p10_tonnes", "p50_tonnes", "p90_tonnes",
              "gap_p50_tonnes", "risk_state", "risk_policy_version", "drivers", "provenance"):
        assert k in d
    assert d["p10_tonnes"] <= d["p50_tonnes"] <= d["p90_tonnes"]
    assert d["gap_p50_tonnes"] == pytest.approx(max(0.0, d["target_tonnes"] - d["p50_tonnes"]), abs=0.2)
    assert d["forecast_horizon"] == "next_period_7_days"
    assert d["period_start"] == "2026-09-25" and d["period_end"] == "2026-10-01"


def test_forecast_is_deterministic(client):
    assert _fc(client).json()["p50_tonnes"] == _fc(client).json()["p50_tonnes"]


def test_unsupported_horizon(client):
    r = _fc(client, horizon_days=30)
    assert r.status_code == 400 and r.json()["error"] == "UNSUPPORTED_HORIZON"


def test_origin_out_of_range(client):
    assert _fc(client, forecast_origin="2027-01-01").json()["error"] == "ORIGIN_OUT_OF_RANGE"
    assert _fc(client, forecast_origin="2021-01-05").json()["error"] == "INSUFFICIENT_HISTORY"
    assert _fc(client, forecast_origin="not-a-date").status_code == 400


def test_retrospective_forecast_reports_actual(client):
    d = _fc(client, forecast_origin="2026-06-05").json()
    assert d["forecast_type"] == "RETROSPECTIVE" and "actual_tonnes" in d


def test_invalid_state_rejected(client):
    r = _fc(client, conditions={"equipment_availability": 1.5})
    assert r.status_code == 400 and r.json()["error"] == "INVALID_INPUT"
    assert _fc(client, conditions={"truck_count": -2}).status_code == 400
    assert _fc(client, target_tonnes=-5).status_code == 400


def test_simulated_override_is_labelled(client):
    d = _fc(client, conditions={"equipment_availability": 0.7}).json()
    assert d["status"] == "SIMULATED_SCENARIO"
    assert d["provenance"]["data_mode"] == "SIMULATED"


def test_worse_conditions_do_not_increase_forecast(client):
    base = _fc(client).json()["p50_tonnes"]
    worse = _fc(client, conditions={"equipment_availability": 0.6, "blast_delay_h": 4, "rainfall_7d_mm": 200}).json()["p50_tonnes"]
    assert worse < base


def test_model_contributions_reconcile_to_p50(client):
    d = _fc(client).json()
    mc = d["model_contributions"]
    assert mc["status"] == "AVAILABLE"
    total = mc["base_value_tonnes"] + sum(i["contribution_tonnes"] for i in mc["items"]) + mc["calibration_adjustment_tonnes"]
    assert total == pytest.approx(d["p50_tonnes"], abs=2.0)
    assert "not causal" in mc["method"]
    assert all(i["contribution_tonnes"] < 0 for i in d["drivers"])


def test_risk_policy(production_svc):
    r = production_svc.risk(10000, 9900, 9950)
    assert r["risk_state"] == "LOW" and r["risk_policy_version"]
    assert production_svc.risk(10000, 5000, 7000)["risk_state"] == "CRITICAL"
    assert production_svc.risk(0, 0, 0)["risk_state"] == "NO_TARGET"


def test_history(client):
    d = client.get("/api/production/history?mine_id=DEMO_MINE&days=90").json()
    assert len(d["rows"]) == 90
    assert {"date", "actual_tonnes", "target_tonnes"} <= set(d["rows"][0])
    assert d["provenance"]["data_mode"] == "SYNTHETIC"
    assert client.get("/api/production/history?days=0").status_code == 400


def test_reconciliation(client):
    d = client.get("/api/production/reconciliation?mine_id=DEMO_MINE").json()
    assert d["status"] == "AVAILABLE"
    assert {"mae", "bias", "latest_error_tonnes"} <= set(d["summary"])
    row = d["rows"][-1]
    assert row["error_tonnes"] == pytest.approx(row["forecast_tonnes"] - row["actual_tonnes"], abs=0.2)
    assert "OUT-OF-SAMPLE" in d["basis"]


def test_reconciliation_not_available(client, production_svc, monkeypatch):
    monkeypatch.setattr(production_svc, "backtest", None)
    d = client.get("/api/production/reconciliation").json()
    assert d["status"] == "NOT_AVAILABLE" and d["rows"] == []


def test_missing_model_returns_503(client, production_svc, monkeypatch):
    monkeypatch.setattr(production_svc, "models", {})
    r = _fc(client)
    assert r.status_code == 503 and r.json()["error"] == "MODEL_UNAVAILABLE"
