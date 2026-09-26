"""Live-satellite failure and cache-distance safety (network-free)."""

import pytest

from services.common import ApiError

INSIDE = (21.8161, 80.1837)


def test_live_disabled_uses_cache_within_limit(exploration_svc):
    d = exploration_svc.predict(*INSIDE)
    assert d["source_mode"] == "CACHED" and d["fallback_used"] is True
    assert d["cache_distance_km"] <= exploration_svc.cfg["cache_max_distance_km"]
    assert d["live_status"] == "LIVE_QUERY_DISABLED"


def test_far_from_cache_is_unavailable(client):
    r = client.post("/api/exploration/predict", json={"lat": 28.6, "lon": 77.2})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "PREDICTION_UNAVAILABLE"
    assert body["cache_distance_km"] > body["max_cache_distance_km"]


def test_just_outside_cache_limit(exploration_svc):
    # 0.03 degrees outside the study-area edge is > 2 km from any grid cell centre
    with pytest.raises(ApiError) as e:
        exploration_svc.predict(20.75 - 0.03, 79.5)
    assert e.value.code == "PREDICTION_UNAVAILABLE"


def test_simulated_live_failure_falls_back(client):
    d = client.post("/api/exploration/predict", json={"lat": INSIDE[0], "lon": INSIDE[1], "mode": "SIMULATE_LIVE_FAILURE"}).json()
    assert d["source_mode"] == "CACHED" and "simulated" in d["live_status"]


def test_live_timeout_falls_back(exploration_svc, monkeypatch):
    monkeypatch.setattr(exploration_svc, "live_enabled", lambda: True)

    def boom(lat, lon):
        raise TimeoutError()

    monkeypatch.setattr(exploration_svc, "_live_features", boom)
    d = exploration_svc.predict(*INSIDE)
    assert d["source_mode"] == "CACHED" and d["live_status"] == "LIVE_QUERY_TIMEOUT"


def test_live_error_falls_back(exploration_svc, monkeypatch):
    monkeypatch.setattr(exploration_svc, "live_enabled", lambda: True)
    monkeypatch.setattr(exploration_svc, "_live_features", lambda lat, lon: (_ for _ in ()).throw(RuntimeError("cloud")))
    d = exploration_svc.predict(*INSIDE)
    assert d["source_mode"] == "CACHED" and d["live_status"].startswith("LIVE_QUERY_FAILED")


def test_live_only_failure_is_503(exploration_svc, monkeypatch):
    monkeypatch.setattr(exploration_svc, "live_enabled", lambda: False)
    with pytest.raises(ApiError) as e:
        exploration_svc.predict(*INSIDE, mode="LIVE_ONLY")
    assert e.value.status == 503


def test_live_success_path(exploration_svc, monkeypatch):
    from tests.conftest import merged_feature_grid

    g = merged_feature_grid()
    row = g[(g["lat"].round(4) == 21.815) & (g["lon"].round(4) == 80.185)].iloc[0].to_dict()
    monkeypatch.setattr(exploration_svc, "live_enabled", lambda: True)
    monkeypatch.setattr(exploration_svc, "_live_features", lambda lat, lon: row)
    d = exploration_svc.predict(*INSIDE)
    assert d["source_mode"] == "LIVE_COORDINATE_QUERY" and d["fallback_used"] is False
    assert d["observation_window"] == "2024-01-01/2024-12-31"
    cached = exploration_svc.predict(*INSIDE, mode="CACHED_ONLY")
    assert d["prospectivity_rank"] == cached["prospectivity_rank"]  # same features -> same rank (train/infer parity)


def test_missing_cache(exploration_svc, monkeypatch):
    monkeypatch.setattr(exploration_svc, "grid_ok", None)
    with pytest.raises(ApiError) as e:
        exploration_svc.predict(*INSIDE)
    assert e.value.status == 503


def test_demo_e_supply_command(client):
    d = client.get("/api/supply-command?mine_id=DEMO_E").json()
    res = d["satellite_fallback_demo"]["results"]
    assert res[0]["source_mode"] == "CACHED" and res[0]["fallback_used"] is True
    assert res[1]["error"] == "PREDICTION_UNAVAILABLE"


def test_fallback_distance_fields(exploration_svc):
    d = exploration_svc.predict(*INSIDE)
    assert d["fallback_used"] is True
    assert d["fallback_distance_km"] == d["cache_distance_km"]
    assert d["max_supported_fallback_distance_km"] == exploration_svc.cfg["cache_max_distance_km"]


def test_fallback_exactly_at_threshold_is_allowed(exploration_svc, monkeypatch):
    lat, lon = 20.75 - 0.012, 79.505   # just south of the grid edge
    _, dist = exploration_svc.nearest_cell(lat, lon)
    monkeypatch.setitem(exploration_svc.cfg, "cache_max_distance_km", dist)
    d = exploration_svc.predict(lat, lon)
    assert d["source_mode"] == "CACHED" and d["fallback_distance_km"] == pytest.approx(dist, abs=1e-3)
    monkeypatch.setitem(exploration_svc.cfg, "cache_max_distance_km", dist - 0.01)
    with pytest.raises(ApiError) as e:
        exploration_svc.predict(lat, lon)
    assert e.value.code == "PREDICTION_UNAVAILABLE"


def test_invalid_coordinates(exploration_svc):
    for lat, lon in ((91, 80), (21, 181), (float("nan"), 80)):
        with pytest.raises(ApiError) as e:
            exploration_svc.predict(lat, lon)
        assert e.value.status == 400
