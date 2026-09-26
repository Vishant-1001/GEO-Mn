"""Applicability / OOD behaviour of the exploration engine (network-free: live features are mocked)."""

from tests.conftest import merged_feature_grid


def _typical_features(exploration_svc):
    g = merged_feature_grid()[exploration_svc.features].dropna()
    return g.median().to_dict()


def test_typical_features_are_applicable(exploration_svc):
    s = exploration_svc.score_features(_typical_features(exploration_svc))
    assert s["feature_applicability"] == "HIGH"
    assert 0 <= s["prospectivity_rank"] <= 100


def test_extreme_features_are_low_applicability(exploration_svc):
    f = _typical_features(exploration_svc)
    f.update({"elevation": 4800.0, "slope": 45.0, "LST_Day_K": 250.0, "NDVI": -0.6})
    assert exploration_svc.score_features(f)["feature_applicability"] == "LOW"


def test_outside_study_area_is_low_applicability(exploration_svc, monkeypatch):
    feats = _typical_features(exploration_svc)
    monkeypatch.setattr(exploration_svc, "live_enabled", lambda: True)
    monkeypatch.setattr(exploration_svc, "_live_features", lambda lat, lon: feats)
    d = exploration_svc.predict(26.0, 86.0)
    assert d["source_mode"] == "LIVE_COORDINATE_QUERY"
    assert d["in_study_area"] is False
    assert d["applicability"] == "LOW"
    assert d["coverage_warning"]


def test_high_rank_with_low_applicability_requires_review(exploration_svc, monkeypatch):
    monkeypatch.setattr(exploration_svc, "live_enabled", lambda: True)
    monkeypatch.setattr(exploration_svc, "_live_features", lambda lat, lon: {"x": 1})
    monkeypatch.setattr(exploration_svc, "score_features", lambda f: {
        "prospectivity_rank": 99.0, "rank_sd": 2.0, "uncertainty": "LOW", "feature_applicability": "LOW"})
    d = exploration_svc.predict(21.8, 80.2)
    assert d["status"] == "REVIEW_REQUIRED"


def test_unmapped_geology_is_allowed_but_eo_is_required(exploration_svc):
    f = _typical_features(exploration_svc)
    geo = [k for k in f if k.startswith(("geom_", "lineament_"))]
    if not geo:
        return
    s = exploration_svc.score_features({**f, **{k: float("nan") for k in geo}})   # outside NRSC MP/MH layers
    assert 0 <= s["prospectivity_rank"] <= 100


def test_incomplete_features_are_not_imputed(exploration_svc):
    f = _typical_features(exploration_svc)
    f["LST_Day_K"] = float("nan")
    try:
        exploration_svc.score_features(f)
    except ValueError:
        return
    raise AssertionError("missing features must not be silently imputed")
