"""Subsurface scenario engine (observed vs SIMULATED) and the REAL MOIL quarterly endpoint."""
import copy
import json

import pytest

from services import subsurface_service
from services.common import DATA_DIR, load_config
from tests.conftest import PROVENANCE_KEYS

FORBIDDEN = ("reserve confirmed", "confirmed reserve", "confirmed mineralisation", "confirmed mineralization",
             "reserve of", "tonnes of ore in", "moil telemetry")


def _first_target(client):
    return client.get("/api/exploration/targets").json()["targets"][0]["target_id"]


def test_subsurface_scenarios_endpoint(client):
    tid = _first_target(client)
    r = client.get(f"/api/exploration/targets/{tid}/subsurface-scenarios")
    assert r.status_code == 200
    d = r.json()
    assert PROVENANCE_KEYS <= set(d["provenance"]) and d["provenance"]["data_mode"] == "SIMULATED"
    sims = d["simulated_scenarios"]
    assert {s["scenario"] for s in sims} == set(load_config("exploration_config.json")["subsurface_fusion"]) - {"_note"}
    for s in sims:
        assert s["label"] == "SIMULATED — NOT OBSERVED"
        assert s["simulated_evidence_level"] <= 3 and "No reserve" in s["not_claimed"]
        assert all(b["provenance"] == "SIMULATED" for b in s["boreholes"])
    text = json.dumps(d).lower()
    assert not any(f in text for f in FORBIDDEN)


def test_negative_drilling_lowers_priority_and_positive_does_not(client):
    tid = _first_target(client)
    d = client.get(f"/api/exploration/targets/{tid}/subsurface-scenarios").json()
    by = {s["scenario"]: s for s in d["simulated_scenarios"]}
    assert by["NEGATIVE_DRILLING_RESULT"]["priority_change"] < 0
    assert by["POSITIVE_DRILLING_INTERSECTION"]["priority_change"] >= 0
    assert by["NO_SUBSURFACE_EVIDENCE"]["priority_change"] == 0
    assert by["AMBIGUOUS_DRILLING_RESULT"]["simulated_uncertainty"] == "HIGH"


def test_simulation_never_modifies_observed_targets(client):
    from services.exploration_service import get_service

    ex = get_service()
    before = copy.deepcopy(ex.targets)
    for t in before[:3]:
        client.get(f"/api/exploration/targets/{t['target_id']}/subsurface-scenarios")
    assert ex.targets == before
    stored = json.loads((DATA_DIR / "exploration_targets.json").read_text())["targets"]
    assert all(t["subsurface_status"] in ("UNAVAILABLE", "REPORTED_BLOCK_LEVEL") for t in stored)


def test_observed_evidence_is_real_government_only(client):
    for t in client.get("/api/exploration/targets").json()["targets"]:
        for o in t.get("observed_ground_evidence") or []:
            assert o["source_mode"] == "REAL_GOVERNMENT" and o["official_url"].startswith("https://")


@pytest.mark.parametrize("path,code", [("/api/exploration/targets/NOPE/subsurface-scenarios", 404),
                                       ("/api/exploration/targets/T01/subsurface-scenarios?scenario=METEOR", 400)])
def test_subsurface_errors(client, path, code):
    assert client.get(path).status_code == code


def test_single_scenario_filter(client):
    tid = _first_target(client)
    d = client.get(f"/api/exploration/targets/{tid}/subsurface-scenarios?scenario=positive_geochemical_support").json()
    assert [s["scenario"] for s in d["simulated_scenarios"]] == ["POSITIVE_GEOCHEMICAL_SUPPORT"]


def test_real_quarterly_endpoint(client):
    r = client.get("/api/production/real-quarterly?last_n=8")
    assert r.status_code == 200
    d = r.json()
    assert d["provenance"]["data_mode"] == "REAL_MOIL_PUBLIC" and len(d["quarters"]) == 8
    v = d["validation"]
    # the verdict must follow the untouched test metrics, whichever way they point
    assert v["model_beats_best_baseline"] == (v["selected_test"]["mae"] < v["best_baseline_test"]["mae"])
    assert ("does NOT beat" in v["verdict"]) != v["model_beats_best_baseline"]
    assert "No mine-level" in d["claims_not_made"]
    assert d["forecast"]["point_t"] and d["baseline_forecasts_t"]["seasonal_naive_x_yoy"] > 0


def test_real_production_selection_precedes_test():
    rep = json.loads((DATA_DIR.parent / "models" / "reports" / "real_production_validation.json").read_text())
    sel_end = int(rep["selection_window"].split("..")[1].strip()[2:6])      # "FY2019-20 .. FY2022-23"
    assert int(rep["test_window"][0][:4]) >= sel_end                        # test starts after selection ends
    assert rep["synthetic_augmentation_helps"] is False or rep["decision"]
