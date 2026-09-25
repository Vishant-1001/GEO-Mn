"""Static regression checks on the frontend source (the UI must not invent semantics)."""

import re

from services.common import ROOT

JS = (ROOT / "app.js").read_text()
HTML = (ROOT / "index.html").read_text()


def test_no_stale_reserve_or_probability_paths():
    assert "/reserve_grid" not in JS
    assert not re.search(r"pick\([^)]*'probability'", JS)
    for txt in (JS, HTML):
        assert "Deposit Probability" not in txt and "reserve probability" not in txt.lower()


def test_no_nominal_coverage_claim():
    for txt in (JS, HTML):
        assert "80% of forecast outcomes fall inside" not in txt
    # interval wording is built from backend validation metadata
    assert "quantile_validation" in JS and "observed_coverage" in JS and "nominal_coverage" in JS


def test_surface_uses_exploration_grid_rank():
    assert "/api/exploration/grid" in JS
    assert "prospectivity_rank" in JS


def test_demo_states_come_from_backend():
    assert "/api/demo/scenarios" in JS
    assert 'id="demoSelect"' in HTML
    assert not re.search(r"['\"]DEMO_[A-F]['\"]", JS)  # no hard-coded demo list


def test_flip_uses_backend_endpoint():
    assert "/api/decision/flip" in JS
    assert "flipped" in JS and "changed_inputs" in JS


def test_no_client_side_decision_logic():
    # the frontend never assigns a decision state itself
    assert not re.search(r"decision_state\s*[:=]\s*['\"](OPERATIONAL|REVIEW)", JS)
    assert "real-time" not in HTML.lower().replace("not real-time", "")


def test_thermal_layer_labelled_as_context():
    assert "MODIS LST &mdash; context layer" in HTML
