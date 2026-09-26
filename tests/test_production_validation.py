import json

import numpy as np
import pandas as pd
import pytest

from ml.production_features import features_at, load_daily, period_conditions
from services.common import DATA_DIR, MODELS_DIR


@pytest.fixture(scope="module")
def report():
    return json.loads((MODELS_DIR / "reports" / "production_validation.json").read_text())


def test_features_use_only_past_data():
    daily = load_daily(DATA_DIR / "production_history.csv")
    origin = pd.Timestamp("2025-06-06")
    f1 = features_at(daily, origin)
    tampered = daily.copy()
    tampered.loc[origin:, ["actual_tonnes", "rainfall_mm", "equipment_availability"]] = 0.0
    assert features_at(tampered, origin) == f1  # no leakage of the future into origin features


def test_period_conditions_cover_the_forecast_period_only():
    daily = load_daily(DATA_DIR / "production_history.csv")
    c = period_conditions(daily, "2025-06-06")
    win = daily.loc["2025-06-06":"2025-06-12"]
    assert c["rainfall_7d_mm"] == pytest.approx(win["rainfall_mm"].sum())


def test_backtest_is_chronological_and_purged():
    bt = pd.read_csv(DATA_DIR / "production_backtest_forecasts.csv")
    origins = pd.to_datetime(bt["origin"])
    assert origins.is_monotonic_increasing
    for _, g in bt.groupby(["window", "fold"]):
        # expanding window: training size never shrinks
        assert g["train_rows"].nunique() == 1
    assert bt.groupby(["window", "fold"])["train_rows"].first().groupby(level=0).is_monotonic_increasing.all()


def test_metrics_reported_and_baseline_compared(report):
    bt = report["backtest"]
    for k in ("mae", "rmse", "r2"):
        assert np.isfinite(bt["model_p50"][k])
    for k in ("baseline_previous_period", "baseline_moving_average_4"):
        assert np.isfinite(bt[k]["mae"])
    best = min(bt["baseline_previous_period"]["mae"], bt["baseline_moving_average_4"]["mae"])
    assert bt["model_beats_best_baseline_mae"] == (bt["model_p50"]["mae"] < best)
    assert set(bt["pinball_loss"]) == {"p10", "p50", "p90"}
    assert 0 <= bt["p10_p90_coverage"] <= 1


def test_quantile_flag_follows_rule(report):
    bt = report["backtest"]
    ok = abs(bt["p10_p90_coverage"] - 0.8) <= 0.08 and all(
        abs(bt["quantile_hit_rate"][k] - q) <= 0.08 for k, q in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)))
    assert report["quantiles_validated"] == ok


def test_selection_and_test_windows_do_not_overlap(report):
    assert report["selection_window"]["end"] <= report["backtest"]["test_start"]


def test_trust_endpoint_matches_report(client, report):
    d = client.get("/api/trust/production").json()
    assert d["mae"] == pytest.approx(report["backtest"]["model_p50"]["mae"], abs=0.1)
    assert d["quantiles_validated"] == report["quantiles_validated"]
    assert any("synthetic" in n.lower() for n in d["notes"])


def test_synthetic_generator_is_deterministic():
    from ml.generate_operations import generate

    a, b = generate()[0], generate()[0]
    pd.testing.assert_frame_equal(a, b)
    stored = pd.read_csv(DATA_DIR / "production_history.csv")
    assert np.allclose(stored["actual_tonnes"], a["actual_tonnes"], atol=0.051)


def test_manifest_documents_production_model():
    man = json.loads((MODELS_DIR / "model_manifest.json").read_text())["production"]
    for k in ("model_version", "model_type", "features", "training_period", "validation_method", "validation_metrics",
              "data_provenance", "uncertainty_method", "applicability_method", "limitations"):
        assert k in man


def test_calibration_window_precedes_test_and_offsets_are_frozen(report):
    from ml.evaluate_production import calibration_offsets

    bt = pd.read_csv(DATA_DIR / "production_backtest_forecasts.csv")
    cal, test = bt[bt["window"] == "calibration"], bt[bt["window"] == "test"]
    test_start = pd.Timestamp(report["protocol"]["test_window"][0])
    # every calibration outcome period ends before the first test origin
    assert pd.to_datetime(cal["period_end"]).max() < test_start
    assert report["protocol"]["calibration_uses_test_data"] is False
    # offsets stored with the model are exactly those recomputed from calibration rows only
    offs = calibration_offsets(cal["actual"].to_numpy(), cal[["raw_p10", "raw_p50", "raw_p90"]].to_numpy())
    stored = json.loads((MODELS_DIR / "production_calibration.json").read_text())["offsets_relative_to_raw_p50"]
    for k in ("p10", "p50", "p90"):
        assert offs[k] == pytest.approx(stored[k], abs=1e-4)  # CSV stores tonnes to 0.1 t
    # test predictions = raw + frozen offsets (no re-estimation from test outcomes)
    raw = test[["raw_p10", "raw_p50", "raw_p90"]].to_numpy()
    adj = np.sort(np.clip(raw + np.column_stack([stored[k] * raw[:, 1] for k in ("p10", "p50", "p90")]), 0, None), axis=1)
    assert np.allclose(adj, test[["p10", "p50", "p90"]].to_numpy(), atol=0.11)


def test_final_artifact_is_declared_post_evaluation_refit(report):
    man = json.loads((MODELS_DIR / "model_manifest.json").read_text())["production"]
    proto = man["validation_protocol"]
    assert proto["final_artifact_is_post_evaluation_refit"] is True
    sel, cal, test = proto["selection_window"], proto["calibration_window"], proto["test_window"]
    assert sel[1] <= cal[0] and cal[1] <= test[0]
