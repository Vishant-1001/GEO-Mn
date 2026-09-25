"""Train the P10/P50/P90 production forecast models and validate them chronologically.

Steps
  1. Build origin-level features from data/production_history.csv (daily).
  Windows are fixed in advance (origins, 7-day periods):
     TRAIN        first origin .. (expanding; each fold trains only on earlier origins)
     SELECTION    2022-07-01 .. 2023-06-30  choose the configuration (lowest raw P50 MAE)
     CALIBRATION  2023-07-01 .. 2024-06-30  estimate quantile offsets once, then FREEZE them
     TEST         2024-07-01 .. end         untouched evaluation with the frozen offsets
  2. Rolling-origin backtest of each candidate on SELECTION only.
  3. Out-of-sample predictions of the selected configuration on CALIBRATION -> offsets.
  4. TEST evaluation with those frozen offsets; baselines on the same origins;
     quantile validity check (coverage / hit rates).
  5. Deployable artifact = POST-EVALUATION REFIT of the selected configuration on
     every complete origin, carrying the frozen calibration offsets. The reported
     metrics describe the backtest procedure, not this refit.

Run:  python -m ml.train_production
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.common import CONFIG_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, SEED, load_json, save_json
from ml.evaluate_production import (
    CANDIDATES, QKEYS, QUANTILES, QuantileForecaster, backtest_predictions, calibration_offsets,
    quantiles_validated, summarise,
)
from ml.production_features import (
    FEATURE_COLUMNS, FEATURE_DOCS, MONOTONIC, PERIOD_DAYS, SCENARIO_FEATURES,
    build_training_frame, load_daily, training_matrix, weekly_origins,
)

SELECTION_START = "2022-07-01"
CALIBRATION_START = "2023-07-01"
TEST_START = "2024-07-01"
MODEL_VERSION = "production-qgbm-1.0"


def main():
    warnings.filterwarnings("ignore")
    demo = load_json(CONFIG_DIR / "demo_config.json")
    policy = load_json(CONFIG_DIR / "risk_policy.json")
    daily = load_daily(DATA_DIR / "production_history.csv")
    frame = build_training_frame(daily, stride_days=1)
    origins = weekly_origins(daily, demo["forecast_origin"])
    tol = policy["shortfall_event_tolerance"]
    print(f"training frame: {len(frame)} origins, {frame['origin'].min().date()} .. {frame['origin'].max().date()}")

    sel_origins = origins[(origins >= pd.Timestamp(SELECTION_START)) & (origins < pd.Timestamp(CALIBRATION_START))]
    cal_origins = origins[(origins >= pd.Timestamp(CALIBRATION_START)) & (origins < pd.Timestamp(TEST_START))]
    test_origins = origins[origins >= pd.Timestamp(TEST_START)]
    method = ("rolling-origin expanding-window backtest, 13-period folds, purged by one period; configuration selected on "
              f"{SELECTION_START}..{CALIBRATION_START}, quantile offsets calibrated on {CALIBRATION_START}..{TEST_START} "
              f"and frozen, metrics reported on origins >= {TEST_START}")

    selection = {}
    for name, cand in CANDIDATES.items():
        bt, nf = backtest_predictions(frame, sel_origins, cand)
        res = summarise(bt, tol, "selection window", nf)
        selection[name] = {"p50_mae_raw": float(np.mean(np.abs(bt["actual"] - bt["raw_p50"]))),
                           "baseline_previous_period_mae": res["baseline_previous_period"]["mae"],
                           "baseline_moving_average_4_mae": res["baseline_moving_average_4"]["mae"],
                           "p10_p90_coverage_raw": res["p10_p90_coverage_before_recalibration"]}
        print(f"[selection:{name}] raw P50 MAE {selection[name]['p50_mae_raw']:.0f}  naive "
              f"{selection[name]['baseline_previous_period_mae']:.0f}  raw coverage {selection[name]['p10_p90_coverage_raw']:.2f}")
    selected = min(selection, key=lambda k: selection[k]["p50_mae_raw"])
    cand = CANDIDATES[selected]
    print(f"selected: {selected}")

    bt_cal, _ = backtest_predictions(frame, cal_origins, cand)
    offsets = calibration_offsets(bt_cal["actual"].to_numpy(), bt_cal[["raw_p10", "raw_p50", "raw_p90"]].to_numpy())
    print(f"calibration offsets (from {len(bt_cal)} periods): {offsets}")

    bt_test, nf = backtest_predictions(frame, test_origins, cand, offsets=offsets)
    res = summarise(bt_test, tol, method, nf)
    validated, rule = quantiles_validated(res)
    print(f"[test] P50 MAE {res['model_p50']['mae']:.0f}  naive {res['baseline_previous_period']['mae']:.0f}  "
          f"MA4 {res['baseline_moving_average_4']['mae']:.0f}  coverage {res['p10_p90_coverage']:.2f} "
          f"(raw {res['p10_p90_coverage_before_recalibration']:.2f})  hits {res['quantile_hit_rate']}  validated {validated}")

    all_bt = pd.concat([bt_cal.assign(window="calibration"), bt_test.assign(window="test")], ignore_index=True)
    model = QuantileForecaster(cand["ratio"], cand["params"], True).fit(training_matrix(frame), frame["y"])
    for k in QKEYS:
        joblib.dump(model.models[k], MODELS_DIR / f"production_{k}.pkl")
    joblib.dump(list(FEATURE_COLUMNS), MODELS_DIR / "production_feature_columns.pkl")
    protocol = {
        "training_window": {"first_origin": str(frame["origin"].min().date()),
                            "rule": "expanding; each backtest fold trains only on origins whose outcome ends before the fold"},
        "selection_window": [SELECTION_START, CALIBRATION_START],
        "calibration_window": [CALIBRATION_START, TEST_START],
        "test_window": [res["test_start"], res["test_end"]],
        "calibration_source": f"out-of-sample raw quantile errors of the selected configuration on {len(bt_cal)} calibration periods",
        "calibration_uses_test_data": False,
        "final_artifact": ("POST-EVALUATION REFIT: selected configuration refit on all complete origins "
                           f"(through {frame['origin'].max().date()}) with the frozen calibration offsets. "
                           "Reported test metrics come from the backtest, not from this refit."),
        "final_artifact_is_post_evaluation_refit": True,
    }
    save_json(MODELS_DIR / "production_calibration.json", {
        "model_version": MODEL_VERSION, "configuration": selected, "ratio_target": cand["ratio"],
        "ratio_base_feature": "prod_roll_mean_4" if cand["ratio"] else None, "params": cand["params"],
        "offsets_relative_to_raw_p50": offsets,
        "offsets_estimated_from": protocol["calibration_source"],
        "calibration_window": protocol["calibration_window"],
    })
    mono, bt = True, all_bt

    S = training_matrix(frame)[SCENARIO_FEATURES]
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(S)
    scores = iso.score_samples(S)
    reference = {
        "isolation_forest": iso,
        "features": list(SCENARIO_FEATURES),
        "score_q01": float(np.quantile(scores, 0.01)),
        "score_q05": float(np.quantile(scores, 0.05)),
        "ranges": {c: {"min": float(S[c].min()), "max": float(S[c].max()),
                       "p01": float(S[c].quantile(0.01)), "p99": float(S[c].quantile(0.99))} for c in SCENARIO_FEATURES},
    }
    joblib.dump(reference, MODELS_DIR / "production_train_reference.pkl", compress=3)

    shap_ok = False
    try:
        import shap

        shap.TreeExplainer(model.models["p50"]).shap_values(frame[FEATURE_COLUMNS].iloc[:5])
        shap_ok = True
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"SHAP unavailable for this model: {e}")

    bt.to_csv(DATA_DIR / "production_backtest_forecasts.csv", index=False, float_format="%.1f")
    report = {
        "model_version": MODEL_VERSION,
        "selected_configuration": selected,
        "protocol": protocol,
        "selection_window": {"start": SELECTION_START, "end": CALIBRATION_START, "candidates": selection},
        "backtest": res,
        "recalibration_offsets": offsets,
        "quantiles_validated": validated,
        "quantile_validation_rule": rule,
        "shap_supported": shap_ok,
        "data_mode": "SYNTHETIC",
        "caveat": "Operational records are synthetic (seed 42); metrics demonstrate pipeline behaviour, not real MOIL accuracy.",
    }
    save_json(REPORTS_DIR / "production_validation.json", report)

    manifest_path = MODELS_DIR / "model_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    manifest["production"] = {
        "model_version": MODEL_VERSION,
        "model_type": ("3 x sklearn HistGradientBoostingRegressor(loss='quantile') for q = 0.10, 0.50, 0.90 "
                       "with monotonic constraints"
                       + ("; target = period tonnes / prod_roll_mean_4 (ratio formulation)" if cand["ratio"] else "")),
        "configuration": selected,
        "hyperparameters": cand["params"],
        "monotonic_constraints": {k: v for k, v in MONOTONIC.items()} if mono else None,
        "target": f"total tonnes produced in the {PERIOD_DAYS}-day period starting at the forecast origin",
        "forecast_procedure": ("conditional model of period production given period operating conditions + origin-known lags; "
                               "forecasts assume the trailing 7-day state persists (persistence assumption) unless a "
                               "SIMULATED scenario overrides it; backtests use exactly these persistence inputs"),
        "forecast_horizon": f"next production period ({PERIOD_DAYS} days)",
        "temporal_granularity": {"inputs": "daily records aggregated to 7-day periods / trailing 7-day windows",
                                 "forecast": "7-day period total"},
        "features": [{"name": c, "unit": FEATURE_DOCS[c][0], "definition": FEATURE_DOCS[c][1]} for c in FEATURE_COLUMNS],
        "missing_value_rule": "forecast refused (HTTP 503/400) if fewer than 84 days of history precede the origin; no imputation",
        "training_period": {"first_origin": str(frame["origin"].min().date()), "last_origin": str(frame["origin"].max().date()),
                            "origins": int(len(frame))},
        "validation_protocol": protocol,
        "validation_method": res["method"],
        "validation_metrics": {
            "p50": res["model_p50"],
            "baseline_previous_period": res["baseline_previous_period"],
            "baseline_moving_average_4": res["baseline_moving_average_4"],
            "pinball_loss": res["pinball_loss"],
            "p10_p90_coverage": res["p10_p90_coverage"],
            "p10_p90_coverage_before_recalibration": res["p10_p90_coverage_before_recalibration"],
            "model_beats_best_baseline_mae": res["model_beats_best_baseline_mae"],
            "mae_improvement_vs_best_baseline_pct": res["mae_improvement_vs_best_baseline_pct"],
            "quantile_hit_rate": res["quantile_hit_rate"],
            "shortfall_classification": res["shortfall_classification"],
            "test_window": [res["test_start"], res["test_end"]],
            "n_test_periods": res["n_test_periods"],
        },
        "quantiles_validated": validated,
        "uncertainty_method": ("direct quantile regression (P10/P50/P90) + additive offsets (relative to P50) estimated once on "
                               "the calibration window and frozen; crossing repaired by sorting; coverage checked on the "
                               "untouched test window"),
        "applicability_method": "IsolationForest on training operating-state + weather features, thresholds at training 1st/5th score percentiles, plus per-feature range checks",
        "explanation_method": "SHAP TreeExplainer on the P50 model (model contributions in tonnes; associative, not causal)" if shap_ok else "unavailable",
        "data_provenance": {
            "operations": "SYNTHETIC — generated by ml/generate_operations.py (seed 42); not MOIL data",
            "weather": "REAL_PUBLIC — ERA5/ERA5-Land reanalysis via Open-Meteo for the demo mine coordinate",
        },
        "limitations": [
            "Operational history is synthetic; real accuracy at a MOIL mine is unknown until trained on real records.",
            "Model contributions and simulated interventions reflect learned associations in the data, not proven causal effects.",
            "Only a 7-day horizon is supported; longer horizons are treated as persistence scenarios by the contingency engine.",
        ],
    }
    save_json(manifest_path, manifest)
    print("saved production models and report")


if __name__ == "__main__":
    main()
