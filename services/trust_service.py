"""Trust layer: validation metrics (only values actually computed), provenance and limitations."""

from __future__ import annotations

import json

from services.common import DATA_DIR, MODELS_DIR, load_config, load_manifest, provenance

NA = "N/A — validation not available for this dataset"


def _report(name):
    p = MODELS_DIR / "reports" / name
    return json.loads(p.read_text()) if p.exists() else None


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def exploration() -> dict:
    rep = _report("exploration_validation.json")
    man = load_manifest().get("exploration", {})
    if rep is None:
        return {"status": "NOT_AVAILABLE", "message": NA}
    cv = rep["spatial_block_cv"]
    ho = rep["region_holdout"]["pooled"]
    return {
        "status": "AVAILABLE",
        "model_version": rep["model_version"],
        "spatial_validation": man.get("validation_method"),
        "validation_method": man.get("validation_method"),
        "roc_auc": _r(cv["roc_auc"]),
        "pr_auc": _r(cv["pr_auc"], 4),
        "pr_auc_prevalence_baseline": _r(cv["pr_auc_prevalence_baseline"], 4),
        "top_area_capture": {k: _r(v) for k, v in cv["top_area_capture"].items()},
        "success_rate_curve": cv["success_rate_curve"],
        "success_rate_auc": _r(cv["success_rate_auc"]),
        "per_fold": cv["per_fold"],
        "region_holdout_method": "train west of 79.8E / test east, and the reverse; pooled within-region ranks",
        "observation_window": man.get("observation_window"),
        "effective_resolution": man.get("effective_resolution"),
        "subsurface_evidence": "UNAVAILABLE — no drilling, assay or geophysical data are used",
        "region_holdout": {"roc_auc": _r(ho["roc_auc"]), "pr_auc": _r(ho["pr_auc"], 4),
                           "top_area_capture": {k: _r(v) for k, v in ho["top_area_capture"].items()},
                           "splits": {k: v for k, v in rep["region_holdout"].items() if k != "pooled"}},
        "feature_ablation": {k: {"roc_auc": _r(v["roc_auc"]), "pr_auc": _r(v["pr_auc"], 4),
                                 "top_10pct_area_capture": _r(v["top_area_capture"]["top_10pct_area"])}
                             for k, v in rep["ablation_spatial_cv"].items()},
        "selected_configuration": rep["selected"],
        "legacy_model_check": rep["legacy_model_check"],
        "labels": rep["labels"],
        "applicability": man.get("applicability_method"),
        "applicability_method": man.get("applicability_method"),
        "uncertainty": man.get("uncertainty_method"),
        "uncertainty_method": man.get("uncertainty_method"),
        "calibration": "NOT CALIBRATED — outputs are relative ranks, not probabilities.",
        "limitations": man.get("limitations", []),
        "notes": [
            "Metrics are computed on held-out spatial blocks / regions only.",
            "Background cells are unlabelled (unknown != barren), so ROC/PR are indicative comparisons.",
            f"PR-AUC must be read against its prevalence baseline ({_r(cv['pr_auc_prevalence_baseline'], 4)}).",
            "Output is a relative prospectivity rank within the study area — not a deposit probability or reserve estimate.",
            "Satellite features come from a fixed 2024 reference window (not real-time imagery).",
        ],
        "provenance": provenance("REAL_PUBLIC", rep["model_version"], man.get("observation_window"), None, False,
                                 labels_mode="REAL_PUBLIC", features_mode="REAL_PUBLIC"),
    }


def production() -> dict:
    rep = _report("production_validation.json")
    man = load_manifest().get("production", {})
    if rep is None:
        return {"status": "NOT_AVAILABLE", "message": NA}
    bt = rep["backtest"]
    return {
        "status": "AVAILABLE",
        "model_version": rep["model_version"],
        "rolling_backtest": bt["method"],
        "validation_method": bt["method"],
        "test_window": [bt["test_start"], bt["test_end"]],
        "n_test_periods": bt["n_test_periods"],
        "mae": _r(bt["model_p50"]["mae"], 1),
        "rmse": _r(bt["model_p50"]["rmse"], 1),
        "r2": _r(bt["model_p50"]["r2"]),
        "mape_pct": _r(bt["model_p50"]["mape_pct"], 2),
        "baseline": {
            "previous_period": {k: _r(v, 3) for k, v in bt["baseline_previous_period"].items()},
            "moving_average_4": {k: _r(v, 3) for k, v in bt["baseline_moving_average_4"].items()},
        },
        "baseline_mae": _r(min(bt["baseline_previous_period"]["mae"], bt["baseline_moving_average_4"]["mae"]), 1),
        "model_beats_best_baseline_mae": bt["model_beats_best_baseline_mae"],
        "mae_improvement_vs_best_baseline_pct": _r(bt["mae_improvement_vs_best_baseline_pct"], 1),
        "pinball_loss": {k: _r(v, 1) for k, v in bt["pinball_loss"].items()},
        "coverage": _r(bt["p10_p90_coverage"]),
        "observed_coverage": _r(bt["p10_p90_coverage"]),
        "nominal_coverage": 0.8,
        "p10_p90_coverage": _r(bt["p10_p90_coverage"]),
        "p10_p90_coverage_before_recalibration": _r(bt["p10_p90_coverage_before_recalibration"]),
        "p10_p90_nominal": 0.8,
        "quantile_hit_rate": {k: _r(v) for k, v in bt["quantile_hit_rate"].items()},
        "quantiles_validated": rep["quantiles_validated"],
        "quantile_validation_rule": rep["quantile_validation_rule"],
        "shortfall_classification": bt["shortfall_classification"],
        "diagnostic_oracle_conditions_p50": bt.get("diagnostic_oracle_conditions_p50"),
        "selection_window": rep["selection_window"],
        "protocol": rep.get("protocol"),
        "selected_configuration": rep["selected_configuration"],
        "forecast_procedure": man.get("forecast_procedure"),
        "uncertainty_method": man.get("uncertainty_method"),
        "applicability_method": man.get("applicability_method"),
        "limitations": man.get("limitations", []),
        "notes": [
            rep["caveat"],
            "Real mine accuracy requires training and validation on mine-level operational data.",
            "Configuration chosen on the selection window; quantile offsets estimated on the separate calibration window "
            "and frozen; all reported metrics come from the later, untouched test window.",
            "The deployed model is a post-evaluation refit on all history; the metrics describe the backtest procedure.",
            (f"P10-P90 observed coverage {_r(bt['p10_p90_coverage'], 2)} vs nominal 0.80; "
             + ("the interval passed the validation rule." if rep["quantiles_validated"] else
                "the quantiles did NOT pass the validation rule, so P10/P90 are indicative only.")),
            "Forecasts assume the trailing 7-day operating/weather state persists (conditional forecast).",
        ],
        "provenance": provenance("SYNTHETIC", rep["model_version"], f"{bt['test_start']}/{bt['test_end']}", None, False,
                                 operations_mode="SYNTHETIC", weather_mode="REAL_PUBLIC"),
    }


def provenance_catalogue() -> dict:
    ex_src = json.loads((DATA_DIR / "exploration_sources.json").read_text()) if (DATA_DIR / "exploration_sources.json").exists() else {}
    wx_src = json.loads((DATA_DIR / "weather_sources.json").read_text()) if (DATA_DIR / "weather_sources.json").exists() else {}
    man = load_manifest()
    demo = load_config("demo_config.json")
    return {
        "sources": {
            "exploration_labels": {"mode": "REAL_PUBLIC", "description": "USGS Mineral Resources Data System (MRDS) Mn records",
                                   "retrieved_utc": ex_src.get("mrds", {}).get("retrieved_utc"),
                                   "records_used": ex_src.get("mrds", {}).get("positives_used")},
            "sentinel2": {"mode": "REAL_PUBLIC", "description": "Sentinel-2 L2A via Microsoft Planetary Computer (annual median composite)",
                          "observation_window": man.get("exploration", {}).get("observation_window")},
            "modis_lst": {"mode": "REAL_PUBLIC", "description": "MODIS MOD11A2 v061 LST (x0.02 K, QC-filtered)",
                          "observation_window": man.get("exploration", {}).get("observation_window")},
            "dem": {"mode": "REAL_PUBLIC", "description": "NASADEM elevation and derived slope"},
            "geology": {"mode": "REAL_PUBLIC", "description": ex_src.get("geology", {}).get("underlying_map", "Macrostrat world geology"),
                        "note": ex_src.get("geology", {}).get("note")},
            "exploration_grid": {"mode": "CACHED", "description": "Precomputed 0.01 degree prospectivity grid",
                                 "cells": ex_src.get("grid", {}).get("cells")},
            "weather": {"mode": "REAL_PUBLIC", "description": wx_src.get("source"), "note": wx_src.get("note"),
                        "observation_window": f"{demo['history_start']}/{demo['history_end']}"},
            "operations": {"mode": "SYNTHETIC", "description": "Generated operational history for DEMO_MINE (seed 42) — NOT MOIL data"},
            "scenarios": {"mode": "SIMULATED", "description": "Disruption scenarios, action portfolios and demo states"},
            "subsurface": {"mode": "UNAVAILABLE", "description": "No drilling, assay or geophysical data available"},
            "reserves": {"mode": "UNAVAILABLE", "description": "No reserve/resource tonnage is produced or implied by GEO-MN"},
        },
        "model_versions": {k: v.get("model_version") for k, v in man.items()},
        "policies": {
            "risk_policy": load_config("risk_policy.json")["version"],
            "exploration_policy": load_config("exploration_config.json")["version"],
            "recovery_config": load_config("recovery_config.json")["version"],
            "note": "Thresholds and weights are project demonstration defaults, not industry standards.",
        },
        "notes": [
            "Satellite composites use a fixed 2024 observation window; nothing is real-time.",
            "Prospectivity is a relative rank; it is not a probability of a deposit or a reserve estimate.",
            "Operational data are synthetic; forecasts and scenario outcomes are model estimates.",
        ],
    }
