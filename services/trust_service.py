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


def _experiments():
    e = _report("exploration_experiments.json")
    if e is None:
        return None
    rob = e.get("seed_robustness", {})
    return {
        "design": e["design"],
        "deployed": e["final_decision"]["deployed"], "decision_rule": e["final_decision"].get("rule"),
        "decision_reason": e["final_decision"]["reason"],
        "selected_feature_set": e["selected_feature_set"], "selected_label_set": e["selected_label_set"],
        "final_test_seed_averaged": {k: {m: _r(v.get(f"{m}_mean"), 4) for m in ("test_roc", "test_pr", "test_cap10")}
                                     | {"test_roc_sd": _r(v.get("test_roc_sd"), 4)} for k, v in rob.items()},
        "final_test_prevalence": e["models"]["A_mrds_baseline_features"]["final_test"]["mrds_labels"].get("prevalence"),
        "ablation_dev_cv": {k: {"roc_auc": _r(v["roc_auc"]), "pr_auc": _r(v["pr_auc"], 4)} for k, v in e["ablation_dev_cv"].items()},
        "model_d": e["models"]["D_real_plus_synthetic"],
        "supplementary_note": e.get("supplementary_note"),
    }


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
        "subsurface_evidence": ("Observed: REPORTED_BLOCK_LEVEL for targets overlapping official NMET blocks (REAL_GOVERNMENT); "
                                "no public collars, logs or assays; every other target is UNAVAILABLE. No subsurface record is simulated."),
        "experiments": _experiments(),
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
                                 labels_mode="REAL_PUBLIC", features_mode="REAL_DERIVED",
                                 geology_features_mode="REAL_GOVERNMENT" if any(
                                     f.startswith("geom_") for f in man.get("features", [])) else None),
    }


def recovery(mine_id: str = "DEMO_MINE") -> dict:
    """Recovery diagnostics for the current state (no 'accuracy': effects are simulator counterfactuals)."""
    from services import recovery_service

    d = recovery_service.evaluate(mine_id)
    ports = d["evaluated_portfolios"]
    sel = next((p for p in ports if p["selected"]), None)
    return {
        "mine_id": d["mine_id"],
        "scenarios_tested": d["disruption_scenarios"],
        "portfolios_evaluated": len(ports),
        "eligible_portfolios": [p["portfolio_id"] for p in ports if p["eligible"]],
        "applicability_blocked_portfolios": [{"portfolio": p["portfolio_id"], "scenarios": p["low_applicability_scenarios"]}
                                             for p in ports if p["low_applicability_scenarios"]],
        "feasibility_blocked_portfolios": [p["portfolio_id"] for p in ports
                                           if p["modelled_feasibility"] not in (recovery_service.FEASIBLE, recovery_service.CONSTRAINED)],
        "constraint_notes": {p["portfolio_id"]: p["constraint_notes"] for p in ports if p["constraint_notes"]},
        "baseline_worst_case_residual_gap_tonnes": d["no_action_worst_case_gap_tonnes"],
        "selected_portfolio": d["selected_portfolio"],
        "selection_status": d["selection_status"],
        "selected_worst_case_residual_gap_tonnes": sel["worst_case_residual_gap_tonnes"] if sel else None,
        "selected_intervention_burden": sel["intervention_burden"] if sel else None,
        "selection_rule": d["selection_rule"],
        "applicability_policy": d["applicability_policy"],
        "note": ("No recovery 'accuracy' is reported: action effects are simulator counterfactuals scored by the "
                 "production model, not historical interventions."),
        "provenance": d["provenance"],
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
        "wape_pct": _r(bt["model_p50"].get("wape_pct"), 2),
        "mape_pct_excl_near_zero": _r(bt["model_p50"].get("mape_pct_excl_near_zero"), 2),
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
            (f"P50 MAE is {bt.get('mae_improvement_vs_best_baseline_pct', 0):.1f} % better than the best simple baseline on the "
             "test window — "
             + ("a material advantage." if bt.get("mae_improvement_vs_best_baseline_pct", 0) >= 2.0 else
                "effectively a tie; the model's value here is the conditional scenario response and calibrated "
                "interval, not point accuracy.")),
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
                                 operations_mode="SYNTHETIC", weather_mode="REAL_GOVERNMENT (IMD) + REAL_PUBLIC (ERA5 soil moisture)"),
    }


def _manifest(rel):
    p = DATA_DIR / rel
    return json.loads(p.read_text()) if p.exists() else {}


def provenance_catalogue() -> dict:
    ex_src = json.loads((DATA_DIR / "exploration_sources.json").read_text()) if (DATA_DIR / "exploration_sources.json").exists() else {}
    era5 = _manifest("raw/weather/era5_open_meteo_sources.json")
    imd = _manifest("manifests/real_imd_gridded.json")
    moil = _manifest("manifests/real_moil_production.json")
    bhuvan = _manifest("manifests/real_bhuvan_geology.json")
    nmet = _manifest("manifests/real_subsurface_nmet.json")
    syn_ops = _manifest("synthetic/production/synthetic_generation_manifest.json")
    syn_rec = _manifest("synthetic/recovery/recovery_generation_manifest.json")
    man = load_manifest()
    demo = load_config("demo_config.json")
    exm = man.get("exploration", {})
    return {
        "modes": {
            "REAL_GOVERNMENT": "Official Government of India data (IMD, NRSC/Bhuvan, NMET/Ministry of Mines)",
            "REAL_PUBLIC": "Public international data (USGS MRDS, Sentinel-2, MODIS, NASADEM, ERA5)",
            "REAL_MOIL_PUBLIC": "MOIL Ltd. public investor disclosures (company level)",
            "REAL_DERIVED": "Features computed from real data by a documented recipe",
            "SYNTHETIC": "Generated by a seeded, domain-constrained simulator; not observed",
            "SIMULATED": "Scenario / counterfactual outputs of a simulator; not observed",
            "CACHED": "Precomputed from the sources above; replayed, not live",
        },
        "sources": {
            "exploration_labels": {"mode": "REAL_PUBLIC", "description": "USGS Mineral Resources Data System (MRDS) Mn records",
                                   "retrieved_utc": ex_src.get("mrds", {}).get("retrieved_utc"),
                                   "records_used": ex_src.get("mrds", {}).get("positives_used")},
            "sentinel2": {"mode": "REAL_PUBLIC", "description": "Sentinel-2 L2A via Microsoft Planetary Computer (annual median composite)",
                          "observation_window": exm.get("observation_window")},
            "modis_lst": {"mode": "REAL_PUBLIC", "description": "MODIS MOD11A2 v061 LST (x0.02 K, QC-filtered)",
                          "observation_window": exm.get("observation_window")},
            "dem": {"mode": "REAL_PUBLIC", "description": "NASADEM elevation and derived slope / relief"},
            "geology_macrostrat": {"mode": "REAL_PUBLIC", "description": ex_src.get("geology", {}).get("underlying_map", "Macrostrat world geology"),
                                   "note": ex_src.get("geology", {}).get("note")},
            "geomorphology_lineaments": {"mode": "REAL_GOVERNMENT", "description": bhuvan.get("dataset_name"),
                                         "provider": bhuvan.get("provider"), "retrieved_utc": bhuvan.get("retrieval_timestamp")},
            "exploration_blocks": {"mode": "REAL_GOVERNMENT", "description": nmet.get("dataset_name"),
                                   "provider": nmet.get("provider"), "blocks": nmet.get("blocks"),
                                   "note": "Block-level reported outcomes only; no public collars, logs or assays."},
            "exploration_grid": {"mode": "CACHED", "description": "Precomputed 0.01 degree prospectivity grid",
                                 "cells": ex_src.get("grid", {}).get("cells")},
            "weather_imd": {"mode": "REAL_GOVERNMENT", "description": imd.get("dataset_name"), "provider": imd.get("provider"),
                            "retrieved_utc": imd.get("retrieval_timestamp"), "unavailable": imd.get("unavailable")},
            "weather_era5": {"mode": "REAL_PUBLIC", "description": era5.get("source"), "note": era5.get("note"),
                             "use": "soil moisture and dates IMD has not yet published (per-row source recorded)",
                             "observation_window": f"{demo['history_start']}/{demo['history_end']}"},
            "production_moil": {"mode": "REAL_MOIL_PUBLIC", "description": moil.get("dataset_name"),
                                "quarters": moil.get("quarters_derived"), "coverage": moil.get("coverage"),
                                "note": "Company-total quarterly production; not mine-level, not weekly."},
            "operations": {"mode": "SYNTHETIC", "description": "Equipment-level simulated operations for SYN_MINE_01 (DEMO_MINE) — NOT MOIL telemetry",
                           "seed": syn_ops.get("seed"), "generator_version": syn_ops.get("generator_version")},
            "recovery_scenarios": {"mode": "SIMULATED", "description": syn_rec.get("dataset_name"), "seed": syn_rec.get("seed"),
                                   "note": "Action effects are simulator counterfactuals, not historical MOIL interventions."},
            "subsurface_observed": {"mode": "REAL_GOVERNMENT", "description": "Reported NMET / DGM / MECL block findings attached to overlapping targets",
                                    "note": "REPORTED_BLOCK_LEVEL only where a target overlaps an official block; otherwise UNAVAILABLE."},
            "next_evidence_sensitivity": {"mode": "SIMULATED", "description": "Rule-based what-if: how a target's priority would change for each possible outcome of the next investigation",
                                          "note": "No boreholes, assays or geophysical values are generated."},
            "demo_states": {"mode": "SIMULATED", "description": "Deterministic demo states DEMO_A..DEMO_F"},
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
            "Demo-mine operational data are synthetic; forecasts and scenario outcomes are model estimates.",
            "Real MOIL production is company-level quarterly; no mine-level MOIL accuracy is claimed.",
        ],
    }
