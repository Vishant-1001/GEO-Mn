"""REAL MOIL company-level quarterly production (public disclosures) and its validated forecast.

Data: data/processed/production/moil_quarterly_production.csv (REAL_MOIL_PUBLIC, built by
scripts/data_acquisition/fetch_moil_production.py). Model report: models/reports/real_production_validation.json
(ml/train_real_production.py). This is a company-total quarterly view; it is NOT the weekly demo-mine engine,
and the demo mine's operational records are SYNTHETIC.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from services.common import DATA_DIR, MODELS_DIR, REAL_MOIL_PUBLIC, ApiError, file_timestamp, provenance

QPATH = DATA_DIR / "processed" / "production" / "moil_quarterly_production.csv"
RPATH = MODELS_DIR / "reports" / "real_production_validation.json"


@lru_cache(maxsize=1)
def _load():
    if not QPATH.exists() or not RPATH.exists():
        return None, None
    return pd.read_csv(QPATH), json.loads(RPATH.read_text())


def _m(d):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (d or {}).items()}


def quarterly(last_n: int = 24) -> dict:
    q, rep = _load()
    if q is None:
        raise ApiError(503, "DATA_UNAVAILABLE", "Real MOIL quarterly data or model report has not been built.")
    tail = q.tail(last_n)
    sel = rep["selected_method"]
    best_base = min(rep["baselines"].items(), key=lambda kv: kv[1]["mae"])
    beats = rep["all_candidates_on_test"][sel]["mae"] < best_base[1]["mae"]
    return {
        "unit": "tonnes of manganese ore, MOIL company total (all mines), fiscal quarter",
        "quarters": [{"quarter_start": r.quarter_start, "fy": r.fy, "fy_quarter": int(r.fy_quarter),
                      "production_t": r.production_t, "basis": r.basis, "source": r.source_documents}
                     for r in tail.itertuples()],
        "coverage": [q["quarter_start"].min(), q["quarter_start"].max()], "n_quarters": int(len(q)),
        "gaps_note": "Quarters that cannot be derived from published disclosures are left missing (not imputed).",
        "forecast": rep["next_quarter_forecast"],
        "baseline_forecasts_t": rep.get("baseline_next_quarter_t"),
        "validation": {
            "selection_window": rep["selection_window"], "test_window": rep["test_window"],
            "selected_method": sel, "selected_test": _m(rep["all_candidates_on_test"][sel]),
            "best_baseline": best_base[0], "best_baseline_test": _m(best_base[1]),
            "model_beats_best_baseline": bool(beats),
            "verdict": ("Selected model beats the best baseline on the untouched real test window." if beats else
                        f"Selected model does NOT beat the {best_base[0]} baseline on the untouched real test window "
                        "(12 quarters); treat the model forecast as one indicative estimate alongside the baselines."),
            "synthetic_augmentation_helps": rep["synthetic_augmentation_helps"],
            "hybrid_comparison": {k: _m(v) if isinstance(v, dict) else v for k, v in rep["hybrid_comparison"].items()},
            "decision": rep["decision"],
        },
        "claims_not_made": "No mine-level or weekly MOIL accuracy is claimed; the weekly engine runs on SYNTHETIC demo-mine operations.",
        "provenance": provenance(REAL_MOIL_PUBLIC, rep["model_version"], f"{q['quarter_start'].min()}..{q['quarter_start'].max()}",
                                 file_timestamp(QPATH), False,
                                 source_url="https://www.moil.nic.in (Investors > Financials > Quantitative Details)"),
    }
