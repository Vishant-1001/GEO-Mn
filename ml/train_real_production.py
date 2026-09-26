"""REAL-data production model: MOIL company-level quarterly production (headline real validation).

Data
  * target: MOIL quarterly manganese-ore production (REAL_MOIL_PUBLIC, derived from cumulative
    disclosures; data/processed/production/moil_quarterly_production.csv)
  * weather: IMD gridded rainfall / Tmax over MOIL's Balaghat-Nagpur-Bhandara belt (REAL_GOVERNMENT)
  * optional (Model B only): quarterly aggregates of the SYNTHETIC demo-mine operations

Forecast unit: next fiscal quarter's total production (tonnes), company level.
Features known at the origin only: previous quarter, same quarter last year, 4-quarter mean,
YoY growth of the last quarter, fiscal-quarter seasonality, previous-quarter rainfall anomaly
(vs a climatology built from TRAINING years only). Target-quarter rainfall is never used for
forecasting; it is evaluated separately as a labelled diagnostic ("if the monsoon were known").

Protocol (quarters, chronological, fixed before evaluation)
  SELECTION: origins FY2019-20 Q1 .. FY2022-23 Q4  (choose model family on one-step-ahead errors)
  TEST:      FY2023-24 Q1 .. latest                 (untouched; expanding refit before each quarter)
  Baselines on the same test quarters: last quarter, same quarter last year, seasonal naive x YoY growth.
  Model B (hybrid) is evaluated on the SAME real test quarters; kept only if it is better there.

Run: python -m ml.train_real_production
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.common import DATA_DIR, MODELS_DIR, REPORTS_DIR, SEED, load_json, save_json

MODEL_VERSION = "real-moil-quarterly-1.0"
SELECTION = (2019, 2022)     # fiscal years (start-year labels) used for model selection
TEST_FROM = 2023             # FY2023-24 Q1 onward is the untouched test window


def quarter_frame():
    q = pd.read_csv(DATA_DIR / "processed" / "production" / "moil_quarterly_production.csv")
    q["qstart"] = pd.to_datetime(q["quarter_start"])
    q = q.sort_values("qstart").reset_index(drop=True)
    # complete quarterly index (gaps stay NaN; never interpolated)
    idx = pd.date_range(q["qstart"].min(), q["qstart"].max(), freq="QS-APR")
    q = q.set_index("qstart").reindex(idx)
    q.index.name = "qstart"
    q["fy_quarter"] = ((q.index.month - 4) % 12) // 3 + 1
    q["fy_start_year"] = np.where(q.index.month >= 4, q.index.year, q.index.year - 1)
    return q.reset_index()


def weather_quarters():
    p = DATA_DIR / "processed" / "weather" / "imd_daily_mine_belt.csv"
    w = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    agg = w.resample("QS-APR").agg({"belt_rainfall_mm": "sum", "belt_tmax_c": "mean"})
    days = w["belt_rainfall_mm"].resample("QS-APR").count()
    agg.loc[days < 80, :] = np.nan            # incomplete quarters are not used
    agg.columns = ["rain_q_mm", "tmax_q_c"]
    return agg


def synthetic_quarters():
    h = pd.read_csv(DATA_DIR / "production_history.csv", parse_dates=["date"]).set_index("date")
    cols = ["equipment_availability", "blast_delay_h", "haulage_delay_h", "drilling_delay_h", "truck_count"]
    a = h[cols].resample("QS-APR").mean()
    a.columns = ["syn_" + c for c in cols]
    return a


def build(q, weather, synth=None):
    f = q.copy()
    y = f["production_t"]
    f["lag1"] = y.shift(1)
    f["lag4"] = y.shift(4)
    f["mean4"] = y.shift(1).rolling(4, min_periods=3).mean()
    f["yoy_growth_lag1"] = y.shift(1) / y.shift(5)
    for k in (1, 2, 3, 4):
        f[f"q{k}"] = (f["fy_quarter"] == k).astype(float)
    f = f.join(weather, on="qstart")
    f["rain_prev_q"] = f["rain_q_mm"].shift(1)
    f["rain_target_q"] = f["rain_q_mm"]                   # diagnostic only (not known at origin)
    if synth is not None:
        f = f.join(synth, on="qstart")
        for c in synth.columns:
            f[c + "_prev_q"] = f[c].shift(1)
    return f


def add_climatology(f, train_mask):
    """Rainfall anomalies vs a per-fiscal-quarter climatology from TRAINING rows only."""
    clim = f[train_mask].groupby("fy_quarter")["rain_q_mm"].mean()
    f = f.copy()
    prev_q = ((f["fy_quarter"] - 2) % 4) + 1
    f["rain_prev_anom"] = f["rain_prev_q"] - prev_q.map(clim)
    f["rain_target_anom"] = f["rain_target_q"] - f["fy_quarter"].map(clim)
    return f


BASE = ["lag1", "lag4", "mean4", "yoy_growth_lag1", "q1", "q2", "q3", "q4"]
REAL_ONLY = BASE + ["rain_prev_anom"]
DIAG_KNOWN_WEATHER = REAL_ONLY + ["rain_target_anom"]
HYBRID_SYN = ["syn_equipment_availability_prev_q", "syn_blast_delay_h_prev_q", "syn_haulage_delay_h_prev_q"]


def make(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=3.0))
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=4, min_samples_leaf=5,
                                         random_state=SEED)


def one_step(f, feats, kind, origins, target_ratio=True):
    """Expanding-window one-step-ahead forecasts at the given row positions."""
    preds = {}
    for i in origins:
        train = f.iloc[:i]
        mask = pd.Series(False, index=f.index)
        mask.iloc[:i] = True
        g = add_climatology(f, mask)
        tr = g.iloc[:i].dropna(subset=feats + ["production_t"])
        if len(tr) < 12 or g.iloc[i][feats].isna().any() or pd.isna(g.iloc[i]["production_t"]):
            continue
        yv = tr["production_t"] / tr["mean4"] if target_ratio else tr["production_t"]
        m = make(kind).fit(tr[feats], yv)
        p = float(m.predict(g.iloc[[i]][feats])[0])
        preds[i] = p * g.iloc[i]["mean4"] if target_ratio else p
    return preds


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(y) > 2 else None,
            "mape_pct": float(np.mean(np.abs(y - p) / y) * 100),
            "smape_pct": float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p))) * 100)}


def main():
    warnings.filterwarnings("ignore")
    q = quarter_frame()
    wq = weather_quarters()
    f = build(q, wq, synthetic_quarters())
    sel_rows = [i for i in range(len(f)) if SELECTION[0] <= f.loc[i, "fy_start_year"] <= SELECTION[1]]
    test_rows = [i for i in range(len(f)) if f.loc[i, "fy_start_year"] >= TEST_FROM and pd.notna(f.loc[i, "production_t"])]

    # Candidates include the simple baselines and a weather-free model: the forecasting METHOD is
    # chosen on the selection window only; the untouched test window then reports every candidate.
    candidates = {"ridge_weather": ("ridge", REAL_ONLY), "ridge_no_weather": ("ridge", BASE),
                  "gbm_weather": ("gbm", REAL_ONLY), "seasonal_naive_x_yoy": ("baseline", "seasonal_naive_x_yoy"),
                  "last_quarter": ("baseline", "last_quarter")}

    def baseline_pred(name, rows):
        rows = [i for i in rows if pd.notna(f.loc[i, "production_t"])]
        if name == "last_quarter":
            v = f.loc[rows, "lag1"]
        else:
            v = f.loc[rows, "lag4"] * f.loc[rows, "yoy_growth_lag1"]
        return {i: float(x) for i, x in zip(rows, v) if pd.notna(x)}

    def predict(name, rows):
        kind_, feats_ = candidates[name]
        return baseline_pred(feats_, rows) if kind_ == "baseline" else one_step(f, feats_, kind_, rows)

    sel = {}
    for name in candidates:
        pr = predict(name, sel_rows)
        rows = list(pr)
        sel[name] = metrics(f.loc[rows, "production_t"], [pr[i] for i in rows]) if rows else None
    best = min((k for k in sel if sel[k]), key=lambda k: sel[k]["mae"])
    kind = "ridge"                       # family used for the Model A / B real-vs-hybrid experiment

    def evaluate(feats, rows):
        pr = one_step(f, feats, kind, rows)
        return pr

    pa = evaluate(REAL_ONLY, test_rows)
    pd_known = evaluate(DIAG_KNOWN_WEATHER, test_rows)
    pb = evaluate(REAL_ONLY + HYBRID_SYN, test_rows)
    rows_a = [i for i in test_rows if i in pa]
    y = f.loc[rows_a, "production_t"].to_numpy()
    base = {
        "last_quarter": f.loc[rows_a, "lag1"].to_numpy(),
        "same_quarter_last_year": f.loc[rows_a, "lag4"].to_numpy(),
        "seasonal_naive_x_yoy": (f.loc[rows_a, "lag4"] * f.loc[rows_a, "yoy_growth_lag1"]).to_numpy(),
    }
    rows_ab = [i for i in rows_a if i in pb]
    rows_dk = [i for i in rows_a if i in pd_known]
    y_ab = f.loc[rows_ab, "production_t"].to_numpy()
    res = {
        "model_version": MODEL_VERSION,
        "data_provenance": {"target": "REAL_MOIL_PUBLIC", "weather": "REAL_GOVERNMENT (IMD)",
                            "hybrid_features": "SYNTHETIC (demo-mine simulator)"},
        "forecast_unit": "next fiscal quarter, MOIL company total manganese-ore production (t)",
        "selection_window": f"FY{SELECTION[0]}-{str(SELECTION[0] + 1)[-2:]} .. FY{SELECTION[1]}-{str(SELECTION[1] + 1)[-2:]}",
        "selection": sel, "selected_family": best,
        "test_window": [str(f.loc[rows_a[0], "qstart"].date()), str(f.loc[rows_a[-1], "qstart"].date())],
        "test_provenance": "REAL_MOIL_PUBLIC quarters, never used for selection",
        "model_a_real_only": metrics(y, [pa[i] for i in rows_a]),
        "all_candidates_on_test": {n: (lambda pr: metrics(f.loc[list(pr), "production_t"], list(pr.values())))(predict(n, rows_a))
                                   for n in candidates},
        "selected_method": best,
        "baselines": {k: metrics(y, v) for k, v in base.items()},
        "diagnostic_target_quarter_weather_known": metrics(f.loc[rows_dk, "production_t"], [pd_known[i] for i in rows_dk]),
        "hybrid_comparison": {
            "quarters": [str(f.loc[i, "qstart"].date()) for i in rows_ab],
            "note": ("Model B needs synthetic operations, which exist only from 2021; A and B are compared on the "
                     "same real quarters where both are available."),
            "model_a_real_only": metrics(y_ab, [pa[i] for i in rows_ab]) if len(rows_ab) > 2 else None,
            "model_b_real_plus_synthetic": metrics(y_ab, [pb[i] for i in rows_ab]) if len(rows_ab) > 2 else None,
        },
        "features": {"model_a": REAL_ONLY, "model_b_extra": HYBRID_SYN},
    }
    best_base = min(res["baselines"].values(), key=lambda m: m["mae"])
    res["model_a_beats_best_baseline"] = res["model_a_real_only"]["mae"] < best_base["mae"]
    hc = res["hybrid_comparison"]
    res["synthetic_augmentation_helps"] = bool(hc["model_b_real_plus_synthetic"] and
                                               hc["model_b_real_plus_synthetic"]["mae"] < hc["model_a_real_only"]["mae"])
    res["model_b_real_plus_synthetic"] = hc["model_b_real_plus_synthetic"]
    res["decision"] = ("synthetic operational augmentation RETAINED" if res["synthetic_augmentation_helps"] else
                       "synthetic operational augmentation NOT used for headline real forecasting (did not improve real "
                       "held-out error); synthetic operations remain for scenario / recovery simulation only")
    res["test_rows"] = [{"quarter_start": str(f.loc[i, "qstart"].date()), "fy": f.loc[i, "fy"], "fy_quarter": int(f.loc[i, "fy_quarter"]),
                         "actual_t": float(f.loc[i, "production_t"]), "model_a_t": round(pa[i], 0),
                         "model_b_t": round(pb[i], 0) if i in pb else None,
                         "same_quarter_last_year_t": float(f.loc[i, "lag4"])} for i in rows_a]
    common = rows_a
    # final forecast with the SELECTED method (selection window only decided it)
    last = f.iloc[-1]
    nxt = pd.Timestamp(last["qstart"]) + pd.DateOffset(months=3)
    fq = ((nxt.month - 4) % 12) // 3 + 1
    mask = pd.Series(True, index=f.index)
    g = add_climatology(f, mask)
    kind_b, feats_b = candidates[best]
    final = None
    fc = None
    if kind_b == "baseline":
        same_q_last_year = f.iloc[-3]["production_t"]             # next quarter minus 4 quarters
        yoy = last["production_t"] / f.iloc[-5]["production_t"]
        fc = float(last["production_t"]) if feats_b == "last_quarter" else float(same_q_last_year * yoy)
    else:
        tr = g.dropna(subset=feats_b + ["production_t"])
        final = make(kind_b).fit(tr[feats_b], tr["production_t"] / tr["mean4"])
        row = {"lag1": last["production_t"], "lag4": f.iloc[-3]["production_t"],
               "mean4": f["production_t"].iloc[-4:].mean(), "yoy_growth_lag1": last["production_t"] / f.iloc[-5]["production_t"]}
        for k in (1, 2, 3, 4):
            row[f"q{k}"] = float(fq == k)
        clim = g.groupby("fy_quarter")["rain_q_mm"].mean()
        prev_rain = wq["rain_q_mm"].get(pd.Timestamp(last["qstart"]))
        row["rain_prev_anom"] = (prev_rain - clim.get(int(last["fy_quarter"]))) if pd.notna(prev_rain) else np.nan
        if not any(pd.isna(row[k]) for k in feats_b):
            fc = float(final.predict(pd.DataFrame([row])[feats_b])[0] * row["mean4"])
    sel_test = predict(best, rows_a)
    resid = np.array([sel_test[i] - f.loc[i, "production_t"] for i in sel_test])
    res["next_quarter_forecast"] = {
        "quarter_start": str(nxt.date()), "fiscal_quarter": int(fq), "method": best,
        "point_t": round(fc, 0) if fc else None,
        "empirical_error_band_t": [round(fc + float(np.quantile(-resid, 0.1)), 0), round(fc + float(np.quantile(-resid, 0.9)), 0)] if fc else None,
        "band_note": "10th-90th percentile of the selected method's real held-out one-step errors (n=12; indicative only, not a calibrated interval)",
        "status": "AVAILABLE" if fc else "UNAVAILABLE — an input for the selected method is not yet published",
    }
    res["baseline_next_quarter_t"] = {
        "seasonal_naive_x_yoy": round(float(f.iloc[-3]["production_t"] * last["production_t"] / f.iloc[-5]["production_t"]), 0),
        "last_quarter": float(last["production_t"]),
        "note": "Shown because the selected model did not beat the best baseline on the untouched real test window."}
    joblib.dump({"method": best, "model": final, "features": feats_b if kind_b != "baseline" else None,
                 "ratio_base": "mean4"}, MODELS_DIR / "real_moil_quarterly.pkl")
    save_json(REPORTS_DIR / "real_production_validation.json", res)
    man_p = MODELS_DIR / "model_manifest.json"
    man = load_json(man_p)
    man["real_production"] = {
        "model_version": MODEL_VERSION, "model_type": f"selected method: {best} (chosen on the selection window)",
        "features": REAL_ONLY, "candidates": list(candidates), "training_period": [str(q['qstart'].min().date()), str(q['qstart'].max().date())],
        "validation_method": "expanding-window one-step-ahead; selection FY2019-22, untouched test FY2023+",
        "validation_metrics": {"model_a": res["model_a_real_only"], "baselines": res["baselines"],
                               "hybrid_comparison": res["hybrid_comparison"]},
        "data_provenance": res["data_provenance"], "uncertainty_method": "empirical held-out error band (indicative)",
        "applicability_method": "company-level quarterly only; not a mine-level or weekly model",
        "limitations": ["~50 quarters of data; metrics have wide sampling uncertainty",
                        "company-level totals mix many mines; no operational drivers are observed",
                        "quarterly values are derived by differencing cumulative disclosures (rounded to 0.01 lakh t)"],
    }
    from ml.common import save_json as _s
    _s(man_p, man)
    print({k: res[k] for k in ("selected_family", "test_window")}, {k: round(v["mae"]) for k, v in sel.items() if v})
    for n, m in res["all_candidates_on_test"].items():
        print("TEST", n, {a: round(b, 3) if isinstance(b, float) else b for a, b in m.items()})
    for k in ("model_a_real_only", "diagnostic_target_quarter_weather_known"):
        print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in res[k].items()})
    print("hybrid comparison", json.dumps(res["hybrid_comparison"], default=str)[:600])
    for k, v in res["baselines"].items():
        print("baseline", k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
    print(res["decision"]); print(res["next_quarter_forecast"])


if __name__ == "__main__":
    main()
