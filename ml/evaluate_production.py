"""Production forecast model definition + rolling-origin backtest.

Model
  * Three HistGradientBoostingRegressor(loss="quantile") models (q = 0.1/0.5/0.9)
    trained on period conditions (see ml/production_features.py) and evaluated
    with persistence inputs that are known at the origin.
  * Optional "ratio" formulation: the models predict y / prod_roll_mean_4 and the
    prediction is multiplied back, so trees do not have to extrapolate a drifting
    production level.
  * Quantile recalibration: additive offsets (relative to the raw P50) chosen so
    that the empirical quantile of PRIOR out-of-sample errors matches the nominal
    level (a split-conformal style correction using only past information).

Backtest
  * Test origins: non-overlapping 7-day periods aligned to the forecast origin,
    grouped into 13-period folds; for each fold the models are trained only on
    origins whose outcome window ends before the fold (purged).
  * Baselines on the same origins: previous-period production and 4-period mean.
  * Metrics are computed on SYNTHETIC operations: they demonstrate that the
    pipeline behaves as designed, not real-world accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.common import SEED
from ml.production_features import FEATURE_COLUMNS, MONOTONIC, PERIOD_DAYS, training_matrix

QUANTILES = (0.1, 0.5, 0.9)
QKEYS = ("p10", "p50", "p90")
FOLD_PERIODS = 13
RATIO_BASE = "prod_roll_mean_4"

CANDIDATES = {
    "level_default": {"ratio": False, "params": dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
                                                      min_samples_leaf=25, l2_regularization=1.0)},
    "ratio_default": {"ratio": True, "params": dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
                                                     min_samples_leaf=25, l2_regularization=1.0)},
    "ratio_regularised": {"ratio": True, "params": dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
                                                         min_samples_leaf=60, l2_regularization=2.0)},
    "ratio_strongly_regularised": {"ratio": True, "params": dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=5,
                                                                  min_samples_leaf=120, l2_regularization=4.0)},
}


class QuantileForecaster:
    """Three quantile GBMs + optional ratio target + recalibration offsets."""

    def __init__(self, ratio=True, params=None, monotonic=True, offsets=None):
        self.ratio = ratio
        self.params = params or CANDIDATES["ratio_regularised"]["params"]
        self.monotonic = monotonic
        self.offsets = offsets or {k: 0.0 for k in QKEYS}
        self.models = {}

    def _base(self, X):
        return np.maximum(X[RATIO_BASE].to_numpy(dtype=float), 1.0) if self.ratio else 1.0

    def fit(self, X, y):
        cst = [MONOTONIC.get(c, 0) for c in FEATURE_COLUMNS] if self.monotonic else None
        target = np.asarray(y, dtype=float) / self._base(X)
        for q, k in zip(QUANTILES, QKEYS):
            self.models[k] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, monotonic_cst=cst, random_state=SEED, **self.params
            ).fit(X[FEATURE_COLUMNS], target)
        return self

    def predict_raw(self, X):
        base = self._base(X)
        P = np.column_stack([self.models[k].predict(X[FEATURE_COLUMNS]) * base for k in QKEYS])
        return np.sort(np.clip(P, 0, None), axis=1)

    def predict(self, X):
        raw = self.predict_raw(X)
        adj = raw + np.column_stack([self.offsets[k] * raw[:, 1] for k in QKEYS])
        return np.sort(np.clip(adj, 0, None), axis=1)


def calibration_offsets(actual, raw):
    """Relative offsets so the empirical quantile of (y - q_raw)/p50_raw hits each nominal level."""
    actual = np.asarray(actual, dtype=float)
    rel = {k: (actual - raw[:, i]) / np.maximum(raw[:, 1], 1.0) for i, k in enumerate(QKEYS)}
    return {k: float(np.quantile(rel[k], q)) for k, q in zip(QKEYS, QUANTILES)}


def pinball(y, p, q):
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def shortfall_scores(actual, pred, target, tol):
    a = actual < target * (1 - tol)
    p = pred < target * (1 - tol)
    tp = int(np.sum(a & p))
    fp = int(np.sum(~a & p))
    fn = int(np.sum(a & ~p))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"precision": prec, "recall": rec, "f1": f1, "events": int(a.sum()), "predicted_events": int(p.sum())}


def point_metrics(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "mape_pct": float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100),
    }


def backtest_predictions(frame, test_origins, candidate, monotonic=True, calib_from=None):
    """Out-of-sample raw + recalibrated predictions for every test origin.

    Recalibration for a fold uses only out-of-sample errors from origins whose
    outcome window ended before that fold (none -> zero offsets). If calib_from
    is given, errors before that date are also eligible (from an earlier run).
    """
    idx = frame.set_index("origin")
    test_origins = test_origins[test_origins.isin(idx.index)]
    folds = [test_origins[i:i + FOLD_PERIODS] for i in range(0, len(test_origins), FOLD_PERIODS)]
    rows = []
    history = calib_from.copy() if calib_from is not None else pd.DataFrame()
    for k, fold in enumerate(folds):
        cutoff = fold[0] - pd.Timedelta(days=PERIOD_DAYS)
        train = frame[frame["origin"] <= cutoff]
        model = QuantileForecaster(candidate["ratio"], candidate["params"], monotonic).fit(training_matrix(train), train["y"])
        te = idx.loc[fold]
        raw = model.predict_raw(te.reset_index())  # persistence inputs: origin-known only
        oracle = model.predict_raw(training_matrix(te.reset_index()))  # diagnostic: true period conditions
        prior = history[pd.to_datetime(history["origin"]) <= cutoff] if len(history) else history
        if len(prior) >= 13:
            model.offsets = calibration_offsets(prior["actual"].to_numpy(), prior[["raw_p10", "raw_p50", "raw_p90"]].to_numpy())
        P = model.predict(te.reset_index())
        fold_rows = []
        for j, o in enumerate(fold):
            r = te.loc[o]
            fold_rows.append({
                "fold": k, "origin": o.strftime("%Y-%m-%d"),
                "period_end": (o + pd.Timedelta(days=PERIOD_DAYS - 1)).strftime("%Y-%m-%d"),
                "train_rows": len(train), "actual": r["y"], "target": r["period_target"],
                "raw_p10": raw[j, 0], "raw_p50": raw[j, 1], "raw_p90": raw[j, 2],
                "p10": P[j, 0], "p50": P[j, 1], "p90": P[j, 2],
                "oracle_p50": oracle[j, 1],
                "naive": r["prod_lag1"], "ma4": r["prod_roll_mean_4"],
                "calibrated": len(prior) >= 13,
            })
        fold_rows = pd.DataFrame(fold_rows)
        history = pd.concat([history, fold_rows], ignore_index=True)
        rows.append(fold_rows)
    return pd.concat(rows, ignore_index=True), len(folds)


def summarise(bt, shortfall_tol, method, n_folds):
    y = bt["actual"].to_numpy()
    res = {
        "method": method,
        "test_start": bt["origin"].iloc[0],
        "test_end": bt["period_end"].iloc[-1],
        "n_folds": n_folds,
        "n_test_periods": int(len(bt)),
        "model_p50": point_metrics(y, bt["p50"].to_numpy()),
        "baseline_previous_period": point_metrics(y, bt["naive"].to_numpy()),
        "baseline_moving_average_4": point_metrics(y, bt["ma4"].to_numpy()),
        "diagnostic_oracle_conditions_p50": {
            **point_metrics(y, bt["oracle_p50"].to_numpy()),
            "note": "NOT a forecast skill: uses the true period conditions. Shows how much of the error comes from unknown future conditions.",
        },
        "pinball_loss": {k: pinball(y, bt[k].to_numpy(), q) for k, q in zip(QKEYS, QUANTILES)},
        "quantile_hit_rate": {k: float(np.mean(y <= bt[k].to_numpy())) for k in QKEYS},
        "p10_p90_coverage": float(np.mean((y >= bt["p10"]) & (y <= bt["p90"]))),
        "p10_p90_coverage_before_recalibration": float(np.mean((y >= bt["raw_p10"]) & (y <= bt["raw_p90"]))),
        "p10_p90_nominal": 0.8,
        "mean_interval_width_tonnes": float(np.mean(bt["p90"] - bt["p10"])),
        "mean_actual_tonnes": float(np.mean(y)),
        "shortfall_classification": {
            "definition": f"shortfall event = actual < target x (1 - {shortfall_tol}); predicted when P50 < target x (1 - {shortfall_tol})",
            **shortfall_scores(y, bt["p50"].to_numpy(), bt["target"].to_numpy(), shortfall_tol),
        },
    }
    m = res["model_p50"]["mae"]
    b = min(res["baseline_previous_period"]["mae"], res["baseline_moving_average_4"]["mae"])
    res["model_beats_best_baseline_mae"] = bool(m < b)
    res["mae_improvement_vs_best_baseline_pct"] = float((b - m) / b * 100)
    return res


def quantiles_validated(res, cov_tol=0.08, hit_tol=0.08):
    cov_ok = abs(res["p10_p90_coverage"] - res["p10_p90_nominal"]) <= cov_tol
    hits = res["quantile_hit_rate"]
    hit_ok = all(abs(hits[k] - q) <= hit_tol for k, q in zip(QKEYS, QUANTILES))
    return bool(cov_ok and hit_ok), {
        "rule": f"|P10-P90 coverage - 0.80| <= {cov_tol} and every quantile hit rate within +/-{hit_tol} of nominal on the untouched test window",
        "coverage_ok": bool(cov_ok),
        "hit_rates_ok": bool(hit_ok),
    }
