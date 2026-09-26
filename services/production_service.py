"""Production forecasting service: history, P10/P50/P90 forecast, risk, drivers, reconciliation.

Forecast unit: total tonnes in the 7-day period starting at the forecast origin.
Operations data are SYNTHETIC; weather is REAL_GOVERNMENT (IMD gridded rainfall / Tmax) with ERA5 soil moisture (see provenance).
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from ml.generate_operations import planned_daily_target
from ml.production_features import (
    FEATURE_COLUMNS, FEATURE_DOCS, HUMAN, PERIOD_DAYS, SCENARIO_FEATURES, features_at, load_daily,
)
from services import demo_service
from services.common import (
    DATA_DIR, MODELS_DIR, SIMULATED, SYNTHETIC, ApiError, file_timestamp, load_config, load_manifest, log, provenance,
)

QKEYS = ("p10", "p50", "p90")

# Accepted request aliases -> model feature names.
STATE_ALIASES = {
    "rainfall_mm": "rainfall_7d_mm",
    "rainfall": "rainfall_7d_mm",
    "rainfall_7d": "rainfall_7d_mm",
    "blast_delay_hours": "blast_delay_h",
    "blast_delay": "blast_delay_h",
    "drilling_delay_hours": "drilling_delay_h",
    "drilling_delay": "drilling_delay_h",
    "haulage_delay_hours": "haulage_delay_h",
    "haulage_delay": "haulage_delay_h",
    "equipment_downtime": "equipment_downtime_h",
    "equipment_downtime_hours": "equipment_downtime_h",
    "maintenance_h": "maintenance_hours",
    "soil_moisture": "soil_moisture_m3m3",
    "temperature": "temperature_max_c",
    "temperature_c": "temperature_max_c",
    "trucks": "truck_count",
}
# (min, max) physically valid ranges for simulation inputs; None = unbounded.
VALID_RANGES = {
    "equipment_availability": (0.0, 1.0),
    "equipment_downtime_h": (0.0, 24.0),
    "maintenance_hours": (0.0, 24.0),
    "drilling_delay_h": (0.0, 24.0),
    "blast_delay_h": (0.0, 24.0),
    "truck_count": (0.0, None),
    "haulage_delay_h": (0.0, 24.0),
    "rainfall_7d_mm": (0.0, 2000.0),
    "soil_moisture_m3m3": (0.0, 1.0),
    "temperature_max_c": (-20.0, 60.0),
}


def normalise_state(raw: dict | None) -> dict:
    """Map aliases to feature names and validate ranges. Raises 400 on invalid values."""
    out = {}
    for k, v in (raw or {}).items():
        key = STATE_ALIASES.get(k, k)
        if key not in VALID_RANGES:
            continue  # unknown keys are ignored (reported back as ignored_inputs by callers if needed)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ApiError(400, "INVALID_INPUT", f"{k} must be numeric")
        if not np.isfinite(fv):
            raise ApiError(400, "INVALID_INPUT", f"{k} must be finite")
        lo, hi = VALID_RANGES[key]
        if (lo is not None and fv < lo) or (hi is not None and fv > hi):
            raise ApiError(400, "INVALID_INPUT", f"{k}={fv} outside the physically valid range [{lo}, {hi}]")
        out[key] = fv
    return out


class ProductionService:
    def __init__(self):
        self.error = None
        self.models = {}
        try:
            self.daily = load_daily(DATA_DIR / "production_history.csv")
        except Exception as e:  # pragma: no cover
            self.daily = None
            self.error = f"history unavailable: {type(e).__name__}"
        try:
            for k in QKEYS:
                self.models[k] = joblib.load(MODELS_DIR / f"production_{k}.pkl")
            self.feature_columns = joblib.load(MODELS_DIR / "production_feature_columns.pkl")
            self.calibration = json.loads((MODELS_DIR / "production_calibration.json").read_text())
            self.reference = joblib.load(MODELS_DIR / "production_train_reference.pkl")
            if list(self.feature_columns) != list(FEATURE_COLUMNS):
                raise ValueError("feature schema mismatch between model and code")
        except Exception as e:
            self.models = {}
            self.error = f"production models unavailable: {type(e).__name__}: {e}"
            log.warning(self.error)
        self._explainer = None
        try:
            self.backtest = pd.read_csv(DATA_DIR / "production_backtest_forecasts.csv")
        except Exception:
            self.backtest = None

    # ------------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        return bool(self.models) and self.daily is not None

    def _require(self):
        if not self.available:
            raise ApiError(503, "MODEL_UNAVAILABLE", "Production forecast models or history are not loaded.")

    @property
    def model_version(self):
        return self.calibration.get("model_version") if self.models else None

    @property
    def quantiles_validated(self) -> bool:
        return bool(load_manifest().get("production", {}).get("quantiles_validated", False))

    def quantile_validation(self) -> dict:
        """Backend-measured interval evidence (never computed or assumed by the client)."""
        man = load_manifest().get("production", {})
        vm = man.get("validation_metrics", {})
        if not vm:
            return {"status": "NOT_AVAILABLE", "nominal_coverage": 0.8, "observed_coverage": None}
        validated = bool(man.get("quantiles_validated", False))
        return {
            "status": "VALIDATED" if validated else "NOT_VALIDATED",
            "nominal_coverage": vm.get("p10_p90_nominal", 0.8),
            "observed_coverage": round(vm["p10_p90_coverage"], 3),
            "quantile_hit_rate": {k: round(v, 3) for k, v in vm.get("quantile_hit_rate", {}).items()},
            "evaluation_window": "/".join(vm.get("test_window", [])),
            "n_periods": vm.get("n_test_periods"),
            "rule": man.get("quantile_validation_rule"),
            "note": ("Observed coverage is the share of untouched backtest periods (synthetic data) whose actual fell "
                     "inside P10-P90; the nominal level is what the interval is designed for."
                     + ("" if validated else " The interval did NOT pass the validation rule: treat P10/P90 as indicative only.")),
        }

    def history_end(self) -> pd.Timestamp:
        return self.daily.index[-1]

    def default_origin(self) -> pd.Timestamp:
        return pd.Timestamp(load_config("demo_config.json")["forecast_origin"])

    # ------------------------------------------------------------ prediction
    def _base(self, X: pd.DataFrame):
        if self.calibration.get("ratio_target"):
            return np.maximum(X[self.calibration["ratio_base_feature"]].to_numpy(dtype=float), 1.0)
        return np.ones(len(X))

    def predict(self, rows: list[dict]) -> np.ndarray:
        """Calibrated, non-crossing (n, 3) array of P10/P50/P90 tonnes."""
        self._require()
        X = pd.DataFrame(rows)[FEATURE_COLUMNS]
        base = self._base(X)
        raw = np.sort(np.clip(np.column_stack([self.models[k].predict(X) * base for k in QKEYS]), 0, None), axis=1)
        off = self.calibration["offsets_relative_to_raw_p50"]
        adj = raw + np.column_stack([off[k] * raw[:, 1] for k in QKEYS])
        return np.sort(np.clip(adj, 0, None), axis=1)

    def explainer(self):
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.models["p50"])
        return self._explainer

    def contributions(self, row: dict, p50: float) -> dict:
        """SHAP contributions of the P50 model, converted to tonnes. Associative, not causal."""
        try:
            X = pd.DataFrame([row])[FEATURE_COLUMNS]
            base = float(self._base(X)[0])
            sv = self.explainer().shap_values(X)[0]
            ev = float(np.ravel(self.explainer().expected_value)[0])
        except Exception as e:
            return {"status": "UNAVAILABLE", "reason": type(e).__name__, "items": []}
        items = []
        for f, v in zip(FEATURE_COLUMNS, sv):
            items.append({
                "feature": f,
                "label": HUMAN[f],
                "value": round(float(row[f]), 4),
                "unit": FEATURE_DOCS[f][0],
                "contribution": round(float(v), 5),
                "contribution_tonnes": round(float(v) * base, 1),
                "direction": "DECREASES_FORECAST" if v < 0 else "INCREASES_FORECAST",
            })
        items.sort(key=lambda d: -abs(d["contribution_tonnes"]))
        explained = ev * base + float(np.sum(sv)) * base
        return {
            "status": "AVAILABLE",
            "method": "SHAP TreeExplainer on the P50 model; contributions in model units (ratio to the 4-period mean) "
                      "and converted to tonnes. Model contributions are associations learned from the data, not causal effects.",
            "base_value_tonnes": round(ev * base, 1),
            "calibration_adjustment_tonnes": round(p50 - explained, 1),
            "items": items,
        }

    def applicability(self, row: dict) -> dict:
        ref = self.reference
        X = pd.DataFrame([row])[ref["features"]]
        score = float(ref["isolation_forest"].score_samples(X)[0])
        level = "HIGH" if score >= ref["score_q05"] else ("MODERATE" if score >= ref["score_q01"] else "LOW")
        viol = []
        for c, r in ref["ranges"].items():
            v = float(row[c])
            if v < r["min"] or v > r["max"]:
                viol.append({"feature": c, "value": round(v, 3), "train_min": round(r["min"], 3), "train_max": round(r["max"], 3)})
        if viol:
            level = "LOW"
        return {"level": level, "score": round(score, 4), "range_violations": viol,
                "method": "IsolationForest on training operating-state/weather features + training min/max range check"}

    # ---------------------------------------------------------------- policy
    def risk(self, target: float, p10: float, p50: float) -> dict:
        pol = load_config("risk_policy.json")
        if target <= 0:
            return {"risk_state": "NO_TARGET", "gap_pct": None, "risk_policy_version": pol["version"]}
        gap_pct = max(0.0, target - p50) / target * 100
        bands = pol["risk_bands"]
        idx = next(i for i, b in enumerate(bands) if b["max_gap_pct"] is None or gap_pct <= b["max_gap_pct"])
        p10_gap = max(0.0, target - p10) / target * 100
        escalated = p10_gap > pol["escalate_one_band_if_p10_gap_pct_above"] and idx < len(bands) - 1
        if escalated:
            idx += 1
        return {"risk_state": bands[idx]["state"], "gap_pct": round(gap_pct, 2), "p10_gap_pct": round(p10_gap, 2),
                "escalated_by_p10": escalated, "risk_policy_version": pol["version"],
                "risk_policy_note": "Demonstration thresholds (config/risk_policy.json); not an industry standard."}

    # --------------------------------------------------------------- context
    def parse_origin(self, origin) -> pd.Timestamp:
        if origin in (None, ""):
            return self.default_origin()
        try:
            o = pd.Timestamp(origin).normalize()
        except Exception:
            raise ApiError(400, "INVALID_INPUT", "forecast_origin must be an ISO date (YYYY-MM-DD)")
        first_ok = self.daily.index[0] + pd.Timedelta(days=PERIOD_DAYS * 12)
        if o < first_ok:
            raise ApiError(400, "INSUFFICIENT_HISTORY", f"forecast_origin must be on or after {first_ok.date()} (84 days of history needed)")
        if o > self.history_end() + pd.Timedelta(days=1):
            raise ApiError(400, "ORIGIN_OUT_OF_RANGE",
                           f"History ends {self.history_end().date()}; the latest supported forecast_origin is "
                           f"{(self.history_end() + pd.Timedelta(days=1)).date()} (no future observations are used).")
        return o

    def period_target(self, origin: pd.Timestamp) -> float:
        dates = pd.date_range(origin, periods=PERIOD_DAYS, freq="D")
        in_hist = self.daily.reindex(dates)["target_tonnes"]
        planned = pd.Series(planned_daily_target(dates), index=dates)
        return float(in_hist.fillna(planned).sum())

    def period_actual(self, origin: pd.Timestamp):
        win = self.daily.loc[origin: origin + pd.Timedelta(days=PERIOD_DAYS - 1)]
        return float(win["actual_tonnes"].sum()) if len(win) == PERIOD_DAYS else None

    def base_row(self, origin: pd.Timestamp) -> dict:
        return {k: float(v) for k, v in features_at(self.daily, origin).items()}

    def context(self, mine_id: str, origin=None, target_tonnes=None, state_overrides=None) -> dict:
        """Resolve a mine + origin into the feature row, target and applied overrides."""
        self._require()
        sc = demo_service.resolve(mine_id)
        o = self.parse_origin(origin)
        row = self.base_row(o)
        demo_over = normalise_state(sc.get("state_overrides"))
        user_over = normalise_state(state_overrides)
        row.update(demo_over)
        row.update(user_over)
        over = {**demo_over, **user_over}
        if "equipment_availability" in over and not ({"equipment_downtime_h", "maintenance_hours"} & set(over)):
            # keep the operating state self-consistent: lost hours follow the overridden availability
            sched = float(sc["mine"]["scheduled_hours_per_day"])
            row["equipment_downtime_h"] = max(0.0, sched * (1.0 - float(row["equipment_availability"])) - float(row["maintenance_hours"]))
        if target_tonnes is not None:
            try:
                target = float(target_tonnes)
            except (TypeError, ValueError):
                raise ApiError(400, "INVALID_INPUT", "target_tonnes must be numeric")
            if not np.isfinite(target) or target < 0:
                raise ApiError(400, "INVALID_INPUT", "target_tonnes must be a non-negative number")
            target_source = "REQUEST"
        else:
            target = self.period_target(o) * float(sc.get("target_multiplier", 1.0))
            target_source = "DEMO_PLAN" + (f" x {sc['target_multiplier']}" if sc.get("target_multiplier", 1.0) != 1.0 else "")
        return {"scenario": sc, "origin": o, "row": row, "target": target, "target_source": target_source,
                "overrides": {**demo_over, **user_over}, "simulated": bool(demo_over or user_over)}

    # ------------------------------------------------------------ endpoints
    def provenance_block(self, simulated=False, fallback=False):
        return provenance(
            SIMULATED if simulated else SYNTHETIC,
            self.model_version,
            f"{self.daily.index[0].date()}/{self.history_end().date()}",
            file_timestamp(DATA_DIR / "production_history.csv"),
            fallback,
            operations_mode=SYNTHETIC,
            weather_mode="REAL_GOVERNMENT",
            note=("Operational records are SYNTHETIC demonstration data (seed 42), not MOIL data; rainfall and Tmax are "
                  "IMD gridded observations (ERA5-Land soil moisture; ERA5 where IMD is not yet published)." + (" Operating-state inputs were overridden: SIMULATED SCENARIO." if simulated else "")),
        )

    def forecast(self, mine_id="DEMO_MINE", origin=None, horizon_days=PERIOD_DAYS, target_tonnes=None,
                 state_overrides=None, explain=True) -> dict:
        self._require()
        if horizon_days is None:
            horizon_days = PERIOD_DAYS
        try:
            horizon_days = int(horizon_days)
        except (TypeError, ValueError):
            raise ApiError(400, "INVALID_INPUT", "horizon_days must be an integer")
        if horizon_days != PERIOD_DAYS:
            raise ApiError(400, "UNSUPPORTED_HORIZON",
                           f"Only horizon_days={PERIOD_DAYS} (one production period) is supported; longer horizons are "
                           "handled as persistence scenarios by /api/contingency/evaluate.")
        ctx = self.context(mine_id, origin, target_tonnes, state_overrides)
        p10, p50, p90 = (float(x) for x in self.predict([ctx["row"]])[0])
        return self._forecast_payload(ctx, p10, p50, p90, explain)

    def _forecast_payload(self, ctx, p10, p50, p90, explain=True):
        o, target, row = ctx["origin"], ctx["target"], ctx["row"]
        risk = self.risk(target, p10, p50)
        contrib = self.contributions(row, p50) if explain else {"status": "SKIPPED", "items": []}
        drivers = [d for d in contrib["items"] if d["contribution_tonnes"] < 0][:5]
        appl = self.applicability(row)
        actual = self.period_actual(o)
        out = {
            "mine_id": ctx["scenario"]["mine_id"],
            "forecast_origin": str(o.date()),
            "period_start": str(o.date()),
            "period_end": str((o + pd.Timedelta(days=PERIOD_DAYS - 1)).date()),
            "forecast_horizon": f"next_period_{PERIOD_DAYS}_days",
            "horizon_days": PERIOD_DAYS,
            "forecast_type": "RETROSPECTIVE" if actual is not None else "FORWARD",
            "target_tonnes": round(target, 1),
            "target_source": ctx["target_source"],
            "p10_tonnes": round(p10, 1),
            "p50_tonnes": round(p50, 1),
            "p90_tonnes": round(p90, 1),
            "gap_p50_tonnes": round(max(0.0, target - p50), 1),
            "gap_p10_tonnes": round(max(0.0, target - p10), 1),
            "gap_pct": risk["gap_pct"],
            "risk_state": risk["risk_state"],
            "risk_policy_version": risk["risk_policy_version"],
            "risk_detail": risk,
            "quantiles_validated": self.quantiles_validated,
            "quantile_validation": self.quantile_validation(),
            "interval_note": ("P10/P90 passed the backtest validation rule on synthetic data."
                              if self.quantiles_validated else "P10-P90 interval not validated — treat as indicative only."),
            "forecast_method": "conditional_production_forecast",
            "persistence_assumption": True,
            "scenario_override_applied": ctx["simulated"],
            "forecast_basis": ("7-day conditional production forecast: trailing 7-day operating and weather conditions are "
                               "assumed to persist through the period unless a scenario overrides them. This is not a "
                               "weather or operations forecast."),
            "drivers": drivers,
            "model_contributions": contrib,
            "applicability": appl,
            "inputs": {k: round(float(row[k]), 4) for k in SCENARIO_FEATURES},
            "overrides_applied": ctx["overrides"],
            "status": "SIMULATED_SCENARIO" if ctx["simulated"] else "MODEL_ESTIMATE",
            "provenance": self.provenance_block(ctx["simulated"]),
        }
        if actual is not None:
            out["actual_tonnes"] = round(actual, 1)
        return out

    def history(self, mine_id="DEMO_MINE", days=90) -> dict:
        self._require()
        sc = demo_service.resolve(mine_id)
        try:
            days = int(days)
        except (TypeError, ValueError):
            raise ApiError(400, "INVALID_INPUT", "days must be an integer")
        if not 1 <= days <= 3650:
            raise ApiError(400, "INVALID_INPUT", "days must be between 1 and 3650")
        h = self.daily.iloc[-days:]
        cols = ["equipment_availability", "equipment_downtime_h", "maintenance_hours", "drilling_delay_h", "blast_delay_h",
                "truck_count", "haulage_delay_h", "rainfall_mm", "soil_moisture_m3m3", "temperature_max_c"]
        rows = [{"date": str(d.date()), "actual_tonnes": float(r["actual_tonnes"]), "target_tonnes": float(r["target_tonnes"]),
                 **{c: float(r[c]) for c in cols}} for d, r in h.iterrows()]
        wk = h[["actual_tonnes", "target_tonnes"]].iloc[len(h) % PERIOD_DAYS:]
        periods = []
        if len(wk) >= PERIOD_DAYS:
            a = wk["actual_tonnes"].to_numpy().reshape(-1, PERIOD_DAYS).sum(1)
            t = wk["target_tonnes"].to_numpy().reshape(-1, PERIOD_DAYS).sum(1)
            starts = wk.index[::PERIOD_DAYS]
            periods = [{"period_start": str(s.date()), "period_end": str((s + pd.Timedelta(days=PERIOD_DAYS - 1)).date()),
                        "actual_tonnes": round(float(x), 1), "target_tonnes": round(float(y), 1)}
                       for s, x, y in zip(starts, a, t)]
        return {
            "mine_id": sc["mine_id"],
            "base_mine": sc["base_mine"],
            "granularity": "daily",
            "rows": rows,
            "periods": periods,
            "provenance": self.provenance_block(),
        }

    def reconciliation(self, mine_id="DEMO_MINE", periods=26) -> dict:
        sc = demo_service.resolve(mine_id)
        if self.backtest is None or self.backtest.empty:
            return {"status": "NOT_AVAILABLE", "mine_id": sc["mine_id"], "rows": [],
                    "reason": "No forecast/actual pairs are stored for this mine."}
        bt = self.backtest[self.backtest["window"] == "test"].tail(int(periods)).copy()
        bt["error"] = bt["p50"] - bt["actual"]
        rows = []
        errs = []
        for r in bt.itertuples():
            errs.append(r.error)
            rows.append({
                "period": f"{r.origin}/{r.period_end}",
                "period_start": r.origin,
                "period_end": r.period_end,
                "forecast_tonnes": round(r.p50, 1),
                "p10_tonnes": round(r.p10, 1),
                "p90_tonnes": round(r.p90, 1),
                "actual_tonnes": round(r.actual, 1),
                "target_tonnes": round(r.target, 1),
                "error_tonnes": round(r.error, 1),
                "absolute_error_tonnes": round(abs(r.error), 1),
                "percentage_error": round(r.error / r.actual * 100, 2) if r.actual else None,
                "rolling_mae_4_tonnes": round(float(np.mean(np.abs(errs[-4:]))), 1),
                "within_p10_p90": bool(r.p10 <= r.actual <= r.p90),
            })
        e = bt["error"].to_numpy()
        return {
            "status": "AVAILABLE",
            "mine_id": sc["mine_id"],
            "basis": ("Historical diagnostic: retrospective OUT-OF-SAMPLE backtest forecasts (each made by a model trained only "
                      "on earlier data, with frozen calibration offsets) compared with SYNTHETIC actuals. Not live operational "
                      "reconciliation and not proof of future accuracy."),
            "error_convention": "error = forecast - actual (positive = over-forecast)",
            "summary": {
                "periods": int(len(e)),
                "mae": round(float(np.mean(np.abs(e))), 1),
                "rolling_mae": round(float(np.mean(np.abs(e[-4:]))), 1),
                "bias": round(float(np.mean(e)), 1),
                "latest_error_tonnes": round(float(e[-1]), 1),
                "over_forecast_count": int(np.sum(e > 0)),
                "under_forecast_count": int(np.sum(e < 0)),
                "p10_p90_coverage": round(float(np.mean([r["within_p10_p90"] for r in rows])), 3),
            },
            "mae": round(float(np.mean(np.abs(e))), 1),
            "rolling_mae": round(float(np.mean(np.abs(e[-4:]))), 1),
            "bias": round(float(np.mean(e)), 1),
            "rows": rows,
            "provenance": self.provenance_block(),
        }


@lru_cache(maxsize=1)
def get_service() -> ProductionService:
    return ProductionService()
