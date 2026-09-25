"""GEO-MN API — uncertainty-aware manganese supply-continuity decision support (SIH 2026, PS 26009).

Modular monolith: routes here, logic in services/, training pipeline in ml/.

Decision loop: forecast -> uncertainty -> model contributions -> robust recovery ->
residual gap -> horizon gate -> (strategic) exploration contingency -> target
priority -> why this target now -> decision flip -> reconciliation.

Environment
  PORT                 server port (default 8000)
  GEOMN_CORS_ORIGINS   comma-separated allowed origins for cross-origin use
                       (default: none — the frontend is served same-origin)
  GEOMN_LIVE_EO        1/0 enable live coordinate feature extraction (default 1)
  GEOMN_DECISION_LOG   path of the decision review log (JSON Lines)
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Optional

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from services import contingency_service, decision_service, demo_service, recovery_service, trust_service
from services.common import API_VERSION, ROOT, ApiError, load_manifest
from services.exploration_service import get_service as exploration
from services.health_service import health as health_status
from services.production_service import get_service as production

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("geomn")

app = FastAPI(
    title="GEO-MN Supply-Continuity Decision Support API",
    version=API_VERSION,
    description="Relative manganese prospectivity, production-risk forecasting, robust recovery and "
                "exploration-contingency decisions. Operational data are SYNTHETIC; see /api/trust/provenance.",
)

_origins = [o.strip() for o in os.environ.get("GEOMN_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], allow_credentials=False)


# ---------------------------------------------------------------------------
# Error handling — structured bodies, never stack traces
# ---------------------------------------------------------------------------

@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status, content={"error": exc.code, "message": exc.message, **exc.extra})


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e.get("loc", []) if p != "body"), "message": e.get("msg")}
               for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "VALIDATION_ERROR",
                                                  "message": "Request body or parameters failed validation.",
                                                  "details": details})


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException):
    code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
    return JSONResponse(status_code=exc.status_code, content={"error": code, "message": str(exc.detail)})


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.exception("unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "message": "Unexpected server error."})


# ---------------------------------------------------------------------------
# Request schemas (unknown fields are ignored so the frontend can evolve)
# ---------------------------------------------------------------------------

class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PredictRequest(_Loose):
    lat: float = Field(..., description="Latitude in decimal degrees")
    lon: float = Field(..., description="Longitude in decimal degrees")
    mode: Optional[str] = Field("AUTO", description="AUTO | LIVE_ONLY | CACHED_ONLY | SIMULATE_LIVE_FAILURE")


class ForecastRequest(_Loose):
    mine_id: str = "DEMO_MINE"
    forecast_origin: Optional[str] = None
    horizon_days: Optional[int] = 7
    target_tonnes: Optional[float] = None
    base_state: Optional[dict[str, Any]] = None
    conditions: Optional[dict[str, Any]] = None


class ScenarioRequest(_Loose):
    """Shared by recovery / contingency. Accepts both the documented contract and the
    compact frontend form ({scenario, actions, conditions})."""
    mine_id: str = "DEMO_MINE"
    target_tonnes: Optional[float] = None
    forecast_origin: Optional[str] = None
    base_state: Optional[dict[str, Any]] = None
    conditions: Optional[dict[str, Any]] = None
    disruption_scenarios: Optional[list[str]] = None
    action_portfolios: Optional[list[Any]] = None
    actions: Optional[list[str]] = None
    scenario: Optional[str] = None
    horizon: Optional[str] = None
    decision_horizon: Optional[str] = None
    forecast: Optional[dict[str, Any]] = None
    recovery: Optional[dict[str, Any]] = None


class FlipRequest(_Loose):
    mine_id: str = "DEMO_MINE"
    horizon: Optional[str] = None
    scenario: Optional[str] = None
    perturbed_scenario: Optional[str] = None
    actions: Optional[list[str]] = None
    target_tonnes: Optional[float] = None
    baseline_conditions: Optional[dict[str, Any]] = None
    perturbed_conditions: Optional[dict[str, Any]] = None


class ReviewRequest(_Loose):
    mine_id: str = "DEMO_MINE"
    decision_state: str
    action: Optional[str] = "ACCEPT"
    target_id: Optional[str] = None
    next_target: Optional[str] = None
    decision_horizon: Optional[str] = None
    selected_portfolio: Optional[str] = None
    scenario: Optional[str] = None
    assumptions: Optional[dict[str, Any]] = None
    conditions: Optional[dict[str, Any]] = None
    notes: Optional[str] = None


def _scenario_kwargs(r: ScenarioRequest) -> dict:
    return dict(target_tonnes=r.target_tonnes, base_state=r.base_state, conditions=r.conditions,
                forecast_origin=r.forecast_origin, disruption_scenarios=r.disruption_scenarios,
                action_portfolios=r.action_portfolios, actions=r.actions, scenario=r.scenario)


# ---------------------------------------------------------------------------
# Cached deterministic computations (same inputs -> same outputs)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _contingency_cached(mine_id: str, horizon: Optional[str]) -> str:
    return json.dumps(contingency_service.evaluate(mine_id, horizon))


def _contingency_default(mine_id: str, horizon: Optional[str] = None) -> dict:
    sc = demo_service.resolve(mine_id)
    h = demo_service.normalise_horizon(horizon, sc.get("horizon", "STRATEGIC"))
    return json.loads(_contingency_cached(sc["mine_id"], h))


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------

@app.get("/api/health")
def api_health():
    return health_status()


@app.get("/api/model/manifest")
def model_manifest():
    return load_manifest()


@app.get("/api/demo/scenarios")
def demo_scenarios():
    s = demo_service.scenarios()
    return {"note": s["_note"], "version": s["version"],
            "scenarios": [{"mine_id": k, **{kk: vv for kk, vv in v.items() if kk != "expected_state"}}
                          for k, v in s["scenarios"].items()]}


# ---------------------------------------------------------------------------
# Supply command (one-screen summary of the whole decision loop)
# ---------------------------------------------------------------------------

@app.get("/api/supply-command")
def supply_command(mine_id: str = Query("DEMO_MINE"), horizon: Optional[str] = Query(None)):
    sc = demo_service.resolve(mine_id)
    con = _contingency_default(mine_id, horizon)
    fc = production().forecast(sc["mine_id"], explain=True)
    out = {
        "mine_id": sc["mine_id"],
        "demo_state": sc.get("demo_state"),
        "scenario_label": sc.get("label"),
        "forecast_origin": fc["forecast_origin"],
        "forecast_horizon": "next_period",
        "forecast_period": {"start": fc["period_start"], "end": fc["period_end"], "days": fc["horizon_days"]},
        "target_tonnes": fc["target_tonnes"],
        "p10_tonnes": fc["p10_tonnes"],
        "p50_tonnes": fc["p50_tonnes"],
        "p90_tonnes": fc["p90_tonnes"],
        "gap_p50_tonnes": fc["gap_p50_tonnes"],
        "gap_pct": fc["gap_pct"],
        "risk_state": fc["risk_state"],
        "risk_policy_version": fc["risk_policy_version"],
        "quantiles_validated": fc["quantiles_validated"],
        "primary_drivers": fc["drivers"],
        "model_contributions": fc["model_contributions"],
        "production_applicability": fc["applicability"]["level"],
        "best_operational_action": con["best_operational_action"],
        "action_required": con["action_required"],
        "action_note": con["action_note"],
        "selected_portfolio": con["selected_portfolio"],
        "expected_recovery_tonnes": con["best_operational_recovery_tonnes"],
        "expected_residual_gap_tonnes": con["expected_residual_gap_tonnes"],
        "worst_case_residual_gap_tonnes": con["worst_case_residual_gap_tonnes"],
        "worst_case_scenario": con["worst_case_scenario"],
        "can_operations_close": con["can_operations_close"],
        "decision_state": con["decision_state"],
        "decision_horizon": con["decision_horizon"],
        "supply_status": con["supply_status"],
        "decision_summary": con["decision_summary"],
        "next_target": con["next_target"],
        "target_priority": con["target_priority"],
        "selected_target_detail": con["selected_target_detail"],
        "why_target_now": con["why_target_now"],
        "reason_codes": con["reason_codes"],
        "review_reasons": con["review_reasons"],
        "strategic_requirement": con["strategic_requirement"],
        "status": con["status"],
        "human_review_required": True,
        "provenance": {**fc["provenance"], "decision": con["provenance"]},
    }
    demo_extra = sc.get("demo_extras") or {}
    if "exploration_query" in demo_extra:
        q = demo_extra["exploration_query"]
        results = []
        for p in q["points"]:
            try:
                results.append(exploration().predict(p["lat"], p["lon"], q.get("mode", "SIMULATE_LIVE_FAILURE")))
            except ApiError as e:
                results.append({"query_lat": p["lat"], "query_lon": p["lon"], "error": e.code, "message": e.message, **e.extra})
        out["satellite_fallback_demo"] = {"description": q.get("description"), "results": results}
    if "decision_flip" in demo_extra:
        f = demo_extra["decision_flip"]
        out["decision_flip"] = contingency_service.decision_flip(
            sc["mine_id"], f.get("baseline_conditions"), f.get("perturbed_conditions"), horizon,
            f.get("scenario"), f.get("perturbed_scenario"))
    return out


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

@app.get("/api/exploration/targets")
def exploration_targets(mine_id: str = Query("DEMO_MINE"), horizon: Optional[str] = Query(None)):
    con = _contingency_default(mine_id, horizon)
    out = exploration().list_targets(con["strategic_requirement"])
    out["mine_id"] = con["mine_id"]
    out["decision_state"] = con["decision_state"]
    out["selected_target"] = con["selected_target"]
    return out


@app.get("/api/exploration/targets/{target_id}")
def exploration_target(target_id: str, mine_id: str = Query("DEMO_MINE"), horizon: Optional[str] = Query(None)):
    con = _contingency_default(mine_id, horizon)
    out = exploration().target_detail(target_id, con["strategic_requirement"])
    out["is_selected_target"] = con["selected_target"] == out["target_id"]
    out["decision_state"] = con["decision_state"]
    return out


@app.post("/api/exploration/predict")
def exploration_predict(req: PredictRequest):
    return exploration().predict(req.lat, req.lon, req.mode)


@app.get("/api/exploration/grid")
def exploration_grid(stride: int = Query(5, ge=1, le=50)):
    pts = exploration().grid_points(stride)
    return {"points": pts, "count": len(pts), "stride_cells": stride,
            "cell_size_deg": 0.01, "value": "prospectivity_rank (relative 0-100, not a probability)",
            "provenance": exploration()._prov("CACHED")}


@app.get("/reserve_grid")
def legacy_reserve_grid():
    """Legacy alias kept for older clients: coarse sample of the new ~1 km prospectivity grid."""
    return exploration().grid_points(10)


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

@app.post("/api/production/forecast")
def production_forecast(req: ForecastRequest):
    return production().forecast(req.mine_id, req.forecast_origin, req.horizon_days, req.target_tonnes,
                                 {**(req.base_state or {}), **(req.conditions or {})})


@app.get("/api/production/history")
def production_history(mine_id: str = Query("DEMO_MINE"), days: int = Query(90)):
    return production().history(mine_id, days)


@app.get("/api/production/reconciliation")
def production_reconciliation(mine_id: str = Query("DEMO_MINE"), periods: int = Query(26, ge=1, le=200)):
    return production().reconciliation(mine_id, periods)


# ---------------------------------------------------------------------------
# Recovery / contingency / decisions
# ---------------------------------------------------------------------------

@app.post("/api/recovery/evaluate")
def recovery_evaluate(req: ScenarioRequest):
    return recovery_service.evaluate(req.mine_id, **_scenario_kwargs(req))


@app.post("/api/contingency/evaluate")
def contingency_evaluate(req: ScenarioRequest):
    return contingency_service.evaluate(req.mine_id, req.horizon or req.decision_horizon, **_scenario_kwargs(req),
                                        client_supplied=bool(req.forecast or req.recovery))


@app.post("/api/decision/flip")
def decision_flip(req: FlipRequest):
    return contingency_service.decision_flip(req.mine_id, req.baseline_conditions, req.perturbed_conditions, req.horizon,
                                             req.scenario, req.perturbed_scenario, req.actions, req.target_tonnes)


@app.post("/api/decision/review")
def decision_review(req: ReviewRequest):
    return decision_service.record(req.model_dump())


@app.get("/api/decision/history")
def decision_history(mine_id: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=500)):
    return decision_service.history(mine_id, limit)


# ---------------------------------------------------------------------------
# Trust
# ---------------------------------------------------------------------------

@app.get("/api/trust/exploration")
def trust_exploration():
    return trust_service.exploration()


@app.get("/api/trust/production")
def trust_production():
    return trust_service.production()


@app.get("/api/trust/provenance")
def trust_provenance():
    return trust_service.provenance_catalogue()


# ---------------------------------------------------------------------------
# Frontend — only the three frontend-owned files are served (never the repo tree)
# ---------------------------------------------------------------------------

_FRONTEND = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}


@app.get("/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
def serve_index():
    return FileResponse(ROOT / "index.html", media_type="text/html")


@app.get("/{name}", include_in_schema=False)
def serve_frontend_file(name: str):
    if name not in _FRONTEND:
        raise ApiError(404, "NOT_FOUND", f"/{name} not found")
    return FileResponse(ROOT / name, media_type=_FRONTEND[name])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
