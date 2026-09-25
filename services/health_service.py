"""Public health status — component availability only, never credentials or paths."""

from __future__ import annotations

from services.common import API_VERSION
from services.exploration_service import get_service as exploration
from services.production_service import get_service as production


def health() -> dict:
    ex, pr = exploration(), production()
    live = ex.live_enabled()
    ok = ex.model_loaded and ex.cache_loaded and pr.available
    return {
        "status": "ok" if ok else "degraded",
        "api_version": API_VERSION,
        "exploration_model": ex.model_loaded,
        "production_models": pr.available,
        "earth_engine": False,
        "live_satellite": live,
        "live_satellite_provider": "Microsoft Planetary Computer (anonymous STAC)" if live else None,
        "cache": ex.cache_loaded,
        "exploration_targets": len(ex.targets),
        "shap": _shap_ok(),
        "notes": ["Earth Engine is not used: live coordinate queries recompute the training feature recipe from "
                  "Planetary Computer so training and inference stay identical."],
    }


def _shap_ok() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False
