"""Deterministic demonstration scenarios (data/demo_scenarios.json).

Every demo mine id maps onto the SYNTHETIC history of DEMO_MINE plus a documented
set of overrides (operating state at the origin, period target, horizon). The
decision each one produces is COMPUTED by the forecast / recovery / contingency
engines — the file never states the outcome, only the inputs; `expected_state`
is used by the tests to check that the engines still produce it.
"""

from __future__ import annotations

import json
from functools import lru_cache

from services.common import DATA_DIR, ApiError, load_config

HORIZONS = ("NEAR_TERM", "STRATEGIC")


@lru_cache(maxsize=1)
def scenarios() -> dict:
    with open(DATA_DIR / "demo_scenarios.json") as fh:
        return json.load(fh)


def list_mines() -> list[str]:
    return list(scenarios()["scenarios"].keys())


def resolve(mine_id: str) -> dict:
    """Return the scenario definition for a mine id, or raise 404."""
    if not isinstance(mine_id, str) or not mine_id:
        raise ApiError(400, "INVALID_INPUT", "mine_id is required")
    sc = scenarios()["scenarios"].get(mine_id.strip().upper())
    if sc is None:
        raise ApiError(404, "MINE_NOT_FOUND", f"Unknown mine_id '{mine_id}'. Known: {', '.join(list_mines())}")
    base = sc.get("base_mine", "DEMO_MINE")
    mine_cfg = load_config("demo_config.json")["mines"][base]
    return {"mine_id": mine_id.strip().upper(), "base_mine": base, "mine": mine_cfg, **sc}


def normalise_horizon(h: str | None, default: str) -> str:
    if h is None or h == "":
        return default
    v = str(h).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {"NEAR": "NEAR_TERM", "SHORT_TERM": "NEAR_TERM", "NEXT_PERIOD": "NEAR_TERM", "OPERATIONAL": "NEAR_TERM",
               "LONG_TERM": "STRATEGIC", "LONG": "STRATEGIC", "STRATEGY": "STRATEGIC"}
    v = aliases.get(v, v)
    if v not in HORIZONS:
        raise ApiError(400, "INVALID_INPUT", f"horizon must be one of {HORIZONS}")
    return v
