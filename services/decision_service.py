"""Human decision review log (append-only JSON Lines).

Path: $GEOMN_DECISION_LOG (default data/runtime/decision_log.jsonl, git-ignored).
Each record stores what the reviewer accepted and the decision the engine
recomputes for the same mine at review time, so disagreements are visible.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from services import demo_service
from services.common import DATA_DIR, ApiError, utc_now

ACTIONS = ("ACCEPT", "REJECT", "DEFER", "ESCALATE")
_LOCK = threading.Lock()


def log_path() -> Path:
    return Path(os.environ.get("GEOMN_DECISION_LOG", DATA_DIR / "runtime" / "decision_log.jsonl"))


def record(payload: dict) -> dict:
    from services.contingency_service import STATES

    mine_id = demo_service.resolve(payload.get("mine_id") or "DEMO_MINE")["mine_id"]
    state = str(payload.get("decision_state") or "").upper()
    if state not in STATES:
        raise ApiError(400, "INVALID_INPUT", f"decision_state must be one of {STATES}")
    action = str(payload.get("action") or "ACCEPT").upper()
    if action not in ACTIONS:
        raise ApiError(400, "INVALID_INPUT", f"action must be one of {ACTIONS}")
    target = payload.get("target_id") or payload.get("next_target")
    notes = payload.get("notes")
    if notes is not None and len(str(notes)) > 2000:
        raise ApiError(400, "INVALID_INPUT", "notes must be at most 2000 characters")
    rec = {
        "decision_id": str(uuid.uuid4()),
        "timestamp": utc_now(),
        "mine_id": mine_id,
        "decision_state": state,
        "decision_horizon": payload.get("decision_horizon") or payload.get("horizon"),
        "action": action,
        "review_status": {"ACCEPT": "ACCEPTED", "REJECT": "REJECTED", "DEFER": "DEFERRED", "ESCALATE": "ESCALATED"}[action],
        "target_id": target,
        "next_target": target,
        "selected_portfolio": payload.get("selected_portfolio"),
        "scenario": payload.get("scenario"),
        "assumptions": payload.get("assumptions") or payload.get("conditions") or {},
        "notes": notes,
        "data_mode": "SIMULATED",
        "note": "Human review record for a simulated decision-support output; not an operational instruction.",
    }
    p = log_path()
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec


def history(mine_id: str | None = None, limit: int = 50) -> dict:
    p = log_path()
    rows = []
    if p.exists():
        with open(p) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if mine_id:
        mid = demo_service.resolve(mine_id)["mine_id"]
        rows = [r for r in rows if r.get("mine_id") == mid]
    rows.reverse()
    return {"history": rows[: max(1, min(int(limit), 500))], "count": len(rows)}
