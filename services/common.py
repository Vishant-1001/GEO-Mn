"""Shared runtime helpers: paths, config, provenance, structured errors."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"

API_VERSION = "1.0"

log = logging.getLogger("geomn")

# Data modes used in every provenance block.
REAL_PUBLIC = "REAL_PUBLIC"
SYNTHETIC = "SYNTHETIC"
SIMULATED = "SIMULATED"
CACHED = "CACHED"
LIVE_COORDINATE_QUERY = "LIVE_COORDINATE_QUERY"
REAL_GOVERNMENT = "REAL_GOVERNMENT"
REAL_MOIL_PUBLIC = "REAL_MOIL_PUBLIC"
REAL_DERIVED = "REAL_DERIVED"


class ApiError(Exception):
    """Raised by services; main.py renders it as {"error": code, "message": ...}."""

    def __init__(self, status: int, code: str, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    with open(CONFIG_DIR / name) as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def load_manifest() -> dict:
    p = MODELS_DIR / "model_manifest.json"
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def file_timestamp(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


def provenance(data_mode: str, model_version: str | None, observation_window: str | None,
               source_timestamp: str | None, fallback_used: bool = False, **extra) -> dict:
    """Uniform provenance block required on every major response."""
    p = {
        "data_mode": data_mode,
        "model_version": model_version,
        "observation_window": observation_window,
        "source_timestamp": source_timestamp,
        "fallback_used": bool(fallback_used),
    }
    p.update({k: v for k, v in extra.items() if v is not None})
    return p


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}
