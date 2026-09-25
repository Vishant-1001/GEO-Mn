"""Shared paths and small helpers for the GEO-MN ML pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"
REPORTS_DIR = ROOT / "models" / "reports"

SEED = 42


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
        fh.write("\n")


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    import numpy as np

    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def round_or_none(x, nd=2):
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), nd)
