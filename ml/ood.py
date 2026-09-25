"""Applicability / out-of-distribution helpers shared by training and the API."""

from __future__ import annotations

import numpy as np

LEVELS = ("HIGH", "MODERATE", "LOW")
LEVEL_SCORE = {"HIGH": 1.0, "MODERATE": 0.6, "LOW": 0.2}


def in_envelope(lat, lon, area):
    return area["lat_min"] <= lat <= area["lat_max"] and area["lon_min"] <= lon <= area["lon_max"]


def applicability_from_score(score, q_moderate, q_low):
    """Map an IsolationForest score (higher = more typical) to HIGH / MODERATE / LOW."""
    score = np.asarray(score, dtype=float)
    out = np.where(score >= q_moderate, "HIGH", np.where(score >= q_low, "MODERATE", "LOW"))
    return out


def range_violations(row: dict, ranges: dict):
    """Features whose value lies outside the training min/max."""
    out = []
    for c, r in ranges.items():
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if v < r["min"] or v > r["max"]:
            out.append({"feature": c, "value": float(v), "train_min": r["min"], "train_max": r["max"]})
    return out


def worst(*levels):
    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    return max(levels, key=lambda x: order[x])
