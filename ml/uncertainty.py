"""Ensemble-disagreement uncertainty for exploration prospectivity.

Each ensemble member is trained on a different bootstrap of the positives and a
different random background sample (PU bagging). A cell's uncertainty is the
standard deviation, across members, of the cell's percentile rank under each
member's own reference distribution over the study-area grid.
"""

from __future__ import annotations

import numpy as np


def percentile_rank(ref_sorted, scores):
    scores = np.asarray(scores, dtype=float)
    return 100.0 * np.searchsorted(ref_sorted, scores, side="right") / len(ref_sorted)


def member_scores(members, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in members])


def ensemble_rank_and_sd(members, member_refs, ensemble_ref, X):
    S = member_scores(members, X)
    member_ranks = np.column_stack([percentile_rank(member_refs[k], S[:, k]) for k in range(S.shape[1])])
    rank = percentile_rank(ensemble_ref, S.mean(axis=1))
    return rank, member_ranks.std(axis=1), S.mean(axis=1)


def uncertainty_level(sd, low_below, high_above):
    sd = np.asarray(sd, dtype=float)
    return np.where(sd < low_below, "LOW", np.where(sd > high_above, "HIGH", "MODERATE"))
