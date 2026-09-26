"""Train and spatially validate the exploration prospectivity model.

Labels (positive-unlabelled design — unknown is NOT barren)
  * Positives: 0.01 degree cells containing a USGS MRDS manganese record
    (occurrence, prospect, producer or past producer) inside the study area,
    excluding regional/duplicate records (see ml/build_exploration_dataset.py).
  * Background: cells sampled at random from the study area at least
    `exclusion_buffer_km` from every positive. They are UNLABELLED background,
    not confirmed barren ground. Scores are therefore only used as a RELATIVE
    rank (0-100) — never as a probability of a deposit.

Model
  * PU-bagging ensemble: each member sees a bootstrap of the positives and a
    fresh background sample; the mean member score is converted to a percentile
    rank over the study-area grid. Member disagreement gives the uncertainty.

Validation (no random pixel splits)
  * Spatial block CV: 0.25 degree blocks, blocks with positives dealt round-robin
    across folds; every metric computed on held-out blocks only.
  * Region holdout: train west of 79.8E and test east, and vice versa.
  * Feature-family ablation and a check of the legacy classifier.

Run:  python -m ml.train_exploration
"""

from __future__ import annotations

import math
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.common import CONFIG_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, ROOT, haversine_km, load_json, save_json
from ml.eo_features import FEATURES, RECIPE
from ml.uncertainty import percentile_rank, uncertainty_level

MODEL_VERSION = "exploration-pu-ensemble-2.0"
FAMILIES = {
    "spectral_only": ["NDVI", "Iron_Oxide_Index", "Clay_Hydroxyl_Index"],
    "thermal_terrain_only": ["LST_Day_K", "elevation", "slope"],
    "all_six": list(FEATURES),
}
REGION_SPLIT_LON = 79.8
AREA_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data(cfg):
    res = cfg["grid_res_deg"]
    grid = pd.read_csv(DATA_DIR / "exploration_grid_features.csv.gz")
    mrds = pd.read_csv(DATA_DIR / "mrds_mn_occurrences.csv")
    pos = mrds[mrds["label_use"] == "POSITIVE"].reset_index(drop=True)

    grid["ci"] = np.floor(grid["lat"] / res + 1e-9).astype(int)
    grid["cj"] = np.floor(grid["lon"] / res + 1e-9).astype(int)
    pkeys = {(math.floor(a / res), math.floor(b / res)) for a, b in zip(pos["lat"], pos["lon"])}
    grid["label"] = [(i, j) in pkeys for i, j in zip(grid["ci"], grid["cj"])]
    grid["complete"] = grid[FEATURES].notna().all(axis=1)

    d = haversine_km(grid["lat"].to_numpy()[:, None], grid["lon"].to_numpy()[:, None],
                     pos["lat"].to_numpy()[None, :], pos["lon"].to_numpy()[None, :])
    grid["dist_to_known_km"] = d.min(axis=1)
    grid["eligible_bg"] = grid["complete"] & ~grid["label"] & (grid["dist_to_known_km"] > cfg["background"]["exclusion_buffer_km"])

    area = cfg["study_area"]
    b = cfg["spatial_cv"]["block_size_deg"]
    grid["block"] = (np.floor((grid["lat"] - area["lat_min"]) / b).astype(int).astype(str) + "_"
                     + np.floor((grid["lon"] - area["lon_min"]) / b).astype(int).astype(str))
    return grid, pos


def make_member(kind, seed):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=10, l2_regularization=1.0,
                                              class_weight="balanced", random_state=seed)
    return RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=5, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=2, random_state=seed)


def train_members(train, feats, n_members, n_bg, seed, kind):
    rng = np.random.default_rng(seed)
    P = train[train["label"] & train["complete"]]
    B = train[train["eligible_bg"]]
    members = []
    for k in range(n_members):
        pi = rng.integers(0, len(P), len(P))
        bi = rng.choice(len(B), size=min(n_bg, len(B)), replace=False)
        X = pd.concat([P.iloc[pi][feats], B.iloc[bi][feats]])
        y = np.r_[np.ones(len(pi)), np.zeros(len(bi))]
        members.append(make_member(kind, seed + k).fit(X, y))
    return members


def mean_score(members, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def region_metrics(cells, score):
    """Metrics on one held-out region. `cells` = all complete cells of the region."""
    pct = percentile_rank(np.sort(score), score)
    is_pos = cells["label"].to_numpy()
    is_bg = cells["eligible_bg"].to_numpy()
    out = {"cells": int(len(cells)), "positives": int(is_pos.sum())}
    if is_pos.sum() == 0 or is_bg.sum() == 0:
        return out, pct
    m = is_pos | is_bg
    out["roc_auc"] = float(roc_auc_score(is_pos[m], pct[m]))
    out["pr_auc"] = float(average_precision_score(is_pos[m], pct[m]))
    out["pr_auc_prevalence_baseline"] = float(is_pos[m].mean())
    return out, pct


def pooled_metrics(pcts, labels, bgs):
    pct = np.concatenate(pcts)
    pos = np.concatenate(labels)
    bg = np.concatenate(bgs)
    m = pos | bg
    out = {
        "positives_evaluated": int(pos.sum()),
        "roc_auc": float(roc_auc_score(pos[m], pct[m])),
        "pr_auc": float(average_precision_score(pos[m], pct[m])),
        "pr_auc_prevalence_baseline": float(pos[m].mean()),
    }
    # top-area capture: share of held-out positive cells whose rank lies in the
    # top x % of their held-out region's area
    out["top_area_capture"] = {f"top_{int(f * 100)}pct_area": float(np.mean(pct[pos] >= 100 * (1 - f)))
                               for f in (0.05, 0.1, 0.2)}
    # success-rate curve (capture vs area), pooled over regions via within-region percentiles
    curve = [{"area_fraction": f, "captured_fraction": float(np.mean(pct[pos] >= 100 * (1 - f)))} for f in AREA_FRACTIONS]
    xs = np.r_[0.0, [c["area_fraction"] for c in curve]]
    ys = np.r_[0.0, [c["captured_fraction"] for c in curve]]
    out["success_rate_curve"] = curve
    out["success_rate_auc"] = float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2))
    return out


def fold_assignment(grid, n_folds, seed):
    rng = np.random.default_rng(seed)
    counts = grid.groupby("block")["label"].sum()
    pos_blocks = counts[counts > 0].sort_values(ascending=False).index.tolist()
    other = counts[counts == 0].index.tolist()
    rng.shuffle(other)
    assign = {}
    for k, b in enumerate(pos_blocks):
        assign[b] = k % n_folds
    for k, b in enumerate(other):
        assign[b] = k % n_folds
    return grid["block"].map(assign)


def spatial_cv(grid, feats, cfg, kind, n_members=5):
    folds = fold_assignment(grid, cfg["spatial_cv"]["n_folds"], cfg["spatial_cv"]["seed"])
    pcts, labels, bgs, per_fold = [], [], [], []
    for f in sorted(folds.unique()):
        train = grid[(folds != f)]
        test = grid[(folds == f) & grid["complete"]]
        members = train_members(train, feats, n_members, cfg["background"]["n_background"], cfg["ensemble"]["seed"], kind)
        m, pct = region_metrics(test, mean_score(members, test[feats]))
        per_fold.append({"fold": int(f), **m})
        pcts.append(pct)
        labels.append(test["label"].to_numpy())
        bgs.append(test["eligible_bg"].to_numpy())
    return {"per_fold": per_fold, **pooled_metrics(pcts, labels, bgs)}


def region_holdout(grid, feats, cfg, kind, n_members=5):
    out = {}
    pcts, labels, bgs = [], [], []
    for name, test_mask in (("train_west_test_east", grid["lon"] >= REGION_SPLIT_LON),
                            ("train_east_test_west", grid["lon"] < REGION_SPLIT_LON)):
        train = grid[~test_mask]
        test = grid[test_mask & grid["complete"]]
        members = train_members(train, feats, n_members, cfg["background"]["n_background"], cfg["ensemble"]["seed"], kind)
        m, pct = region_metrics(test, mean_score(members, test[feats]))
        out[name] = m
        pcts.append(pct)
        labels.append(test["label"].to_numpy())
        bgs.append(test["eligible_bg"].to_numpy())
    out["pooled"] = pooled_metrics(pcts, labels, bgs)
    return out


def legacy_check(grid):
    path = MODELS_DIR / "legacy" / "manganese_model.pkl"
    if not path.exists():
        return {"status": "NOT_AVAILABLE"}
    model = joblib.load(path)
    cols = joblib.load(MODELS_DIR / "legacy" / "feature_columns.pkl")
    cells = grid[grid["complete"]]
    X = cells.rename(columns={"LST_Day_K": "LST_Day_1km"}).copy()
    X["LST_Day_1km"] = X["LST_Day_1km"] / 0.02  # legacy model used raw MODIS DN
    m, _ = region_metrics(cells, model.predict_proba(X[cols])[:, 1])
    pct = percentile_rank(np.sort(model.predict_proba(X[cols])[:, 1]), model.predict_proba(X[cols])[:, 1])
    pos = cells["label"].to_numpy()
    m["top_area_capture"] = {f"top_{int(f * 100)}pct_area": float(np.mean(pct[pos] >= 100 * (1 - f))) for f in (0.05, 0.1, 0.2)}
    m["status"] = "EVALUATED"
    m["caveat"] = ("Legacy training data (X_train.pkl) carry no labels or coordinates, so it is unknown whether these "
                   "MRDS locations were in its training set; this is NOT a clean holdout and is likely optimistic. "
                   "It also used SRTM (not NASADEM) and a different Sentinel-2 compositing recipe.")
    return m


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    cfg = load_json(CONFIG_DIR / "exploration_config.json")
    exp_path = REPORTS_DIR / "exploration_experiments.json"
    experiments = load_json(exp_path) if exp_path.exists() else None
    if experiments is not None:
        # Feature set and label set were chosen by ml/exploration_experiments.py on the development region only.
        from ml.exploration_experiments import ABLATION, BASELINE_SIX, load_grid, with_labels

        dep = experiments["final_decision"]["deployed"]
        m = experiments["models"][dep]
        feats = ABLATION.get(m["features"], BASELINE_SIX)
        raw, mrds, nmet = load_grid(cfg)
        grid = with_labels(raw, m["labels"])
        pos = pd.concat([mrds[["lat", "lon"]], nmet[["lat", "lon"]]]) if m["labels"] == "B" else mrds
        kind, best = "hgb", f"hgb:{dep}"
        print(f"cells {len(grid)}, complete {int(grid['complete'].sum())}, positive cells {int(grid['label'].sum())}; "
              f"deployed configuration {dep} ({len(feats)} features, label set {m['labels']})")
        ablation = {best: spatial_cv(grid, feats, cfg, kind)}
    else:
        grid, pos = load_data(cfg)
        print(f"cells {len(grid)}, complete {int(grid['complete'].sum())}, positive cells {int(grid['label'].sum())}, "
              f"eligible background {int(grid['eligible_bg'].sum())}")
        ablation = {}
        for kind in ("hgb", "rf"):
            for fam, fs in FAMILIES.items():
                if kind == "rf" and fam != "all_six":
                    continue
                ablation[f"{kind}:{fam}"] = spatial_cv(grid, fs, cfg, kind)
        best = max(ablation, key=lambda k: (round(ablation[k]["roc_auc"], 3), k.endswith("all_six")))
        kind, fam = best.split(":")
        feats = FAMILIES[fam]
    for key, a in ablation.items():
        print(f"[{key}] spatial-CV ROC-AUC {a['roc_auc']:.3f} PR-AUC {a['pr_auc']:.3f} "
              f"(base {a['pr_auc_prevalence_baseline']:.3f}) capture@10% {a['top_area_capture']['top_10pct_area']:.2f}")
    print(f"selected {best}")
    # applicability uses only features that exist for every scored cell (geology may be missing -> NaN)
    iso_feats = [f for f in feats if not f.startswith(("geom_", "lineament_"))]
    holdout = region_holdout(grid, feats, cfg, kind)
    print(f"region holdout pooled ROC-AUC {holdout['pooled']['roc_auc']:.3f}")
    legacy = legacy_check(grid) if experiments is None else {"status": "SKIPPED (feature set differs from the legacy model)"}
    print(f"legacy check: {legacy}")

    # ---- final ensemble on all labelled data ------------------------------
    n_members = cfg["ensemble"]["n_members"]
    members = train_members(grid, feats, n_members, cfg["background"]["n_background"], cfg["ensemble"]["seed"], kind)
    cells = grid[grid["complete"]].copy()
    S = np.column_stack([m.predict_proba(cells[feats])[:, 1] for m in members])
    qs = np.linspace(0, 1, 1001)
    member_refs = [np.quantile(S[:, k], qs) for k in range(n_members)]
    ens = S.mean(axis=1)
    ensemble_ref = np.quantile(ens, qs)
    rank = percentile_rank(ensemble_ref, ens)
    member_ranks = np.column_stack([percentile_rank(member_refs[k], S[:, k]) for k in range(n_members)])
    sd = member_ranks.std(axis=1)

    ut = cfg["uncertainty_thresholds_rank_sd"]
    iso_rng = np.random.default_rng(cfg["ensemble"]["seed"])
    iso_idx = iso_rng.choice(len(cells), size=min(10000, len(cells)), replace=False)
    iso = IsolationForest(n_estimators=300, random_state=cfg["ensemble"]["seed"]).fit(cells.iloc[iso_idx][iso_feats])
    iso_scores = iso.score_samples(cells[iso_feats])
    ap = cfg["applicability"]
    q_mod = float(np.quantile(iso_scores[iso_idx], ap["moderate_below_train_quantile"]))
    q_low = float(np.quantile(iso_scores[iso_idx], ap["low_below_train_quantile"]))
    appl = np.where(iso_scores >= q_mod, "HIGH", np.where(iso_scores >= q_low, "MODERATE", "LOW"))

    out = grid[["lat", "lon"]].copy()
    out["prospectivity_rank"] = np.nan
    out["rank_sd"] = np.nan
    out["uncertainty"] = "UNAVAILABLE"
    out["applicability"] = "UNAVAILABLE"
    out["data_status"] = "INSUFFICIENT_DATA"
    idx = cells.index
    out.loc[idx, "prospectivity_rank"] = np.round(rank, 1)
    out.loc[idx, "rank_sd"] = np.round(sd, 1)
    out.loc[idx, "uncertainty"] = uncertainty_level(sd, ut["low_below"], ut["high_above"])
    out.loc[idx, "applicability"] = appl
    out.loc[idx, "data_status"] = "OK"
    out["known_mn_record_in_cell"] = grid["label"].astype(int)
    if "label_mrds" in grid:
        out["known_mn_record_in_cell"] = grid["label_mrds"].astype(int)
        out["indian_government_mn_evidence_in_cell"] = grid["label_nmet"].astype(int)
    out.to_csv(DATA_DIR / "exploration_grid.csv", index=False)

    joblib.dump(members, MODELS_DIR / "exploration_model.pkl", compress=3)
    joblib.dump(list(feats), MODELS_DIR / "exploration_feature_columns.pkl")
    joblib.dump({
        "member_refs": member_refs,
        "ensemble_ref": ensemble_ref,
        "isolation_forest": iso,
        "iso_q_moderate": q_mod,
        "iso_q_low": q_low,
        "feature_ranges": {c: {"min": float(cells[c].min()), "max": float(cells[c].max())} for c in feats},
        "features": list(feats),
        "iso_features": list(iso_feats),
    }, MODELS_DIR / "exploration_train_reference.pkl", compress=3)

    unc_counts = out.loc[idx, "uncertainty"].value_counts().to_dict()
    appl_counts = out.loc[idx, "applicability"].value_counts().to_dict()
    report = {
        "model_version": MODEL_VERSION,
        "labels": {
            "positive_cells": int(grid["label"].sum()),
            "positive_records": int(len(pos)),
            "eligible_background_cells": int(grid["eligible_bg"].sum()),
            "background_per_member": cfg["background"]["n_background"],
            "negative_construction": ("Positive-unlabelled: background cells are drawn at random from the study area "
                                      f">{cfg['background']['exclusion_buffer_km']} km from any MRDS Mn record. "
                                      "They are unlabelled, not proven barren; unknown != barren."),
        },
        "ablation_spatial_cv": ablation,
        "experiments_report": "models/reports/exploration_experiments.json" if experiments is not None else None,
        "selected": best,
        "spatial_block_cv": ablation[best],
        "region_holdout": holdout,
        "legacy_model_check": legacy,
        "grid_summary": {"cells": int(len(out)), "scored": int(len(idx)),
                         "uncertainty_counts": unc_counts, "applicability_counts": appl_counts,
                         "rank_sd_quantiles": {str(q): float(np.quantile(sd, q)) for q in (0.1, 0.5, 0.9)}},
    }
    save_json(REPORTS_DIR / "exploration_validation.json", report)

    manifest_path = MODELS_DIR / "model_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    cv = ablation[best]
    manifest["exploration"] = {
        "model_version": MODEL_VERSION,
        "model_type": (f"PU-bagging ensemble of {n_members} "
                       + ("HistGradientBoostingClassifier" if kind == "hgb" else "RandomForestClassifier")
                       + " members; output = percentile rank of the mean member score over the study-area grid"),
        "output_semantics": "prospectivity_rank 0-100 = relative rank within the study area. It is NOT a probability of a deposit, reserve or tonnage.",
        "features": list(feats),
        "feature_recipe": RECIPE,
        "study_area": cfg["study_area"],
        "effective_resolution": "0.01 degree cells (~1.1 km x 1.0 km)",
        "observation_window": RECIPE["observation_window"],
        "training_labels": report["labels"],
        "validation_method": (f"spatial block CV ({cfg['spatial_cv']['block_size_deg']} degree blocks, "
                              f"{cfg['spatial_cv']['n_folds']} folds) + east/west region holdout at {REGION_SPLIT_LON}E"),
        "validation_metrics": {
            "spatial_cv_roc_auc": cv["roc_auc"],
            "spatial_cv_pr_auc": cv["pr_auc"],
            "spatial_cv_pr_auc_prevalence_baseline": cv["pr_auc_prevalence_baseline"],
            "spatial_cv_top_area_capture": cv["top_area_capture"],
            "spatial_cv_success_rate_auc": cv["success_rate_auc"],
            "region_holdout_roc_auc": holdout["pooled"]["roc_auc"],
            "region_holdout_top_area_capture": holdout["pooled"]["top_area_capture"],
        },
        "uncertainty_method": (f"std-dev of member percentile ranks across {n_members} PU-bagging members; "
                               f"LOW < {ut['low_below']}, HIGH > {ut['high_above']} rank points (project thresholds)"),
        "applicability_method": (f"geographic envelope + IsolationForest on study-area feature space; MODERATE below the "
                                 f"{ap['moderate_below_train_quantile']:.0%} and LOW below the {ap['low_below_train_quantile']:.0%} "
                                 "training score quantile"),
        "data_provenance": {
            "labels": ("REAL_PUBLIC — USGS MRDS manganese records (mrdata.usgs.gov)"
                       + ("; REAL_GOVERNMENT — NMET sample / block evidence cells" if "label_nmet" in grid else "")),
            "features": "REAL_DERIVED — Sentinel-2 L2A, MODIS MOD11A2, NASADEM via Microsoft Planetary Computer",
            **({"geology_features": "REAL_GOVERNMENT — NRSC Bhuvan 1:50k geomorphology (MP, MH layers)"}
               if any(f.startswith("geom_") for f in feats) else {}),
        },
        "selection": ("ml/exploration_experiments.py: feature set and label set chosen on the eastern development "
                      "region, confirmed once on the untouched western region (models/reports/exploration_experiments.json)"
                      if experiments is not None else "spatial-CV ROC-AUC over feature families"),
        "limitations": [
            "Positive labels are MRDS point records of mixed location precision; they cluster in two belts.",
            "Background is unlabelled, not barren, so ROC/PR values are indicative lower-bound style comparisons.",
            "Surface spectra, terrain and geomorphology cannot see subsurface ore; no drilling, assay or geophysics is used as a feature.",
            "Geomorphology is mapped for MP and MH only (~85 % of cells); elsewhere it is missing, not zero.",
            "Evidence for the richer feature stack is mixed (see supplementary_full_area in the experiments report).",
            "Ranks are relative to this study area only and are not transferable as absolute scores.",
            "Observation window is the fixed 2024 composite, not real-time imagery.",
        ],
    }
    save_json(manifest_path, manifest)
    print("saved exploration model, grid and report")


if __name__ == "__main__":
    main()
