"""Exploration experiments: label sources (Models A-D) and feature-family ablation (A-F).

Design (all REAL data; nothing synthetic is used as a training label):
  * Development region  = east of 79.8E (57 MRDS positive cells). Every selection decision
    (feature family, label set) is taken ONLY from spatial block CV inside this region.
  * Final test region   = west of 79.8E, never used for selection. It is scored ONCE per model
    after selection. It also contains most official NMET / DGM manganese blocks, which act as an
    independent Indian-government check (block-level locations).
  * Metrics per held-out set: ROC-AUC, PR-AUC with its prevalence baseline, precision / recall / F1
    when the top 10 % of the held-out area is flagged, top-k area capture (5 / 10 / 20 %), and
    precision in the top-50 cells. Background is UNLABELLED (positive-unlabelled design), so
    precision values are lower-bound style comparisons, not deposit hit rates.

Label sets
  Model A  MRDS positives only (USGS, REAL_PUBLIC) — the baseline label source.
  Model B  MRDS + Indian-government positives (REAL_GOVERNMENT): NMET Katori XRF samples with
           Mn >= 10 %, and the centroid cells of blocks with a REPORTED drilling intersection or
           reported ore grades from old workings (Nagardhan, Kawalewada-Sakkardara).
  Model C  label set chosen from A/B + the full real feature stack chosen by the ablation.
  Model D  real + synthetic augmentation — NOT RUN: the only synthetic exploration-side data are
           the simulated subsurface scenarios, which are generated FROM this model's targets;
           training on them would be circular leakage. Recorded as such in the report.

Run: python -m ml.exploration_experiments   (needs data/processed/exploration/*_grid.csv*)
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.common import CONFIG_DIR, DATA_DIR, REPORTS_DIR, haversine_km, load_json, save_json
from ml.eo_features import EXTENDED_S2_FEATURES, FEATURES
from ml.train_exploration import REGION_SPLIT_LON, fold_assignment, mean_score, train_members
from ml.uncertainty import percentile_rank

PROC = DATA_DIR / "processed" / "exploration"
SPECTRAL = ["NDVI", "Iron_Oxide_Index", "Clay_Hydroxyl_Index"] + EXTENDED_S2_FEATURES
TERRAIN = ["LST_Day_K", "elevation", "slope", "elevation_sd"]
GEOMORPH = ["geom_structural_origin", "geom_hills_valleys", "geom_plateau", "geom_pediplain", "geom_dissection",
            "geom_water_fluvial"]
STRUCTURAL = ["lineament_dist_km", "lineament_density"]
ABLATION = {
    "A_s2_spectral": SPECTRAL,
    "B_s2_plus_terrain": SPECTRAL + TERRAIN,
    "C_s2_plus_geomorphology": SPECTRAL + GEOMORPH,
    "D_s2_terrain_geomorphology": SPECTRAL + TERRAIN + GEOMORPH,
    "E_structural_geomorphology_only": GEOMORPH + STRUCTURAL,
    "F_full_real_stack": SPECTRAL + TERRAIN + GEOMORPH + STRUCTURAL,
}
BASELINE_SIX = list(FEATURES)
TOP_FLAG = 0.10          # share of held-out area flagged for precision / recall / F1
TOP_CELLS = 50
SEEDS = (42, 7, 13, 21)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def nmet_positive_points():
    """Indian-government positive locations (REAL_GOVERNMENT) with their basis."""
    sub = DATA_DIR / "processed" / "subsurface"
    pts = []
    s = pd.read_csv(sub / "surface_geochem_samples.csv")
    for r in s[s["mn_pct"] >= 10].itertuples():
        pts.append({"lat": r.lat, "lon": r.lon, "source": f"NMET {r.block_id} XRF sample {r.sample_id} ({r.mn_pct}% Mn)"})
    blocks = pd.read_csv(sub / "exploration_blocks.csv").set_index("block_id")
    f = pd.read_csv(sub / "reported_findings.csv")
    strong = f[(f["evidence_class"] == "DRILLING_INTERSECTION_REPORTED")
               | ((f["evidence_class"] == "GRADE_REPORTED") & f["mn_pct_min"].fillna(0).ge(25))]
    for b in sorted(set(strong["block_id"])):
        pts.append({"lat": float(blocks.loc[b, "lat"]), "lon": float(blocks.loc[b, "lon"]),
                    "source": f"NMET/DGM block {b} centroid (reported: "
                              + "; ".join(strong[strong['block_id'] == b]['evidence_class'].unique()) + ")"})
    return pd.DataFrame(pts)


def load_grid(cfg):
    res = cfg["grid_res_deg"]
    eo = pd.read_csv(PROC / "eo_features_grid.csv.gz")
    geo = pd.read_csv(PROC / "geology_features_grid.csv")
    for d in (eo, geo):
        d["ci"] = np.floor(d["lat"] / res + 1e-9).astype(int)
        d["cj"] = np.floor(d["lon"] / res + 1e-9).astype(int)
    geo_cols = GEOMORPH + STRUCTURAL + ["geomorphology_class"]
    grid = eo.merge(geo[["ci", "cj"] + geo_cols], on=["ci", "cj"], how="left")

    mrds = pd.read_csv(DATA_DIR / "mrds_mn_occurrences.csv")
    mrds = mrds[mrds["label_use"] == "POSITIVE"]
    nmet = nmet_positive_points()

    def cell_keys(df):
        return {(math.floor(a / res + 1e-9), math.floor(b / res + 1e-9)) for a, b in zip(df["lat"], df["lon"])}

    keys = list(zip(grid["ci"], grid["cj"]))
    km, kn = cell_keys(mrds), cell_keys(nmet)
    grid["label_mrds"] = [k in km for k in keys]
    grid["label_nmet"] = [k in kn for k in keys]
    grid["complete"] = grid[SPECTRAL + TERRAIN].notna().all(axis=1)

    allpos = pd.concat([mrds[["lat", "lon"]], nmet[["lat", "lon"]]])
    d = haversine_km(grid["lat"].to_numpy()[:, None], grid["lon"].to_numpy()[:, None],
                     allpos["lat"].to_numpy()[None, :], allpos["lon"].to_numpy()[None, :])
    grid["dist_to_known_km"] = d.min(axis=1)
    grid["bg_ok"] = grid["complete"] & ~grid["label_mrds"] & ~grid["label_nmet"] & (
        grid["dist_to_known_km"] > cfg["background"]["exclusion_buffer_km"])
    area, b = cfg["study_area"], cfg["spatial_cv"]["block_size_deg"]
    grid["block"] = (np.floor((grid["lat"] - area["lat_min"]) / b).astype(int).astype(str) + "_"
                     + np.floor((grid["lon"] - area["lon_min"]) / b).astype(int).astype(str))
    grid["dev"] = grid["lon"] >= REGION_SPLIT_LON
    return grid, mrds, nmet


def with_labels(grid, label_set):
    g = grid.copy()
    g["label"] = g["label_mrds"] | (g["label_nmet"] if label_set == "B" else False)
    g["eligible_bg"] = g["bg_ok"]
    return g


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def heldout_metrics(pct, pos, bg):
    m = pos | bg
    out = {"positives": int(pos.sum()), "background": int(bg.sum())}
    if pos.sum() == 0 or bg.sum() == 0:
        out["status"] = "NO_POSITIVES_IN_SET"
        return out
    y, s = pos[m], pct[m]
    flag = s >= 100 * (1 - TOP_FLAG)
    tp = int((flag & y).sum())
    prec = tp / max(int(flag.sum()), 1)
    rec = tp / int(y.sum())
    order = np.argsort(-s)[:TOP_CELLS]
    out.update({
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "prevalence": float(y.mean()),
        f"precision_top{int(TOP_FLAG * 100)}pct": prec,
        f"recall_top{int(TOP_FLAG * 100)}pct": rec,
        f"f1_top{int(TOP_FLAG * 100)}pct": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        f"precision_top{TOP_CELLS}_cells": float(y[order].mean()),
        "top_area_capture": {f"top_{int(f * 100)}pct": float(np.mean(pct[pos] >= 100 * (1 - f))) for f in (0.05, 0.1, 0.2)},
    })
    return out


def dev_spatial_cv(grid, feats, cfg, n_members=5):
    dev = grid[grid["dev"]]
    folds = fold_assignment(dev, cfg["spatial_cv"]["n_folds"], cfg["spatial_cv"]["seed"])
    pcts, pos, bg = [], [], []
    for f in sorted(folds.unique()):
        train = dev[folds != f]
        test = dev[(folds == f) & dev["complete"]]
        members = train_members(train, feats, n_members, cfg["background"]["n_background"], cfg["ensemble"]["seed"], "hgb")
        s = mean_score(members, test[feats])
        pcts.append(percentile_rank(np.sort(s), s))
        pos.append(test["label_mrds"].to_numpy())      # same evaluation labels for every model
        bg.append(test["eligible_bg"].to_numpy())
    return heldout_metrics(np.concatenate(pcts), np.concatenate(pos), np.concatenate(bg))


def final_test(grid, feats, cfg, n_members=5):
    """Train on the whole development region, score the untouched test region once."""
    train = grid[grid["dev"]]
    test = grid[~grid["dev"] & grid["complete"]]
    members = train_members(train, feats, n_members, cfg["background"]["n_background"], cfg["ensemble"]["seed"], "hgb")
    s = mean_score(members, test[feats])
    pct = percentile_rank(np.sort(s), s)
    bg = test["eligible_bg"].to_numpy()
    return {
        "mrds_labels": heldout_metrics(pct, test["label_mrds"].to_numpy(), bg),
        "indian_government_labels": heldout_metrics(pct, test["label_nmet"].to_numpy(), bg),
        "nmet_cell_ranks": [round(float(p), 1) for p in pct[test["label_nmet"].to_numpy()]],
    }


def _r(d):
    if isinstance(d, dict):
        return {k: _r(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_r(v) for v in d]
    return round(d, 4) if isinstance(d, float) else d


def main():
    warnings.filterwarnings("ignore")
    cfg = load_json(CONFIG_DIR / "exploration_config.json")
    grid, mrds, nmet = load_grid(cfg)
    dev, test = grid[grid["dev"]], grid[~grid["dev"]]
    print(f"cells {len(grid)} complete {int(grid['complete'].sum())}; dev MRDS pos {int(dev['label_mrds'].sum())}, "
          f"test MRDS pos {int(test['label_mrds'].sum())}; NMET pos cells dev {int(dev['label_nmet'].sum())} "
          f"test {int(test['label_nmet'].sum())}")

    # ---- feature ablation on the development region (label set A) -------------
    gA = with_labels(grid, "A")
    ablation = {"baseline_six": dev_spatial_cv(gA, BASELINE_SIX, cfg)}
    for name, feats in ABLATION.items():
        ablation[name] = dev_spatial_cv(gA, feats, cfg)
    for k, v in ablation.items():
        print(f"[ablation {k}] devCV ROC {v['roc_auc']:.3f} PR {v['pr_auc']:.4f} (prev {v['prevalence']:.4f}) "
              f"cap10 {v['top_area_capture']['top_10pct']:.2f}")
    # selection: dev-CV PR-AUC (rare positives), ties -> fewer features
    cands = {k: v for k, v in ablation.items()}
    best_fs = max(cands, key=lambda k: (round(cands[k]["pr_auc"], 4), round(cands[k]["roc_auc"], 3),
                                        -len(ABLATION.get(k, BASELINE_SIX))))
    best_feats = ABLATION.get(best_fs, BASELINE_SIX)

    # ---- label-source models --------------------------------------------------
    gB = with_labels(grid, "B")
    models = {
        "A_mrds_baseline_features": {"labels": "A", "features": "baseline_six", "dev_cv": ablation["baseline_six"]},
        "B_mrds_plus_indian_government_labels": {"labels": "B", "features": "baseline_six",
                                                 "dev_cv": dev_spatial_cv(gB, BASELINE_SIX, cfg)},
    }
    lab = "B" if models["B_mrds_plus_indian_government_labels"]["dev_cv"]["pr_auc"] > ablation["baseline_six"]["pr_auc"] else "A"
    gC = gB if lab == "B" else gA
    models["C_full_real_stack"] = {"labels": lab, "features": best_fs,
                                   "dev_cv": ablation[best_fs] if lab == "A" else dev_spatial_cv(gC, best_feats, cfg)}
    models["D_real_plus_synthetic"] = {
        "status": "NOT_RUN",
        "reason": ("The only synthetic exploration-side data are simulated subsurface scenarios generated FROM this "
                   "model's targets; using them as labels or features would be circular leakage. No synthetic "
                   "augmentation is used for exploration."),
    }
    # ---- final test region: scored once per model after all selection ----------
    for k, m in models.items():
        if m.get("status") == "NOT_RUN":
            continue
        g = gB if m["labels"] == "B" else gA
        feats = ABLATION.get(m["features"], BASELINE_SIX)
        m["final_test"] = final_test(g, feats, cfg)
        ft = m["final_test"]["mrds_labels"]
        print(f"[{k}] devCV ROC {m['dev_cv']['roc_auc']:.3f} PR {m['dev_cv']['pr_auc']:.4f} | FINAL TEST ROC "
              f"{ft.get('roc_auc', float('nan')):.3f} PR {ft.get('pr_auc', float('nan')):.4f} (prev {ft.get('prevalence', 0):.4f}) "
              f"cap10 {ft.get('top_area_capture', {}).get('top_10pct', float('nan')):.2f} | NMET cell ranks {m['final_test']['nmet_cell_ranks']}")

    # ---- seed robustness: PU bootstrap / background draws change with the seed and with the number of
    # positives, so single-seed differences can be resampling noise. Repeat A, B and C over 4 seeds.
    robust = {}
    for k in ("A_mrds_baseline_features", "B_mrds_plus_indian_government_labels", "C_full_real_stack"):
        m = models[k]
        g = gB if m["labels"] == "B" else gA
        feats = ABLATION.get(m["features"], BASELINE_SIX)
        runs = []
        for seed in SEEDS:
            c2 = json.loads(json.dumps(cfg))
            c2["ensemble"]["seed"] = seed
            cv = dev_spatial_cv(g, feats, c2)
            ft = final_test(g, feats, c2)["mrds_labels"]
            runs.append({"seed": seed, "dev_roc": cv["roc_auc"], "dev_pr": cv["pr_auc"], "test_roc": ft["roc_auc"],
                         "test_pr": ft["pr_auc"], "test_cap10": ft["top_area_capture"]["top_10pct"]})
        df = pd.DataFrame(runs)
        robust[k] = {"runs": runs, **{f"{c}_mean": float(df[c].mean()) for c in df.columns if c != "seed"},
                     **{f"{c}_sd": float(df[c].std()) for c in df.columns if c != "seed"}}
        r = robust[k]
        print(f"[robust {k}] dev ROC {r['dev_roc_mean']:.3f}±{r['dev_roc_sd']:.3f} PR {r['dev_pr_mean']:.4f}±{r['dev_pr_sd']:.4f} | "
              f"test ROC {r['test_roc_mean']:.3f}±{r['test_roc_sd']:.3f} PR {r['test_pr_mean']:.4f}±{r['test_pr_sd']:.4f} "
              f"cap10 {r['test_cap10_mean']:.2f}±{r['test_cap10_sd']:.2f}")

    sel = "C_full_real_stack"
    ra, rc = robust["A_mrds_baseline_features"], robust[sel]
    # confirmation gate on the untouched test (not a search): C must beat A on seed-averaged PR-AUC and ROC-AUC
    improves = rc["test_pr_mean"] > ra["test_pr_mean"] and rc["test_roc_mean"] >= ra["test_roc_mean"] - 0.01
    report = {
        "design": {
            "development_region": f"lon >= {REGION_SPLIT_LON}E (selection by spatial block CV inside this region only)",
            "final_test_region": f"lon < {REGION_SPLIT_LON}E (untouched until scored once per model)",
            "evaluation_labels": "MRDS positive cells (same for every model); Indian-government cells reported separately",
            "background": "unlabelled cells > exclusion buffer from every known positive (PU design; not proven barren)",
            "selection_metric": "development spatial-CV PR-AUC (then ROC-AUC, then fewer features)",
            "flag_rule": f"precision / recall / F1 when the top {int(TOP_FLAG * 100)} % of held-out area is flagged",
        },
        "label_counts": {"mrds_positive_cells": int(grid["label_mrds"].sum()),
                         "indian_government_positive_cells": int(grid["label_nmet"].sum()),
                         "indian_government_sources": nmet.to_dict(orient="records")},
        "feature_sets": {k: v for k, v in {**ABLATION, "baseline_six": BASELINE_SIX}.items()},
        "ablation_dev_cv": ablation,
        "selected_feature_set": best_fs,
        "selected_label_set": lab,
        "models": models,
        "seed_robustness": robust,
        "final_decision": {
            "deployed": sel if improves else "A_mrds_baseline_features",
            "rule": "confirmation gate: deploy C only if its seed-averaged final-test PR-AUC beats A without losing ROC-AUC",
            "reason": ("Full real stack improves the seed-averaged untouched final-test PR-AUC without losing ROC-AUC."
                       if improves else
                       "Full real stack did not improve the untouched final test; the baseline configuration is kept."),
        },
        "caveats": [
            "Positive-unlabelled labels: background is not barren, so metrics compare models; they are not hit rates.",
            "Few positives in the final test region: metric sampling uncertainty is large.",
            "Indian-government positives are block-level / sample-level points (location precision ~ block size).",
            "Geomorphology covers Madhya Pradesh + Maharashtra layers only (~85 % of cells); the rest is missing, not zero.",
        ],
    }
    save_json(REPORTS_DIR / "exploration_experiments.json", _r(report))
    print(json.dumps(report["final_decision"]), "selected features", best_fs, "labels", lab)


def supplementary():
    """Full-area context for the deployed choice (both label sets x baseline / selected features):
    5-fold spatial block CV over the whole area and both east/west holdout directions. Reported, not used
    for selection."""
    from ml.train_exploration import region_holdout, spatial_cv

    warnings.filterwarnings("ignore")
    cfg = load_json(CONFIG_DIR / "exploration_config.json")
    rep = load_json(REPORTS_DIR / "exploration_experiments.json")
    raw, _, _ = load_grid(cfg)
    out = {}
    for lab in ("A", "B"):
        g = with_labels(raw, lab)
        for name, f in (("baseline_six", BASELINE_SIX), (rep["selected_feature_set"], ABLATION[rep["selected_feature_set"]])):
            cv, ho = spatial_cv(g, f, cfg, "hgb"), region_holdout(g, f, cfg, "hgb")
            out[f"labels_{lab}:{name}"] = {
                "full_area_cv_roc": cv["roc_auc"], "full_area_cv_pr": cv["pr_auc"],
                "full_area_cv_prevalence": cv["pr_auc_prevalence_baseline"],
                "full_area_cv_cap10": cv["top_area_capture"]["top_10pct_area"],
                "train_east_test_west_roc": ho["train_east_test_west"].get("roc_auc"),
                "train_west_test_east_roc": ho["train_west_test_east"].get("roc_auc"),
                "holdout_pooled_cap10": ho["pooled"]["top_area_capture"]["top_10pct_area"]}
            print(lab, name, {k: round(v, 4) for k, v in out[f"labels_{lab}:{name}"].items()})
    rep["supplementary_full_area"] = _r(out)
    rep["supplementary_note"] = (
        "Mixed evidence: the selected stack improves ROC-AUC and top-10 % capture and, trained on the data-rich east, "
        "generalises much better to the west; but full-area PR-AUC (dominated by a few top-ranked hits among ~70 "
        "positives) is higher for the six-feature baseline, and training on the 15 western positives shows no gain "
        "in the east. Deployment followed the pre-declared confirmation gate.")
    save_json(REPORTS_DIR / "exploration_experiments.json", rep)


if __name__ == "__main__":
    import sys

    supplementary() if "--supplementary" in sys.argv else main()
