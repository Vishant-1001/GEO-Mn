"""REAL data products: normalisation, schema, provenance and physical sanity (no network)."""
import importlib.util
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from services.common import DATA_DIR, ROOT

PROC = DATA_DIR / "processed"
MAN = DATA_DIR / "manifests"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- MOIL production (REAL_MOIL_PUBLIC)

@pytest.fixture(scope="module")
def moil():
    return _load("scripts/data_acquisition/fetch_moil_production.py", "fetch_moil")


def test_moil_fiscal_helpers(moil):
    assert moil.fy_of(date(2024, 3, 31)) == 2023 and moil.fy_of(date(2024, 4, 1)) == 2024
    assert moil.months_into_fy(date(2024, 6, 30)) == 3 and moil.months_into_fy(date(2025, 3, 31)) == 12


def test_moil_quarterly_series_is_consistent():
    q = pd.read_csv(PROC / "production" / "moil_quarterly_production.csv")
    assert set(q["provenance"]) == {"REAL_MOIL_PUBLIC"}
    assert not q.duplicated(["fy_start_year", "fy_quarter"]).any()
    assert (q["production_t"] > 0).all() and q["fy_quarter"].between(1, 4).all()
    # quarter start month follows the Indian fiscal year (Apr / Jul / Oct / Jan)
    months = pd.to_datetime(q["quarter_start"]).dt.month
    assert (months == q["fy_quarter"].map({1: 4, 2: 7, 3: 10, 4: 1})).all()
    # derived quarters equal the difference of two stated cumulative figures and say so
    assert q["basis"].isin(["stated quarter", "stated cumulative", "derived: cumulative difference"]).all()
    assert q.loc[q["basis"].str.startswith("derived"), "source_documents"].str.contains(" minus ").all()
    # no quarter is more than 40 % of its fiscal year where the whole year is present
    full = q.groupby("fy_start_year").filter(lambda g: len(g) == 4)
    share = full["production_t"] / full.groupby("fy_start_year")["production_t"].transform("sum")
    assert share.between(0.1, 0.4).all()


def test_moil_quarters_match_stated_fy_totals():
    q = pd.read_csv(PROC / "production" / "moil_quarterly_production.csv")
    d = pd.read_csv(PROC / "production" / "moil_production_disclosures.csv")
    fy = d[d["period_kind"] == "FY"].groupby("fy")["grand_total_t"].first()
    full = q.groupby("fy_start_year").filter(lambda g: len(g) == 4).groupby("fy_start_year")["production_t"].sum()
    common = full.index.intersection(fy.index)
    assert len(common) >= 8
    # disclosures are rounded to 0.01 lakh t per figure
    assert np.allclose(full.loc[common], fy.loc[common], atol=4 * 1000.0 + 1)


# ---------------------------------------------------------------- IMD (REAL_GOVERNMENT)

def test_imd_reader_decodes_layout(tmp_path):
    imd = _load("scripts/data_acquisition/fetch_imd_gridded.py", "fetch_imd")
    spec = dict(imd.RF)
    ndays = 365
    a = np.zeros((ndays, spec["ny"], spec["nx"]), dtype="<f4")
    i, j = imd.cell_index(spec, 21.75, 80.0)
    a[10, i, j] = 42.0
    a[11, i, j] = spec["missing"]
    p = tmp_path / "rain.grd"
    a.tofile(p)
    out = imd.read(spec, p, 2023)
    assert out[10, i, j] == pytest.approx(42.0) and np.isnan(out[11, i, j])
    with pytest.raises(ValueError):
        imd.read(spec, p, 2024)      # leap year: wrong size must be rejected, never silently reshaped


def test_imd_processed_series_is_plausible():
    d = pd.read_csv(PROC / "weather" / "imd_daily_mine_belt.csv", parse_dates=["date"])
    assert set(d["provenance"]) == {"REAL_GOVERNMENT"}
    assert d["date"].is_monotonic_increasing and not d["date"].duplicated().any()
    assert d[["belt_rainfall_mm", "mine_cell_rainfall_mm", "belt_tmax_c"]].notna().all().all()
    assert (d["mine_cell_rainfall_mm"] >= 0).all() and d["belt_tmax_c"].between(10, 50).all()
    annual = d[d["date"].dt.year < 2026].groupby(d["date"].dt.year)["belt_rainfall_mm"].sum()
    assert annual.between(700, 2200).all()                     # central-India belt climatology
    monthly = d.groupby(d["date"].dt.month)["belt_rainfall_mm"].mean()
    assert monthly.loc[[6, 7, 8, 9]].sum() > 5 * monthly.loc[[11, 12, 1, 2]].sum()   # monsoon dominated


def test_demo_weather_uses_imd_and_records_source():
    w = pd.read_csv(PROC / "weather" / "demo_mine_daily_weather.csv")
    assert w["rainfall_source"].notna().all()
    assert w["rainfall_source"].str.startswith("IMD").mean() > 0.8
    h = pd.read_csv(DATA_DIR / "production_history.csv")
    assert set(h["weather_data_mode"]) <= {"REAL_GOVERNMENT", "REAL_PUBLIC"} and set(h["ops_data_mode"]) == {"SYNTHETIC"}


# ---------------------------------------------------------------- geology / subsurface (REAL_GOVERNMENT)

def test_geology_feature_encoding():
    geo = _load("scripts/data_processing/build_geology_features.py", "build_geo")
    e = geo.encode("Structural Origin-Moderately Dissected Hills and Valleys")
    assert e["geom_structural_origin"] == 1 and e["geom_hills_valleys"] == 1 and e["geom_dissection"] == 2
    assert all(np.isnan(v) for v in geo.encode(None).values())     # unmapped stays missing, never 0


def test_geology_grid_coverage_and_ranges():
    g = pd.read_csv(PROC / "exploration" / "geology_features_grid.csv")
    assert len(g) == 50000 and 0.7 < g["geomorphology_class"].notna().mean() <= 1.0
    assert (g["lineament_dist_km"] >= 0).all() and g["lineament_density"].between(0, 1).all()
    unmapped = g["geomorphology_class"].isna()
    assert g.loc[unmapped, "geom_dissection"].isna().all()


def test_nmet_blocks_and_findings():
    b = pd.read_csv(PROC / "subsurface" / "exploration_blocks.csv")
    assert set(b["provenance"]) <= {"REAL_GOVERNMENT"} and b["in_study_area"].all()
    f = pd.read_csv(PROC / "subsurface" / "reported_findings.csv")
    # only one block reports a drilling intersection; proposals are never counted as executed work
    drill = f[f["evidence_class"] == "DRILLING_INTERSECTION_REPORTED"]
    assert set(drill["block_id"]) == {"NAGARDHAN"}
    assert f.loc[f["evidence_class"] == "PROPOSAL_ONLY", "value_text"].str.contains("not drilled").all()
    c = json.loads((PROC / "subsurface" / "geological_constraints.json").read_text())
    assert c["proposed_borehole_depths_m"] and all(20 <= x <= 200 for x in c["proposed_borehole_depths_m"])


@pytest.mark.parametrize("name", ["real_moil_production.json", "real_imd_gridded.json", "real_bhuvan_geology.json",
                                  "real_subsurface_nmet.json"])
def test_real_manifests_have_provenance(name):
    m = json.loads((MAN / name).read_text())
    assert m.get("provenance", "").startswith(("REAL_", "REAL"))
    assert m.get("dataset_name") and (m.get("official_url") or m.get("documents"))


def test_nmet_documents_have_checksums():
    d = pd.read_csv(DATA_DIR / "raw" / "subsurface" / "nmet_documents.csv")
    assert d["sha256"].str.fullmatch(r"[0-9a-f]{64}").all() and d["official_url"].str.startswith("https://").all()


def test_moil_raw_pdfs_match_index():
    idx = json.loads((DATA_DIR / "raw" / "production" / "moil" / "index.json").read_text())
    assert len(idx) >= 50
    for it in idx:
        p = DATA_DIR / "raw" / "production" / "moil" / it["file"]
        assert p.read_bytes()[:4] == b"%PDF", it["file"]
        assert it["source_url"].startswith("https://backend.moil.nic.in/")
