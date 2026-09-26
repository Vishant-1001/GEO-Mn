"""SYNTHETIC / SIMULATED datasets: schema, seed reproducibility, physical and geological consistency,
manifests, and proof that each dataset is consumed by an engine."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.synthetic_ops import SCENARIOS, check_day, counterfactual_week, simulate
from services.common import DATA_DIR

SYN = DATA_DIR / "synthetic"


@pytest.fixture(scope="module")
def ops_cfg():
    return yaml.safe_load((SYN / "production" / "synthetic_generation_config.yaml").read_text())


def _weather(n=60, rain=0.0, start="2024-06-01"):
    return pd.DataFrame({"date": pd.date_range(start, periods=n), "rainfall_mm": rain,
                         "soil_moisture_m3m3": 0.3, "temperature_max_c": 33.0})


# ---------------------------------------------------------------- operations simulator

def test_simulator_is_seed_reproducible(ops_cfg):
    a = simulate(ops_cfg, _weather(), seed=7)[0]
    b = simulate(ops_cfg, _weather(), seed=7)[0]
    c = simulate(ops_cfg, _weather(), seed=8)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not a["actual_tonnes"].equals(c["actual_tonnes"])


def test_simulator_daily_invariants(ops_cfg):
    days = simulate(ops_cfg, _weather(90, rain=5.0), seed=3)[0]
    sched = ops_cfg["schedule"]["hours_per_day"] if "schedule" in ops_cfg else 20.0
    for d in days.to_dict(orient="records"):
        check_day(d, sched)
    assert (days["actual_tonnes"] >= 0).all() and days["equipment_availability"].between(0, 1).all()
    assert (days["truck_count"] <= ops_cfg["fleet_changes"]["max_trucks"] + 1e-6).all()


def test_heavy_rain_reduces_output_same_random_numbers(ops_cfg):
    w = _weather(40)
    _, _, snaps, sim = simulate(ops_cfg, w, seed=11, snapshot_every=7)
    d0 = sorted(snaps)[2]
    state, seed = snaps[d0]
    wk = w[w["date"] >= d0].head(7).reset_index(drop=True)
    wk["scenario_rain_mm"] = 30.0
    base, _ = counterfactual_week(sim, state, wk, seed, "NORMAL")
    for s in ("HEAVY_RAIN", "EQUIPMENT_DEGRADATION", "HAULAGE_DISRUPTION", "COMBINED_DISRUPTION"):
        assert counterfactual_week(sim, state, wk, seed, s)[0] < base, s
    again, _ = counterfactual_week(sim, state, wk, seed, "NORMAL")
    assert again == pytest.approx(base)          # common random numbers -> identical replay


def test_committed_history_schema_and_labels():
    h = pd.read_csv(DATA_DIR / "production_history.csv")
    assert set(h["ops_data_mode"]) == {"SYNTHETIC"} and (h["synthetic_mine_id"] == "SYN_MINE_01").all()
    assert h["date"].is_monotonic_increasing and not h["date"].duplicated().any()
    lost = h["equipment_downtime_h"] + h["maintenance_hours"]
    assert (lost <= 24).all() and h["equipment_availability"].between(0, 1).all()
    e = pd.read_csv(SYN / "production" / "synthetic_equipment_events.csv")
    assert set(e["provenance"]) == {"SYNTHETIC"} and set(e["event_type"]) >= {"BREAKDOWN", "PREVENTIVE_MAINTENANCE"}
    assert e["equipment_id"].str.match(r"^(SYN_|SITE)").all()      # never real MOIL equipment ids


def test_synthetic_relationships_have_expected_sign():
    h = pd.read_csv(DATA_DIR / "production_history.csv")
    att = h["actual_tonnes"] / h["target_tonnes"]
    assert np.corrcoef(h["equipment_availability"], att)[0, 1] > 0.3
    assert np.corrcoef(h["blast_delay_h"], att)[0, 1] < 0
    assert np.corrcoef(h["haulage_delay_h"], att)[0, 1] < 0


# ---------------------------------------------------------------- recovery matrix (SIMULATED)

def test_recovery_matrix_complete_and_consumed():
    from services import recovery_service as rs

    m = pd.read_csv(SYN / "recovery" / "recovery_scenario_matrix.csv")
    assert set(m["scenario"]) == set(SCENARIOS) and set(m["provenance"]) == {"SIMULATED"}
    assert (m.groupby("scenario")["action_portfolio"].nunique() == 8).all()
    normal = m[(m["scenario"] == "NORMAL") & (m["action_portfolio"] == "NO_ACTION")]
    assert (normal.filter(like="scenario_delta_").abs() < 1e-9).all(axis=None)
    # the recovery engine reads its scenario / action effects from this file
    assert rs.scenario_matrix() is not None
    row = rs._row("HAULAGE_DISRUPTION", "NO_ACTION")
    assert row["scenario_delta_haulage_delay_h"] > 0 and row["scenario_delta_truck_count"] < 0
    assert set(rs.ALL_SCENARIOS) == set(SCENARIOS)


def test_action_outcomes_arithmetic():
    a = pd.read_csv(SYN / "recovery" / "synthetic_action_outcomes.csv")
    assert set(a["provenance"]) == {"SIMULATED"}
    assert np.allclose(a["gap_after_t"], np.maximum(0, a["target_t"] - a["production_after_t"]), atol=0.11)
    assert np.allclose(a["recovery_t"], a["production_after_t"] - a["production_before_t"], atol=0.11)


# ---------------------------------------------------------------- no fabricated subsurface

def test_no_fabricated_subsurface_records_exist():
    """Boreholes, assays and geophysics are never generated (they would be fabricated evidence)."""
    assert not (SYN / "subsurface").exists()
    import importlib.util
    assert importlib.util.find_spec("ml.synthetic_subsurface") is None


# ---------------------------------------------------------------- manifests

@pytest.mark.parametrize("manifest,base", [
    ("synthetic/recovery/recovery_generation_manifest.json", {"synthetic_disruption_scenarios.csv": "synthetic/production",
                                                              "synthetic_action_outcomes.csv": "synthetic/recovery",
                                                              "recovery_scenario_matrix.csv": "synthetic/recovery"}),
])
def test_manifest_checksums_match_files(manifest, base):
    m = json.loads((DATA_DIR / manifest).read_text())
    assert m["seed"] is not None and m.get("consumer")
    folder = (DATA_DIR / manifest).parent
    for name, meta in m["files"].items():
        p = DATA_DIR / base[name] / name if base else folder / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"], name
        assert len(pd.read_csv(p)) == meta["rows"]


def test_operations_manifest():
    m = json.loads((SYN / "production" / "synthetic_generation_manifest.json").read_text())
    assert m["seed"] == 42 and "SYNTHETIC" in json.dumps(m)
    where = {"production_history.csv": DATA_DIR}
    for name, meta in m["files"].items():
        p = where.get(name, SYN / "production") / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"], name
