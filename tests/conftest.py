import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEOMN_LIVE_EO", "0")  # tests never touch the network
os.environ.setdefault("OMP_NUM_THREADS", "2")

PROVENANCE_KEYS = {"data_mode", "model_version", "observation_window", "source_timestamp", "fallback_used"}


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    os.environ["GEOMN_DECISION_LOG"] = str(tmp_path_factory.mktemp("log") / "decisions.jsonl")
    from fastapi.testclient import TestClient

    import main

    # Context manager => the client's portal/event-loop thread is shut down after the session,
    # so the full pytest process exits cleanly.
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def exploration_svc():
    from services.exploration_service import get_service

    return get_service()


@pytest.fixture
def production_svc():
    from services.production_service import get_service

    return get_service()


def merged_feature_grid():
    """The REAL feature grid the deployed exploration model uses (EO + NRSC geology), keyed by cell centre."""
    import pandas as pd

    from services.common import DATA_DIR

    proc = DATA_DIR / "processed" / "exploration"
    eo = pd.read_csv(proc / "eo_features_grid.csv.gz")
    geo = pd.read_csv(proc / "geology_features_grid.csv").drop(columns=["provenance", "geomorphology_class"])
    return eo.merge(geo, on=["lat", "lon"], how="left")
