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

    return TestClient(main.app)


@pytest.fixture
def exploration_svc():
    from services.exploration_service import get_service

    return get_service()


@pytest.fixture
def production_svc():
    from services.production_service import get_service

    return get_service()
