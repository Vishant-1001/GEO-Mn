"""Rebuild every GEO-MN data product and model in order.

  python -m ml.run_pipeline            # offline steps only (uses committed raw / processed data)
  python -m ml.run_pipeline --fetch    # also re-download public data (network; IMD, MOIL, Bhuvan, MPC)

Offline steps are deterministic (seeds in config / yaml): re-running them reproduces the
committed artifacts. Fetch steps depend on the upstream archives (see docs/DATA_SOURCE_AUDIT.md).
Order: real data -> synthetic operations -> models -> simulated recovery -> targets.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

S = "scripts"
FETCH = [
    ["ml.fetch_weather"],                                            # ERA5 / ERA5-Land (REAL_PUBLIC)
    ["ml.build_exploration_dataset"],                                # MRDS labels + baseline EO grid
    [f"{S}/data_acquisition/fetch_imd_gridded.py"],                  # IMD rainfall / Tmax (REAL_GOVERNMENT)
    [f"{S}/data_acquisition/fetch_moil_production.py"],              # MOIL disclosures (REAL_MOIL_PUBLIC)
    [f"{S}/data_acquisition/fetch_bhuvan_geology.py"],               # NRSC geomorphology / lineaments
    [f"{S}/data_processing/build_geology_features.py"],
    [f"{S}/data_processing/build_exploration_features.py"],          # extended EO grid (MPC)
]
OFFLINE = [
    [f"{S}/data_acquisition/fetch_moil_production.py", "--offline"],  # re-parse committed PDFs
    [f"{S}/data_processing/build_real_subsurface.py"],               # NMET / MOIL transcriptions
    ["ml.generate_operations"],                                      # SYNTHETIC operations (seed 42)
    ["ml.train_production"],
    ["ml.train_real_production"],                                    # REAL MOIL quarterly
    ["ml.exploration_experiments"],                                  # Models A-D, ablation A-F
    ["ml.exploration_experiments", "--supplementary"],               # full-area context (not selection)
    ["ml.train_exploration"],
    [f"{S}/synthetic/generate_recovery.py"],                         # SIMULATED recovery matrix
    ["ml.target_engine"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="re-download public source data first")
    args = ap.parse_args()
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")}
    for step in (FETCH if args.fetch else []) + OFFLINE:
        print(f"\n=== {' '.join(step)} ===", flush=True)
        cmd = [sys.executable, "-u", *step] if step[0].endswith(".py") else [sys.executable, "-u", "-m", *step]
        subprocess.run(cmd, check=True, env=env, cwd=ROOT)


if __name__ == "__main__":
    main()
