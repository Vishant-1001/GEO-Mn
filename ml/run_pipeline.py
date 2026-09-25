"""Rebuild every GEO-MN data product and model in order.

  python -m ml.run_pipeline            # offline steps only (uses committed raw data)
  python -m ml.run_pipeline --fetch    # also re-download public data (network)

Offline steps are deterministic (seed 42): re-running them reproduces the
committed artifacts. Fetch steps depend on the upstream archives.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

FETCH = [
    ["ml.fetch_weather"],
    ["ml.build_exploration_dataset"],
]
OFFLINE = [
    ["ml.generate_operations"],
    ["ml.train_production"],
    ["ml.train_exploration"],
    ["ml.target_engine"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="re-download public source data first")
    args = ap.parse_args()
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")}
    for step in (FETCH if args.fetch else []) + OFFLINE:
        print(f"\n=== {' '.join(step)} ===", flush=True)
        subprocess.run([sys.executable, "-u", "-m", *step], check=True, env=env)


if __name__ == "__main__":
    main()
