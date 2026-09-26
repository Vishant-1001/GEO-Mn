"""Regenerate every SYNTHETIC / SIMULATED dataset in dependency order (deterministic, seed from the yaml configs).

  1. operations  ml.generate_operations          -> data/synthetic/production/* and data/production_history.csv
  2. recovery    scripts/synthetic/generate_recovery.py   -> disruption scenarios, action outcomes, recovery matrix
  3. subsurface  scripts/synthetic/generate_subsurface.py -> simulated boreholes / intervals / geochem / geophysics

Step 3 reads data/exploration_targets.json, so run it after the exploration model and target engine.
Run: python scripts/synthetic/generate_all.py [--only operations|recovery|subsurface]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEPS = {
    "operations": [sys.executable, "-u", "-m", "ml.generate_operations"],
    "recovery": [sys.executable, "-u", str(ROOT / "scripts" / "synthetic" / "generate_recovery.py")],
    "subsurface": [sys.executable, "-u", str(ROOT / "scripts" / "synthetic" / "generate_subsurface.py")],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(STEPS))
    a = ap.parse_args()
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")}
    for name, cmd in STEPS.items():
        if a.only and name != a.only:
            continue
        print(f"\n=== synthetic: {name} ===", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
