"""Generate the SIMULATED subsurface-evidence scenarios for every exploration target.

Run after the exploration targets exist (python -m ml.target_engine).
Outputs (data/synthetic/subsurface/): synthetic_boreholes.csv, synthetic_borehole_intervals.csv,
synthetic_geochemistry.csv, synthetic_geophysics.csv, synthetic_subsurface_targets.csv,
subsurface_generation_manifest.json. Consumer: services/subsurface_service.py (scenario / evidence-fusion).
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.synthetic_subsurface import SCENARIOS, check_subsurface, generate_target  # noqa: E402

OUT = ROOT / "data" / "synthetic" / "subsurface"


def main():
    cfg = yaml.safe_load((OUT / "subsurface_generation_config.yaml").read_text())
    constraints = json.loads((ROOT / "data" / "processed" / "subsurface" / "geological_constraints.json").read_text())
    targets = json.loads((ROOT / "data" / "exploration_targets.json").read_text())["targets"]
    rng = np.random.default_rng(cfg["seed"])
    H, I, G, P, T = [], [], [], [], []
    for t in targets:
        for sc in SCENARIOS:
            h, i, g, p = generate_target(t, sc, constraints, cfg, rng)
            H += h; I += i; G += g; P += p
            T.append({"target_id": t["target_id"], "scenario": sc, "lat": t["lat"], "lon": t["lon"],
                      "prospectivity_rank": t["prospectivity_rank"], "observed_evidence_level": t["evidence_level"],
                      "geological_unit": (t.get("geological_evidence") or {}).get("unit_name"),
                      "nmet_block_id": t.get("nmet_block_id"), "n_boreholes": len(h), "n_intervals": len(i),
                      "n_geochem": len(g), "n_geophys": len(p),
                      "mn_bearing_intervals": sum(1 for x in i if x["manganese_presence"]),
                      "max_mn_pct": max((x["mn_grade_pct"] for x in i if x["manganese_presence"]), default=None),
                      "provenance": "SIMULATED"})
    frames = {"synthetic_boreholes.csv": pd.DataFrame(H), "synthetic_borehole_intervals.csv": pd.DataFrame(I),
              "synthetic_geochemistry.csv": pd.DataFrame(G), "synthetic_geophysics.csv": pd.DataFrame(P),
              "synthetic_subsurface_targets.csv": pd.DataFrame(T)}
    check_subsurface(frames["synthetic_boreholes.csv"], frames["synthetic_borehole_intervals.csv"])
    for name, df in frames.items():
        df.to_csv(OUT / name, index=False)
    man = {"dataset_name": "Synthetic subsurface evidence scenarios", "generator_name": "ml/synthetic_subsurface.py",
           "generator_version": cfg["generator_version"], "schema_version": cfg["schema_version"], "seed": cfg["seed"],
           "generation_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source_constraints": "data/processed/subsurface/geological_constraints.json (NMET / DGM / MECL / IBM, cited)",
           "assumptions": "values marked 'assumption' in subsurface_generation_config.yaml",
           "scenarios": list(SCENARIOS),
           "files": {n: {"rows": int(len(df)), "sha256": hashlib.sha256((OUT / n).read_bytes()).hexdigest()} for n, df in frames.items()},
           "consumer": "services/subsurface_service.py -> /api/exploration/targets/{id}/subsurface-scenarios",
           "provenance": "SIMULATED — not observed drilling, assay or geophysical data; no reserve or tonnage is derived"}
    (OUT / "subsurface_generation_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print({n: len(df) for n, df in frames.items()})


if __name__ == "__main__":
    main()
