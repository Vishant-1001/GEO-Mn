# Simulated subsurface evidence: method

Code: `ml/synthetic_subsurface.py`, `scripts/synthetic/generate_subsurface.py`. Configuration:
`data/synthetic/subsurface/subsurface_generation_config.yaml` (seed 42, `subsurface-sim-1.0`).
Consumer: `services/subsurface_service.py` → `GET /api/exploration/targets/{id}/subsurface-scenarios`
and the **Subsurface scenario simulator** panel in the UI.

## Why it exists

The real subsurface audit (`DATA_SOURCE_AUDIT.md` §2) found no public borehole collars, logs,
assays or geophysics in the study area. Decision makers still need to see how the exploration
priority of a target would change after the next investigation. The simulator answers that
what-if question. Its outputs are not evidence.

## Real constraints used (from `data/processed/subsurface/geological_constraints.json`)

| Parameter | Value | Source |
|---|---|---|
| Host | Sausar Group, Mansar Formation (gondite / oxide ore) | NMET Kawalewada, Katori |
| Horizons H1–H3 | Stacked manganiferous horizons in schist / calc-gneiss | NMET Rongha, Kawalewada |
| Bed dip | 45–70° S (Kawalewada), 55–75° (Rongha), 60–80° (Lanjera) | NMET documents |
| Reef true thickness | 1.5–2 m | NMET Kawalewada |
| Overburden | 2–3 m | NMET Nagardhan |
| Ore grade | 26–38 % Mn (reported), samples to 45 % | NMET Nagardhan, Kawalewada |
| Hole depths | Proposed scout depths (50–100 m) | NMET proposals |
| Grade classes | IBM Indian Minerals Yearbook grade classes | IBM IMY (manganese chapter) |

Values without a source are marked `assumption` in the config: supergene uplift, assay precision,
IP chargeability background and anomaly factors, and the Mn-bearing cutoff.

## Scenarios (per exploration target)

| Scenario | Generated records | Fusion rule (project-configured) |
|---|---|---|
| NO_SUBSURFACE_EVIDENCE | none | unchanged |
| POSITIVE_GEOPHYSICAL_SUPPORT | 1 IP chargeability anomaly | level ≥ L2, uncertainty −1 step |
| POSITIVE_GEOCHEMICAL_SUPPORT | 8 surface samples (lognormal around 15 % Mn) | level ≥ L2, uncertainty −1 step |
| POSITIVE_DRILLING_INTERSECTION | 3 holes; 1–2 horizons intersected at ore grade | level ≥ L3, uncertainty −1 step |
| NEGATIVE_DRILLING_RESULT | 3 holes; host rock only; weak IP | level L3 ("tested"), prospectivity ×0.4 |
| AMBIGUOUS_DRILLING_RESULT | 3 holes; thin (×0.2–0.5), low-grade intersections | level L3, uncertainty HIGH, prospectivity ×0.85 |

**Geometry.** Holes are drilled at −60°. Apparent interval thickness follows from the true
thickness and the angle between the hole and the dipping bed. Intervals are ordered, never
overlap, never exceed hole depth, and the thickness of each interval equals `to − from`. These
are enforced by `check_subsurface()` and by the tests.

The fusion rules live in `config/exploration_config.json → subsurface_fusion`. The engine applies
them to a copy of the target and recomputes priority with the same scoring function used for real
targets (`ExplorationService.score_target`). The observed record is never modified.

## Hard limits

* Every row has `provenance = SIMULATED` and an ID starting with `SIM_`.
* The UI labels it "SIMULATED — NOT OBSERVED".
* It is never used for training, and never used to derive a reserve, resource or tonnage.
* Simulated drilling does not confirm mineralisation. A positive simulated intersection means
  "if drilling found this, the next step would be…".
