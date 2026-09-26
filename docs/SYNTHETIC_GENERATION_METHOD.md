# Synthetic operations and simulated recovery: generation method

Code: `ml/synthetic_ops.py` (simulator), `ml/generate_operations.py` (history),
`scripts/synthetic/generate_recovery.py` (counterfactuals). Configuration:
`data/synthetic/production/synthetic_generation_config.yaml` (seed 42, `ops-sim-2.0`).

The mine is **SYN_MINE_01**, shown as `DEMO_MINE` in the product. It is a generic demonstration
mine located at 21.70 °N, 79.95 °E only so that it can use real weather. It is not a MOIL mine,
and nothing it produces is MOIL telemetry.

## 1. Inputs that are real

| Input | Source | Use |
|---|---|---|
| Daily rainfall, Tmax 2021–2025 | IMD gridded (mine cell 21.75 N / 80.0 E, 0.25°; Tmax 1°) | Weather driver |
| Daily rainfall, Tmax 2026 | ERA5 (IMD not yet published). Recorded per row in `rainfall_source`. | Weather driver |
| Soil moisture | ERA5-Land | Wet-ground effects |
| Heavy-rain threshold 64.5 mm/day | IMD category definition | Weather-state labels |
| Heavy-rain scenario intensity | 95th percentile of 7-day IMD mine-cell rainfall (148 mm) | Recovery scenarios |
| Scale | MOIL company output (~18–19 lakh t/yr across its mines) | Single-mine magnitude kept plausible |

## 2. Simulator (one day = three shifts: A 7 h, B 7 h, C 6 h)

* **Fleet**: 2 excavators (loading capacity 50 t/h), 18 dumpers plus 4 pool trucks (8 t payload,
  60 min base cycle), 2 drills (8.5 m/h). Every unit has an ID (`SYN_EXC_001`, …).
* **Failures**: each unit's hazard per operating hour is `base × (1 + age/PM interval)`. It is
  multiplied by 1.6 for dumpers and drills on wet days (rain > 20 mm or soil moisture > 0.38).
  Repair times are lognormal. Preventive maintenance at the PM interval resets age, so
  maintenance and breakdowns interact.
* **Shared shocks**: logistic site stoppage in heavy rain (midpoint 70 mm), power outages
  (2.2 %/day), and explosive-supply episodes (0.9 %/day, 5–21 days, with a 25 % chance of blasting
  during an episode).
* **Process chain**: drilled metres → daily blast (6.5 t/m, blasted inventory capped at 3 days of
  nominal output) → loading (excavator hours) → haulage (truck-hours × payload ÷ cycle). Rain
  and wet ground lengthen the haul cycle. Output is `min(loading, haulage, blasted inventory) ×
  utilisation`, where utilisation carries Sunday-crew and heat losses.
* **Regimes**: truck redeployments (every ~100 days, 13–22 trucks) and a capacity decline of
  −1.2 %/yr.
* **Derived daily columns**: availability = available excavator hours ÷ scheduled;
  downtime / maintenance hours per excavator; `drilling_delay_h` = lost hours per drill, averaged
  per shift; `blast_delay_h` = excavator hours waiting on blasted ore; `haulage_delay_h` =
  excavator hours waiting on trucks, plus half of site stoppage; `truck_count` = time-averaged
  available trucks.
* **Invariants** (`check_day`, enforced every day): non-negative hours, availability in [0, 1],
  lost hours ≤ scheduled hours, production ≥ 0.

The resulting relationships were checked, not imposed. Correlation with attainment is +0.64 for
availability, −0.55 for blast delay and −0.28 for haulage delay. Attainment is ~0.83–0.90 in normal
years.

## 3. Outputs

`production_history.csv` holds daily aggregates. The per-shift unit records
(`synthetic_mine_operational_history.*`) and the event log (`synthetic_equipment_events.csv`) come
from the same run, so every daily figure can be traced to unit-level events.

## 4. Simulated recovery (counterfactuals)

For 143 weekly origins, the simulator state is snapshotted. The next 7 days are then re-run under
each of the **7 scenarios**: NORMAL, HEAVY_RAIN, EQUIPMENT_DEGRADATION, BLAST_DELAY, DRILL_DELAY,
HAULAGE_DISRUPTION and COMBINED. Each scenario is run with each of the **8 action portfolios**
(none, A1, A2, A3 and their combinations), using **common random numbers**: the same seed and the
same regimes, so only the scenario and actions differ.

| Scenario / action | Mechanism in the simulator |
|---|---|
| HEAVY_RAIN | Rain raised to the IMD p95 7-day intensity; soil moisture ≥ 0.46 |
| EQUIPMENT_DEGRADATION | Excavator and dumper hazard ×3 |
| BLAST_DELAY | Explosive-supply episode active |
| DRILL_DELAY | Drill hazard ×4 |
| HAULAGE_DISRUPTION | −4 trucks, haul cycle ×1.4 |
| A1 EQUIPMENT_RECOVERY | Excavator and dumper repair time ×0.4 |
| A2 SCHEDULE_ADJUSTMENT | +3 trucks, haul cycle ×0.9 |
| A3 BLAST_DRILL_DELAY_REDUCTION | Drill repair ×0.5, wet-hole loss halved, blast probability ≥ 0.9 in an episode |

`recovery_scenario_matrix.csv` stores the median (and p10 / p90) change in each production-model
input caused by each scenario and each portfolio. The recovery engine applies these deltas to the
current state and scores the result with the production model. The simulator therefore defines
the direction and size of every scenario and action, and the production model estimates the
tonnes.

**Limitation**: these are simulated effects of simulated actions. They are not historical MOIL
interventions, and recovery is never guaranteed.
