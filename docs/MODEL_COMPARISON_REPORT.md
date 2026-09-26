# Model comparison report

All numbers come from `models/reports/*.json` and are reproducible with `python -m ml.run_pipeline`.

## 1. Exploration: label sources and feature families

**Protocol.** Development region: east of 79.8 °E (57 MRDS positive cells). All selection was done
there with 5-fold 0.25° spatial block CV. The final test region is west of 79.8 °E (15 MRDS positive
cells). It was never used for selection and served only as a confirmation gate: deploy the
candidate only if its seed-averaged test PR-AUC beats the baseline without losing ROC-AUC.

Evaluation labels are the MRDS cells for every model. Background is unlabelled (positive-unlabelled
design). Test prevalence is 0.0006, so PR-AUC must be read against that. Model C's test PR-AUC of
0.0128 is about 21× random, yet still a small number in absolute terms.

### Feature ablation (development spatial CV, label set A)

| Set | Features | ROC-AUC | PR-AUC (prev. 0.0025) | Top-10 % area capture |
|---|---|---|---|---|
| baseline six | NDVI, Fe-oxide, clay, LST, elevation, slope | 0.629 | 0.0074 | 21 % |
| A | S2 spectral (8) | 0.560 | 0.0037 | 14 % |
| B | S2 + terrain / thermal | **0.667** | 0.0078 | 18 % |
| C | S2 + geomorphology | 0.576 | 0.0038 | 18 % |
| D | S2 + terrain + geomorphology | 0.644 | **0.0091** | 18 % |
| E | geomorphology + lineaments only | 0.634 | 0.0036 | 12 % |
| F | full stack incl. lineaments | 0.567 | 0.0085 | 18 % |

Selection rule: PR-AUC (rare positives), then ROC-AUC. **D** was selected. S2 spectral ratios alone
carry little signal. Terrain and thermal context carries most of it. Lineament distance / density
did not help (F < D).

### Label sources and final test (mean ± SD over 4 ensemble seeds)

| Model | Labels | Features | Dev CV ROC | Test ROC | Test PR-AUC | Top-10 % capture |
|---|---|---|---|---|---|---|
| A | MRDS | baseline six | 0.624 ± 0.018 | 0.697 ± 0.016 | 0.0021 | 25 % |
| B | MRDS + NMET (3 cells) | baseline six | 0.687 ± 0.015 | 0.701 ± 0.037 | 0.0022 | 30 % |
| **C (deployed)** | MRDS + NMET | D | 0.733 ± 0.016 | **0.803 ± 0.030** | **0.0128** | **43 %** |
| D | real + synthetic | — | not run (circular: simulated subsurface is generated from model targets) | | | |

Single-seed run of C on the test region: precision / recall / F1 in the top 10 % of area are
0.003 / 0.53 / 0.007, and top-50-cell precision is 0.04 (2 of 50). NMET cells rank 43.6 under A and
64.2 under C.

**Findings.**

* The improvement comes from the **features**, not the Indian-government labels. B's dev-CV gain
  (PR 0.0074 → 0.0190 at one seed) did not survive the test region (0.0021 → 0.0022). That is
  consistent with resampling noise from adding only 3 positive cells.
* C passed the confirmation gate and is deployed as `exploration-pu-ensemble-2.0`.

### Supplementary full-area check (reported, not used for selection)

| Labels : features | Full-area CV ROC | Full-area CV PR (prev. 0.0015) | E→W ROC | W→E ROC |
|---|---|---|---|---|
| A : baseline six | 0.684 | **0.0213** | 0.685 | 0.665 |
| A : D | 0.727 | 0.0077 | 0.789 | 0.659 |
| B : baseline six | 0.712 | 0.0061 | 0.712 | 0.645 |
| B : D (deployed) | 0.719 | 0.0044 | **0.813** | 0.663 |

**The evidence is mixed.** The richer stack improves ROC and top-area capture, and it transfers well
from the data-rich east to the west. However, full-area PR-AUC is highest for the six-feature
baseline (a few very top-ranked hits among ~70 positives dominate PR-AUC), and training on the 15
western positives gives no gain in the east. The deployed model is a modest improvement in
ranking, not a step change.

### Other consequences

* Ensemble rank SD is larger with 18 features (median 14 rank points). Under the project thresholds
  (LOW < 5, HIGH > 10), 78 % of grid cells are HIGH uncertainty. High-rank target cells are more
  stable. The thresholds were not retuned to hide this.
* The applicability check now has a hard training-range envelope in addition to IsolationForest,
  because a multivariate score dilutes a few extreme values across 12 features.

## 2. Production: weekly demo-mine engine (SYNTHETIC operations)

| | MAE (t) | RMSE (t) | WAPE |
|---|---|---|---|
| P50 (selected: ratio_regularised) | **1144** | 1589 | 13.2 % |
| Previous period | 1197 | 1698 | — |
| 4-period moving average | 1150 | 1569 | — |

The P50 is 0.5 % better than the best baseline (a tie). Calibrated P10–P90 coverage is 0.80 against
a nominal 0.80, with hit rates 0.12 / 0.48 / 0.92, so the quantiles are validated. MAPE is undefined
because one simulated stoppage week had zero output.

## 3. Production: real MOIL company quarterly (REAL_MOIL_PUBLIC)

| Method (12 untouched real quarters) | MAE (t) | MAPE |
|---|---|---|
| ridge, no weather (selected on FY2019–22) | 52,703 | 11.7 % |
| ridge + IMD rain anomaly | 52,841 | 11.7 % |
| GBM + weather | 61,478 | 14.2 % |
| **seasonal naive × YoY** | **38,837** | **8.6 %** |
| last quarter | 40,500 | 9.2 % |

Real-only vs real + synthetic, on the same 8 real quarters: MAE 44,772 vs **60,486** t, so
synthetic augmentation hurts and is excluded. The selected model does not beat the baselines, and
the UI and API say so.

## 4. Recovery

The recovery engine has no fitted skill metric. Its effects are simulator counterfactuals
(`recovery_scenario_matrix.csv`) scored by the production model. Median simulated 7-day recovery
(t):

| Scenario | A1 | A2 | A3 | A1+A2+A3 |
|---|---|---|---|---|
| NORMAL | 402 | 0 | 0 | 620 |
| HEAVY_RAIN | 536 | 583 | 0 | 1,449 |
| EQUIPMENT_DEGRADATION | 938 | 0 | 0 | 1,464 |
| BLAST_DELAY | 282 | 0 | **1,377** | 2,457 |
| DRILL_DELAY | 467 | 0 | 28 | 973 |
| HAULAGE_DISRUPTION | 223 | **1,798** | 0 | 2,343 |
| COMBINED_DISRUPTION | 846 | 42 | 531 | 3,007 |

Each action works only where its constraint binds:

* A2 (trucks) matters under haulage disruption and heavy rain.
* A3 matters under explosive-supply delays.
* A3 barely helps under drill delay (+28 t), because the blasted-inventory buffer absorbs most drill
  outages within a week.

These are simulated effects of simulated actions, not historical MOIL outcomes.
