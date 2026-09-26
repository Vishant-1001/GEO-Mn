# Data quality report

Checks are automated in `tests/test_real_data.py` and `tests/test_synthetic_data.py`, unless a row
says otherwise.

## Real data

| Dataset | Coverage | Quality checks | Issues and handling |
|---|---|---|---|
| MOIL production (REAL_MOIL_PUBLIC) | 52 disclosures → 142 stated figures → 54 quarters, 2012-13 Q3 … 2026-27 Q1 | Header-based cumulative vs quarterly rule; units by magnitude (lakh t vs t); every derived quarter traced to two documents; full fiscal years sum to the stated FY total (±4,000 t rounding); each quarter is 10–40 % of its FY | **0 cross-check conflicts.** One quarter missing (FY2014-15 Q3, starting 2014-10-01): no disclosure lets it be derived, so it is left missing, not imputed. Figures are rounded to 0.01 lakh t and partly unaudited (MOIL note). One document is filed under the wrong date (named 31.12.2014, contains 30.09.2015); the content date is used. |
| IMD rainfall / Tmax (REAL_GOVERNMENT) | 5,114 days, 2012–2025, 40 rain cells, 6 Tmax cells | Binary size checked against days × grid (leap years); −999 / 99.9 sentinels → NaN; 0 missing days in the belt series; annual totals 1,040–1,554 mm; June–September ≫ November–February | 2026 not yet published by IMD → ERA5 used for 2026 dates, source recorded per row (267 of 2,093 demo-mine days). Tmax is 1° (coarse). |
| NRSC geomorphology / lineaments (REAL_GOVERNMENT) | 4,109 polygons; 607,263 lineament pixels at 0.0005° | 0 failed WMS requests; point-in-polygon at cell centres | MP and MH layers only: **85.0 %** of cells covered. The rest (Chhattisgarh edge) stays NaN, never 0. Lineaments are rasterised from rendered WMS tiles, so positions are ±1 pixel (~55 m) plus 1:50k mapping precision. The classes are geomorphology, not lithology. |
| NMET blocks (REAL_GOVERNMENT) | 5 blocks, 5 surface samples, 15 findings | Block areas recomputed from transcribed corners vs stated areas (Katori 141.03 vs 140.22 km²; Nagardhan 2.01 vs 2.00 km²); PDF sha256 recorded | Transcribed by hand from the PDFs, with quotes kept. Katori "Mansar terrain" was downgraded from GEOLOGICAL_MAPPING to GEOLOGICAL_CONTEXT, because it describes terrain rather than a mapped Mn horizon. Proposals are never counted as executed work. |
| USGS MRDS (REAL_PUBLIC) | 73 positive records | Regional / duplicate records excluded | Mixed location precision; clustered in two belts. |
| EO feature grid (REAL_DERIVED) | 50,000 cells (0.01°) | Recipe shared with live queries; offset correction for processing baseline ≥ 04.00; SCL cloud mask; minimum valid scenes | See §EO completeness below. |

## Synthetic data

| Dataset | Checks | Notes |
|---|---|---|
| Operations (2,093 days, 115,530 unit-shifts, 9,428 events) | Daily invariants on every simulated day (`check_day`); seed reproducibility; sha256 in manifest | Events: 7,275 breakdowns, 2,047 PM, 49 power outages, 43 weather stoppages, 14 explosive-supply episodes. Attainment 0.83–0.90 per year from 2022; 2021 is 1.00 because the simulation starts with fresh equipment and full inventory (burn-in). One test week (2024-08-16) has zero output because an explosive-supply episode exhausted the blasted inventory. That is physically consistent, so the production metrics use WAPE instead of MAPE. |
| Recovery matrix (56 rows from 8,008 counterfactuals) | NORMAL / NO_ACTION deltas exactly 0; the arithmetic `gap_after = max(0, target − after)` holds; the matrix is read by the recovery engine | Common random numbers, so the differences are scenario and action effects only. |
| Subsurface (360 holes, 1,316 intervals, 1,636 geochemistry samples, 160 geophysics readings) | `check_subsurface`: ordered, non-overlapping intervals within hole depth, thickness = to − from, grades 0–60 %; scenario semantics (negative = no Mn-bearing interval) | All rows SIMULATED; never used for training. |

## EO completeness

* **Cells**: 49,922 of 50,000 (99.84 %) have every EO feature. The other 78 are
  `INSUFFICIENT_DATA` and are not filled.
* **Valid Sentinel-2 scenes per cell**: median 8.9 (range 3–12, minimum required 3).
* **Reproducibility**: the six baseline features reproduce the committed grid exactly for LST,
  elevation and slope. For the S2 indices, ≥ 99.99 % of cells agree within 1e-4 (max difference
  0.009).

**Incident found and fixed.** Earlier runs dropped scenes silently when a band read failed. Two
causes were network stalls, and Planetary Computer SAS tokens expiring about 1 hour after the STAC
search. Five tiles were empty or partial, and the completeness of the grid fell to 36 %.

The fix:

* Tokens are re-signed per tile.
* Any failed read now fails the tile, which is retried, instead of being composited from a reduced
  scene set.
* Remote reads carry timeouts.

After the fix, tiles that had looked healthy gained scenes (e.g. 43QHE: median 5 → 7), which shows
the silent losses were not limited to the failed tiles. The grid was recomputed from scratch.
