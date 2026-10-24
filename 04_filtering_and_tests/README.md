# Step 4: Filtering and tests (paper §2.7–2.8, §3)

> Paths: inputs are read from the data folder and outputs written under the output folder (`--data-dir` / `--out-dir`, see the main README). The "original location" is where each script lived in the project it was developed in.


Original locations: `experiment_9_23/src/` (scripts) and `experiment_9_23/results/ephemeris_drift/` (protocols).

| File | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `33_filtered_mann_whitney_v2.py` | Uses one window for every satellite (15:29–20:03 UTC, Aug 12) and computes each satellite's forecast gap at 20:03. Applies the three filters: altitude ≥ 440 km, no thruster burn (a sudden rise or drop in orbit size over 3 hours, IQR × 3 limits), and an IQR × 3 outlier check on the gap. Runs the shell-stratified Mann-Whitney test (effect size A, permutation p-value) for dosed vs no-dose satellites, and the same test on the placebo days (Aug 11 and Aug 13). | `satellite_dosage.csv`, `satellite_planes.csv`, `dose_pre_post_predictions_v3.csv`, `states.parquet` | `filtered_mw_v2_per_satellite.csv`, `filtered_mw_v2_results.md` |
| `37_filtered_mw_v2_save_all_days.py` | Same as 33, but saves the per-satellite table for each day, for the figures. | same as 33 | `filtered_mw_v2_per_satellite_aug11.csv`, `_aug12.csv`, `_aug13.csv` |
| `PROTOCOL_filtered_mann_whitney.md` | The test design, written down before the test was run. | – | – |
| `PROTOCOL_filtered_mann_whitney_addendum.md` | The filter change (altitude cut and two-sided burn check), recorded before the rerun and labeled exploratory. | – | – |

Headline result (Aug 12): 701 dosed vs 161 no-dose satellites, A = 0.746, p = 0.0002. Placebo: Aug 11 A = 0.462 (p = 0.25), Aug 13 A = 0.503 (p = 0.93).
