# Step 3: Forecast gap (paper §2.6, Figure 6)

> Paths: inputs are read from the data folder and outputs written under the output folder (`--data-dir` / `--out-dir`, see the main README). The "original location" is where each script lived in the project it was developed in.


Original location: `experiment_9_23/src/`.

For each dose span, these scripts compare where the **pre-dose** forecast (made before the shadow) placed the satellite with where the **post-dose** forecast (made after) places it at the same moment, measured along the direction of travel.

| Script | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `22_along_track_gap_all.py` | Finds each satellite's dose passes (start, end, dose) and a first along-track gap. | `satellite_dosage.csv`, `satellite_planes.csv`, `states.parquet`, `cohorts_closest_quartile.csv` (see note) | `along_track_gap_per_satellite.csv` |
| `23_delta_pre_post_by_pass.py` | Gap for every dose pass; extends the post-dose forecast back in time with the fitted formula D(τ) = c₀ + c₁τ + c₂τ² + c₃cos(nτ) + c₄sin(nτ), and **tests that extension** against passes where the answer can be read directly. | step 22 output, `states.parquet` | `delta_pre_post_by_dose_pass.csv`, validation report |
| `24_dose_span_predictions.py` | Stores the pre-dose and post-dose positions for each dose span. | step 23 output | `dose_span_predictions.csv` |
| `25_dose_pre_post_file.py` | Tidy per-dose-span table. | step 24 output | `dose_pre_post_predictions.csv` |
| `27_dose_pre_post_v3.py` | Rebuilds the table using each forecast file's own creation time: pre = last file made before the dose starts, post = first file made after it ends. Its dose spans give each satellite's total time in shadow, used in step 4. | step 25 output, `states.parquet`, raw ephemeris headers | `dose_pre_post_predictions_v3.csv` |
| `28_sma_check.py` | **Drag-paradox check:** satellites that got ahead of their forecast should have a smaller orbit. Result: 92% did. | step 27 output, `states.parquet` | `sma_paradox_check.csv`, `.md` |

Run order: 22 → 23 → 24 → 25 → 27 → 28.

Note: `22_along_track_gap_all.py` can read an optional `cohorts_closest_quartile.csv` in the output folder (from an earlier part of the project, not produced here). It only fills an `in_matched_cohort` flag column that none of the later steps use; without the file the column is simply all False.
