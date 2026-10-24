# Step 5: Figures (paper Figures 5–14)

> Paths: inputs are read from the data folder and outputs written under the output folder (`--data-dir` / `--out-dir`, see the main README). The "original location" is where each script lived in the project it was developed in.


Original location: `experiment_9_23/src/`. Figures 1–4 are made in step 2. Run these after step 4.

| Script | Figure | Output |
|---|---|---|
| `39_dose_distribution.py` | Fig 5: dose received by the satellites in the test | `docs/dose_distribution.png` |
| `42_forward_backward_explainer_v4.py` | Fig 6: projecting forward and extrapolating backward; what the sign of the gap means | `docs/forward_backward_explainer_v4.png` |
| `36_filter_funnel.py` | Fig 7: filtering funnel | `docs/filter_funnel.png` |
| `34_filtered_mw_v2_figures.py` | Figs 8–10: altitude, thruster-burn and IQR filters | `docs/mw2_fig1_altitude_filter.png`, `mw2_fig2_burn_filter.png`, `mw2_fig3_iqr_filter.png` |
| `35_dosed_vs_undosed_figure.py` | Fig 11: dosed vs no-dose Mann-Whitney, Aug 12 | `docs/dosed_vs_undosed_mann_whitney.png` |
| `43_finding_in_one_picture_v3.py` | Fig 12: visual summary of the main result (about 80 m, 11 ms) | `docs/finding_in_one_picture_v3.png` |
| `38_three_day_mann_whitney.py` | Fig 13: Aug 11 / 12 / 13 comparison (placebo) | `docs/mw_dosed_vs_undosed_three_days.png` |
| `40_gap_vs_dose_charts.py` | Fig 14: forecast gap by dose band | `docs/gap_vs_dose_2_boxes.png` (plus four extra views) |
| `44_orbital_tilts.py` | Orbital tilts and satellite counts per tilt (extra, not yet in the paper) | `docs/orbital_tilts.png` |

Inputs: the step 4 outputs (`filtered_mw_v2_per_satellite*.csv`); `36_filter_funnel.py` also reads `eclipse_targets.csv` and `satellite_dosage.csv`; `42_forward_backward_explainer_v4.py` uses fixed example times and needs no data.
