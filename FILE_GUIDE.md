# File Guide: What's in This Folder

This folder holds the code for *Detecting Solar Eclipse Effects on Neutral Density Using Starlink Ephemerides* (Vivienne Shea, 2026). The project asks whether Starlink satellites that flew through the Moon's shadow on 12 August 2026 met thinner air (lower neutral density) and so felt less drag than satellites that did not.

The code runs in five steps, one folder per step, in order. Each step uses what the step before it produced. No data is included, only code.

**A few words used below**
- **Ephemeris:** SpaceX's forecast of where each satellite will be, minute by minute, for the next few days.
- **TLE (two-line element set):** a short public summary of a satellite's orbit.
- **Umbra / penumbra:** the Moon's full shadow / partial shadow.
- **Dose:** how much shadow a satellite went through, in umbra-seconds (1 second in full shadow = 1).
- **Forecast gap:** how far apart the forecast made before the eclipse and the forecast made after it put the satellite at 20:03 UTC, measured along its path.
- **Tilt:** the angle of a satellite's orbit to the equator (43°, 53°, 70° or 97°).
- **Neutral density:** how thick the uncharged gas in the upper atmosphere is. Thinner gas means less drag.

**What TLE data is used for:** only two things. (1) The first list of satellites expected to cross the shadow, later replaced by the shadow dose. (2) Each satellite's tilt, which the test needs so it compares satellites at the same tilt. The TLE scripts also save a drag number called BSTAR, but the paper does not use it.

---

## Files at the top level

| File | What it is |
|---|---|
| `README.md` | The main instructions: what the project is, the run order, where data goes, and how to install. |
| `FILE_GUIDE.md` | This file: a plain-language tour of every file. |
| `starlink_paths.py` | Tells every script where to find the data and where to save results. Run it by itself to see which folders will be used. |
| `requirements.txt` | The list of Python packages to install. |
| `LICENSE` | The MIT license, which lets others reuse the code. |
| `.gitignore` | Tells Git which files never to upload (data, outputs, passwords). |

---

## Step 1: Collect the data (`01_data_collection/`)

| File | What it does |
|---|---|
| `README.md` | Details for this step. |
| `ephemeris/starlink_ephem.py` | Downloads SpaceX's ephemeris forecasts for every Starlink satellite. It was run 11 times, from August 10 to 14, to save "snapshots" before, during and after the eclipse. |
| `target_selection/eclipse_forecast.py` | Before the eclipse, predicts which satellites will cross the full or partial shadow. Used to choose the 1,257 target satellites. |
| `target_selection/eclipse_shadow.py` | Helper that works out where the Sun, the Moon and the shadow are at any moment. Used by the forecast script. |
| `tle_celestrak/build_tle_observations.py` | Turns the TLE files saved with each snapshot into one clean table. Later used to find each satellite's tilt, which the test needs. |
| `tle_spacetrack/download_spacetrack_tle.py` | Downloads each target satellite's TLE history from Space-Track. Needs a free account. The login is read from the computer, never written in the code. |
| `tle_spacetrack/consolidate_spacetrack_tle.py` | Combines those downloads into tidy tables. |

---

## Step 2: Shadow and dose (`02_shadow_geometry_and_dose/`)

| File | What it does |
|---|---|
| `README.md` | Details for this step. |
| `eclipse_shadow.py` | The same shadow helper as in Step 1, used by the scripts here. |
| `01_shadow_geometry.py` | Works out where the shadow falls throughout the eclipse, at ground level and at Starlink height (465.6 km). |
| `02_render_shadow_map.py` | Draws the shadow maps (paper Figures 1 and 2). |
| `03_satellite_shadow_passes.py` | Finds each time a satellite entered and left the shadow. |
| `04_shadow_dose_heatmap.py` | Builds a map of the total shadow dose at Starlink height (Figure 3). Slow: about 30 minutes. |
| `05_satellite_paths_on_dose_field.py` | Draws the satellites' paths over that dose map (Figure 4). |
| `02_satellite_dosage.py` | **The dose used in the test:** follows each satellite second by second and adds up how much of the Sun was covered. |
| `build_satellite_planes.py` | Finds each satellite's tilt and orbital plane from the TLE data. |

---

## Step 3: The forecast gap (`03_forecast_gap/`)

| File | What it does |
|---|---|
| `README.md` | Details for this step. |
| `22_along_track_gap_all.py` | First pass: finds each satellite's shadow passes and a first forecast gap. |
| `23_delta_pre_post_by_pass.py` | Fits the formula that runs the after-eclipse forecast backwards in time, then **checks that it works** on cases where the answer can be read directly. |
| `24_dose_span_predictions.py` | Saves the before and after forecast positions for each shadow pass. |
| `25_dose_pre_post_file.py` | Turns that into a tidy table. |
| `27_dose_pre_post_v3.py` | Rebuilds the table using when each forecast file was actually made, so "before" really means before the shadow and "after" really means after. |
| `28_sma_check.py` | **Drag-paradox check:** satellites that got ahead of their forecast should have dropped to a smaller orbit. About 9 in 10 did. |

---

## Step 4: Filters and the test (`04_filtering_and_tests/`)

| File | What it does |
|---|---|
| `README.md` | Details for this step. |
| `33_filtered_mann_whitney_v2.py` | **The main result.** Finds each satellite's forecast gap at 20:03 UTC, applies the three filters (altitude of 440 km or higher, no thruster firing, no extreme gap), and runs the Mann-Whitney test comparing satellites with and without a dose at the same tilt. Then repeats everything on August 11 and 13 as a placebo check. |
| `37_filtered_mw_v2_save_all_days.py` | The same as 33, but saves a table for each of the three days for the figures. |
| `PROTOCOL_filtered_mann_whitney.md` | The test plan, written down before the test was run. |
| `PROTOCOL_filtered_mann_whitney_addendum.md` | A later change to the filters, recorded before the rerun and labeled as exploratory. |

**Result:** on eclipse day, A = 0.746 and p = 0.0002. On the placebo days, A = 0.462 (Aug 11) and 0.503 (Aug 13), meaning no difference.

---

## Step 5: Figures (`05_figures/`)

| File | Paper figure |
|---|---|
| `README.md` | Details for this step. |
| `39_dose_distribution.py` | Fig 5: how much dose the tested satellites got |
| `42_forward_backward_explainer_v4.py` | Fig 6: how the forecast gap works and what its sign means |
| `36_filter_funnel.py` | Fig 7: how many satellites each filter removed |
| `34_filtered_mw_v2_figures.py` | Figs 8–10: the altitude, thruster-firing and outlier filters |
| `35_dosed_vs_undosed_figure.py` | Fig 11: dosed vs undosed satellites on eclipse day |
| `43_finding_in_one_picture_v3.py` | Fig 12: visual summary of the main result (about 80 m) |
| `38_three_day_mann_whitney.py` | Fig 13: August 11, 12 and 13 side by side |
| `40_gap_vs_dose_charts.py` | Fig 14: forecast gap by amount of dose |
| `44_orbital_tilts.py` | Extra chart of satellites per tilt (not in the paper) |
