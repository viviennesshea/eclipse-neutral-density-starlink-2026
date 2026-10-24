# Detecting Solar Eclipse Effects on Neutral Density Using Starlink Ephemerides

Code for the paper *Detecting Solar Eclipse Effects on Neutral Density Using Starlink Ephemerides* (Vivienne Shea, 2026), a student research project that looks for the 12 August 2026 total solar eclipse in public Starlink satellite data.

The project computes how much of the Moon's shadow each satellite flew through (a "shadow dose" in equivalent umbra-seconds), then compares SpaceX position forecasts made before and after the eclipse. Satellites that flew through the shadow ended up about 80 m further back along their path than satellites that did not, relative to their own forecasts, which is the direction lower neutral density (thinner air) in the shadow would cause: less drag. A placebo test on the day before and the day after showed no difference.

Author: Vivienne Shea. License: MIT (see `LICENSE`).

## What is and is not included

- **Included:** the Python code for data collection, shadow geometry and dose, the forecast-gap calculation, the filters and statistical tests, and the figures.
- **Not included:** any data. SpaceX does not archive its ephemeris files, so they had to be downloaded at the time of the eclipse; Space-Track TLE data is covered by its user agreement. No outputs (CSV files, figures) are included either. The geomagnetic check (paper §2.5) used the GFZ Potsdam Kp index directly and has no script here.
- **How TLE data is used:** TLEs (CelesTrak and Space-Track) are used for two things only: (1) the initial list of satellites expected to cross the shadow (step 1), which the paper later replaces with the continuous shadow dose (step 2); and (2) each satellite's orbital tilt (inclination), which the statistical test uses to compare like with like (step 2, `build_satellite_planes.py`). The TLE scripts also record the BSTAR drag term as a by-product; BSTAR is not used in the paper.

## Folder layout and run order

| Step | Folder | What it does | Paper |
|---|---|---|---|
| 1 | `01_data_collection/` | Download SpaceX ephemeris, pick target satellites, collect TLE data (used for the target list and orbital tilt) | §2.1–2.2 |
| 2 | `02_shadow_geometry_and_dose/` | Moon's shadow geometry, dose heatmap, per-satellite 1-second dose, orbital tilt per satellite | §2.3–2.4, Figs 1–4 |
| 3 | `03_forecast_gap/` | Pre-dose vs post-dose forecast gap, extension check, drag-paradox check | §2.6, Fig 6 |
| 4 | `04_filtering_and_tests/` | Altitude, thruster-burn and IQR filters; stratified Mann-Whitney test; placebo days | §2.7–2.8, §3 |
| 5 | `05_figures/` | Figures 5–14 | §2–3 |

Each folder has its own README with every script's inputs and outputs. Run the steps in this order; each step reads the previous steps' outputs from the output folder.

## Folders: where data is read and results are written

Every script uses two folders, set in `starlink_paths.py`:

- **Data folder** (`--data-dir`): the downloaded data, laid out as
  ```
  <data-dir>/
  ├── eclipse_targets.csv                 target satellites and group labels
  └── snapshots/<snapshot name>/          one folder per download
      ├── states.parquet                  parsed ephemeris (position/velocity every minute)
      ├── raw/MEME_*.txt                  raw SpaceX ephemeris files
      └── tle/eclipse_targets_*.tle, *omm*.json
  ```
- **Output folder** (`--out-dir`): everything the scripts produce, in subfolders `shadow/`, `tle/`, `tle_spacetrack/`, `results/`, `results/ephemeris_drift/` and `docs/` (figures), plus `satellite_planes.csv`.

Each folder is taken from, in order: the command-line flag, the environment variable (`STARLINK_DATA_DIR`, `STARLINK_OUT_DIR`), a prompt if you run the script in a terminal (press Enter for the default), or the default (`./starlink_data` and `./outputs` in this folder). For example:

```
python 04_filtering_and_tests/33_filtered_mann_whitney_v2.py --data-dir ~/eclipse/starlink_data --out-dir ~/eclipse/outputs
```

or set them once:

```
export STARLINK_DATA_DIR=~/eclipse/starlink_data
export STARLINK_OUT_DIR=~/eclipse/outputs
```

`python starlink_paths.py` prints the folders that will be used.

The full pipeline was tested end to end from this folder against the original data and reproduces the paper's numbers exactly (for example, eclipse day A = 0.746, p = 0.0002; placebo days A = 0.462 and 0.503).

**Snapshot names.** Some scripts refer to specific downloads by folder name, for example `20260812T155855Z_2026-08-12_preeclipse` (the last download before the eclipse). These are the names of our 11 downloads; if your snapshot folders are named differently, edit the names at the top of those scripts.

## Requirements

Python 3.9+ and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The Space-Track downloader needs a free Space-Track account; it reads the login from the environment variables `SPACETRACK_USER` / `SPACETRACK_PASS` or from `~/.config/spacetrack/credentials`. Never commit credentials.

## AI assistance

Much of this code was written with Claude Code (Anthropic) under the author's direction; see the AI tooling disclosure in the paper.
