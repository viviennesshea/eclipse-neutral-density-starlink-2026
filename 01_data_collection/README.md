# Step 1: Data collection (paper §2.1–2.2)

> Paths: inputs are read from the data folder and outputs written under the output folder (`--data-dir` / `--out-dir`, see the main README). The "original location" is where each script lived in the project it was developed in.


| Script | Original location | What it does | Main inputs | Main outputs |
|---|---|---|---|---|
| `ephemeris/starlink_ephem.py` | `starlink_ephem.py` | Downloads the SpaceX Starlink ephemeris files (positions and velocities every minute for about 3 days) and parses them. Run once per snapshot; we took 11 snapshots from Aug 10 to Aug 14, 2026. | SpaceX public manifest (internet) | `<data-dir>/snapshots/<UTC timestamp>/` with `states.parquet` and raw `MEME_*.txt` files |
| `target_selection/eclipse_forecast.py` | `eclipse_forecast.py` | Propagates current Starlink TLEs through the eclipse and predicts which satellites cross the umbra or penumbra. Used to pick the target satellites. | CelesTrak TLE set (internet) | `<data-dir>/eclipse_transits.csv` |
| `target_selection/eclipse_shadow.py` | `eclipse_shadow.py` | Shadow geometry helper (Sun/Moon positions, umbra/penumbra test). Imported by `eclipse_forecast.py`. Identical to the copy in step 2. | – | – |
| `tle_celestrak/build_tle_observations.py` | `experiment_9_23/tle/` | Parses the TLE files saved with each snapshot into a de-duplicated table (one row per distinct orbit fit). Its summary table is used in step 2 for each satellite's orbital tilt. Also writes a reconciliation report against two files from an earlier version of the project (`experiment/data/01_panel_table.csv`, `experiment/results/03b_dose_response_robust_data.csv`), which are not included; the script runs without them and skips that report section (put them in `<data-dir>/legacy/` to include it). | `<data-dir>/snapshots/*/tle/`, `eclipse_targets.csv` | `<out-dir>/tle/output/tle_satellite_summary.csv` and others |
| `tle_spacetrack/download_spacetrack_tle.py` | `experiment_9_23/tle_spacetrack/` | Bulk download of the Space-Track TLE history (`gp_history`) for the target satellites. Needs a free Space-Track account (login from environment variables or `~/.config/spacetrack/credentials`). | `eclipse_targets.csv` | `<out-dir>/tle_spacetrack/raw/batch_*.json` |
| `tle_spacetrack/consolidate_spacetrack_tle.py` | `experiment_9_23/tle_spacetrack/` | Turns the downloaded batches into tidy tables and a report. | `raw/batch_*.json` | `<out-dir>/tle_spacetrack/output/spacetrack_all_rows.csv` and others |

Run order: `starlink_ephem.py` (at each snapshot time) → `eclipse_forecast.py` → the two TLE steps.

**Why the TLE steps are still needed:** the paper's statistical test compares dosed and undosed satellites within the same orbital tilt, and each satellite's tilt comes from these TLE tables (via `build_satellite_planes.py` in step 2). The BSTAR columns these scripts also write are not used in the paper.

The target list `<data-dir>/eclipse_targets.csv` (1,257 satellites with group labels) is read by most later steps.
