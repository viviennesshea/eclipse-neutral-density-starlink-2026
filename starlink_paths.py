#!/usr/bin/env python3
"""
starlink_paths.py
=================
One place that decides where every script reads data from and writes results to.

Two folders:
  DATA_DIR  holds the downloaded data:
              eclipse_targets.csv
              snapshots/<snapshot name>/states.parquet, raw/MEME_*.txt, tle/...
  OUT_DIR   receives everything the scripts produce (tables, reports, figures).

Each folder is chosen in this order:
  1. a command-line flag:      --data-dir PATH   /   --out-dir PATH
  2. an environment variable:  STARLINK_DATA_DIR  /   STARLINK_OUT_DIR
  3. if running interactively, the script asks (press Enter to accept the default)
  4. the default:              ./starlink_data    /   ./outputs   (next to this file)

Example:
  python 04_filtering_and_tests/33_filtered_mann_whitney_v2.py --data-dir ~/eclipse/starlink_data --out-dir ~/eclipse/outputs

Every script imports this module first (as SP). It removes --data-dir / --out-dir from the command line,
so each script's own options keep working.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(REPO, "starlink_data")
DEFAULT_OUT = os.path.join(REPO, "outputs")


def _take_flag(flag: str):
    """Return the value of --flag PATH or --flag=PATH and remove it from sys.argv."""
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            del sys.argv[i + 1:i + 3]
            return args[i + 1]
        if a.startswith(flag + "="):
            del sys.argv[i + 1]
            return a.split("=", 1)[1]
    return None


def _choose(flag: str, env: str, default: str, what: str) -> str:
    value = _take_flag(flag) or os.environ.get(env)
    if not value and sys.stdin is not None and sys.stdin.isatty():
        try:
            answer = input(f"{what} [{default}]: ").strip()
        except EOFError:
            answer = ""
        value = answer or default
    return os.path.abspath(os.path.expanduser(value or default))


DATA_DIR = _choose("--data-dir", "STARLINK_DATA_DIR", DEFAULT_DATA, "Data folder (eclipse_targets.csv and snapshots/)")
OUT_DIR = _choose("--out-dir", "STARLINK_OUT_DIR", DEFAULT_OUT, "Output folder")

# ---- inputs (inside DATA_DIR)
TARGETS_CSV = os.path.join(DATA_DIR, "eclipse_targets.csv")
SNAP_DIR = os.path.join(DATA_DIR, "snapshots")

# ---- outputs (inside OUT_DIR)
SHADOW_DATA = os.path.join(OUT_DIR, "shadow", "data")          # step 2 shadow tables
SHADOW_RESULTS = os.path.join(OUT_DIR, "shadow", "results")    # step 2 shadow reports and Figures 1-4
TLE_OUT = os.path.join(OUT_DIR, "tle", "output")               # step 1 CelesTrak TLE tables
ST_RAW = os.path.join(OUT_DIR, "tle_spacetrack", "raw")        # step 1 Space-Track downloads
ST_OUT = os.path.join(OUT_DIR, "tle_spacetrack", "output")     # step 1 Space-Track tables
RESULTS = os.path.join(OUT_DIR, "results")                     # satellite_dosage.csv
ED = os.path.join(RESULTS, "ephemeris_drift")                  # steps 3-4 tables
DOCS = os.path.join(OUT_DIR, "docs")                           # step 5 figures
PLANES_CSV = os.path.join(OUT_DIR, "satellite_planes.csv")     # orbital tilt per satellite
COHORT_CSV = os.path.join(OUT_DIR, "cohorts_closest_quartile.csv")  # optional, not produced by this repository

# ---- code (inside this repository)
SHADOW_SRC = os.path.join(REPO, "02_shadow_geometry_and_dose")  # eclipse_shadow.py, 04_shadow_dose_heatmap.py

for _d in (SHADOW_DATA, SHADOW_RESULTS, TLE_OUT, ST_RAW, ST_OUT, RESULTS, ED, DOCS):
    os.makedirs(_d, exist_ok=True)


def snapshot(name: str) -> str:
    """Folder of one ephemeris snapshot, e.g. snapshot('20260812T155855Z_2026-08-12_preeclipse')."""
    return os.path.join(SNAP_DIR, name)


def load_dose_module():
    """Import 04_shadow_dose_heatmap.py (its name starts with a digit, so a normal import doesn't work)."""
    import importlib.util
    if SHADOW_SRC not in sys.path:
        sys.path.insert(0, SHADOW_SRC)
    spec = importlib.util.spec_from_file_location("dm", os.path.join(SHADOW_SRC, "04_shadow_dose_heatmap.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR: ", OUT_DIR)
