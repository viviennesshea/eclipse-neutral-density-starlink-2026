#!/usr/bin/env python3
"""
01_shadow_geometry.py
========================
First piece of the rebuilt analysis: an accurate description of the
umbra/penumbra footprint at SATELLITE ALTITUDE (not ground level) for the
2026-08-12 total solar eclipse, across the full duration the shadow
touches Earth anywhere.

Why altitude matters
---------------------
The Moon's shadow is a cone in space, not a flat shape stamped on the
ground. A satellite at ~466 km doesn't sit on the ground -- it sits
higher up, closer to the Moon, so the cone it intersects is narrower than
the cone that reaches the ground. Classifying shadow status directly at
satellite altitude (rather than computing the ground path and assuming
it applies unchanged higher up) is the physically correct approach for a
paper about satellite drag.

Altitude used: 465.6 km -- the median semi-major-axis-derived altitude of
our umbra-candidate satellites (established in the earlier `experiment/`
pipeline, step 5/6). Kept as a single named constant here so it's easy to
re-run this at a different altitude later if needed.

Method
------
A regular latitude/longitude grid, evaluated at each grid cell's true
geodetic position AT ALTITUDE, transformed to GCRS via astropy's
EarthLocation.get_gcrs() (proper Earth orientation -- sidereal time and
polar motion handled by astropy, not approximated), then classified
against the real Sun/Moon shadow cone with the already-validated
eclipse_shadow.classify_shadow() (conical shadow test, Vallado-style).

Grid design, and why:
  - Full 360 degrees of longitude (-180 to 180, half-open) -- a partial
    range leaves a blank, un-evaluated wedge with no data in it at all.
  - Latitude stops at 89.5 degrees, not 90 -- longitude is mathematically
    undefined exactly at the pole, and including that row produces
    degenerate, zero-width cells that break area calculations and plotting.
  - Latitude extent -20 to 90 -- confirmed empirically (see
    ANALYSIS_LOG.md) that the real penumbra for this eclipse reaches as
    far south as -6 degrees around 19:45 UTC as it exits over the
    Atlantic/Africa, so -20 gives real margin without wasting compute on
    the whole globe.
  - 0.5 degree resolution, 1-minute cadence, 15:00-20:15 UTC (confirmed
    empirically to be the full window any shadow touches this altitude
    anywhere on the grid, with margin on both ends).

Surface area, computed correctly
----------------------------------
A lat/lon grid cell is NOT constant area -- cells near the pole are much
smaller than cells near the equator for the same lat/lon step. Each
cell's true area on the sphere of radius R = R_EARTH + ALTITUDE_KM is:

    area = R^2 * dlon_rad * (sin(lat_top_rad) - sin(lat_bottom_rad))

This is computed exactly (not approximated with cos(lat) * dlat * dlon)
so that summed umbra/penumbra areas reported in the stats file are
correct, not just visually plausible.

Storage format -- two files, two different jobs
--------------------------------------------------
Storing every individual cell at full (0.5 deg, 1-min) resolution turned
out to produce an 8.2-million-row, ~500 MB file -- accurate, but not a
reasonable "start here" artifact for mapping. So this splits into two
outputs with different jobs:

  1. AREA ACCURACY (`01_shadow_geometry_area_timeseries.csv`): computed at
     the full fine grid (0.5 deg, 1-min cadence) so the reported umbra/
     penumbra areas are exact, but only totals per timestep are kept --
     tiny file (~20 KB), no per-cell rows.

  2. MAP RENDERING (`01_shadow_geometry_map_layer.csv`): a deliberately
     coarser grid (MAP_LAT_STEP/MAP_LON_STEP degrees, MAP_TIME_STEP_MIN
     cadence) computed independently (not resampled from the fine grid,
     to keep it a clean, self-consistent, directly-reproducible dataset
     in its own right) -- still uses the same exact shadow classification
     and exact spherical cell-area formula, just fewer cells/timesteps,
     sized to actually be usable in a plotting tool. Columns: time_utc,
     lat_deg, lon_deg (cell center), shadow_type, cell_area_km2 -- same
     schema as before, just smaller. Only shadow-touched cells are kept
     ("none" cells dropped; the fixed grid definition above means "not
     present at this time" unambiguously means "no shadow contact").

Read-only: this script only computes from Sun/Moon ephemeris (via
astropy) and eclipse_shadow.py. Nothing in starlink_data/ or experiment/
is read or modified. No downloads.

Output: <out-dir>/shadow/data/01_shadow_geometry.csv,
<out-dir>/shadow/results/01_shadow_geometry_stats.txt
"""

from __future__ import annotations

import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eclipse_shadow import classify_shadow, sun_moon_gcrs_km, UMBRA, PENUMBRA, ANTUMBRA, NONE  # noqa: E402


# --altitude lets this same script produce a ground-truth (0 km) run for
# validating against a published eclipse-path reference, as well as the
# satellite-altitude run this script was originally built for. Default
# (465.6 km, the satellite altitude) keeps the ORIGINAL filenames
# unchanged, so anything already validated/logged at that altitude isn't
# silently renamed; any other altitude gets a tag suffix instead.
_parser = argparse.ArgumentParser()
_parser.add_argument("--altitude", type=float, default=465.6, help="altitude in km (default: 465.6, satellite altitude)")
_parser.add_argument("--tag", type=str, default=None, help="output filename tag (default: auto from altitude)")
_args, _unknown = _parser.parse_known_args()

ALTITUDE_KM = _args.altitude
_is_default_altitude = abs(ALTITUDE_KM - 465.6) < 1e-9
TAG = _args.tag or ("" if _is_default_altitude else (
    "ground" if ALTITUDE_KM == 0 else f"alt{ALTITUDE_KM:g}km"
))
_suffix = f"_{TAG}" if TAG else ""

OUT_AREA_CSV = os.path.join(SP.SHADOW_DATA, f"01_shadow_geometry_area_timeseries{_suffix}.csv")
OUT_MAP_CSV = os.path.join(SP.SHADOW_DATA, f"01_shadow_geometry_map_layer{_suffix}.csv")
OUT_UMBRA_FINE_CSV = os.path.join(SP.SHADOW_DATA, f"01_shadow_geometry_umbra_fine{_suffix}.csv")
OUT_STATS_TXT = os.path.join(SP.SHADOW_RESULTS, f"01_shadow_geometry_stats{_suffix}.txt")

R_EARTH_KM = 6378.137
R_SHELL_KM = R_EARTH_KM + ALTITUDE_KM

# Fine grid: used ONLY to compute exact per-timestep areas (totals kept, cells discarded).
FINE_LAT_STEP = 0.5
FINE_LON_STEP = 0.5
FINE_LAT = np.arange(-20.0, 89.51, FINE_LAT_STEP)   # empirically confirmed shadow extent, see ANALYSIS_LOG.md
FINE_LON = np.arange(-180.0, 180.0, FINE_LON_STEP)  # full 360 degrees, half-open (no double-count at wrap)
FINE_TIME_STEP_MIN = 1

# Coarse grid: used to produce the actual per-cell map-rendering file (kept small on purpose).
MAP_LAT_STEP = 1.0
MAP_LON_STEP = 1.0
MAP_LAT = np.arange(-20.0, 89.01, MAP_LAT_STEP)
MAP_LON = np.arange(-180.0, 180.0, MAP_LON_STEP)
MAP_TIME_STEP_MIN = 5

TIME_START = "2026-08-12T15:00:00"
TIME_END = "2026-08-12T20:15:00"


def cell_areas_km2(lat_bins: np.ndarray, lat_step_deg: float, lon_step_deg: float, radius_km: float) -> np.ndarray:
    """Exact area (km^2) of each latitude BAND's cells at this grid's lon step,
    on a sphere of the given radius. lat_bins are cell CENTERS; cell edges are
    +/- half the lat step. Returns one area value per lat_bins entry (constant
    across longitude, since area only depends on latitude band + lon width)."""
    half = lat_step_deg / 2.0
    lat_top = np.radians(lat_bins + half)
    lat_bot = np.radians(lat_bins - half)
    dlon_rad = np.radians(lon_step_deg)
    return radius_km**2 * dlon_rad * (np.sin(lat_top) - np.sin(lat_bot))


def build_time_grid(start: str, end: str, step_min: int):
    from astropy.time import Time, TimeDelta

    t0 = Time(start, scale="utc")
    t1 = Time(end, scale="utc")
    n = int((t1 - t0).sec // (step_min * 60)) + 1
    return t0 + TimeDelta(np.arange(n) * step_min * 60, format="sec")


def run_pass(lat_bins, lat_step, lon_bins, lon_step, time_step_min, keep_rows: bool, label: str,
             keep_types=None):
    """Classify shadow status on a (lat_bins x lon_bins) grid across the shared
    TIME_START/TIME_END window, at time_step_min cadence. Always returns the
    per-timestep area summary; if keep_rows, also returns a long-format
    per-cell DataFrame of every shadow-touched cell at every timestep
    (restricted to `keep_types` if given -- e.g. UMBRA only, since the
    umbra touches far fewer cells than the penumbra and can affordably be
    stored at the SAME fine resolution used for the area totals, instead
    of the coarser grid needed to keep the full penumbra layer small)."""
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    LAT, LON = np.meshgrid(lat_bins, lon_bins, indexing="ij")
    flat_lat = LAT.ravel()
    flat_lon = LON.ravel()
    n_cells = flat_lat.size
    print(f"[{label}] Grid: {len(lat_bins)} lat x {len(lon_bins)} lon = {n_cells} cells, "
          f"altitude {ALTITUDE_KM} km", file=sys.stderr)

    area_by_lat = cell_areas_km2(lat_bins, lat_step, lon_step, R_SHELL_KM)
    flat_area = area_by_lat[np.searchsorted(lat_bins, flat_lat)]

    times = build_time_grid(TIME_START, TIME_END, time_step_min)
    print(f"[{label}] Time grid: {len(times)} steps, {TIME_START} to {TIME_END} UTC, "
          f"{time_step_min}-min cadence", file=sys.stderr)

    loc = EarthLocation.from_geodetic(lon=flat_lon * u.deg, lat=flat_lat * u.deg, height=ALTITUDE_KM * u.km)

    rows = []
    summary_rows = []

    for i, t in enumerate(times):
        gcrs = loc.get_gcrs(obstime=t)
        r_obs = gcrs.cartesian.xyz.to("km").value.T

        r_sun, r_moon = sun_moon_gcrs_km(t)
        status, *_ = classify_shadow(r_obs, np.tile(r_sun, (n_cells, 1)), np.tile(r_moon, (n_cells, 1)))

        shadowed = status != NONE
        n_umbra = int(np.sum(status == UMBRA))
        n_penumbra = int(np.sum(status == PENUMBRA))
        n_antumbra = int(np.sum(status == ANTUMBRA))
        area_umbra = float(flat_area[status == UMBRA].sum())
        area_penumbra = float(flat_area[status == PENUMBRA].sum())
        area_antumbra = float(flat_area[status == ANTUMBRA].sum())

        summary_rows.append({
            "time_utc": t.isot + "Z",
            "n_umbra_cells": n_umbra, "n_penumbra_cells": n_penumbra, "n_antumbra_cells": n_antumbra,
            "umbra_area_km2": area_umbra, "penumbra_area_km2": area_penumbra, "antumbra_area_km2": area_antumbra,
        })

        keep_mask = shadowed if keep_types is None else np.isin(status, keep_types)
        if keep_rows and keep_mask.any():
            rows.append(pd.DataFrame({
                "time_utc": t.isot + "Z",
                "lat_deg": flat_lat[keep_mask],
                "lon_deg": flat_lon[keep_mask],
                "shadow_type": status[keep_mask],
                "cell_area_km2": flat_area[keep_mask],
            }))

        if i % 30 == 0 or i == len(times) - 1:
            print(f"[{label}]   step {i+1}/{len(times)}  {t.isot}  "
                  f"umbra_cells={n_umbra} penumbra_cells={n_penumbra} "
                  f"umbra_km2={area_umbra:.0f} penumbra_km2={area_penumbra:.0f}", file=sys.stderr)

    summary = pd.DataFrame(summary_rows)
    if not keep_rows:
        return summary, None
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["time_utc", "lat_deg", "lon_deg", "shadow_type", "cell_area_km2"]
    )
    return summary, out


def main():
    # Pass 1: fine grid, area accuracy only (no per-cell rows kept -- would be ~500MB)
    summary, _ = run_pass(FINE_LAT, FINE_LAT_STEP, FINE_LON, FINE_LON_STEP, FINE_TIME_STEP_MIN,
                          keep_rows=False, label="fine/area")

    # Pass 1b: SAME fine grid/cadence, but keep only UMBRA rows. The umbra
    # touches far fewer cells than the penumbra (hundreds vs tens of
    # thousands per timestep), so storing it at full fine resolution is
    # cheap -- and necessary, since the coarse map-layer grid (1 deg /
    # 5-min, sized for the huge penumbra) is too coarse for the umbra's
    # ~100-300km width and produces a gappy, dashed-looking path instead
    # of a continuous band.
    _, umbra_fine = run_pass(FINE_LAT, FINE_LAT_STEP, FINE_LON, FINE_LON_STEP, FINE_TIME_STEP_MIN,
                             keep_rows=True, label="fine/umbra-only", keep_types=[UMBRA])
    os.makedirs(os.path.dirname(OUT_UMBRA_FINE_CSV), exist_ok=True)
    umbra_fine.to_csv(OUT_UMBRA_FINE_CSV, index=False)
    print(f"Fine-resolution umbra-only layer ({len(umbra_fine)} rows) -> {OUT_UMBRA_FINE_CSV}", file=sys.stderr)
    os.makedirs(os.path.dirname(OUT_AREA_CSV), exist_ok=True)
    summary.to_csv(OUT_AREA_CSV, index=False)
    print(f"\nArea time series ({len(summary)} rows) -> {OUT_AREA_CSV}", file=sys.stderr)

    # Pass 2: coarse grid, sized to actually be usable for map rendering
    _, map_layer = run_pass(MAP_LAT, MAP_LAT_STEP, MAP_LON, MAP_LON_STEP, MAP_TIME_STEP_MIN,
                            keep_rows=True, label="coarse/map")
    map_layer.to_csv(OUT_MAP_CSV, index=False)
    print(f"Map layer ({len(map_layer)} rows) -> {OUT_MAP_CSV}", file=sys.stderr)
    peak_umbra = summary.loc[summary["umbra_area_km2"].idxmax()]
    peak_penumbra = summary.loc[summary["penumbra_area_km2"].idxmax()]

    lines = []
    lines.append("Shadow geometry at satellite altitude (step 1)")
    lines.append("=" * 48)
    lines.append(f"Altitude: {ALTITUDE_KM} km (shell radius {R_SHELL_KM:.1f} km)")
    lines.append(f"Fine grid (area accuracy): {FINE_LAT_STEP} deg lat x {FINE_LON_STEP} deg lon, "
                 f"lat range [{FINE_LAT.min()}, {FINE_LAT.max()}], {FINE_TIME_STEP_MIN}-min cadence")
    lines.append(f"Coarse grid (map layer): {MAP_LAT_STEP} deg lat x {MAP_LON_STEP} deg lon, "
                 f"lat range [{MAP_LAT.min()}, {MAP_LAT.max()}], {MAP_TIME_STEP_MIN}-min cadence")
    lines.append(f"Time window: {TIME_START} to {TIME_END} UTC")
    lines.append("")
    lines.append(f"Peak instantaneous umbra area: {peak_umbra['umbra_area_km2']:.0f} km^2 "
                 f"at {peak_umbra['time_utc']}")
    lines.append(f"Peak instantaneous penumbra area: {peak_penumbra['penumbra_area_km2']:.0f} km^2 "
                 f"at {peak_penumbra['time_utc']}")
    lines.append(f"Map-layer rows written: {len(map_layer)}")
    lines.append(f"Fine-resolution umbra-only rows written: {len(umbra_fine)}")
    lines.append(f"Shadow types present in map layer: "
                 f"{map_layer['shadow_type'].value_counts().to_dict() if len(map_layer) else {}}")
    lines.append("")
    lines.append("Per-timestep area summary (fine grid; first 5, peak, last 5):")
    lines.append(summary.head(5).to_string(index=False))
    lines.append("...")
    lines.append(summary.loc[[summary['umbra_area_km2'].idxmax()]].to_string(index=False))
    lines.append("...")
    lines.append(summary.tail(5).to_string(index=False))

    os.makedirs(os.path.dirname(OUT_STATS_TXT), exist_ok=True)
    with open(OUT_STATS_TXT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nStats: {OUT_STATS_TXT}")


if __name__ == "__main__":
    main()
