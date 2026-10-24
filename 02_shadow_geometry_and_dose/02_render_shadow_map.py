#!/usr/bin/env python3
"""
02_render_shadow_map.py
==========================
Render the shadow geometry stored in
`data/01_shadow_geometry_map_layer.csv` as a North Polar Stereographic
map -- the umbra and penumbra footprint at satellite altitude (465.6 km),
swept across several snapshots to show the shadow's path.

This reads the ALREADY-COMPUTED map-layer data (does not recompute shadow
geometry itself) so the figure is a direct visualization of the stored
step-1 output, not a separate computation path.

Known pitfalls, deliberately handled here (each one broke an earlier
attempt at this exact figure -- see ../../experiment/docs/ANALYSIS_LOG.md
for the history):
  1. Partial longitude range in the underlying grid -> blank wedge.
     `01_shadow_geometry.py`'s grid is already full 360 degrees, so this
     is inherited as fixed; nothing to do here except not crop it.
  2. Grid row exactly at the pole -> degenerate cells / seam artifacts.
     Also already avoided upstream (grid stops at 89 degrees).
  3. No cyclic closure at the +/-180 meridian -> a visible seam line once
     reprojected through the polar transform. THIS script adds the
     cyclic point back in (via cartopy.util.add_cyclic_point) when it
     pivots each snapshot into a 2D grid for contourf, since the stored
     CSV itself doesn't duplicate the wrap column (no reason to store
     redundant rows).
  4. `ax.set_extent` given a longitude range narrower than 360 degrees,
     or a rectangular (non-circular) axes boundary -> pie-slice wedge cut
     in the rendered circle, independent of the data being correct. Fixed
     with a full [-180, 180] extent and an explicit circular boundary
     clip (the standard Cartopy polar-stereographic recipe).

Read-only: reads data/01_shadow_geometry_map_layer.csv (already derived,
computed in step 1). Nothing in starlink_data/ or experiment/ touched.

Output: results/02_shadow_map.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point


_parser = argparse.ArgumentParser()
_parser.add_argument("--tag", type=str, default=None,
                     help="matches 01_shadow_geometry.py's --tag/--altitude output suffix, "
                          "e.g. 'ground' to read *_ground.csv. Default: the satellite-altitude run.")
_parser.add_argument("--altitude-label", type=str, default="satellite altitude, 465.6 km",
                     help="text shown in the figure title/caption")
_args, _unknown = _parser.parse_known_args()
_suffix = f"_{_args.tag}" if _args.tag else ""

IN_CSV = os.path.join(SP.SHADOW_DATA, f"01_shadow_geometry_map_layer{_suffix}.csv")
IN_UMBRA_FINE_CSV = os.path.join(SP.SHADOW_DATA, f"01_shadow_geometry_umbra_fine{_suffix}.csv")
OUT_PNG = os.path.join(SP.SHADOW_RESULTS, f"02_shadow_map{_suffix}.png")

# Must match 01_shadow_geometry.py's MAP_* grid definition exactly, since
# we're re-gridding the stored (sparse, touched-cells-only) rows back into
# a dense 2D array. Used for PENUMBRA only now -- see FINE_* below for umbra.
MAP_LAT_STEP = 1.0
MAP_LON_STEP = 1.0
MAP_LAT = np.arange(-20.0, 89.01, MAP_LAT_STEP)
MAP_LON = np.arange(-180.0, 180.0, MAP_LON_STEP)

# Must match 01_shadow_geometry.py's FINE_* grid definition. The umbra is
# rendered from this finer grid/cadence (via 01_shadow_geometry_umbra_fine*.csv)
# instead of the coarse map layer above -- at 1deg/5-min the umbra's
# ~100-300km width and fast ground speed leave real gaps between
# snapshots, rendering as a dashed/streaked line instead of a continuous
# band. The umbra touches few enough cells that fine resolution is cheap.
FINE_LAT_STEP = 0.5
FINE_LON_STEP = 0.5
FINE_LAT = np.arange(-20.0, 89.51, FINE_LAT_STEP)
FINE_LON = np.arange(-180.0, 180.0, FINE_LON_STEP)

CODE = {"none": 0, "penumbra": 1, "antumbra": 2, "umbra": 3}


def union_grid_from(df: pd.DataFrame, lat_bins: np.ndarray, lon_bins: np.ndarray) -> np.ndarray:
    """Long-format (time, lat, lon, shadow_type) rows -> dense (lat x lon)
    grid of the STRONGEST shadow code any cell ever reached across all
    stored timesteps."""
    grid = np.zeros((len(lat_bins), len(lon_bins)), dtype=float)
    if len(df) == 0:
        return grid
    lat_idx = np.searchsorted(lat_bins, df["lat_deg"].values)
    lon_idx = np.searchsorted(lon_bins, df["lon_deg"].values)
    codes = df["shadow_type"].map(CODE).values
    np.maximum.at(grid, (lat_idx, lon_idx), codes)
    return grid



def main():
    df = pd.read_csv(IN_CSV, parse_dates=["time_utc"])
    all_times = sorted(df["time_utc"].unique())
    print(f"Loaded {len(df)} penumbra-layer rows across {len(all_times)} timesteps "
          f"(1deg/5-min grid).")

    df_umbra = pd.read_csv(IN_UMBRA_FINE_CSV, parse_dates=["time_utc"])
    umbra_times = sorted(df_umbra["time_utc"].unique()) if len(df_umbra) else []
    print(f"Loaded {len(df_umbra)} fine-resolution umbra rows across {len(umbra_times)} "
          f"timesteps (0.5deg/1-min grid).")

    # Union across ALL timesteps: for each grid cell, the STRONGEST shadow
    # status it ever reached during the whole event. This draws the path as
    # one solid band (matching how published totality-path maps look),
    # rather than stacking many translucent per-snapshot layers -- which,
    # tried first, produced two problems: (1) a sparse subset of snapshots
    # left visible gaps since the umbra is narrow and moves fast, and (2)
    # overlapping semi-transparent layers where snapshots DO overlap
    # stack their alpha and get progressively darker/muddier, which is not
    # a meaningful visual (it doesn't mean "more shadow", just "more
    # snapshots happened to overlap here").
    #
    # Penumbra and umbra are unioned SEPARATELY, from different grids: the
    # coarse map layer (fine enough for the huge, slow-moving penumbra) and
    # the fine umbra-only layer (needed for the narrow, fast-moving umbra
    # -- at the coarse grid's 5-min cadence it left real gaps between
    # snapshots and rendered as a dashed/streaked line instead of a
    # continuous band).
    penumbra_grid = union_grid_from(df, MAP_LAT, MAP_LON)
    umbra_grid = union_grid_from(df_umbra, FINE_LAT, FINE_LON)
    print(f"Penumbra union (coarse grid): {int((penumbra_grid>=1).sum())} cells ever touched.")
    print(f"Umbra union (fine grid): {int((umbra_grid>=3).sum())} cells ever touched.")

    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cfeature.OCEAN, facecolor="#eef3f8", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f7f5f0", zorder=0)
    ax.coastlines(resolution="110m", linewidth=0.6, color="#555555", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#aaaaaa", zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#cccccc", linestyle="--", zorder=1)

    pen_c, pen_lon_c = add_cyclic_point(penumbra_grid, coord=MAP_LON, axis=1)
    ax.contourf(pen_lon_c, MAP_LAT, np.where(pen_c >= 1.0, 1, 0), levels=[0.5, 1.5],
                colors=["#7FA8D9"], alpha=0.45, transform=ccrs.PlateCarree(), zorder=2)

    if umbra_grid.max() > 0:
        umb_c, umb_lon_c = add_cyclic_point(umbra_grid, coord=FINE_LON, axis=1)
        ax.contourf(umb_lon_c, FINE_LAT, np.where(umb_c >= 3.0, 1, 0), levels=[0.5, 1.5],
                    colors=["#8B0000"], alpha=0.9, transform=ccrs.PlateCarree(), zorder=3)
        ax.contour(umb_lon_c, FINE_LAT, np.where(umb_c >= 3.0, 1, 0), levels=[0.5],
                   colors=["#4A0000"], linewidths=0.6, transform=ccrs.PlateCarree(), zorder=4)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#4A0000", alpha=0.55, label="Umbra (total shadow)"),
        Patch(facecolor="#7FA8D9", alpha=0.35, label="Penumbra (partial shadow)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)

    t0, t1 = str(all_times[0])[11:16], str(all_times[-1])[11:16]
    ax.set_title(
        f"Moon's shadow, 2026-08-12 total solar eclipse ({_args.altitude_label})\n"
        f"penumbra: {len(all_times)} timesteps (1deg/5-min); umbra: {len(umbra_times)} "
        f"timesteps (0.5deg/1-min); {t0}-{t1} UTC",
        fontsize=11,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
