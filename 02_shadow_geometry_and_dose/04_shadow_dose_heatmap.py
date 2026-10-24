#!/usr/bin/env python3
"""
04_shadow_dose_heatmap.py
============================
"Volume of air" dose field: for every point in space (a lat/lon grid at
satellite altitude), how much cumulative shadow exposure did that point
experience across the whole eclipse -- not "was it ever in the umbra"
(binary), but a single continuous number that treats 1 second at 10%
solar obscuration the same as 2 seconds at 5% (i.e. the time-integral of
obscuration fraction).

Unit: "equivalent umbra-seconds" = integral of obscuration_fraction(t) dt.
A point under total eclipse for 1 second contributes 1.0; a point under
50% obscuration for 2 seconds also contributes 1.0. This is dimensionally
just seconds (fraction is unitless) and is backward-compatible with the
old umbra_seconds concept (a pure-100%-obscuration span's dose equals its
duration exactly). Values keep accumulating with time and are NOT capped
at 100 -- a point under partial shadow for a long stretch can accumulate
a dose far larger than any single instant's obscuration percentage.

Obscuration fraction itself (0-1, the fraction of the Sun's disk area
the Moon covers, as seen from that point) is computed from the SAME
theta/a_sun/a_moon that classify_shadow() already produces, via the
standard circle-circle intersection ("lens") area formula used in real
eclipse calculators -- this fully replaces the umbra/penumbra/antumbra
CATEGORIES with one continuous physical quantity. See the 2026-09-06
"is umbra really special" discussion in ANALYSIS_LOG.md for why: the
actual physical driver (solar heating) plausibly scales continuously
with % obscuration, not as a step function at the umbra boundary.

Performance design -- two real problems found and fixed while building
this (both documented in ANALYSIS_LOG.md, both instances of the same
underlying lesson: astropy calls are relatively expensive individually,
so never call astropy at the FULL fine time resolution):

  1. Computing each grid cell's true position (EarthLocation.get_gcrs)
     at every single fine second would need ~20,000+ individual astropy
     calls -- hours of runtime. First attempted fix: replace astropy
     entirely with an analytical Earth-rotation-rate formula (rotate a
     single real anchor position by Earth's sidereal rotation rate).
     THIS TURNED OUT TO BE WRONG: verified against fresh astropy calls
     and found ~22km of error after 5.5 hours, growing linearly with
     time -- a rotation-RATE mismatch. Tried calibrating the rate
     empirically from two real anchors instead of trusting a textbook
     sidereal-rate constant -- STILL wrong (~15-27km residual error,
     non-monotonic, because different grid points don't share a single
     effective rotation rate -- likely polar motion / latitude-dependent
     effects a pure single-axis rotation model can't capture).
  2. Working fix: compute REAL astropy positions at a moderate cadence
     (60s) for the whole grid at once (vectorized, ~0.45s per call
     regardless of grid size), then LINEARLY INTERPOLATE grid positions
     between those real anchors for the intermediate 1-second steps --
     the same method already validated for satellite trajectories in
     step 3. A fixed geographic point's arc over 60 real seconds of
     Earth's rotation is tiny (~0.25 degrees) relative to Earth's
     radius, so linear interpolation error is a few tens of meters --
     far below anything that matters for ~100+ km shadow features.
  Sun/Moon positions are interpolated the same way, from the same 60s
  anchors (their position barely changes over 60s regardless).

Grid: reuses 01_shadow_geometry.py's MAP_LAT/MAP_LON definition (1 deg,
-20 to 89 latitude, full 360 longitude -- already validated, no pole
singularity, no wrap-seam issues).

Window: 14:45-20:30 UTC, padding around the 15:00-20:15 UTC window
step 1 empirically confirmed is when any shadow touches this altitude
anywhere on the grid.

Read-only: computes only from Sun/Moon ephemeris (astropy) and
eclipse_shadow.py. Nothing in starlink_data/ or experiment/ touched. No
downloads.

Output: <out-dir>/shadow/data/04_shadow_dose_field.csv,
<out-dir>/shadow/results/04_shadow_dose_field_stats.txt,
<out-dir>/shadow/results/04_shadow_dose_heatmap.png
"""

from __future__ import annotations

import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eclipse_shadow import sun_moon_gcrs_km  # noqa: E402

OUT_CSV = os.path.join(SP.SHADOW_DATA, "04_shadow_dose_field.csv")
OUT_STATS_TXT = os.path.join(SP.SHADOW_RESULTS, "04_shadow_dose_field_stats.txt")
OUT_PNG = os.path.join(SP.SHADOW_RESULTS, "04_shadow_dose_heatmap.png")

ALTITUDE_KM = 465.6
GRID_LAT_STEP = 1.0
GRID_LON_STEP = 1.0
GRID_LAT = np.arange(-20.0, 89.01, GRID_LAT_STEP)
GRID_LON = np.arange(-180.0, 180.0, GRID_LON_STEP)

WINDOW_START = "2026-08-12T14:45:00"
WINDOW_END = "2026-08-12T20:30:00"
ANCHOR_STEP_S = 60.0   # real astropy calls at this cadence
FINE_STEP_S = 1.0      # dose accumulated at this cadence, via interpolation between anchors

# Heatmap display cap (per the "capped + extend" design decision) -- picked
# after seeing the actual data distribution (set below main(), before plotting).
HEATMAP_CAP_PERCENTILE = 99.0


def obscuration_fraction(theta: np.ndarray, a_sun: np.ndarray, a_moon: np.ndarray) -> np.ndarray:
    """Fraction (0-1) of the Sun's disk AREA covered by the Moon, as seen
    from a point with Sun-Moon angular separation theta and angular radii
    a_sun/a_moon (all radians). Standard circle-circle intersection (lens)
    area formula, normalized by the Sun's disk area."""
    theta = np.asarray(theta, float)
    a_sun = np.asarray(a_sun, float)
    a_moon = np.asarray(a_moon, float)

    frac = np.zeros_like(theta)

    no_overlap = theta >= (a_sun + a_moon)
    full_contain = theta <= np.abs(a_sun - a_moon)
    partial = ~no_overlap & ~full_contain

    moon_covers_sun = full_contain & (a_moon >= a_sun)
    frac[moon_covers_sun] = 1.0
    moon_inside_sun = full_contain & (a_moon < a_sun)
    frac[moon_inside_sun] = (a_moon[moon_inside_sun] / a_sun[moon_inside_sun]) ** 2

    d, r1, r2 = theta[partial], a_sun[partial], a_moon[partial]
    d1 = (d**2 + r1**2 - r2**2) / (2 * d)
    d2 = d - d1
    d1c = np.clip(d1, -r1, r1)
    d2c = np.clip(d2, -r2, r2)
    area1 = r1**2 * np.arccos(d1c / r1) - d1c * np.sqrt(np.maximum(r1**2 - d1c**2, 0))
    area2 = r2**2 * np.arccos(d2c / r2) - d2c * np.sqrt(np.maximum(r2**2 - d2c**2, 0))
    lens_area = area1 + area2
    frac[partial] = lens_area / (np.pi * r1**2)

    return np.clip(frac, 0.0, 1.0)


def build_time_grid(start: str, end: str, step_s: float):
    from astropy.time import Time, TimeDelta

    t0 = Time(start, scale="utc")
    t1 = Time(end, scale="utc")
    n = int((t1 - t0).sec // step_s) + 1
    return t0 + TimeDelta(np.arange(n) * step_s, format="sec")


def main():
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    LAT, LON = np.meshgrid(GRID_LAT, GRID_LON, indexing="ij")
    flat_lat = LAT.ravel()
    flat_lon = LON.ravel()
    n_cells = flat_lat.size
    print(f"Grid: {len(GRID_LAT)} lat x {len(GRID_LON)} lon = {n_cells} cells, "
          f"altitude {ALTITUDE_KM} km", file=sys.stderr)

    anchor_times = build_time_grid(WINDOW_START, WINDOW_END, ANCHOR_STEP_S)
    n_anchors = len(anchor_times)
    print(f"Anchors: {n_anchors} real astropy positions ({ANCHOR_STEP_S}s cadence)", file=sys.stderr)

    loc = EarthLocation.from_geodetic(lon=flat_lon * u.deg, lat=flat_lat * u.deg, height=ALTITUDE_KM * u.km)

    anchor_r = np.empty((n_anchors, n_cells, 3))
    anchor_sun = np.empty((n_anchors, 3))
    anchor_moon = np.empty((n_anchors, 3))
    for i, t in enumerate(anchor_times):
        anchor_r[i] = loc.get_gcrs(obstime=t).cartesian.xyz.to("km").value.T
        r_sun, r_moon = sun_moon_gcrs_km(t)
        anchor_sun[i] = r_sun
        anchor_moon[i] = r_moon
        if i % 50 == 0:
            print(f"  anchor {i}/{n_anchors}", file=sys.stderr)

    dose = np.zeros(n_cells)
    peak_obscuration = np.zeros(n_cells)
    peak_time_unix = np.full(n_cells, np.nan)  # WHEN each cell's own peak occurred -- a property
                                                 # of the location, needed to compare against when
                                                 # any given satellite actually passes through it

    anchor_unix = np.array([t.unix for t in anchor_times])

    print("Fine-resolution dose accumulation...", file=sys.stderr)
    for i in range(n_anchors - 1):
        n_sub = int(round(ANCHOR_STEP_S / FINE_STEP_S))
        for k in range(n_sub):
            f = k / n_sub
            r_obs = anchor_r[i] * (1 - f) + anchor_r[i + 1] * f
            r_sun = anchor_sun[i] * (1 - f) + anchor_sun[i + 1] * f
            r_moon = anchor_moon[i] * (1 - f) + anchor_moon[i + 1] * f

            v_sun = r_sun - r_obs
            v_moon = r_moon - r_obs
            d_sun = np.linalg.norm(v_sun, axis=-1)
            d_moon = np.linalg.norm(v_moon, axis=-1)
            cos_theta = np.clip(np.einsum("ij,ij->i", v_sun, v_moon) / (d_sun * d_moon), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            a_sun = np.arcsin(np.clip(696000.0 / d_sun, -1.0, 1.0))
            a_moon = np.arcsin(np.clip(1737.4 / d_moon, -1.0, 1.0))

            frac = obscuration_fraction(theta, a_sun, a_moon)
            # Earth-occlusion check (same physical requirement as
            # classify_shadow's in_earth_daylight -- a point on the far
            # side of Earth from the Sun can't be "obscured", it's just
            # in ordinary orbital night with no direct sunlight to block)
            sun_dir = r_sun / np.linalg.norm(r_sun)
            along = r_obs @ sun_dir
            perp = r_obs - along[:, None] * sun_dir
            perp_dist = np.linalg.norm(perp, axis=-1)
            night_side_behind_earth = (along < 0) & (perp_dist < 6378.137)
            frac = np.where(night_side_behind_earth, 0.0, frac)

            dose += frac * FINE_STEP_S
            improved = frac > peak_obscuration
            if improved.any():
                t_now_unix = anchor_unix[i] + f * (anchor_unix[i + 1] - anchor_unix[i])
                peak_time_unix[improved] = t_now_unix
            peak_obscuration = np.maximum(peak_obscuration, frac)
        if i % 50 == 0:
            print(f"  anchor interval {i}/{n_anchors - 1}", file=sys.stderr)

    peak_time_utc = pd.to_datetime(peak_time_unix, unit="s", utc=True)

    out = pd.DataFrame({
        "lat_deg": flat_lat, "lon_deg": flat_lon,
        "dose_equiv_umbra_s": dose, "peak_obscuration_pct": peak_obscuration * 100.0,
        "peak_time_utc": peak_time_utc,
    })
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_CSV}", file=sys.stderr)

    touched = out[out["dose_equiv_umbra_s"] > 0]
    lines = []
    lines.append("Shadow dose field (step 4)")
    lines.append("=" * 40)
    lines.append(f"Altitude: {ALTITUDE_KM} km")
    lines.append(f"Grid: {GRID_LAT_STEP} deg x {GRID_LON_STEP} deg, lat [{GRID_LAT.min()},{GRID_LAT.max()}]")
    lines.append(f"Window: {WINDOW_START} to {WINDOW_END} UTC, anchors every {ANCHOR_STEP_S}s, "
                 f"dose accumulated every {FINE_STEP_S}s")
    lines.append(f"Cells with nonzero dose: {len(touched)}/{len(out)}")
    lines.append("")
    lines.append("Dose (equivalent umbra-seconds) distribution, cells with dose>0:")
    lines.append(touched["dose_equiv_umbra_s"].describe().to_string())
    lines.append("")
    lines.append("Peak instantaneous obscuration % distribution, cells with dose>0:")
    lines.append(touched["peak_obscuration_pct"].describe().to_string())

    with open(OUT_STATS_TXT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nStats: {OUT_STATS_TXT}")

    render_heatmap(out)


def render_heatmap(out: pd.DataFrame):
    """Capped + 'extend' colorbar (per the design discussion): the cap is
    set at the 99th percentile of nonzero dose so the bulk of the field's
    variation is visible, with an arrow on the colorbar showing values run
    higher (the exact numbers are in the CSV, not meant to be read off
    color here)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.path as mpath
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point

    grid = out.pivot(index="lat_deg", columns="lon_deg", values="dose_equiv_umbra_s")
    grid = grid.reindex(index=GRID_LAT, columns=GRID_LON, fill_value=0.0)
    grid_c, lon_c = add_cyclic_point(grid.values, coord=GRID_LON, axis=1)

    vmax = np.percentile(out.loc[out["dose_equiv_umbra_s"] > 0, "dose_equiv_umbra_s"], HEATMAP_CAP_PERCENTILE)

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

    mesh = ax.pcolormesh(lon_c, GRID_LAT, grid_c, transform=ccrs.PlateCarree(),
                          cmap="inferno", vmin=0, vmax=vmax, zorder=1)
    ax.coastlines(resolution="110m", linewidth=0.6, color="#888888", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#777777", zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#aaaaaa", linestyle="--", zorder=2)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, extend="max")
    cbar.set_label("Cumulative shadow dose (equivalent umbra-seconds)")

    ax.set_title(
        "Shadow dose field at satellite altitude, 2026-08-12 total solar eclipse\n"
        f"465.6 km altitude, {WINDOW_START[11:16]}-{WINDOW_END[11:16]} UTC, "
        f"color capped at {HEATMAP_CAP_PERCENTILE:.0f}th percentile ({vmax:.0f}s)",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Heatmap: {OUT_PNG}")


if __name__ == "__main__":
    main()
