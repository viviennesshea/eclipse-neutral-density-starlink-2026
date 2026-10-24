#!/usr/bin/env python3
"""
05_satellite_paths_on_dose_field.py
======================================
For every pass found in step 3 (03_satellite_shadow_passes.csv), compute:
  - pass_distance_km: real path length (sum of consecutive position
    deltas from actual ephemeris, not assumed constant orbital speed)
  - pass_dose_equiv_umbra_s: continuous dose along the satellite's OWN
    moving trajectory during its own pass (trapezoidal integral of the
    same obscuration_fraction() formula from step 4) -- deliberately NOT
    sampled from step 4's precomputed spatial field, since that field is
    an all-day fixed-point quantity and would misattribute dose from
    hours the satellite wasn't present. This is the satellite's own,
    separate integral, over its own pass window only.

Then renders every pass's real ground track on top of step 4's dose
heatmap, holding the decay/persistence timing question aside per the
user's request -- this is purely "does a longer path through a
higher-dose region look like it should produce more drag change,"
visually, before running the actual correlation test.

Read-only: reads <out-dir>/shadow/data/03_satellite_shadow_passes.csv and
04_shadow_dose_field.csv (already-derived), plus the pre-eclipse
snapshot's states.parquet (already-used-elsewhere ephemeris). Nothing in
starlink_data/ or experiment/ touched. No downloads.

Output: <out-dir>/shadow/data/05_satellite_pass_dose.csv,
<out-dir>/shadow/results/05_satellite_paths_heatmap.png
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
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "dose_heatmap", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_shadow_dose_heatmap.py")
)
dose_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dose_mod)
from eclipse_shadow import sun_moon_gcrs_km  # noqa: E402

PASSES_CSV = os.path.join(SP.SHADOW_DATA, "03_satellite_shadow_passes.csv")
DOSE_FIELD_CSV = os.path.join(SP.SHADOW_DATA, "04_shadow_dose_field.csv")
EPHEM_PARQUET = os.path.join(SP.snapshot("20260812T155855Z_2026-08-12_preeclipse"), "states.parquet")
OUT_CSV = os.path.join(SP.SHADOW_DATA, "05_satellite_pass_dose.csv")
OUT_PNG = os.path.join(SP.SHADOW_RESULTS, "05_satellite_paths_heatmap.png")

PAD_MIN = 2.0
R_EARTH_KM = 6378.137


def main():
    passes = pd.read_csv(PASSES_CSV, dtype={"norad": str},
                         parse_dates=["penumbra_enter_utc", "penumbra_exit_utc"])
    print(f"Loaded {len(passes)} passes", file=sys.stderr)

    win_start = (passes["penumbra_enter_utc"].min() - pd.Timedelta(minutes=PAD_MIN))
    win_end = (passes["penumbra_exit_utc"].max() + pd.Timedelta(minutes=PAD_MIN))
    eph = pd.read_parquet(EPHEM_PARQUET, columns=["norad", "epoch_utc", "x_km", "y_km", "z_km"])
    eph["epoch_utc"] = pd.to_datetime(eph["epoch_utc"], utc=True)
    eph["norad"] = eph["norad"].astype(str)
    eph = eph[(eph["epoch_utc"] >= win_start) & (eph["epoch_utc"] <= win_end)].copy()
    print(f"Ephemeris rows loaded: {len(eph)}", file=sys.stderr)

    # Batch Sun/Moon + geodetic conversion ONCE per unique timestamp across
    # the whole dataset (every pass's points are a subset of the same
    # handful of hundred native ephemeris timestamps) -- cheap regardless
    # of how many passes/satellites reference them.
    unique_times = sorted(eph["epoch_utc"].unique())
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, GCRS, ITRS, CartesianRepresentation
    import astropy.units as u

    sun_by_t, moon_by_t, geo_by_t = {}, {}, {}
    print(f"Batch Sun/Moon + geodetic for {len(unique_times)} unique timestamps...", file=sys.stderr)
    for i, t in enumerate(unique_times):
        t_astropy = Time(pd.Timestamp(t).to_pydatetime(), scale="utc")
        r_sun, r_moon = sun_moon_gcrs_km(t_astropy)
        sun_by_t[t] = r_sun
        moon_by_t[t] = r_moon
        rows = eph[eph["epoch_utc"] == t]
        xyz = rows[["x_km", "y_km", "z_km"]].values
        rep = CartesianRepresentation(xyz.T * u.km)
        gcrs = GCRS(rep, obstime=t_astropy)
        itrs = gcrs.transform_to(ITRS(obstime=t_astropy))
        geo = itrs.earth_location.geodetic
        geo_by_t[t] = dict(zip(rows["norad"], zip(geo.lat.deg, geo.lon.deg)))
        if i % 50 == 0:
            print(f"  {i}/{len(unique_times)}", file=sys.stderr)

    eph = eph.sort_values(["norad", "epoch_utc"])
    results = []
    for _, prow in passes.iterrows():
        t0 = prow["penumbra_enter_utc"] - pd.Timedelta(minutes=PAD_MIN)
        t1 = prow["penumbra_exit_utc"] + pd.Timedelta(minutes=PAD_MIN)
        sat_rows = eph[(eph["norad"] == prow["norad"]) & (eph["epoch_utc"] >= t0) & (eph["epoch_utc"] <= t1)]
        if len(sat_rows) < 2:
            continue

        xyz = sat_rows[["x_km", "y_km", "z_km"]].values
        times = sat_rows["epoch_utc"].tolist()
        r_sun = np.array([sun_by_t[t] for t in times])
        r_moon = np.array([moon_by_t[t] for t in times])

        v_sun = r_sun - xyz
        v_moon = r_moon - xyz
        d_sun = np.linalg.norm(v_sun, axis=-1)
        d_moon = np.linalg.norm(v_moon, axis=-1)
        cos_theta = np.clip(np.einsum("ij,ij->i", v_sun, v_moon) / (d_sun * d_moon), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        a_sun = np.arcsin(np.clip(696000.0 / d_sun, -1.0, 1.0))
        a_moon = np.arcsin(np.clip(1737.4 / d_moon, -1.0, 1.0))
        frac = dose_mod.obscuration_fraction(theta, a_sun, a_moon)

        t_sec = np.array([pd.Timestamp(t).value / 1e9 for t in times])
        dose = float(np.trapz(frac, t_sec))
        distance_km = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))

        lats, lons = [], []
        for t, norad in zip(times, sat_rows["norad"]):
            lat, lon = geo_by_t[t][norad]
            lats.append(lat)
            lons.append(lon)

        results.append({
            "norad": prow["norad"], "name": prow["name"], "group": prow["group"],
            "pass_number": prow["pass_number"], "pass_distance_km": distance_km,
            "pass_dose_equiv_umbra_s": dose, "n_points": len(times),
            "lats": ";".join(f"{v:.4f}" for v in lats), "lons": ";".join(f"{v:.4f}" for v in lons),
        })

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} passes with dose+distance+path -> {OUT_CSV}", file=sys.stderr)
    print(out[["pass_distance_km", "pass_dose_equiv_umbra_s"]].describe(), file=sys.stderr)

    render(out)


def render(out: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.path as mpath
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point

    field = pd.read_csv(DOSE_FIELD_CSV)
    grid = field.pivot(index="lat_deg", columns="lon_deg", values="dose_equiv_umbra_s")
    grid = grid.reindex(index=dose_mod.GRID_LAT, columns=dose_mod.GRID_LON, fill_value=0.0)
    grid_c, lon_c = add_cyclic_point(grid.values, coord=dose_mod.GRID_LON, axis=1)
    vmax = np.percentile(field.loc[field["dose_equiv_umbra_s"] > 0, "dose_equiv_umbra_s"],
                         dose_mod.HEATMAP_CAP_PERCENTILE)

    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + [0.5, 0.5])
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cfeature.OCEAN, facecolor="#0a0a12", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#141414", zorder=0)
    mesh = ax.pcolormesh(lon_c, dose_mod.GRID_LAT, grid_c, transform=ccrs.PlateCarree(),
                         cmap="inferno", vmin=0, vmax=vmax, zorder=1)
    ax.coastlines(resolution="110m", linewidth=0.5, color="#666666", zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#444444", linestyle="--", zorder=2)

    # linewidth/alpha scaled by pass dose -- higher-dose passes stand out
    # more, same accent color throughout so it reads against the inferno
    # background regardless of what's underneath.
    #
    # Two fixes vs. the first attempt: (1) 1836 overlapping lines at
    # moderate-to-high alpha washed the heatmap out almost completely --
    # thinned and dimmed substantially so the background stays visible
    # through the path overlay. (2) ~281 passes cross the +/-180 meridian;
    # plotting lon/lat directly across that jump drew a spurious straight
    # line clear across the map instead of wrapping around the back of the
    # globe. Fixed by breaking each path into segments at any >100 degree
    # single-step longitude jump (real satellite motion never jumps that
    # far in one ~1-minute ephemeris step; only the wrap discontinuity does).
    dose_vals = out["pass_dose_equiv_umbra_s"].clip(lower=0)
    rank = dose_vals.rank(pct=True)  # 0-1 percentile rank, robust to outliers
    for (_, row), pct in zip(out.iterrows(), rank):
        lats = np.array([float(v) for v in row["lats"].split(";")])
        lons = np.array([float(v) for v in row["lons"].split(";")])
        lw = 0.15 + 0.55 * pct
        alpha = 0.06 + 0.22 * pct

        jump_idx = np.where(np.abs(np.diff(lons)) > 100)[0]
        segments = np.split(np.arange(len(lons)), jump_idx + 1)
        for seg in segments:
            if len(seg) < 2:
                continue
            ax.plot(lons[seg], lats[seg], transform=ccrs.PlateCarree(), color="#4FD8E8",
                   linewidth=lw, alpha=alpha, zorder=3)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, extend="max")
    cbar.set_label("Volume dose field, background (equivalent umbra-seconds)")

    ax.set_title(
        "Satellite paths through the shadow dose field\n"
        f"{len(out)} passes; path brightness/thickness = pass dose percentile "
        "(cyan lines, background = dose field)",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
