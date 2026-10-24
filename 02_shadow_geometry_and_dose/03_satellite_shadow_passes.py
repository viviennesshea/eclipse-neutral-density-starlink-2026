#!/usr/bin/env python3
"""
03_satellite_shadow_passes.py
================================
For every satellite in our target list, find every time it PHYSICALLY
enters and exits the Moon's penumbra/umbra -- i.e. its own real 3D
trajectory tested directly against the shadow cone at each moment, using
the same validated geometry as step 1/2 (`classify_shadow`). This is
different from the persistence-dose "residual" bookkeeping built in the
earlier `experiment/` pipeline (which asked "did this satellite fly over
ground that WAS RECENTLY eclipsed" for a lag/decay effect) -- this script
only asks "was the satellite itself, right now, inside the shadow cone."
The point is to get an honest list of direct physical passes (there can
be more than one, since a satellite orbits every ~94 min and the shadow
itself is only in any one place for a limited window) as raw input for
studying the "shadow's gone but the air hasn't recovered yet" question
separately, rather than baking a decay assumption into this step.

Data source: real SpaceX-predicted ephemeris (GCRS/J2000 positions,
~1-min native cadence) -- NOT SGP4/TLE. This is the same ephemeris file
used throughout the `experiment/` pipeline, confirmed to already cover
the needed window natively (no propagation, no new downloads). Ephemeris
was chosen over TLE/SGP4 here because it covers more of the target list
(1250/1257 vs ~1045/1257 for the classic TLE file) and is SpaceX's own
higher-precision predicted trajectory rather than a re-propagated one.

Window: 2026-08-12 13:30 UTC (a little before the earliest possible
shadow contact anywhere, ~15:00 UTC) through 22:00 UTC (90 minutes past
the latest possible contact anywhere, ~20:15 UTC) -- confirmed empirically
in step 1. Long enough to catch a satellite re-entering the shadow's
(by-then-moved) location on a later orbit (~94-min period), which is
exactly the "goes around and goes through again" case, which is
captured as a separate row rather than a wider table.

The aliasing trap this deliberately avoids: the Moon's umbra is only
~100-300km wide, so a satellite crosses it in as little as ~3-70 seconds
(established in the earlier pipeline). Classifying shadow status directly
on the native ~1-minute ephemeris samples can EASILY miss a real umbra
crossing entirely (no sample happens to land inside the narrow window) --
this exact bug was found and fixed once already in the earlier
`experiment/` pipeline by switching to 1-second-resolution sampling.
Penumbra passes last many minutes to tens of minutes, so 1-min sampling
is fine for finding THOSE -- but not for pinning down a nested umbra
sub-interval within one.

Method, two passes:
  1. COARSE: classify shadow status for every satellite at the native
     ~1-min ephemeris cadence across the whole window. This reliably
     finds every PENUMBRA-OR-STRONGER contiguous "pass" (long enough to
     not need finer sampling to detect in the first place).
  2. FINE REFINEMENT: for each detected coarse pass (with a few minutes
     of padding on each side), linearly interpolate the satellite's real
     ephemeris position between its surrounding true samples at 1-second
     resolution, and re-classify at that resolution. This is what
     actually pins down precise penumbra enter/exit and finds/pins down
     any nested umbra sub-interval, without re-sampling the ENTIRE
     8.5-hour window at 1-second resolution for every satellite (which
     would be needlessly expensive -- the fine pass only touches the
     small fraction of each satellite's time that's already known, from
     the coarse pass, to contain shadow contact).
  Sun/Moon positions at each fine timestamp are cached globally (keyed by
  rounded UTC second) since many satellites' passes overlap in absolute
  time and the Sun/Moon position doesn't depend on which satellite is
  being tested.

Output schema: one row per PASS (not per satellite) -- a satellite with
two separate passes gets two rows, exactly as requested ("just do
another line entry to keep the columns manageable"), rather than
widening the table with pass-2 columns.

Read-only: reads starlink_data/eclipse_targets.csv and the pre-eclipse
snapshot's states.parquet (both already-downloaded, already-used-elsewhere
files). Nothing in starlink_data/ modified. No downloads.

Output: <out-dir>/shadow/data/03_satellite_shadow_passes.csv,
<out-dir>/shadow/results/03_satellite_shadow_passes_stats.txt
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
from eclipse_shadow import classify_shadow, sun_moon_gcrs_km, UMBRA, PENUMBRA, ANTUMBRA, NONE  # noqa: E402

TARGETS_CSV = SP.TARGETS_CSV
EPHEM_PARQUET = os.path.join(SP.snapshot("20260812T155855Z_2026-08-12_preeclipse"), "states.parquet")
OUT_CSV = os.path.join(SP.SHADOW_DATA, "03_satellite_shadow_passes.csv")
OUT_STATS_TXT = os.path.join(SP.SHADOW_RESULTS, "03_satellite_shadow_passes_stats.txt")

WINDOW_START = "2026-08-12T13:30:00"
WINDOW_END = "2026-08-12T22:00:00"
FINE_STEP_S = 1.0     # matches the earlier pipeline's established accuracy floor for umbra timing
PAD_MIN = 3.0         # minutes of padding around each coarse pass before fine-refining it

# Sun/Moon position barely changes over the span of one coarse (~1-min)
# ephemeris step -- far too slowly to matter for pinning down a shadow
# boundary to the second. So the coarse pass's Sun/Moon evaluations (one
# per unique ~1-min timestamp, already needed for coarse classification)
# are cached here, keyed by epoch-seconds, and REUSED for fine refinement
# via linear interpolation -- no new astropy Sun/Moon calls happen during
# fine refinement at all. This is what keeps the fine 1-second refinement
# step affordable: astropy's per-call overhead (~20-200ms) would make
# calling it individually at 1-second resolution across every satellite's
# padded pass window prohibitively slow (easily an hour or more); the
# actual position that changes meaningfully at 1-second resolution is the
# SATELLITE's, not the Sun/Moon's, and that's what's freshly interpolated.
_coarse_t_sec: np.ndarray | None = None
_coarse_r_sun: np.ndarray | None = None
_coarse_r_moon: np.ndarray | None = None


def sun_moon_interp(fine_t_sec: np.ndarray):
    r_sun = np.column_stack([np.interp(fine_t_sec, _coarse_t_sec, _coarse_r_sun[:, i]) for i in range(3)])
    r_moon = np.column_stack([np.interp(fine_t_sec, _coarse_t_sec, _coarse_r_moon[:, i]) for i in range(3)])
    return r_sun, r_moon


def coarse_classify(df: pd.DataFrame) -> pd.DataFrame:
    """df: norad, epoch_utc, x_km, y_km, z_km, one row per (satellite, coarse time).
    Returns the same df with a 'status' column, classified in batches per
    unique timestamp (Sun/Moon computed once per timestamp, not per row).
    Also populates the module-level Sun/Moon cache used by fine refinement."""
    global _coarse_t_sec, _coarse_r_sun, _coarse_r_moon
    from astropy.time import Time

    status = np.empty(len(df), dtype=object)
    groups = df.groupby("epoch_utc").indices  # {timestamp: positional index array}
    times = sorted(groups.keys())
    print(f"Coarse pass: {len(times)} unique timestamps, {len(df)} total rows", file=sys.stderr)

    t_sec_list, r_sun_list, r_moon_list = [], [], []
    for i, t in enumerate(times):
        idx = groups[t]
        r_obs = df.iloc[idx][["x_km", "y_km", "z_km"]].values
        ts = pd.Timestamp(t)
        t_astropy = Time(ts.to_pydatetime(), scale="utc")
        r_sun, r_moon = sun_moon_gcrs_km(t_astropy)
        n = r_obs.shape[0]
        st, *_ = classify_shadow(r_obs, np.tile(r_sun, (n, 1)), np.tile(r_moon, (n, 1)))
        status[idx] = st
        t_sec_list.append(ts.value / 1e9)
        r_sun_list.append(r_sun)
        r_moon_list.append(r_moon)
        if i % 100 == 0:
            print(f"  coarse step {i}/{len(times)}", file=sys.stderr)

    _coarse_t_sec = np.array(t_sec_list)
    _coarse_r_sun = np.array(r_sun_list)
    _coarse_r_moon = np.array(r_moon_list)

    df = df.copy()
    df["status"] = status
    return df


def find_coarse_passes(sat_df: pd.DataFrame) -> list[tuple]:
    """sat_df: one satellite's rows, sorted by epoch_utc, with 'status' column.
    Returns list of (pass_start_time, pass_end_time) for each maximal
    contiguous run where status != NONE."""
    exposed = (sat_df["status"] != NONE).values
    times = sat_df["epoch_utc"].tolist()  # keep tz-aware Timestamp objects (.values strips tz)
    passes = []
    in_run = False
    run_start = None
    for i, e in enumerate(exposed):
        if e and not in_run:
            in_run = True
            run_start = times[i]
        elif not e and in_run:
            in_run = False
            passes.append((run_start, times[i - 1]))
    if in_run:
        passes.append((run_start, times[-1]))
    return passes


def refine_pass(sat_rows: pd.DataFrame, pass_start, pass_end) -> dict:
    """Linear-interpolate this satellite's real ephemeris positions at
    FINE_STEP_S resolution between pass_start-PAD and pass_end+PAD, and
    re-classify. Returns enter/exit times for penumbra-or-stronger and,
    if reached, umbra."""
    pad = pd.Timedelta(minutes=PAD_MIN)
    t0 = pd.Timestamp(pass_start) - pad
    t1 = pd.Timestamp(pass_end) + pad

    bracket = sat_rows[(sat_rows["epoch_utc"] >= t0) & (sat_rows["epoch_utc"] <= t1)]
    if len(bracket) < 2:
        bracket = sat_rows  # pass near the edge of our window; use whatever we have
    bt = bracket["epoch_utc"].astype("int64").values / 1e9  # seconds since epoch
    bx, by, bz = bracket["x_km"].values, bracket["y_km"].values, bracket["z_km"].values

    fine_t = np.arange(bt.min(), bt.max(), FINE_STEP_S)
    fx = np.interp(fine_t, bt, bx)
    fy = np.interp(fine_t, bt, by)
    fz = np.interp(fine_t, bt, bz)
    r_obs = np.column_stack([fx, fy, fz])

    r_sun_arr, r_moon_arr = sun_moon_interp(fine_t)

    status, *_ = classify_shadow(r_obs, r_sun_arr, r_moon_arr)

    exposed_mask = status != NONE
    umbra_mask = status == UMBRA
    if not exposed_mask.any():
        return None  # fine refinement found nothing (shouldn't normally happen)

    exposed_times = fine_t[exposed_mask]
    pen_enter = pd.Timestamp(exposed_times.min(), unit="s", tz="UTC")
    pen_exit = pd.Timestamp(exposed_times.max(), unit="s", tz="UTC")

    if umbra_mask.any():
        umbra_idx = np.where(umbra_mask)[0]
        # A pass can dip into full umbra MORE THAN ONCE (confirmed real,
        # not a bug, via independent SGP4 verification -- see
        # ANALYSIS_LOG.md). Sum each contiguous run's own duration rather
        # than taking (last_umbra_sample - first_umbra_sample), which
        # would silently count any penumbra-only gap BETWEEN two separate
        # umbra dips as if it were continuous totality.
        breaks = np.where(np.diff(umbra_idx) > 1)[0]
        run_starts = np.concatenate(([umbra_idx[0]], umbra_idx[breaks + 1]))
        run_ends = np.concatenate((umbra_idx[breaks], [umbra_idx[-1]]))
        n_umbra_intervals = len(run_starts)
        umbra_duration_s = float(np.sum(fine_t[run_ends] - fine_t[run_starts]) + n_umbra_intervals * FINE_STEP_S)
        umb_enter = pd.Timestamp(fine_t[run_starts[0]], unit="s", tz="UTC")
        umb_exit = pd.Timestamp(fine_t[run_ends[-1]], unit="s", tz="UTC")
    else:
        umb_enter = umb_exit = None
        n_umbra_intervals = 0
        umbra_duration_s = 0.0

    return {
        "penumbra_enter_utc": pen_enter, "penumbra_exit_utc": pen_exit,
        "umbra_enter_utc": umb_enter, "umbra_exit_utc": umb_exit,
        "penumbra_duration_s": (pen_exit - pen_enter).total_seconds(),
        "umbra_duration_s": umbra_duration_s,
        "n_umbra_intervals": n_umbra_intervals,
    }


def main():
    targets = pd.read_csv(TARGETS_CSV, dtype={"norad": str})
    print(f"Target list: {len(targets)} satellites", file=sys.stderr)

    eph = pd.read_parquet(EPHEM_PARQUET, columns=["norad", "epoch_utc", "x_km", "y_km", "z_km"])
    eph["epoch_utc"] = pd.to_datetime(eph["epoch_utc"], utc=True)
    eph["norad"] = eph["norad"].astype(str)
    win_start = pd.Timestamp(WINDOW_START, tz="UTC")
    win_end = pd.Timestamp(WINDOW_END, tz="UTC")
    eph = eph[(eph["epoch_utc"] >= win_start) & (eph["epoch_utc"] <= win_end)].copy()
    eph = eph.merge(targets[["norad", "name", "group"]], on="norad", how="inner")
    print(f"Ephemeris rows in window: {len(eph)}, satellites with coverage: {eph['norad'].nunique()}/"
          f"{len(targets)}", file=sys.stderr)

    eph = coarse_classify(eph)
    eph = eph.sort_values(["norad", "epoch_utc"])

    rows = []
    n_sats_with_passes = 0
    for norad, sat_rows in eph.groupby("norad", sort=False):
        coarse_passes = find_coarse_passes(sat_rows)
        if not coarse_passes:
            continue
        n_sats_with_passes += 1
        name = sat_rows["name"].iloc[0]
        group = sat_rows["group"].iloc[0]
        for pass_num, (p_start, p_end) in enumerate(coarse_passes, start=1):
            refined = refine_pass(sat_rows, p_start, p_end)
            if refined is None:
                continue
            rows.append({
                "norad": norad, "name": name, "group": group, "pass_number": pass_num,
                **refined,
            })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} pass-rows for {n_sats_with_passes} satellites -> {OUT_CSV}", file=sys.stderr)

    lines = []
    lines.append("Satellite shadow passes (step 3)")
    lines.append("=" * 40)
    lines.append(f"Window: {WINDOW_START} to {WINDOW_END} UTC")
    lines.append(f"Coarse cadence: native ephemeris (~1-min); fine refinement: {FINE_STEP_S}s "
                 f"within +/-{PAD_MIN}min of each coarse pass")
    lines.append(f"Satellites in target list: {len(targets)}")
    lines.append(f"Satellites with ephemeris coverage in window: {eph['norad'].nunique()}")
    lines.append(f"Satellites with at least one shadow pass: {n_sats_with_passes}")
    lines.append(f"Total pass-rows: {len(out)}")
    if len(out):
        lines.append(f"Satellites with 2+ passes: {(out.groupby('norad').size() >= 2).sum()}")
        lines.append(f"Passes reaching umbra: {(out['umbra_duration_s'] > 0).sum()}")
        lines.append("")
        lines.append("By group:")
        lines.append(out.groupby("group").agg(
            n_passes=("norad", "count"),
            n_umbra_passes=("umbra_duration_s", lambda s: (s > 0).sum()),
            median_penumbra_s=("penumbra_duration_s", "median"),
            median_umbra_s=("umbra_duration_s", lambda s: s[s > 0].median() if (s > 0).any() else 0),
        ).to_string())

    with open(OUT_STATS_TXT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nStats: {OUT_STATS_TXT}")


if __name__ == "__main__":
    main()
