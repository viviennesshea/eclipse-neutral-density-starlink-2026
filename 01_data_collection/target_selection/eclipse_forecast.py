#!/usr/bin/env python3
"""
eclipse_forecast.py
====================
Forecast which Starlink satellites pass through the Moon's shadow (umbra /
antumbra / penumbra) during the 2026-08-12 total solar eclipse.

Why TLEs instead of the downloaded MEME ephemeris files
---------------------------------------------------------
starlink_ephem.py's published ephemeris files are a rolling ~3-day window
import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
from generation time and, as of this run, don't yet reach the eclipse's
peak (2026-08-12 17:47:06 UTC, per published circumstances). TLEs +
SGP4 are explicitly designed for multi-day propagation, so this script
instead:

  1. pulls the CURRENT public Starlink TLE set from CelesTrak (free, no
     login — unlike Space-Track's historical gp_history used by
     get_tle_history.py),
  2. propagates every satellite with SGP4 across a time grid spanning the
     eclipse,
  3. rotates each TEME-frame result into GCRS (same frame family as the Sun/
     Moon vectors from eclipse_shadow.py) using one rotation matrix per
     timestep (not per satellite — keeps this fast at ~10k satellites),
  4. classifies each (satellite, time) sample against the Moon's shadow cone.

Caveat on accuracy: SGP4 propagation error a few days out from epoch is
typically on the order of single-digit km for LEO — small compared to the
~300 km-wide umbral path, but not zero. This is a forecast, re-run it closer
to 2026-08-12 as fresh TLEs/ephemerides publish for a tighter answer.

Usage
-----
    python3 eclipse_forecast.py                      # full run, default window
    python3 eclipse_forecast.py --start 17:00 --end 18:35 --step 30
    python3 eclipse_forecast.py --limit 500           # quick test on a subset
    python3 eclipse_forecast.py --tle-cache my.tle    # reuse a saved TLE file
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

from eclipse_shadow import classify_shadow, sun_moon_gcrs_km, UMBRA, ANTUMBRA, PENUMBRA, NONE

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
USER_AGENT = "starlink-eclipse-forecast/1.0 (+personal research)"
ECLIPSE_DATE = "2026-08-12"
GREATEST_ECLIPSE_UTC = "2026-08-12T17:47:06"


def fetch_tles(cache_path: Path | None) -> list[tuple[str, str, str]]:
    """Return list of (name, line1, line2). Uses cache_path if given & exists."""
    if cache_path and cache_path.exists():
        text = cache_path.read_text()
    else:
        req = urllib.request.Request(CELESTRAK_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        if cache_path:
            cache_path.write_text(text)

    lines = [l.rstrip("\n") for l in text.splitlines() if l.strip()]
    triples = []
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            triples.append((name.strip(), l1, l2))
    return triples


def build_time_grid(date: str, start_hm: str, end_hm: str, step_s: int):
    from astropy.time import Time, TimeDelta

    t0 = Time(f"{date}T{start_hm}:00", scale="utc")
    t1 = Time(f"{date}T{end_hm}:00", scale="utc")
    n = int((t1 - t0).sec // step_s) + 1
    times = t0 + TimeDelta(np.arange(n) * step_s, format="sec")
    return times


def teme_to_gcrs_matrix(time) -> np.ndarray:
    """3x3 rotation matrix R such that r_gcrs = R @ r_teme, at a single astropy Time."""
    from astropy.coordinates import TEME, GCRS, CartesianRepresentation
    import astropy.units as u

    basis = CartesianRepresentation(np.eye(3) * u.km)
    teme = TEME(basis, obstime=time)
    gcrs = teme.transform_to(GCRS(obstime=time))
    return gcrs.cartesian.xyz.to("km").value  # columns are transformed basis vectors


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Forecast Starlink satellites in the Moon's shadow.")
    p.add_argument("--date", default=ECLIPSE_DATE)
    p.add_argument("--start", default="16:30", help="UTC HH:MM window start")
    p.add_argument("--end", default="19:00", help="UTC HH:MM window end")
    p.add_argument("--step", type=int, default=60, help="time step, seconds")
    p.add_argument("--limit", type=int, default=None, help="only propagate first N satellites (testing)")
    p.add_argument("--tle-cache", type=Path, default=Path(SP.DATA_DIR) / "starlink_latest.tle")
    p.add_argument("--out", type=Path, default=Path(SP.DATA_DIR) / "eclipse_transits.csv")
    args = p.parse_args(argv)

    from sgp4.api import Satrec, SatrecArray, jday

    print(f"Fetching current Starlink TLEs...", file=sys.stderr)
    triples = fetch_tles(args.tle_cache)
    if args.limit:
        triples = triples[: args.limit]
    print(f"{len(triples)} satellites loaded.", file=sys.stderr)

    names, norads, satrecs = [], [], []
    for name, l1, l2 in triples:
        sat = Satrec.twoline2rv(l1, l2)
        satrecs.append(sat)
        names.append(name)
        norads.append(str(sat.satnum))

    arr = SatrecArray(satrecs)

    times = build_time_grid(args.date, args.start, args.end, args.step)
    print(f"Time grid: {len(times)} steps, {args.start}-{args.end} UTC on {args.date}", file=sys.stderr)

    jd = np.array([t.jd1 for t in times])
    fr = np.array([t.jd2 for t in times])
    err, r_teme, v_teme = arr.sgp4(jd, fr)  # shapes: (nsat, ntime), (nsat, ntime, 3), (nsat, ntime, 3)

    n_sat, n_time = err.shape
    status = np.full((n_sat, n_time), NONE, dtype=object)

    print("Rotating TEME->GCRS and classifying shadow status per timestep...", file=sys.stderr)
    for ti in range(n_time):
        t = times[ti]
        R = teme_to_gcrs_matrix(t)
        r_gcrs = r_teme[:, ti, :] @ R.T  # (nsat,3)

        r_sun, r_moon = sun_moon_gcrs_km(t)
        ok = err[:, ti] == 0
        st, *_ = classify_shadow(
            r_gcrs[ok], np.tile(r_sun, (ok.sum(), 1)), np.tile(r_moon, (ok.sum(), 1))
        )
        status[ok, ti] = st

    # Summarize per satellite: any umbra/antumbra/penumbra samples?
    rows = []
    for si in range(n_sat):
        col = status[si]
        for cat in (UMBRA, ANTUMBRA, PENUMBRA):
            idx = np.where(col == cat)[0]
            if idx.size == 0:
                continue
            rows.append(
                {
                    "norad": norads[si],
                    "name": names[si],
                    "category": cat,
                    "first_utc": times[idx[0]].iso,
                    "last_utc": times[idx[-1]].iso,
                    "n_samples": int(idx.size),
                    "approx_duration_s": int(idx.size * args.step),
                }
            )

    rows.sort(key=lambda r: ({"umbra": 0, "antumbra": 1, "penumbra": 2}[r["category"]], r["first_utc"]))

    n_umbra = len({r["norad"] for r in rows if r["category"] == UMBRA})
    n_antumbra = len({r["norad"] for r in rows if r["category"] == ANTUMBRA})
    n_penumbra = len({r["norad"] for r in rows if r["category"] == PENUMBRA})
    print(
        f"\nSatellites with samples in: umbra={n_umbra} antumbra={n_antumbra} penumbra={n_penumbra} "
        f"(out of {n_sat} propagated)",
        file=sys.stderr,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.out.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
