#!/usr/bin/env python3
"""
starlink_ephem.py
=================
Download and parse publicly-released SpaceX Starlink ephemeris data so you can
analyze it.

Background
----------
SpaceX publishes predicted ephemerides for every Starlink satellite. As of
28 July 2025 these are no longer hosted on Space-Track's public-files page; the
live source is SpaceX's own mirror:

    Manifest (list of the newest file per satellite):
        https://api.starlink.com/public-files/ephemerides/MANIFEST.txt
    Individual files:
        https://api.starlink.com/public-files/ephemerides/<filename>
    Format/README:
        https://starlink.com/public-files/ephemerides/README.md

Each file is in the "Modified ITC" / JSpOC-format ephemeris report:
  * a short text header,
  * repeating blocks of one state-vector line followed by (usually) three
    covariance lines.

The state line is:
    <epoch>  X  Y  Z  Vx  Vy  Vz
where positions are km and velocities km/s in the MEME (Mean Equator, Mean
Equinox) J2000 frame, and <epoch> is packed as  YYDDDHHMMSS.sss
(2-digit year, day-of-year, hour, minute, seconds). The covariance (when
present) is the 21-element lower triangle of the 6x6 pos/vel covariance,
expressed in the UVW/RTN frame.

Filenames look like:
    MEME_44714_STARLINK-1008_1981230_Operational_1373891460_UNCLASSIFIED.txt
    ^frame ^norad ^name       ^epoch  ^status      ^created    ^classification

Usage examples
--------------
    # Download 5 satellites' newest ephemerides into ./starlink_data and parse them
    python starlink_ephem.py --limit 5

    # Only satellites whose name matches a pattern
    python starlink_ephem.py --name STARLINK-3     # 3xxx series

    # Specific NORAD catalog IDs
    python starlink_ephem.py --norad 44714 51804

    # Everything (thousands of files — be patient / kind to the server)
    python starlink_ephem.py --all

    # See what date range is available without downloading full files
    python starlink_ephem.py --dates --limit 20
    python starlink_ephem.py --dates --norad 44714 51804

Outputs, per satellite, go to the output dir:
    raw/<original>.txt         the untouched downloaded file
    parsed/<norad>_<name>.parquet   tidy table (falls back to .csv)
And a combined  states.parquet/.csv  across all downloaded satellites.

This script only uses the Python standard library to fetch data. pandas /
pyarrow are optional and only used to write nicer output files.
"""

from __future__ import annotations

import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
import io
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, Optional

BASE = "https://api.starlink.com/public-files/ephemerides"
MANIFEST_URL = f"{BASE}/MANIFEST.txt"
USER_AGENT = "starlink-ephem-analyzer/1.0 (+personal research)"

# Filename:  MEME_<norad>_<name>_<epoch>_<status>_<created>_<classification>.txt
FNAME_RE = re.compile(
    r"^MEME_(?P<norad>\d+)_(?P<name>[^_]+)_(?P<epoch>\d+)_"
    r"(?P<status>[^_]+)_(?P<created>\d+)_(?P<classification>[^.]+)\.txt$"
)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SatelliteFile:
    filename: str
    norad: str
    name: str
    status: str

    @property
    def url(self) -> str:
        return f"{BASE}/{self.filename}"

    @classmethod
    def parse(cls, filename: str) -> Optional["SatelliteFile"]:
        m = FNAME_RE.match(filename.strip())
        if not m:
            return None
        return cls(
            filename=filename.strip(),
            norad=m.group("norad"),
            name=m.group("name"),
            status=m.group("status"),
        )


# --------------------------------------------------------------------------- #
# Networking
# --------------------------------------------------------------------------- #
def _http_get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as e:  # pragma: no cover
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to GET {url}: {last_err}")


def _http_get_range(url: str, nbytes: int = 700, retries: int = 3, timeout: int = 30) -> str:
    """Fetch only the first `nbytes` of a URL (the header), via HTTP Range.

    Ephemeris files start with a small text header stating the exact
    ephemeris_start/ephemeris_stop coverage, so we don't need the whole file
    (which can be megabytes) just to know what dates are available.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Range": f"bytes=0-{nbytes - 1}"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:  # pragma: no cover
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to GET {url}: {last_err}")


def fetch_manifest() -> list[SatelliteFile]:
    raw = _http_get(MANIFEST_URL).decode("utf-8", errors="replace")
    files: list[SatelliteFile] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        sf = SatelliteFile.parse(line)
        if sf:
            files.append(sf)
    return files


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _decode_epoch(token: str) -> dt.datetime:
    """Decode a packed epoch into a UTC datetime.

    SpaceX/JSpOC ephemeris epochs are fixed-width fields, easiest read from the
    right:  ...<year> DDD HH MM SS.sss

    Real files use a 4-digit year; some sample/legacy files use a 2-digit year.
    Parsing DDD/HH/MM/SS off the right end and treating the remaining leading
    digits as the year handles both cases.

        '2025207120000.000' -> 2025-07-26 12:00:00 UTC   (day 207 of 2025)
        '25145000000.000'   -> 2025-05-25 00:00:00 UTC   (day 145 of 2025)
    """
    whole, _, frac = token.partition(".")
    if len(whole) < 9:
        raise ValueError(f"epoch token too short to decode: {token!r}")
    ss = int(whole[-2:])
    mm = int(whole[-4:-2])
    hh = int(whole[-6:-4])
    doy = int(whole[-9:-6])
    year_part = whole[:-9]
    year = int(year_part) + (2000 if len(year_part) <= 2 else 0)
    micro = int(round(float("0." + frac) * 1_000_000)) if frac else 0
    # Build with timedelta so any field roll-over (e.g. a leap second ss=60)
    # normalizes cleanly instead of raising a range error.
    return dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss, microseconds=micro
    )


HEADER_DATES_RE = re.compile(
    r"created:(?P<created>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC.*?"
    r"ephemeris_start:(?P<start>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\s+"
    r"ephemeris_stop:(?P<stop>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC",
    re.S,
)


@dataclasses.dataclass
class DateCoverage:
    norad: str
    name: str
    status: str
    created: Optional[str] = None
    start: Optional[str] = None
    stop: Optional[str] = None
    error: Optional[str] = None


def fetch_date_coverage(sf: SatelliteFile) -> DateCoverage:
    """Peek at a file's header (partial download) to see what dates it covers."""
    try:
        head = _http_get_range(sf.url)
    except Exception as e:  # pragma: no cover - network dependent
        return DateCoverage(sf.norad, sf.name, sf.status, error=str(e))
    m = HEADER_DATES_RE.search(head)
    if not m:
        return DateCoverage(sf.norad, sf.name, sf.status, error="couldn't find date header in response")
    return DateCoverage(
        sf.norad, sf.name, sf.status,
        created=m.group("created"), start=m.group("start"), stop=m.group("stop"),
    )


@dataclasses.dataclass
class StateVector:
    epoch: dt.datetime
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    covariance: list[float] = dataclasses.field(default_factory=list)  # up to 21 terms


@dataclasses.dataclass
class Ephemeris:
    norad: str
    name: str
    header: list[str]
    states: list[StateVector]


def _looks_like_state(tokens: list[str]) -> bool:
    """A state line has 7 numeric tokens whose first is a packed epoch (large)."""
    if len(tokens) != 7:
        return False
    try:
        vals = [float(t) for t in tokens]
    except ValueError:
        return False
    # Packed epoch YYDDD... is always >= ~1e9 for years 2020+.
    return vals[0] >= 1e9


def parse_ephemeris(text: str, norad: str = "", name: str = "") -> Ephemeris:
    header: list[str] = []
    states: list[StateVector] = []
    current: Optional[StateVector] = None
    in_header = True

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()

        if _looks_like_state(tokens):
            in_header = False
            vals = [float(t) for t in tokens]
            current = StateVector(
                epoch=_decode_epoch(tokens[0]),
                x=vals[1], y=vals[2], z=vals[3],
                vx=vals[4], vy=vals[5], vz=vals[6],
            )
            states.append(current)
            continue

        if in_header:
            header.append(line)
            continue

        # Not a state line, past the header -> treat as covariance floats that
        # belong to the most recent state vector.
        if current is not None:
            try:
                nums = [float(t) for t in tokens]
            except ValueError:
                # Some files repeat a textual header block between records.
                continue
            current.covariance.extend(nums)

    return Ephemeris(norad=norad, name=name, header=header, states=states)


# --------------------------------------------------------------------------- #
# Light analysis helpers (no third-party deps required)
# --------------------------------------------------------------------------- #
EARTH_RADIUS_KM = 6378.137  # WGS-84 equatorial radius


def geocentric_altitude_km(sv: StateVector) -> float:
    """Rough altitude = distance from Earth's center minus equatorial radius.

    Note: MEME J2000 is Earth-centered inertial, so this is a spherical-Earth
    altitude approximation (fine for a quick sanity plot, not for precision work
    where you'd convert to ECEF and use an ellipsoid + geodetic latitude).
    """
    r = (sv.x ** 2 + sv.y ** 2 + sv.z ** 2) ** 0.5
    return r - EARTH_RADIUS_KM


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _try_import_pandas():
    try:
        import pandas as pd  # noqa: F401
        return pd
    except Exception:
        return None


def ephemeris_to_rows(eph: Ephemeris) -> list[dict]:
    rows = []
    for sv in eph.states:
        rows.append(
            {
                "norad": eph.norad,
                "name": eph.name,
                "epoch_utc": sv.epoch.isoformat(),
                "x_km": sv.x, "y_km": sv.y, "z_km": sv.z,
                "vx_kms": sv.vx, "vy_kms": sv.vy, "vz_kms": sv.vz,
                "altitude_km": geocentric_altitude_km(sv),
                "n_cov_terms": len(sv.covariance),
            }
        )
    return rows


def write_parsed(eph: Ephemeris, out_dir: Path) -> Path:
    parsed_dir = out_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{eph.norad}_{eph.name}"
    rows = ephemeris_to_rows(eph)
    pd = _try_import_pandas()
    if pd is not None:
        df = pd.DataFrame(rows)
        target = parsed_dir / f"{stem}.parquet"
        try:
            df.to_parquet(target, index=False)
            return target
        except Exception:
            target = parsed_dir / f"{stem}.csv"
            df.to_csv(target, index=False)
            return target
    # stdlib CSV fallback
    import csv
    target = parsed_dir / f"{stem}.csv"
    with target.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return target


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def select_files(
    manifest: list[SatelliteFile],
    name: Optional[str],
    norad: Optional[list[str]],
    status: Optional[str],
    limit: Optional[int],
    take_all: bool,
) -> list[SatelliteFile]:
    sel = manifest
    if status:
        sel = [s for s in sel if s.status.lower() == status.lower()]
    if norad:
        wanted = set(norad)
        sel = [s for s in sel if s.norad in wanted]
    if name:
        pat = re.compile(re.escape(name), re.IGNORECASE)
        sel = [s for s in sel if pat.search(s.name)]
    # --limit exists to cap open-ended browsing (--name/--status scans of the
    # whole manifest). An explicit --norad list is already a bounded,
    # deliberate selection, so the default --limit=5 must not silently
    # truncate it — that would drop satellites without any error or warning.
    if not take_all and not norad and limit:
        sel = sel[:limit]
    return sel


def download_and_parse(sf: SatelliteFile, out_dir: Path) -> tuple[SatelliteFile, Optional[Ephemeris], Optional[str]]:
    try:
        raw = _http_get(sf.url).decode("utf-8", errors="replace")
    except Exception as e:  # pragma: no cover - network dependent
        return sf, None, str(e)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / sf.filename).write_text(raw, encoding="utf-8")
    eph = parse_ephemeris(raw, norad=sf.norad, name=sf.name)
    return sf, eph, None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download & parse Starlink ephemeris data.")
    p.add_argument("-o", "--out", default=None, help="output directory (default: <data-dir>/snapshots/<UTC timestamp>)")
    p.add_argument("--name", help="substring/pattern to match satellite name, e.g. STARLINK-3")
    p.add_argument("--norad", nargs="*", help="one or more NORAD catalog IDs")
    p.add_argument("--status", help="filter by status, e.g. Operational")
    p.add_argument("--limit", type=int, default=5, help="max satellites (ignored with --all or --norad)")
    p.add_argument("--all", action="store_true", help="download every satellite in the manifest")
    p.add_argument("--workers", type=int, default=8, help="parallel download workers")
    p.add_argument("--list-only", action="store_true", help="just print the manifest selection and exit")
    p.add_argument(
        "--dates", action="store_true",
        help="show the date range each selected satellite's ephemeris covers, "
             "via lightweight partial requests (no full download)",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(SP.SNAP_DIR) / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching manifest: {MANIFEST_URL}", file=sys.stderr)
    manifest = fetch_manifest()
    print(f"Manifest lists {len(manifest)} satellites.", file=sys.stderr)

    selection = select_files(
        manifest, name=args.name, norad=args.norad,
        status=args.status, limit=args.limit, take_all=args.all,
    )
    print(f"Selected {len(selection)} satellite(s).", file=sys.stderr)

    if args.list_only:
        for s in selection:
            print(f"{s.norad}\t{s.name}\t{s.status}\t{s.filename}")
        return 0

    if args.dates:
        if not selection:
            print("Nothing to check with those filters.", file=sys.stderr)
            return 1
        print(f"Fetching header dates for {len(selection)} satellite(s) (partial requests only)...", file=sys.stderr)
        coverages: list[DateCoverage] = []
        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(fetch_date_coverage, s) for s in selection]
            for fut in cf.as_completed(futures):
                coverages.append(fut.result())
        coverages.sort(key=lambda c: c.name)

        ok = [c for c in coverages if c.error is None]
        failed = [c for c in coverages if c.error is not None]

        print(f"{'NORAD':<10}{'NAME':<20}{'STATUS':<14}{'CREATED (UTC)':<21}{'START (UTC)':<21}{'STOP (UTC)':<21}")
        for c in ok:
            print(f"{c.norad:<10}{c.name:<20}{c.status:<14}{c.created:<21}{c.start:<21}{c.stop:<21}")

        if ok:
            earliest_start = min(c.start for c in ok)
            latest_stop = max(c.stop for c in ok)
            print(f"\n{len(ok)} satellite(s) checked: available data spans {earliest_start} -> {latest_stop} UTC")

        if failed:
            print(f"\n{len(failed)} failed to check:", file=sys.stderr)
            for c in failed:
                print(f"  {c.norad} {c.name}: {c.error}", file=sys.stderr)

        return 0

    if not selection:
        print("Nothing to download with those filters.", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(download_and_parse, s, out_dir): s for s in selection}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            sf, eph, err = fut.result()
            if err or eph is None:
                failures.append((sf.filename, err or "parse returned None"))
                print(f"[{i}/{len(selection)}] FAILED {sf.name}: {err}", file=sys.stderr)
                continue
            target = write_parsed(eph, out_dir)
            all_rows.extend(ephemeris_to_rows(eph))
            print(
                f"[{i}/{len(selection)}] {sf.name}: {len(eph.states)} states "
                f"-> {target.name}",
                file=sys.stderr,
            )

    # Combined table across everything we pulled
    if all_rows:
        pd = _try_import_pandas()
        if pd is not None:
            df = pd.DataFrame(all_rows)
            combined = out_dir / "states.parquet"
            try:
                df.to_parquet(combined, index=False)
            except Exception:
                combined = out_dir / "states.csv"
                df.to_csv(combined, index=False)
        else:
            import csv
            combined = out_dir / "states.csv"
            with combined.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                w.writeheader()
                w.writerows(all_rows)
        print(f"\nWrote combined table: {combined}  ({len(all_rows)} rows)", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} file(s) failed:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
