#!/usr/bin/env python3
"""
download_spacetrack_tle.py
=============================
Fresh, bulk download of historical element sets (Space-Track `gp_history`) for the eclipse targets,
covering days before, during and after the 2026-08-12 eclipse.

Why: our CelesTrak snapshots only show whichever fit was current at each of 11 downloads, and the feed stalled for
hours (three downloads were identical). Space-Track keeps every published set, so this shows the real update timing
and fills the gaps. NOTE it is a different product from the CelesTrak sets, so treat it as a separate dataset.

Design (all deliberate):
  * Credentials: env vars SPACETRACK_USER / SPACETRACK_PASS, or a file ~/.config/spacetrack/credentials with two lines
        SPACETRACK_USER=you@example.com
        SPACETRACK_PASS=your-password
    The file must not be readable by others (chmod 600). The password is never printed or logged.
  * Polite: batches of IDs per request, a pause between requests, back-off on errors, a re-login if the session lapses.
  * Resumable: each batch is saved to raw/ and skipped on re-run. A manifest records every request.
  * Satellites with 5-digit IDs are queried by NORAD ID. The 205 newly launched satellites carry provisional 6-digit IDs
    that Space-Track does not use, so they are queried by name (STARLINK-nnnnn); the real IDs come back in the data.
  * `--dry-run` prints the plan and makes no network calls. `--limit-batches 1` is a cheap first test.

Read-only on starlink_data/. Output goes to <out-dir>/tle_spacetrack/.
"""

from __future__ import annotations

import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import http.cookiejar
import json
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

TARGETS_CSV = SP.TARGETS_CSV
BASE = "https://www.space-track.org"
CRED_FILE = os.path.expanduser("~/.config/spacetrack/credentials")


def load_credentials():
    user, pw = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASS")
    if user and pw:
        return user, pw
    if os.path.isfile(CRED_FILE):
        mode = os.stat(CRED_FILE).st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise SystemExit(f"{CRED_FILE} is readable by other users. Run: chmod 600 {CRED_FILE}")
        vals = {}
        for line in open(CRED_FILE):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                vals[k.strip()] = v.strip().strip("'\"")
        if vals.get("SPACETRACK_USER") and vals.get("SPACETRACK_PASS"):
            return vals["SPACETRACK_USER"], vals["SPACETRACK_PASS"]
    raise SystemExit("No Space-Track credentials found. Set SPACETRACK_USER / SPACETRACK_PASS, or create "
                     f"{CRED_FILE} (see the top of this script). Nothing was downloaded.")


class Session:
    def __init__(self, user, pw):
        self.user, self.pw = user, pw
        self.login()

    def login(self):
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.opener.addheaders = [("User-Agent", "student-research-tle-fetcher/1.0 (eclipse study)")]
        data = urllib.parse.urlencode({"identity": self.user, "password": self.pw}).encode()
        try:
            with self.opener.open(f"{BASE}/ajaxauth/login", data, timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Space-Track login failed (HTTP {e.code}).")
        if "failed" in body.lower():
            raise SystemExit("Space-Track login rejected. Check the username and password.")

    def get(self, url, retries=4):
        wait = 30
        for attempt in range(retries + 1):
            try:
                with self.opener.open(url, timeout=180) as r:
                    return r.status, r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt < retries:
                    self.login()
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    print(f"   HTTP {e.code}; waiting {wait}s then retrying", flush=True)
                    time.sleep(wait)
                    wait = min(wait * 2, 300)
                    continue
                raise SystemExit(f"Query failed (HTTP {e.code}). Stopping; re-run to resume.")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < retries:
                    print(f"   network error ({e}); waiting {wait}s then retrying", flush=True)
                    time.sleep(wait)
                    wait = min(wait * 2, 300)
                    continue
                raise SystemExit(f"Network failure: {e}. Stopping; re-run to resume.")


def load_targets():
    rows = list(csv.DictReader(open(TARGETS_CSV, newline="")))
    by_id = [r for r in rows if int(r["norad"]) < 100000]
    by_name = [r for r in rows if int(r["norad"]) >= 100000]
    return by_id, by_name


def make_batches(by_id, by_name, size):
    batches = []
    ids = sorted(int(r["norad"]) for r in by_id)
    for i in range(0, len(ids), size):
        batches.append({"kind": "norad", "keys": [str(x) for x in ids[i:i + size]]})
    names = sorted({r["name"].strip() for r in by_name})
    for i in range(0, len(names), size):
        batches.append({"kind": "name", "keys": names[i:i + size]})
    for n, b in enumerate(batches):
        b["index"] = n
    return batches


def build_url(batch, start, end):
    field = "NORAD_CAT_ID" if batch["kind"] == "norad" else "OBJECT_NAME"
    keys = ",".join(urllib.parse.quote(k, safe="") for k in batch["keys"])
    return (f"{BASE}/basicspacedata/query/class/gp_history/{field}/{keys}/EPOCH/{start}--{end}"
            f"/orderby/NORAD_CAT_ID%20asc,EPOCH%20asc/format/json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-08-10", help="first epoch date (UTC), default 2026-08-10")
    ap.add_argument("--end", default="2026-08-15", help="end epoch date, EXCLUSIVE (UTC); default 2026-08-15 = through the end of Aug 14")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=10.0, help="seconds to pause between requests (default 10)")
    ap.add_argument("--limit-batches", type=int, default=None, help="only fetch the first N batches (a cheap test)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; no network, no login")
    ap.add_argument("--raw-dir", "--out-dir-raw", dest="out_dir", default=SP.ST_RAW, help="where the batches are saved (default: <out-dir>/tle_spacetrack/raw)")
    a = ap.parse_args()

    by_id, by_name = load_targets()
    batches = make_batches(by_id, by_name, a.batch_size)
    todo = batches[: a.limit_batches] if a.limit_batches else batches
    print(f"Targets: {len(by_id)} with normal IDs (queried by NORAD ID) + {len(by_name)} newly launched with provisional IDs (queried by name)")
    print(f"Window: {a.start} to {a.end} (epochs, UTC) | batches: {len(batches)} total, {len(todo)} selected, {a.batch_size} per request")
    print(f"Estimated time at {a.sleep:.0f}s between requests: about {len(todo) * (a.sleep + 5) / 60:.0f} minutes")
    if a.dry_run:
        for b in todo[:3] + todo[-2:]:
            print(f"  batch {b['index']:>3} [{b['kind']}] {len(b['keys'])} keys, e.g. {b['keys'][:3]}")
        print("  first URL:", build_url(todo[0], a.start, a.end)[:200] + " ...")
        print("Dry run only. No login, no download.")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    manifest_path = os.path.join(a.out_dir, "manifest.jsonl")
    user, pw = load_credentials()
    sess = Session(user, pw)
    print("Logged in to Space-Track (credentials not shown).", flush=True)
    done = 0
    for b in todo:
        fn = os.path.join(a.out_dir, f"batch_{b['index']:04d}_{b['kind']}.json")
        if os.path.exists(fn):
            print(f" batch {b['index']:>3}: already downloaded, skipping")
            continue
        url = build_url(b, a.start, a.end)
        status, body = sess.get(url)
        try:
            recs = json.loads(body) if body.strip() else []
        except json.JSONDecodeError:
            raise SystemExit(f"Batch {b['index']}: response was not JSON (first 200 chars): {body[:200]!r}")
        if isinstance(recs, dict):
            raise SystemExit(f"Batch {b['index']}: Space-Track returned a message instead of data: {str(recs)[:300]}")
        with open(fn, "w") as f:
            f.write(body)
        with open(manifest_path, "a") as m:
            m.write(json.dumps({"batch": b["index"], "kind": b["kind"], "n_keys": len(b["keys"]), "keys": b["keys"],
                                "start": a.start, "end": a.end, "http_status": status, "n_records": len(recs),
                                "sha256": hashlib.sha256(body.encode()).hexdigest(), "file": os.path.basename(fn),
                                "fetched_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        done += 1
        print(f" batch {b['index']:>3} [{b['kind']}]: {len(recs):>6} element sets for {len(b['keys'])} keys", flush=True)
        time.sleep(a.sleep)
    print(f"Done: {done} new batch(es) saved in {a.out_dir}. Next: python3 consolidate_spacetrack_tle.py")


if __name__ == "__main__":
    main()
