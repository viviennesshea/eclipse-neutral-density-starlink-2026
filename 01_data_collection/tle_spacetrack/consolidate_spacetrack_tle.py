#!/usr/bin/env python3
"""
consolidate_spacetrack_tle.py
================================
Turns the raw Space-Track batches (raw/batch_*.json) into tidy tables and a report.

Outputs (output/):
  spacetrack_all_rows.csv        every element set downloaded, parsed
  spacetrack_distinct_fits.csv   one row per (satellite, epoch); the earliest CREATION_DATE is kept and duplicates counted
  spacetrack_satellite_summary.csv  per satellite: fits before / during / after the eclipse, update gaps
  spacetrack_report.txt          counts, cadence, the eclipse-window check, and a comparison with the CelesTrak snapshots

Buckets use the FIT EPOCH (same rule as ../tle): before < 16:30 UTC on 2026-08-12 <= during <= 19:00 < after.
Read-only on raw/, starlink_data/ and ../tle/output/.
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
import glob
import json
import os

import numpy as np
import pandas as pd

PEAK = pd.Timestamp("2026-08-12T17:47:06Z")
BEFORE_END, AFTER_START = pd.Timestamp("2026-08-12T16:30:00Z"), pd.Timestamp("2026-08-12T19:00:00Z")
lines = []


def log(s=""):
    lines.append(s)
    print(s)


def bucket(ep):
    return "before" if ep < BEFORE_END else ("after" if ep > AFTER_START else "during")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=SP.ST_RAW)
    ap.add_argument("--table-dir", dest="out_dir", default=SP.ST_OUT)
    ap.add_argument("--celestrak-fits", default=os.path.join(SP.TLE_OUT, "tle_distinct_fits.csv"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    recs = []
    files = sorted(glob.glob(os.path.join(a.raw_dir, "batch_*.json")))
    for fn in files:
        data = json.load(open(fn))
        recs.extend(data if isinstance(data, list) else [])
    if not recs:
        raise SystemExit(f"No records found in {a.raw_dir}. Run download_spacetrack_tle.py first.")
    df = pd.DataFrame(recs)
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce")
    out = pd.DataFrame({
        "norad": df["NORAD_CAT_ID"].astype(str), "name": df.get("OBJECT_NAME"), "intl_des": df.get("OBJECT_ID"),
        "epoch_utc": pd.to_datetime(df["EPOCH"], utc=True, format="ISO8601"),
        "creation_date": pd.to_datetime(df.get("CREATION_DATE"), utc=True, format="ISO8601", errors="coerce"),
        "originator": df.get("ORIGINATOR"), "bstar": num("BSTAR"), "mean_motion": num("MEAN_MOTION"),
        "eccentricity": num("ECCENTRICITY"), "inclination_deg": num("INCLINATION"),
        "element_set_no": num("ELEMENT_SET_NO"), "rev_at_epoch": num("REV_AT_EPOCH"), "gp_id": df.get("GP_ID"),
    })
    targets = pd.read_csv(SP.TARGETS_CSV, dtype={"norad": str})
    prov = targets[targets["norad"].astype(int) >= 100000][["norad", "name", "group"]].rename(columns={"norad": "provisional_id"})
    norm = targets[targets["norad"].astype(int) < 100000][["norad", "group"]]
    out = out.merge(norm, on="norad", how="left").merge(prov, on="name", how="left", suffixes=("", "_p"))
    out["group"] = out["group"].fillna(out.get("group_p")); out = out.drop(columns=[c for c in ["group_p"] if c in out])
    out["sat_key"] = np.where(out["provisional_id"].notna(), out["provisional_id"], out["norad"])
    out["epoch_bucket"] = out["epoch_utc"].map(bucket)
    out["hours_from_peak"] = (out["epoch_utc"] - PEAK).dt.total_seconds() / 3600
    out = out.sort_values(["sat_key", "epoch_utc", "creation_date"]).reset_index(drop=True)
    out.to_csv(os.path.join(a.out_dir, "spacetrack_all_rows.csv"), index=False)

    log("Space-Track gp_history: consolidation")
    log("=" * 60)
    log(f"batches read: {len(files)} | element sets: {len(out)} | satellites returned: {out['sat_key'].nunique()} of {len(targets)} targets")
    got = set(out["sat_key"]); miss = targets[~targets["norad"].isin(got)]
    log(f"targets with no element sets in this window: {len(miss)} ({miss['group'].value_counts().to_dict()})")
    log(f"originator(s): {out['originator'].value_counts(dropna=False).to_dict()}")
    log(f"element-set numbers: {out['element_set_no'].nunique()} distinct values (CelesTrak files had exactly 1: 999)")
    log(f"epoch range: {out['epoch_utc'].min():%Y-%m-%d %H:%M} to {out['epoch_utc'].max():%Y-%m-%d %H:%M}")

    dup = out.groupby(["sat_key", "epoch_utc"]).size()
    fits = out.drop_duplicates(["sat_key", "epoch_utc"], keep="first").copy()
    fits["n_rows_for_this_epoch"] = fits.set_index(["sat_key", "epoch_utc"]).index.map(dup)
    fits.to_csv(os.path.join(a.out_dir, "spacetrack_distinct_fits.csv"), index=False)
    log(f"distinct (satellite, epoch) fits: {len(fits)} | epochs published more than once: {int((dup > 1).sum())}")
    log(f"fits per satellite: median {fits.groupby('sat_key').size().median():.0f}, "
        f"range {fits.groupby('sat_key').size().min()}-{fits.groupby('sat_key').size().max()}")
    log(f"fits by bucket: {fits['epoch_bucket'].value_counts().to_dict()}")
    fits = fits.sort_values(["sat_key", "epoch_utc"])
    fits["gap_h"] = fits.groupby("sat_key")["epoch_utc"].diff().dt.total_seconds() / 3600
    g = fits["gap_h"].dropna()
    log(f"gap between consecutive fits (h): median {g.median():.1f}, 25th {g.quantile(.25):.1f}, 75th {g.quantile(.75):.1f}, 90th {g.quantile(.9):.1f}, max {g.max():.1f}")
    e12 = fits[(fits["epoch_utc"] >= pd.Timestamp("2026-08-12T00:00:00Z")) & (fits["epoch_utc"] < pd.Timestamp("2026-08-13T00:00:00Z"))]
    log(f"fits made on 2026-08-12 by UTC hour: {e12['epoch_utc'].dt.hour.value_counts().sort_index().to_dict()}")
    log(f"fits made DURING the shadow crossing (16:30-19:00 UTC): {int((fits['epoch_bucket'] == 'during').sum())}")

    piv = fits.pivot_table(index="sat_key", columns="epoch_bucket", values="bstar", aggfunc=["count", "median"])
    piv.columns = [f"{x}_{y}" for x, y in piv.columns]
    for c in ["count_before", "count_during", "count_after", "median_before", "median_after"]:
        if c not in piv:
            piv[c] = np.nan
    summ = piv.rename(columns={"count_before": "n_fits_before", "count_during": "n_fits_during", "count_after": "n_fits_after",
                               "median_before": "bstar_median_before", "median_after": "bstar_median_after"})
    summ["bstar_change_distinct"] = summ["bstar_median_after"] - summ["bstar_median_before"]
    summ = summ.join(fits.groupby("sat_key").agg(name=("name", "first"), group=("group", "first"), n_fits=("epoch_utc", "count"),
                                                 median_gap_h=("gap_h", "median"))).reset_index()
    summ.to_csv(os.path.join(a.out_dir, "spacetrack_satellite_summary.csv"), index=False)
    ok1 = ((summ["n_fits_before"] >= 1) & (summ["n_fits_after"] >= 1)).sum()
    ok2 = ((summ["n_fits_before"] >= 2) & (summ["n_fits_after"] >= 2)).sum()
    log(f"satellites with >=1 fit before AND after: {ok1} | >=2 each side: {ok2}")

    if os.path.isfile(a.celestrak_fits):
        c = pd.read_csv(a.celestrak_fits, dtype={"norad": str})
        c["e"] = pd.to_datetime(c["tle_epoch_utc"], utc=True, format="ISO8601")
        c["sat_key"] = c["norad"]
        st = fits[["sat_key", "epoch_utc", "bstar"]].sort_values("epoch_utc")
        m = pd.merge_asof(c.sort_values("e"), st.rename(columns={"epoch_utc": "st_epoch", "bstar": "st_bstar"}),
                          left_on="e", right_on="st_epoch", by="sat_key", direction="nearest", tolerance=pd.Timedelta(minutes=1))
        hit = m["st_epoch"].notna()
        log("")
        log(f"COMPARISON with the CelesTrak fits we already have (../tle): {len(c)} fits")
        log(f"  found in Space-Track with the same epoch (within 1 minute): {int(hit.sum())} ({100 * hit.mean():.1f}%)")
        if hit.any():
            same = np.isclose(m.loc[hit, "bstar"], m.loc[hit, "st_bstar"], rtol=1e-3)
            log(f"  of those, BSTAR agrees to 0.1%: {int(same.sum())} ({100 * same.mean():.1f}%)")
            inwin = m["e"] >= fits["epoch_utc"].min()   # m (not c): merge_asof returns a re-indexed, re-sorted frame
            hit_w = hit & inwin
            log(f"  restricted to CelesTrak fits inside the Space-Track window: {int(inwin.sum())} fits, found {int(hit_w.sum())} ({100 * hit_w.sum() / max(int(inwin.sum()), 1):.1f}%)")
            log("  -> high match with identical BSTAR means both feeds carry the same fits; a low match would mean a different product")
    with open(os.path.join(a.out_dir, "spacetrack_report.txt"), "w") as f:
        f.write("\n".join(lines))
    log(f"\nWrote tables and report to {a.out_dir}")


if __name__ == "__main__":
    main()
