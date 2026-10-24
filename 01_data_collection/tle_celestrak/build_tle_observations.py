#!/usr/bin/env python3
"""
build_tle_observations.py
===========================
Build a de-duplicated but honest record of the BSTAR data.

The problem this addresses: a TLE is a fit made at some EPOCH. CelesTrak keeps serving the same
fit until a new one is made, so 11 downloads contain many repeated copies of the same fit. Treating
each download as a separate measurement (as the earlier panel table does) counts a stale fit again
and again. Some satellites were refreshed more often than others, and that differs by group.

What "honest" means here:
  * Nothing is dropped silently. Every (satellite, download) sighting is kept in tle_all_sightings.csv,
    with a flag saying whether it is the first time that fit was seen or a repeat.
  * The unit of de-duplication is the fit itself: (satellite, TLE epoch). Not the BSTAR value (two different
    fits can share a value) and not the panel's is_new_tle flag (NaN for almost half of the rows).
  * Before/after is decided by when the FIT was made (its epoch), not by when it was downloaded. A fit made
    before the eclipse but downloaded afterwards is still a "before" fit.
  * A fit's BSTAR reflects the fitting period BEFORE its epoch (length not recorded in the TLE), so a fit
    made just after the eclipse still contains mostly pre-eclipse data. hours_from_peak is provided so the
    analysis can require fits to be well after the eclipse.
  * Every count is reconciled against the earlier panel table; disagreements are reported, not hidden.

Sources mirror experiment/src/01_build_panel_table.py exactly (eclipse_targets_*.tle first, then OMM JSON
for satellites not found there), so the row counts can be compared like-for-like.

Read-only: The data folder (and optional legacy files) are only read. All output goes to <out-dir>/tle/output/.
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

import csv
import glob
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import stats

SNAP_DIR = SP.SNAP_DIR
TARGETS_CSV = SP.TARGETS_CSV
PANEL_CSV = os.path.join(SP.DATA_DIR, "legacy", "01_panel_table.csv")  # optional: earlier-version file, used only for the reconciliation report
ROBUST_CSV = os.path.join(SP.DATA_DIR, "legacy", "03b_dose_response_robust_data.csv")  # optional, same
OUT = SP.TLE_OUT

MU = 398600.4418
PEAK = pd.Timestamp("2026-08-12T17:47:06Z")            # greatest eclipse
BEFORE_END = pd.Timestamp("2026-08-12T16:30:00Z")      # shadow reaches satellites ~16:37 UTC
AFTER_START = pd.Timestamp("2026-08-12T19:00:00Z")     # last shadow crossings ~18:38 UTC

lines_out = []


def log(s=""):
    lines_out.append(s)
    print(s)


# ---------------------------------------------------------------- parsing
def assumed_decimal(s):
    s = s.strip()
    if not s:
        return 0.0
    sign = -1.0 if s[0] == "-" else 1.0
    return sign * float("0." + s.lstrip("+-"))


def parse_bstar(field):
    m = re.match(r"([+-]?)(\d*)([+-]\d)$", field.strip())
    if not m or not m.group(2):
        return 0.0
    sign, mant, exp = m.groups()
    return assumed_decimal(sign + mant) * 10 ** int(exp)


def sma_km(mm):
    n = mm * 2 * math.pi / 86400.0
    return (MU / n ** 2) ** (1.0 / 3.0)


def tle_epoch(l1):
    yy = int(l1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(l1[20:32])
    return pd.Timestamp(datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0))


def parse_tle_file(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    i = 0
    while i + 2 < len(lines):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            i += 1
            continue
        mm = float(l2[52:63])
        yield dict(norad=l1[2:7].strip(), tle_name=name.strip(), epoch=tle_epoch(l1), bstar=parse_bstar(l1[53:61]),
                   mean_motion=mm, eccentricity=assumed_decimal(l2[26:33]), inclination_deg=float(l2[8:16]),
                   sma_km=sma_km(mm))
        i += 3


def parse_omm(path):
    for r in json.load(open(path)):
        mm = float(r.get("MEAN_MOTION", 0.0))
        ep = pd.Timestamp(r["EPOCH"])
        ep = ep.tz_localize("UTC") if ep.tzinfo is None else ep.tz_convert("UTC")
        yield dict(norad=str(r["NORAD_CAT_ID"]), tle_name=r.get("OBJECT_NAME", ""), epoch=ep, bstar=float(r.get("BSTAR", 0.0)),
                   mean_motion=mm, eccentricity=float(r.get("ECCENTRICITY", 0.0)),
                   inclination_deg=float(r.get("INCLINATION", np.nan)), sma_km=sma_km(mm) if mm else np.nan)


def file_stamp(path):
    m = re.search(r"(\d{8}T\d{6}Z)", os.path.basename(path))
    return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)) if m else pd.NaT


# ------------------------------------------------------------------ build
def collect():
    targets = pd.read_csv(TARGETS_CSV, dtype={"norad": str}).set_index("norad")
    rows, per_snap = [], []
    for d in sorted(glob.glob(os.path.join(SNAP_DIR, "*"))):
        tle_dir = os.path.join(d, "tle")
        label = os.path.basename(d)
        tles = sorted(glob.glob(os.path.join(tle_dir, "eclipse_targets_*.tle")))
        if not os.path.isdir(tle_dir) or not tles:
            continue
        omms = sorted(glob.glob(os.path.join(tle_dir, "*omm*.json")))
        seen, n_tle, n_omm = set(), 0, 0
        for rec in parse_tle_file(tles[0]):
            if rec["norad"] in targets.index:
                seen.add(rec["norad"]); n_tle += 1
                rows.append({**rec, "snapshot_label": label, "source": "tle", "source_file": os.path.basename(tles[0]),
                             "download_utc_filestamp": file_stamp(tles[0])})
        for f in omms:
            for rec in parse_omm(f):
                if rec["norad"] in targets.index and rec["norad"] not in seen:
                    seen.add(rec["norad"]); n_omm += 1
                    rows.append({**rec, "snapshot_label": label, "source": "omm_json", "source_file": os.path.basename(f),
                                 "download_utc_filestamp": file_stamp(f)})
        per_snap.append((label, n_tle, n_omm))
    df = pd.DataFrame(rows)
    df = df.join(targets[["name", "group", "inclination"]].rename(columns={"inclination": "shell_inclination"}), on="norad")
    return df, per_snap


def bucket(ep):
    return "before" if ep < BEFORE_END else ("after" if ep > AFTER_START else "during")


def main():
    os.makedirs(OUT, exist_ok=True)
    log("De-duplicated, honest TLE/BSTAR record (tle)")
    log("=" * 64)
    df, per_snap = collect()
    log("Sightings parsed per snapshot (same sources as the earlier panel table):")
    for label, a, b in per_snap:
        log(f"  {label[:40]:<40} tle={a:>5}  omm_json={b:>4}")
    # The file-name stamp is not always the true download time. Trust it only if it is within 1 h of the
    # snapshot's own label time AND no fit in that snapshot has an epoch after the stamp (impossible).
    df["label_time"] = pd.to_datetime(df["snapshot_label"].str[:15], format="%Y%m%dT%H%M%S", utc=True)
    log("")
    log("DOWNLOAD TIMES (file-name stamp vs snapshot label time)")
    used, src = {}, {}
    for label, g in df.groupby("snapshot_label"):
        stamp, lab = g["download_utc_filestamp"].iloc[0], g["label_time"].iloc[0]
        n_future = int((g["epoch"] > stamp).sum())
        ok = abs((stamp - lab).total_seconds()) <= 3600 and n_future == 0
        used[label], src[label] = (stamp if ok else lab), ("file_stamp" if ok else "snapshot_label_time")
        if not ok:
            log(f"  {label[:40]}: stamp {stamp:%m-%d %H:%M} vs label {lab:%m-%d %H:%M}; {n_future} of {len(g)} fits have epochs AFTER the stamp"
                f" -> file stamp judged unreliable, using label time")
    if all(v == "file_stamp" for v in src.values()):
        log("  all file stamps consistent with snapshot labels")
    df["download_utc_used"] = df["snapshot_label"].map(used)
    df["download_time_source"] = df["snapshot_label"].map(src)
    df["epoch_key"] = df["epoch"].dt.round("ms")
    df = df.sort_values(["norad", "snapshot_label"]).reset_index(drop=True)
    df["fit_id"] = df["norad"] + "_" + df["epoch_key"].dt.strftime("%Y%m%dT%H%M%S.%f").str[:-3]
    df["sighting_number_of_fit"] = df.groupby("fit_id").cumcount() + 1
    df["first_sighting"] = df["sighting_number_of_fit"] == 1
    df["epoch_age_at_download_h"] = (df["download_utc_used"] - df["epoch"]).dt.total_seconds() / 3600
    df["hours_from_peak"] = (df["epoch"] - PEAK).dt.total_seconds() / 3600
    df["epoch_bucket"] = df["epoch"].map(bucket)
    df["bstar_is_zero"] = df["bstar"] == 0

    panel = (pd.read_csv(PANEL_CSV, dtype={"norad": str})[["norad", "snapshot_label", "bstar", "is_new_tle"]] if os.path.exists(PANEL_CSV)
             else pd.DataFrame({"norad": pd.Series(dtype=str), "snapshot_label": pd.Series(dtype=str), "bstar": pd.Series(dtype=float), "is_new_tle": pd.Series(dtype=float)}))
    panel = panel.rename(columns={"bstar": "panel_bstar", "is_new_tle": "panel_is_new_tle"})
    df = df.merge(panel, on=["norad", "snapshot_label"], how="left")

    # ---------------- reconciliation
    log("")
    log("RECONCILIATION with the earlier panel table (experiment/data/01_panel_table.csv)")
    log(f"  sightings here: {len(df)} | rows in panel: {len(panel)} | in both: {int(df['panel_bstar'].notna().sum())}")
    both = df.dropna(subset=["panel_bstar"])
    same = np.isclose(both["bstar"], both["panel_bstar"], rtol=1e-9, atol=0)
    log(f"  BSTAR identical for {int(same.sum())} of {len(both)} shared rows ({100 * same.mean():.2f}%); differing rows: {int((~same).sum())}")
    log(f"  sightings with BSTAR exactly 0 (missing or true zero): {int(df['bstar_is_zero'].sum())}")
    fut = df[df["epoch"] > df["download_utc_used"]]
    log(f"  fits whose epoch is AFTER the download time used (should be 0): {len(fut)}")

    fits = (df.groupby("fit_id")
              .agg(norad=("norad", "first"), name=("name", "first"), group=("group", "first"),
                   shell_inclination=("shell_inclination", "first"), tle_epoch_utc=("epoch", "first"),
                   bstar=("bstar", "first"), bstar_last_seen=("bstar", "last"), mean_motion=("mean_motion", "first"), eccentricity=("eccentricity", "first"),
                   inclination_deg=("inclination_deg", "first"), sma_km=("sma_km", "first"), source=("source", "first"),
                   first_snapshot=("snapshot_label", "first"), last_snapshot=("snapshot_label", "last"),
                   n_snapshots_carrying_it=("snapshot_label", "nunique"), first_download_utc=("download_utc_used", "min"),
                   n_distinct_bstar_in_fit=("bstar", "nunique"), n_distinct_mm_in_fit=("mean_motion", "nunique"))
              .reset_index())
    fits["content_changed_between_copies"] = (fits["n_distinct_bstar_in_fit"] > 1) | (fits["n_distinct_mm_in_fit"] > 1)
    fits = fits.rename(columns={"bstar": "bstar_first_seen"})
    fits["bstar"] = fits["bstar_first_seen"]          # the value used in the summary is the first one seen
    fits["age_when_first_seen_h"] = (fits["first_download_utc"] - fits["tle_epoch_utc"]).dt.total_seconds() / 3600
    fits["hours_from_peak"] = (fits["tle_epoch_utc"] - PEAK).dt.total_seconds() / 3600
    fits["epoch_bucket"] = fits["tle_epoch_utc"].map(bucket)
    fits = fits.sort_values(["norad", "tle_epoch_utc"]).reset_index(drop=True)

    log("")
    log("DE-DUPLICATION (unit = one fit = satellite + TLE epoch)")
    log(f"  sightings: {len(df)} -> distinct fits: {len(fits)}  (repeat sightings: {len(df) - len(fits)}, {100 * (len(df) - len(fits)) / len(df):.1f}%)")
    conflict = fits[(fits["n_distinct_bstar_in_fit"] > 1) | (fits["n_distinct_mm_in_fit"] > 1)]
    log(f"  fits whose content changed between copies (same epoch, different BSTAR or mean motion): {len(conflict)} "
        f"(flagged content_changed_between_copies; the first-seen BSTAR is used, last-seen is kept in bstar_last_seen)")
    if len(conflict):
        dv = (conflict["bstar_last_seen"] - conflict["bstar_first_seen"]).abs() / conflict["bstar_first_seen"].abs()
        log(f"     relative BSTAR change within those fits: median {100 * dv.median():.1f}%, max {100 * dv.max():.1f}%")
    log(f"  fits carried by N downloads: {fits['n_snapshots_carrying_it'].value_counts().sort_index().to_dict()}")
    x = pd.crosstab(df["panel_is_new_tle"].astype(object).where(df["panel_is_new_tle"].notna(), "NaN"),
                    df["first_sighting"].map({True: "first sighting of fit", False: "repeat of an earlier fit"}))
    log("  earlier panel flag is_new_tle vs epoch-based truth (rows):")
    for line in x.to_string().split("\n"):
        log("     " + line)

    # ---------------- per-satellite outcome, by FIT epoch
    log("")
    log(f"BUCKETS BY FIT EPOCH: before < {BEFORE_END:%m-%d %H:%M} <= during <= {AFTER_START:%H:%M} < after (UTC, 2026-08-12)")
    log(f"  distinct fits by bucket: {fits['epoch_bucket'].value_counts().to_dict()}")
    e12 = fits[(fits["tle_epoch_utc"] >= pd.Timestamp("2026-08-12T00:00:00Z")) & (fits["tle_epoch_utc"] < pd.Timestamp("2026-08-13T00:00:00Z"))]
    log(f"  new fits on 2026-08-12 by UTC hour: {e12['tle_epoch_utc'].dt.hour.value_counts().sort_index().to_dict()}")
    gap = fits[(fits["tle_epoch_utc"] > pd.Timestamp("2026-08-12T15:00:00Z")) & (fits["tle_epoch_utc"] < pd.Timestamp("2026-08-12T21:00:00Z"))]["tle_epoch_utc"].sort_values()
    if len(gap):
        dgap = gap.diff().dt.total_seconds().div(3600)
        log(f"  longest gap with no new fit in 15:00-21:00: {dgap.max():.1f} h, ending {gap[dgap.idxmax()]:%H:%M}; "
            f"the 'during' bucket is empty because no fit was made while the shadow was crossing the constellation")
    piv = fits.pivot_table(index="norad", columns="epoch_bucket", values="bstar", aggfunc=["count", "median"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    for c in ["count_before", "count_after", "count_during", "median_before", "median_after"]:
        if c not in piv:
            piv[c] = np.nan
    summ = piv[["count_before", "count_during", "count_after", "median_before", "median_after"]].rename(columns={
        "count_before": "n_fits_before", "count_during": "n_fits_during", "count_after": "n_fits_after",
        "median_before": "bstar_median_before", "median_after": "bstar_median_after"})
    for c in ["n_fits_before", "n_fits_during", "n_fits_after"]:
        summ[c] = summ[c].fillna(0).astype(int)
    summ["bstar_change_distinct"] = summ["bstar_median_after"] - summ["bstar_median_before"]
    meta = df.groupby("norad").agg(name=("name", "first"), group=("group", "first"), shell_inclination=("shell_inclination", "first"),
                                   n_sightings=("snapshot_label", "count")).join(fits.groupby("norad").size().rename("n_distinct_fits"))
    latest_before = (fits[fits.epoch_bucket == "before"].sort_values("tle_epoch_utc").groupby("norad")
                     .agg(latest_before_epoch=("tle_epoch_utc", "last")))
    first_after = (fits[fits.epoch_bucket == "after"].sort_values("tle_epoch_utc").groupby("norad")
                   .agg(first_after_hours_from_peak=("hours_from_peak", "first")))
    summ = meta.join(summ).join(latest_before).join(first_after).reset_index()
    orig = (pd.read_csv(ROBUST_CSV, dtype={"norad": str}) if os.path.exists(ROBUST_CSV) else pd.DataFrame({"norad": pd.Series(dtype=str), "bstar_delta_med": pd.Series(dtype=float)}))[["norad", "bstar_delta_med"]].rename(columns={"bstar_delta_med": "bstar_change_original_panel"})
    summ = summ.merge(orig, on="norad", how="left")
    summ["usable_1plus_each_side"] = (summ["n_fits_before"] >= 1) & (summ["n_fits_after"] >= 1)
    summ["usable_2plus_each_side"] = (summ["n_fits_before"] >= 2) & (summ["n_fits_after"] >= 2)

    log("")
    log("PER-SATELLITE SUMMARY")
    log(f"  satellites: {len(summ)} | with >=1 distinct fit before AND after: {int(summ['usable_1plus_each_side'].sum())} | with >=2 each side: {int(summ['usable_2plus_each_side'].sum())}")
    dirs = summ.assign(direct=summ["group"].isin(["umbra_candidate", "penumbra_buffer"]))
    log("  distinct fits per satellite, mean by side, direct-contact vs other (the update-frequency difference):")
    for side in ["n_fits_before", "n_fits_during", "n_fits_after"]:
        m = dirs.groupby("direct")[side].mean().round(2).to_dict()
        log(f"     {side:<15} other={m.get(False)}  direct-contact={m.get(True)}")
    ok = summ.dropna(subset=["bstar_change_distinct", "bstar_change_original_panel"])
    if len(ok) > 2:
        log(f"  agreement of the new distinct-fit outcome with the original panel outcome: n={len(ok)}, Spearman {stats.spearmanr(ok['bstar_change_distinct'], ok['bstar_change_original_panel'])[0]:.3f}")

    df_out = df[["norad", "name", "group", "shell_inclination", "snapshot_label", "source", "source_file", "download_utc_filestamp",
                 "download_utc_used", "download_time_source", "fit_id",
                 "epoch", "epoch_age_at_download_h", "hours_from_peak", "epoch_bucket", "first_sighting", "sighting_number_of_fit",
                 "bstar", "bstar_is_zero", "mean_motion", "eccentricity", "inclination_deg", "sma_km", "panel_bstar", "panel_is_new_tle"]]
    df_out = df_out.rename(columns={"epoch": "tle_epoch_utc"})
    fmt = lambda c: c.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    for c in ["download_utc_filestamp", "download_utc_used", "tle_epoch_utc"]:
        df_out[c] = fmt(df_out[c])
    for c in ["tle_epoch_utc", "first_download_utc"]:
        fits[c] = fmt(fits[c])
    summ["latest_before_epoch"] = fmt(summ["latest_before_epoch"]) if summ["latest_before_epoch"].notna().any() else summ["latest_before_epoch"]
    df_out.to_csv(os.path.join(OUT, "tle_all_sightings.csv"), index=False)
    fits.to_csv(os.path.join(OUT, "tle_distinct_fits.csv"), index=False)
    summ.to_csv(os.path.join(OUT, "tle_satellite_summary.csv"), index=False)
    with open(os.path.join(OUT, "reconciliation_report.txt"), "w") as f:
        f.write("\n".join(lines_out))
    log("")
    log(f"Wrote: tle_all_sightings.csv ({len(df_out)} rows), tle_distinct_fits.csv ({len(fits)} rows), "
        f"tle_satellite_summary.csv ({len(summ)} rows), reconciliation_report.txt -> {OUT}")


if __name__ == "__main__":
    main()
