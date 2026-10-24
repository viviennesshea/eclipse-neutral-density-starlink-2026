#!/usr/bin/env python3
"""
build_satellite_planes.py
============================
One simple table: every target satellite, the orbital plane it travels in, and how many TLE data points we have for it.

Plane rule (a proxy, not an official Starlink plane list):
  * Each Space-Track fit gives RAAN (where the orbit plane sits around the Earth). RAAN drifts a few degrees a day, so each
    fit's RAAN is moved to a common time (2026-08-12 12:00 UTC) using the drift rate measured for that inclination shell;
    the satellite's value is the median over its fits.
  * Within an inclination shell, satellites are sorted by that RAAN and a new plane starts wherever the gap to the next
    satellite is more than 1.5 degrees (wrapping around 360).
Satellites with no Space-Track data (all newly launched) have no RAAN, so no plane.

Output: <out-dir>/satellite_planes.csv.  Read-only on everything else.
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
import glob, json, os
import numpy as np
import pandas as pd

REF = pd.Timestamp("2026-08-12T12:00:00Z"); GAP = 1.5; SPAN = 2.0

def shell_of(inc):
    return "43" if inc < 48 else ("53" if inc < 60 else ("70" if inc < 80 else "97"))

def main():
    tg = pd.read_csv(SP.TARGETS_CSV, dtype={"norad": str}).rename(columns={"norad": "sat_key"})
    tg["shell"] = tg["inclination"].map(shell_of)
    allr = pd.read_csv(os.path.join(SP.ST_OUT, "spacetrack_all_rows.csv"), dtype={"sat_key": str, "norad": str, "gp_id": str})
    allr["epoch_utc"] = pd.to_datetime(allr["epoch_utc"], utc=True, format="ISO8601")
    raan = {}
    for fn in glob.glob(os.path.join(SP.ST_RAW, "batch_*.json")):
        for r in json.load(open(fn)):
            raan[str(r["GP_ID"])] = float(r["RA_OF_ASC_NODE"])
    allr["raan"] = allr["gp_id"].map(raan)
    allr = allr.merge(tg[["sat_key", "shell"]], on="sat_key", how="left")
    allr["t_days"] = (allr["epoch_utc"] - REF).dt.total_seconds() / 86400

    # drift rate per shell: median of per-satellite slopes of unwrapped RAAN vs time
    rates = {}
    for sh, g in allr.groupby("shell"):
        sl = []
        for _, s in g.sort_values("epoch_utc").groupby("sat_key"):
            if len(s) >= 4 and s["t_days"].max() - s["t_days"].min() > 1:
                sl.append(np.polyfit(s["t_days"], np.degrees(np.unwrap(np.radians(s["raan"]))), 1)[0])
        rates[sh] = float(np.median(sl))
    allr["raan_ref"] = (allr["raan"] + allr["shell"].map(rates) * (0 - allr["t_days"])) % 360
    def circ_median(x):
        x = np.asarray(x); c = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(x))))) % 360
        d = ((x - c + 180) % 360) - 180
        return (c + np.median(d)) % 360
    per = allr.groupby("sat_key").agg(raan_deg_at_ref=("raan_ref", circ_median), spread=("raan_ref", lambda x: np.ptp(((x - x.iloc[0] + 180) % 360) - 180)),
                                      launch_designator=("intl_des", "first"), norad_catalog_id=("norad", "first"),
                                      n_tle_fits_spacetrack=("epoch_utc", "nunique"))
    out = tg.merge(per, on="sat_key", how="left")
    out["n_tle_fits_spacetrack"] = out["n_tle_fits_spacetrack"].fillna(0).astype(int)
    ce = pd.read_csv(os.path.join(SP.TLE_OUT, "tle_satellite_summary.csv"), dtype={"norad": str})[["norad", "n_distinct_fits"]]
    out = out.merge(ce.rename(columns={"norad": "sat_key", "n_distinct_fits": "n_tle_fits_celestrak"}), on="sat_key", how="left")
    out["n_tle_fits_celestrak"] = out["n_tle_fits_celestrak"].fillna(0).astype(int)
    out["launch_batch"] = out["launch_designator"].str.extract(r"^(\d{4}-\d{3})")

    # planes: gap rule within each shell, circular
    pid = {}
    for sh, g in out.dropna(subset=["raan_deg_at_ref"]).groupby("shell"):
        v = g.sort_values("raan_deg_at_ref"); vals = v["raan_deg_at_ref"].values; keys = v["sat_key"].values
        labels = np.zeros(len(vals), int); start = vals[0]
        for i in range(1, len(vals)):
            labels[i] = labels[i - 1]
            if vals[i] - vals[i - 1] > GAP or vals[i] - start > SPAN:   # gap, or plane would get too wide
                labels[i] += 1; start = vals[i]
        if len(vals) > 1 and (vals[0] + 360 - vals[-1]) <= GAP and labels[-1] > 0:   # wrap-around
            labels[labels == labels[-1]] = 0
        for k, l in zip(keys, labels):
            pid[k] = f"{sh}-{l + 1:02d}"
    out["plane_id"] = out["sat_key"].map(pid)
    out["n_satellites_in_plane"] = out.groupby("plane_id")["sat_key"].transform("count")
    cols = ["sat_key", "norad_catalog_id", "name", "group", "shell", "plane_id", "n_satellites_in_plane", "raan_deg_at_ref",
            "launch_designator", "launch_batch", "n_tle_fits_spacetrack", "n_tle_fits_celestrak"]
    out = out[cols].sort_values(["shell", "plane_id", "name"], na_position="last").reset_index(drop=True)
    out["raan_deg_at_ref"] = out["raan_deg_at_ref"].round(3)
    out.to_csv(SP.PLANES_CSV, index=False)
    print(f"satellites: {len(out)} | with a plane: {out['plane_id'].notna().sum()} | planes: {out['plane_id'].nunique()}")
    print("drift rates used (deg/day):", {k: round(v, 3) for k, v in rates.items()})
    print("planes per shell:", out.groupby("shell")["plane_id"].nunique().to_dict())
    print("satellites per plane: median", out.groupby("plane_id").size().median(), "| min", out.groupby("plane_id").size().min(), "| max", out.groupby("plane_id").size().max())
    print("no plane (no Space-Track data):", out["plane_id"].isna().sum(), out[out.plane_id.isna()].group.value_counts().to_dict())
    print("TLE fits per satellite: Space-Track median", out.n_tle_fits_spacetrack.median(), "| CelesTrak median", out.n_tle_fits_celestrak.median())
    print(out.head(6).to_string(index=False))

if __name__ == "__main__":
    main()
