#!/usr/bin/env python3
"""
02_satellite_dosage.py
======================
Shadow dose per satellite, in equivalent umbra-seconds, sampled every 1 second.
See results/satellite_dosage.md for the steps. Read-only on the data folder.
Output: <out-dir>/results/satellite_dosage.csv
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
import glob, importlib.util, os, warnings
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.interpolate import CubicSpline

warnings.filterwarnings("ignore")
SNAP = SP.SNAP_DIR
sys.path.insert(0, SP.SHADOW_SRC)
from eclipse_shadow import sun_moon_gcrs_km  # noqa: E402
dm = SP.load_dose_module()

R_EARTH, R_SUN, R_MOON = 6378.137, 696000.0, 1737.4
T0, T1 = pd.Timestamp("2026-08-12T15:00:00Z"), pd.Timestamp("2026-08-12T20:30:00Z")   # 1-second window (dose is zero outside the eclipse)
PREFERRED = "20260812T155855Z_2026-08-12_preeclipse"
OUT = os.path.join(SP.RESULTS, "satellite_dosage.csv")

def main():
    tg = pd.read_csv(SP.TARGETS_CSV, dtype={"norad": str})[["norad", "name", "group"]]
    secs = pd.date_range(T0, T1, freq="1s"); ts = secs.astype("int64").values / 1e9
    t_ast = Time(secs.to_pydatetime(), scale="utc")
    rs, rm = sun_moon_gcrs_km(t_ast)                                   # Sun and Moon at every second, (N,3)
    ds_all, dmm = np.linalg.norm(rs, axis=1), np.linalg.norm(rm, axis=1)
    a_moon = np.arcsin(np.clip(R_MOON / dmm, -1, 1)); shat = rs / ds_all[:, None]

    snaps = [os.path.basename(p.rstrip("/")) for p in sorted(glob.glob(os.path.join(SNAP, "*", "")))]
    order = [PREFERRED] + [s for s in snaps if s != PREFERRED]
    result = {}
    for s in order:
        left = [n for n in tg.norad if n not in result]
        if not left:
            break
        st = pd.read_parquet(os.path.join(SNAP, s, "states.parquet"), columns=["norad", "epoch_utc", "x_km", "y_km", "z_km"])
        st["norad"] = st["norad"].astype(str); st["epoch_utc"] = pd.to_datetime(st["epoch_utc"], utc=True)
        st = st[st.norad.isin(left)].sort_values(["norad", "epoch_utc"])
        for n, g in st.groupby("norad"):
            t = g["epoch_utc"].astype("int64").values / 1e9
            if t[0] > ts[0] or t[-1] < ts[-1] or len(t) < 4:      # must cover the whole 1-second window
                continue
            xyz = CubicSpline(t, g[["x_km", "y_km", "z_km"]].values, axis=0)(ts)      # 1-second positions
            vs, vm = rs - xyz, rm - xyz
            dsv, dmv = np.linalg.norm(vs, axis=1), np.linalg.norm(vm, axis=1)
            th = np.arccos(np.clip((vs * vm).sum(1) / (dsv * dmv), -1, 1))
            f = dm.obscuration_fraction(th, np.arcsin(np.clip(R_SUN / dsv, -1, 1)), np.arcsin(np.clip(R_MOON / dmv, -1, 1)))
            proj = (xyz * shat).sum(1); perp = np.sqrt(np.maximum((xyz ** 2).sum(1) - proj ** 2, 0))
            f = np.where((proj < 0) & (perp < R_EARTH), 0.0, f)                       # Earth blocks the Sun: no sunlight to lose
            result[n] = (float(f.sum()), int((f > 0).sum()), s)                       # 1-second samples: sum of fractions = seconds
        print(f"{s}: {len(result)}/{len(tg)} satellites done", flush=True)
    out = tg.copy()
    out["umbra_seconds"] = out.norad.map(lambda n: result[n][0] if n in result else np.nan)
    out["ephemeris_snapshot"] = out.norad.map(lambda n: result[n][2] if n in result else "")
    out["has_ephemeris"] = out.norad.isin(result)
    out = out.rename(columns={"norad": "sat_key"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, float_format="%.4f")
    d = out.dropna(subset=["umbra_seconds"])
    print(f"rows {len(out)} | with dose {len(d)} | no ephemeris {int((~out.has_ephemeris).sum())}")
    print("umbra_seconds: >0:", int((d.umbra_seconds > 0).sum()), "| max %.1f | median of >0 %.1f" % (d.umbra_seconds.max(), d.umbra_seconds[d.umbra_seconds > 0].median()))
    print(d.groupby("group").umbra_seconds.agg(["count", "median", "max"]).round(1))

if __name__ == "__main__":
    main()
