#!/usr/bin/env python3
"""
28_sma_check.py
===============
Drag-paradox check on dose_pre_post_predictions_v3.csv: does 'ahead of the forecast' go with 'smaller orbit'?
Same pre/post file selection as 27_dose_pre_post_v3.py (pre = latest created before dose start; post = earliest created after dose end).
Orbit size = semi-major axis from vis-viva, a = 1 / (2/r - v^2/mu), from each file's own position AND velocity at a COMMON time
(short-period gravity terms are nearly identical for both files at the same time and place, so they cancel in the difference).
  common time: dose end (direct rows) and, for every row, the post file's first hour (mean) where both files have real data (no extension).
  da_km = a_pre - a_post (positive = the post model has the smaller orbit).  along_km = (r_pre - r_post) . unit(v_post) (negative = satellite ahead of the forecast).
Paradox expectation: along < 0 goes with da > 0  (negative correlation).
Output (new): results/ephemeris_drift/sma_paradox_check.csv, sma_paradox_check.md
"""
import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import sys
sys.dont_write_bytecode = True
import glob, os, re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline

OUT = SP.ED; SNAP = SP.SNAP_DIR
MU = 398600.4418
FILES = ["20260810T055748Z_baseline", "20260810T162156Z_2026-08-10_afternoon", "20260811T064712Z_2026-08-11_morning", "20260812T060837Z_2026-08-12_morning",
         "20260812T155855Z_2026-08-12_preeclipse", "20260812T200617Z_2026-08-12_T-minus22min", "20260813T014750Z_2026-08-13_posteclipse_full",
         "20260813T054149Z_2026-08-13_Tplus12h", "20260813T165333Z_2026-08-13_Tplus24h"]
V = pd.read_csv(os.path.join(OUT, "dose_pre_post_predictions_v3.csv"), dtype={"sat_key": str}); keys = set(V.sat_key)
ts = lambda s: pd.Timestamp(s, tz="UTC").timestamp()
created = {}
for f in FILES:
    for p in glob.glob(os.path.join(SNAP, f, "raw", "MEME_*")):
        k = re.match(r"MEME_(\d+)_", os.path.basename(p)).group(1)
        if k in keys:
            with open(p) as fh: created[(f, k)] = pd.Timestamp(fh.readline().split("created:")[1].strip().replace(" UTC", ""), tz="UTC").timestamp()
data = {}
for f in FILES:
    d = pd.read_parquet(os.path.join(SNAP, f, "states.parquet"), columns=["norad", "epoch_utc", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"])
    d["norad"] = d.norad.astype(str); d = d[d.norad.isin(keys)].copy(); d["t"] = pd.to_datetime(d.epoch_utc, utc=True).astype("int64") / 1e9
    data[f] = {k: x.sort_values("t") for k, x in d.groupby("norad")}
_c = {}
def spl(f, k):
    if (f, k) not in _c:
        x = data[f][k]; _c[(f, k)] = (x.t.values, CubicSpline(x.t.values, x[["x_km", "y_km", "z_km"]].values, axis=0), CubicSpline(x.t.values, x[["vx_kms", "vy_kms", "vz_kms"]].values, axis=0))
    return _c[(f, k)]
sma = lambda r, v: 1.0 / (2.0 / np.linalg.norm(r, axis=-1) - (v ** 2).sum(-1) / MU)
def compare(fp, fq, k, t):
    _, sp, vp = spl(fp, k); _, sq, vq = spl(fq, k); rP, vP, rQ, vQ = sp(t), vp(t), sq(t), vq(t)
    u = vQ / np.linalg.norm(vQ, axis=-1, keepdims=True)
    return np.mean(((rP - rQ) * u).sum(-1)), np.mean(sma(rP, vP) - sma(rQ, vQ))
rows = []
for _, s in V.iterrows():
    k, t_s, t_e = s.sat_key, ts(s.dose_start_time), ts(s.dose_end_time)
    fp = max((created[(f, k)], f) for f in FILES if (f, k) in created and k in data[f] and created[(f, k)] < t_s)[1]
    fq = min((created[(f, k)], f) for f in FILES if (f, k) in created and k in data[f] and created[(f, k)] > t_e)[1]
    tq = spl(fq, k)[0]; tp = spl(fp, k)[0]; w = tq[(tq < tq[0] + 3600) & (tq <= tp[-1])]
    r = dict(sat_key=k, name=s["name"], dose_number=s.dose_number, dose_umbra_seconds=s.dose_umbra_seconds, post_method=s.post_method,
             hours_dose_end_to_common_time=(w.mean() - t_e) / 3600, along_km_common=np.nan, da_km_common=np.nan, along_km_dose_end=np.nan, da_km_dose_end=np.nan)
    if len(w): r["along_km_common"], r["da_km_common"] = compare(fp, fq, k, w)
    if s.post_method == "direct": r["along_km_dose_end"], r["da_km_dose_end"] = compare(fp, fq, k, np.array([t_e]))
    rows.append(r)
O = pd.DataFrame(rows); O.to_csv(os.path.join(OUT, "sma_paradox_check.csv"), index=False, float_format="%.6g")

L = ["# Drag-paradox check: along-track gap vs orbit-size (semi-major axis) gap", "",
     "da = a_pre - a_post (positive = post model has the smaller orbit). along < 0 = satellite ahead of the pre forecast. Paradox: along < 0 goes with da > 0.", "",
     "| comparison | n | Spearman(along, da) | ahead (along<0): share with smaller post orbit | behind (along>0): share with smaller post orbit | median da (m) |", "|---|---|---|---|---|---|"]
for lab, a, b in (("direct rows, at the dose end", "along_km_dose_end", "da_km_dose_end"), ("all rows, post file's first hour", "along_km_common", "da_km_common")):
    x = O.dropna(subset=[a, b]); rho = stats.spearmanr(x[a], x[b])[0]
    L.append(f"| {lab} | {len(x)} | {rho:+.3f} | {100 * (x[x[a] < 0][b] > 0).mean():.0f}% | {100 * (x[x[a] > 0][b] > 0).mean():.0f}% | {1000 * x[b].median():.2f} |")
open(os.path.join(OUT, "sma_paradox_check.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
