#!/usr/bin/env python3
"""
27_dose_pre_post_v3.py
======================
Rebuild of the dose pre/post file using each ephemeris file's OWN creation time (header 'created:'), per satellite.
  pre file  = the latest file CREATED BEFORE the dose START  (a true forecast made before the dose)
  post file = the earliest file CREATED AFTER the dose END   (the first model made after the dose)
Positions at the dose end time: pre = read from the pre file (forecast). post = read from the post file if its start <= dose end ('direct'),
otherwise the pre-minus-post gap is fitted over the post file's first 8 h and extended back to the dose end ('extended').
Dose spans from dose_pre_post_predictions.csv. Output (new): results/ephemeris_drift/dose_pre_post_predictions_v3.csv
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
from scipy.interpolate import CubicSpline

OUT = SP.ED; SNAP = SP.SNAP_DIR
FILES = ["20260810T055748Z_baseline", "20260810T162156Z_2026-08-10_afternoon", "20260811T064712Z_2026-08-11_morning", "20260812T060837Z_2026-08-12_morning",
         "20260812T155855Z_2026-08-12_preeclipse", "20260812T200617Z_2026-08-12_T-minus22min", "20260813T014750Z_2026-08-13_posteclipse_full",
         "20260813T054149Z_2026-08-13_Tplus12h", "20260813T165333Z_2026-08-13_Tplus24h"]
V = pd.read_csv(os.path.join(OUT, "dose_pre_post_predictions.csv"), dtype={"sat_key": str})
spans = V[["sat_key", "name", "dose_number", "dose_start_time", "dose_end_time", "dose_delta_time", "dose_umbra_seconds"]].copy(); keys = set(spans.sat_key)
ts = lambda s: pd.Timestamp(s, tz="UTC").timestamp()

created = {}
for f in FILES:
    for p in glob.glob(os.path.join(SNAP, f, "raw", "MEME_*")):
        k = re.match(r"MEME_(\d+)_", os.path.basename(p)).group(1)
        if k not in keys: continue
        with open(p) as fh: c = fh.readline()
        created[(f, k)] = pd.Timestamp(c.split("created:")[1].strip().replace(" UTC", ""), tz="UTC").timestamp()
data = {}
for f in FILES:
    d = pd.read_parquet(os.path.join(SNAP, f, "states.parquet"), columns=["norad", "epoch_utc", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"])
    d["norad"] = d.norad.astype(str); d = d[d.norad.isin(keys)].copy(); d["t"] = pd.to_datetime(d.epoch_utc, utc=True).astype("int64") / 1e9
    data[f] = {k: x.sort_values("t") for k, x in d.groupby("norad")}
print("loaded", flush=True)
_c = {}
def spl(f, k):
    if (f, k) not in _c:
        x = data[f][k]; _c[(f, k)] = (x.t.values, CubicSpline(x.t.values, x[["x_km", "y_km", "z_km"]].values, axis=0), CubicSpline(x.t.values, x[["vx_kms", "vy_kms", "vz_kms"]].values, axis=0))
    return _c[(f, k)]
def comps(rp, rm, vm):
    u = vm / np.linalg.norm(vm, axis=-1, keepdims=True); rad = rm / np.linalg.norm(rm, axis=-1, keepdims=True); cr = np.cross(u, rad); d = rp - rm
    return np.stack([(d * u).sum(-1), (d * rad).sum(-1), (d * cr).sum(-1)], -1)

rows, drop = [], {"no pre": 0, "no post": 0, "pre does not cover": 0, "fit failed": 0}
for _, s in spans.iterrows():
    k, t_s, t_e = s.sat_key, ts(s.dose_start_time), ts(s.dose_end_time)
    pre = [(created[(f, k)], f) for f in FILES if (f, k) in created and k in data[f] and created[(f, k)] < t_s]
    post = [(created[(f, k)], f) for f in FILES if (f, k) in created and k in data[f] and created[(f, k)] > t_e]
    if not pre: drop["no pre"] += 1; continue
    if not post: drop["no post"] += 1; continue
    cp, fp = max(pre); cq, fq = min(post)
    tp, sp, vp_ = spl(fp, k); tq, sq, vq = spl(fq, k)
    if not (tp[0] <= t_e <= tp[-1]): drop["pre does not cover"] += 1; continue
    rp, vp = sp(t_e), vp_(t_e)
    if tq[0] <= t_e:
        rq = sq(t_e); dv = comps(rp, rq, vq(t_e)); method = "direct"
    else:
        sel = tq[(tq <= tq[0] + 8 * 3600) & (tq <= tp[-1])]
        if len(sel) < 60: drop["fit failed"] += 1; continue
        D = comps(sp(sel), sq(sel), vq(sel)); tau = (sel - t_e) / 3600
        n = np.mean(np.linalg.norm(vq(sel), axis=1) / np.linalg.norm(sq(sel), axis=1)) * 3600
        X = np.column_stack([np.ones_like(tau), tau, tau ** 2, np.cos(n * tau), np.sin(n * tau)])
        dv = np.array([(lambda c: c[0] + c[3])(np.linalg.lstsq(X, D[:, j], rcond=None)[0]) for j in range(3)])
        u = vp / np.linalg.norm(vp); rad = rp / np.linalg.norm(rp); cr = np.cross(u, rad); rq = rp - (dv[0] * u + dv[1] * rad + dv[2] * cr); method = "extended"
    iso = lambda t: pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S")
    rows.append({"sat_key": k, "name": s["name"], "dose_number": s.dose_number, "dose_start_time": s.dose_start_time, "dose_end_time": s.dose_end_time,
                 "dose_delta_time": s.dose_delta_time, "dose_umbra_seconds": s.dose_umbra_seconds,
                 "pre_dose_ephemeris_prediction_x_km": rp[0], "pre_dose_ephemeris_prediction_y_km": rp[1], "pre_dose_ephemeris_prediction_z_km": rp[2],
                 "pre_dose_ephemeris_snapshot_time": iso(cp), "pre_dose_ephemeris_delta_time": (t_e - cp) / 3600,
                 "post_dose_ephemeris_prediction_x_km": rq[0], "post_dose_ephemeris_prediction_y_km": rq[1], "post_dose_ephemeris_prediction_z_km": rq[2],
                 "post_dose_ephemeris_snapshot_time": iso(cq), "post_dose_ephemeris_delta_time": (cq - t_e) / 3600,
                 "delta_predictions_distance_km": float(np.linalg.norm(dv)), "delta_along_track_km": dv[0], "post_method": method})
O = pd.DataFrame(rows)
assert (pd.to_datetime(O.pre_dose_ephemeris_snapshot_time, utc=True) < pd.to_datetime(O.dose_start_time, utc=True)).all()
assert (pd.to_datetime(O.post_dose_ephemeris_snapshot_time, utc=True) > pd.to_datetime(O.dose_end_time, utc=True)).all()
O.to_csv(os.path.join(OUT, "dose_pre_post_predictions_v3.csv"), index=False, float_format="%.6g")
print(f"rows {len(O)} of {len(spans)} | satellites {O.sat_key.nunique()} | dropped {drop}")
print("post_method:", O.post_method.value_counts().to_dict())
for c in ("pre_dose_ephemeris_delta_time", "post_dose_ephemeris_delta_time", "delta_predictions_distance_km"):
    print(f"{c}: median {O[c].median():.3f}, 25-75% {O[c].quantile(.25):.3f}-{O[c].quantile(.75):.3f}, max {O[c].max():.3f}")
