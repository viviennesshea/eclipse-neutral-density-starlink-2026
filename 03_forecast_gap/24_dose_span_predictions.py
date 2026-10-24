#!/usr/bin/env python3
"""
24_dose_span_predictions.py
===========================
One row per dose span (pass) per satellite: dose start, end, span (minutes), dose amount (umbra-seconds); the PRE-dose ephemeris predicted position at the dose end time;
the POST-dose ephemeris position at the same time (the post file whose start is closest to the dose end; read directly if it covers the dose end, otherwise the pre-minus-post gap
was fitted over the post file's first 8 h and extended back, and the position is pre position minus that gap); and the delta (pre - post) in along-track / radial / cross-track km.
Built from results/ephemeris_drift/delta_pre_post_by_dose_pass.csv (script 23) plus the ephemeris files. New file: results/ephemeris_drift/dose_span_predictions.csv
Pseudo-passes (pass_number 0, no-dose satellites, ending at the eclipse end 18:38) are included at the bottom and flagged; filter them out for dose spans only.
"""
import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import sys
sys.dont_write_bytecode = True
import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

OUT = SP.ED; SNAP = SP.SNAP_DIR
F = {"pre": "20260812T155855Z_2026-08-12_preeclipse", "A": "20260812T200617Z_2026-08-12_T-minus22min", "B": "20260813T014750Z_2026-08-13_posteclipse_full"}
D = pd.read_csv(os.path.join(OUT, "delta_pre_post_by_dose_pass.csv"), dtype={"sat_key": str}); keys = set(D.sat_key)
def load(label):
    d = pd.read_parquet(os.path.join(SNAP, F[label], "states.parquet"), columns=["norad", "epoch_utc", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"])
    d["norad"] = d.norad.astype(str); d = d[d.norad.isin(keys)].copy(); d["t"] = pd.to_datetime(d.epoch_utc, utc=True).astype("int64") / 1e9
    return {k: (CubicSpline(x.sort_values("t").t.values, x.sort_values("t")[["x_km", "y_km", "z_km"]].values, axis=0), CubicSpline(x.sort_values("t").t.values, x.sort_values("t")[["vx_kms", "vy_kms", "vz_kms"]].values, axis=0),
                 x.t.min(), x.t.max()) for k, x in d.groupby("norad")}
S = {k: load(k) for k in F}; ts = lambda s: pd.Timestamp(s, tz="UTC").timestamp()
rows = []
for _, r in D.iterrows():
    k = r.sat_key; te = ts(r.pass_end_utc)
    if k not in S["pre"]: continue
    rp = S["pre"][k][0](te); vp = S["pre"][k][1](te)
    # closest post file by start time to the dose end (only files with a delta)
    cand = []
    for m in ("A", "B"):
        if pd.notna(r.get(f"delta_pre_post{m}_along_km")): cand.append((abs(r[f"hours_pass_end_to_post{m}_start"]), m))
    if not cand: continue
    m = min(cand)[1]; tag = f"post{m}"; method = r[f"{tag}_method"]
    if method == "direct" and k in S[m]:
        pp = S[m][k][0](te)
    else:
        u = vp / np.linalg.norm(vp); rad = rp / np.linalg.norm(rp); cr = np.cross(u, rad)
        pp = rp - (r[f"delta_pre_post{m}_along_km"] * u + r[f"delta_pre_post{m}_radial_km"] * rad + r[f"delta_pre_post{m}_cross_km"] * cr)
    ps_ = ts(r.pass_start_utc) if isinstance(r.pass_start_utc, str) and r.pass_start_utc else np.nan
    rows.append(dict(sat_key=k, name=r["name"], group=r["group"], shell=r["shell"], plane_id=r["plane_id"], in_matched_cohort=r["in_matched_cohort"], pass_number=r.pass_number, is_pseudo_no_dose=r.pass_is_pseudo_no_dose,
                     dose_start_utc=r.pass_start_utc, dose_end_utc=r.pass_end_utc, dose_span_min=(te - ps_) / 60 if ps_ == ps_ else np.nan, dose_amount_umbra_s=r.pass_dose_s, satellite_total_umbra_s=r.umbra_seconds,
                     pre_file_start_utc=r.pre_file_start_utc, hours_pre_start_to_dose_end=r.hours_pre_file_start_to_pass_end, pre_pred_x_km=rp[0], pre_pred_y_km=rp[1], pre_pred_z_km=rp[2],
                     post_file=r[f"{tag}_file"], post_file_start_utc=r[f"{tag}_start_utc"], hours_dose_end_to_post_start=r[f"hours_pass_end_to_{tag}_start"], post_method=method,
                     post_start_inside_this_dose=r[f"{tag}_start_inside_this_pass"], post_fit_rms_km=r[f"{tag}_fit_rms_km"], post_back_x_km=pp[0], post_back_y_km=pp[1], post_back_z_km=pp[2],
                     delta_along_km=r[f"delta_pre_post{m}_along_km"], delta_radial_km=r[f"delta_pre_post{m}_radial_km"], delta_cross_km=r[f"delta_pre_post{m}_cross_km"], delta_total_km=r[f"delta_pre_post{m}_total_km"]))
O = pd.DataFrame(rows).sort_values(["is_pseudo_no_dose", "sat_key", "pass_number"])
# Freshness columns: when each file was downloaded (from the snapshot folder names) and how far that is from the dose end.
DL = {"pre": pd.Timestamp("2026-08-12T15:58:55Z"), "T-22min": pd.Timestamp("2026-08-12T20:06:17Z"), "posteclipse_full": pd.Timestamp("2026-08-13T01:47:50Z")}
_te = pd.to_datetime(O.dose_end_utc, utc=True)
O["hours_abs_dose_end_to_post_start"] = O.hours_dose_end_to_post_start.abs()
O["post_file_download_utc"] = O.post_file.map(lambda f: DL[f].strftime("%Y-%m-%d %H:%M:%S"))
O["hours_dose_end_to_post_download"] = [(DL[f] - t).total_seconds() / 3600 for f, t in zip(O.post_file, _te)]
O["pre_file_download_utc"] = DL["pre"].strftime("%Y-%m-%d %H:%M:%S")
O["hours_pre_download_to_dose_end"] = [(t - DL["pre"]).total_seconds() / 3600 for t in _te]
O.to_csv(os.path.join(OUT, "dose_span_predictions.csv"), index=False, float_format="%.6g")
d = O[~O.is_pseudo_no_dose]
print(f"rows {len(O)} | dose spans {len(d)} from {d.sat_key.nunique()} satellites | pseudo rows {int(O.is_pseudo_no_dose.sum())}")
print("post file used:", O.post_file.value_counts().to_dict(), "| method:", O.post_method.value_counts().to_dict(), "| post start inside the dose:", int(d.post_start_inside_this_dose.sum()))
print("dose span (min): median %.1f, 25-75%% %.1f-%.1f, max %.1f" % (d.dose_span_min.median(), d.dose_span_min.quantile(.25), d.dose_span_min.quantile(.75), d.dose_span_min.max()))
print("dose amount per span (umbra-s): median %.1f, max %.1f" % (d.dose_amount_umbra_s.median(), d.dose_amount_umbra_s.max()))
print("hours dose end -> post start: median %.1f, range %.1f to %.1f" % (d.hours_dose_end_to_post_start.median(), d.hours_dose_end_to_post_start.min(), d.hours_dose_end_to_post_start.max()))
print("delta_along_km: median %.3f, 25-75%% %.3f to %.3f" % (d.delta_along_km.median(), d.delta_along_km.quantile(.25), d.delta_along_km.quantile(.75)))
print("check |pre-post| vector vs delta_total: max diff %.4f km" % np.abs(np.linalg.norm(d[["pre_pred_x_km","pre_pred_y_km","pre_pred_z_km"]].values-d[["post_back_x_km","post_back_y_km","post_back_z_km"]].values,axis=1)-d.delta_total_km).max())
