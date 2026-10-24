#!/usr/bin/env python3
"""
23_delta_pre_post_by_pass.py
============================
delta_pre_post per DOSE PASS: the pre-dose prediction (pre-eclipse ephemeris, treated as 'no dose in the model') vs post-dose models (T-22min file = model A, may start mid-dose;
post-eclipse_full file = model B), evaluated at the pass END time t_end.
 - pre position at t_end: read (cubic spline) from the pre-eclipse file.
 - post model at t_end: if the post file's start <= t_end it is read directly ('direct'); otherwise the pre-minus-post gap d(t) = r_pre(t) - r_post(t), expressed in the post file's along/radial/cross
   basis, is fitted over the post file's first 8 h with  c0 + c1*tau + c2*tau^2 + A cos(n tau) + B sin(n tau)  (tau = hours from t_end, n = orbital rate) and extended back to tau = 0
   ('extended'): this transports the DIFFERENCE, so unmodelled gravity, which both files share, cancels.
 - Sign: positive along-track = the pre-dose prediction is AHEAD of the post model.
Also pseudo-passes (pass_number 0) at the global eclipse end 18:38 for satellites with no dose. Validation of the extension on the passes model A covers directly is written to the results md.
Outputs: results/ephemeris_drift/delta_pre_post_by_dose_pass.csv, delta_pre_post_by_dose_pass.md (new)
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
FLAB = {"A": "T-22min", "B": "posteclipse_full"}; FIT_H = 8.0
g = pd.read_csv(os.path.join(OUT, "along_track_gap_per_satellite.csv"), dtype={"sat_key": str}); keys = list(g.sat_key)
def load(label):
    d = pd.read_parquet(os.path.join(SNAP, F[label], "states.parquet"), columns=["norad", "epoch_utc", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"])
    d["norad"] = d.norad.astype(str); d = d[d.norad.isin(keys)].copy(); d["t"] = pd.to_datetime(d.epoch_utc, utc=True).astype("int64") / 1e9
    return {k: x.sort_values("t") for k, x in d.groupby("norad")}
files = {k: load(k) for k in F}; print("files loaded", flush=True)
splP = {k: CubicSpline(x.t.values, x[["x_km", "y_km", "z_km"]].values, axis=0) for k, x in files["pre"].items()}
splM = {m: {k: (CubicSpline(x.t.values, x[["x_km", "y_km", "z_km"]].values, axis=0), CubicSpline(x.t.values, x[["vx_kms", "vy_kms", "vz_kms"]].values, axis=0)) for k, x in files[m].items()} for m in ("A", "B")}
iso = lambda t: pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S")
E1 = pd.Timestamp("2026-08-12T18:38:00Z").timestamp()

def basis(r, v):
    u = v / np.linalg.norm(v, axis=-1, keepdims=True); rad = r / np.linalg.norm(r, axis=-1, keepdims=True); return u, rad, np.cross(u, rad)
def delta_at(k, m, t):                        # direct: pre minus post model at time(s) t, components (along, radial, cross)
    rP = splP[k](t); rM = splM[m][k][0](t); vM = splM[m][k][1](t); u, rad, cr = basis(rM, vM); d = rP - rM
    return np.stack([(d * u).sum(-1), (d * rad).sum(-1), (d * cr).sum(-1)], -1)
def extended(k, m, t_end, t_from, fit_h=FIT_H):
    """fit d(t) over [t_from, t_from + fit_h h] and extend to t_end (t_end < t_from)."""
    tt = files[m][k].t.values; sel = tt[(tt >= t_from) & (tt <= t_from + fit_h * 3600)]
    if len(sel) < 60: return None
    D = delta_at(k, m, sel); tau = (sel - t_end) / 3600.0
    r = splM[m][k][0](sel); v = splM[m][k][1](sel); n = np.mean(np.linalg.norm(v, axis=1) / np.linalg.norm(r, axis=1)) * 3600.0
    X = np.column_stack([np.ones_like(tau), tau, tau ** 2, np.cos(n * tau), np.sin(n * tau)]); out, rms = [], []
    for j in range(3):
        c, *_ = np.linalg.lstsq(X, D[:, j], rcond=None); out.append(c[0] + c[3]); rms.append(np.sqrt(np.mean((X @ c - D[:, j]) ** 2)))
    return np.array(out), max(rms)

rows = []
for _, s in g.iterrows():
    k = s.sat_key; passes = []
    for i in range(1, 6):
        if isinstance(s[f"pass_{i}_end_utc"], str) and s[f"pass_{i}_end_utc"]:
            passes.append((i, pd.Timestamp(s[f"pass_{i}_start_utc"], tz="UTC").timestamp(), pd.Timestamp(s[f"pass_{i}_end_utc"], tz="UTC").timestamp(), s[f"pass_{i}_dose_s"]))
    if not passes: passes = [(0, np.nan, E1, 0.0)]
    if k not in files["pre"]: continue
    P0 = files["pre"][k].t.iloc[0]
    for (i, ps, pe, pd_) in passes:
        r = dict(sat_key=k, name=s["name"], group=s["group"], shell=s["shell"], plane_id=s["plane_id"], in_matched_cohort=s["in_matched_cohort"], umbra_seconds=s["umbra_seconds"], pass_number=i,
                 pass_is_pseudo_no_dose=(i == 0), pass_start_utc=iso(ps) if ps == ps else "", pass_end_utc=iso(pe), pass_dose_s=pd_, pre_file="pre-eclipse", pre_file_start_utc=iso(P0),
                 hours_pre_file_start_to_pass_end=(pe - P0) / 3600)
        for m in ("A", "B"):
            if k not in files[m] or k not in splM[m]: continue
            M0 = files[m][k].t.iloc[0]; last = files[m][k].t.iloc[-1]; tag = f"post{m}"
            r[f"{tag}_file"] = FLAB[m]; r[f"{tag}_start_utc"] = iso(M0); r[f"hours_pass_end_to_{tag}_start"] = (M0 - pe) / 3600
            r[f"{tag}_start_inside_this_pass"] = bool(ps == ps and ps <= M0 <= pe); r[f"{tag}_start_before_pass_start"] = bool(ps == ps and M0 < ps)
            if M0 <= pe <= last and pe >= P0:
                dv = delta_at(k, m, np.array([pe]))[0]; r[f"{tag}_method"] = "direct"; r[f"{tag}_fit_rms_km"] = 0.0
            else:
                e = extended(k, m, pe, M0)
                if e is None: continue
                dv, rms = e; r[f"{tag}_method"] = "extended"; r[f"{tag}_fit_rms_km"] = rms
            r[f"delta_pre_post{m}_along_km"], r[f"delta_pre_post{m}_radial_km"], r[f"delta_pre_post{m}_cross_km"] = dv
            r[f"delta_pre_post{m}_total_km"] = float(np.linalg.norm(dv))
            dst = delta_at(k, m, files[m][k].t.values[:60]).mean(0); r[f"delta_at_post{m}_start_along_km"] = dst[0]        # no extension: mean of the first hour
        rows.append(r)
    if len(rows) % 500 < len(passes): print(len(rows), "rows", flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "delta_pre_post_by_dose_pass.csv"), index=False, float_format="%.6g")

# ---- validation of the extension: passes where model A covers t_end directly; refit starting h hours after t_end and extend back
val = {1: [], 2: [], 4: []}
for _, r in D[(D.postA_method == "direct") & ~D.pass_is_pseudo_no_dose].iterrows():
    k = r.sat_key; pe = pd.Timestamp(r.pass_end_utc, tz="UTC").timestamp(); truth = r.delta_pre_postA_along_km
    for h in val:
        if files["A"][k].t.values[-1] < pe + (h + FIT_H) * 3600: continue
        e = extended(k, "A", pe, pe + h * 3600)
        if e is not None: val[h].append(e[0][0] - truth)
L = ["# delta_pre_post by dose pass", "",
     "Pre-dose prediction = pre-eclipse ephemeris ('no dose in the model'); post models: A = T-22min file (may start mid-dose), B = post-eclipse_full file. Evaluated at each pass's end time. Positive along-track = the pre prediction is ahead of the post model.", "",
     f"- rows: {len(D)} ({int((~D.pass_is_pseudo_no_dose).sum())} dose passes from {D[~D.pass_is_pseudo_no_dose].sat_key.nunique()} satellites; {int(D.pass_is_pseudo_no_dose.sum())} pseudo-passes for no-dose satellites at 18:38)",
     f"- model A: method direct {int((D.postA_method == 'direct').sum())}, extended {int((D.postA_method == 'extended').sum())}; A file starts INSIDE the pass (mid-dose state): {int(D.postA_start_inside_this_pass.sum())}; A starts before the pass starts: {int(D.postA_start_before_pass_start.sum())}",
     f"- model B: method direct {int((D.postB_method == 'direct').sum())}, extended {int((D.postB_method == 'extended').sum())}", "",
     "## Validation of the 'extended' method (model A, passes it covers directly; refit from h hours after t_end and extend back; error = extended - direct, along-track, km)", "", "| h extended back | n | median abs error | 90th pct abs | max abs |", "|---|---|---|---|---|"]
for h, v in val.items():
    v = np.abs(np.array(v)); L.append(f"| {h} | {len(v)} | {np.median(v):.4f} | {np.percentile(v, 90):.4f} | {v.max():.4f} |" if len(v) else f"| {h} | 0 | | | |")
x = D.dropna(subset=["delta_pre_postA_along_km"]); y = D.dropna(subset=["delta_pre_postB_along_km"])
L += ["", "## Size of the deltas (along-track km, median and middle half)", "", "| model | rows | median | 25th to 75th |", "|---|---|---|---|"]
for nm, d, c in (("A", x, "delta_pre_postA_along_km"), ("B", y, "delta_pre_postB_along_km")): L.append(f"| {nm} | {len(d)} | {d[c].median():.3f} | {d[c].quantile(.25):.3f} to {d[c].quantile(.75):.3f} |")
L += ["", "Columns: see the CSV header; `hours_pass_end_to_postA_start` is negative when the post file starts before the pass end. `delta_at_post{A,B}_start_along_km` is the gap at the post file's own start (no extension)."]
open(os.path.join(OUT, "delta_pre_post_by_dose_pass.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
