#!/usr/bin/env python3
"""
22_along_track_gap_all.py
=========================
One CSV, one row per satellite with ephemeris: its dose passes (start/end/dose), the post-eclipse ephemeris state chosen from its own last dose end, and the
along-track prediction gap between the PRE-eclipse ephemeris (prediction) and that post state.
Definitions
  dose pass   : run of seconds with obscuration > 0.01 (Earth-shadow seconds zeroed), 1-second series from the pre-eclipse file; runs separated by > 10 min are separate passes;
                a pass needs >= 1 umbra-second. pass_k_start/end (UTC) and pass_k_dose_s (equivalent umbra-seconds); k up to 5.
  prediction  : the pre-eclipse ephemeris download (2026-08-12 15:58, each satellite's file starts ~10:07 on Aug 12).
  post state  : the first later download whose start (for that satellite) is at or after the satellite's LAST dose end; for satellites with no dose, at or after the global eclipse end 18:38.
                Candidates: T-22min (start ~18:09), post-eclipse_full (~21:14), T+12h (~02:03 Aug 13), T+24h (~10:13), T+48h (Aug 14 ~10:08).
  gap         : e_along_km = mean over the post file's first hour of (r_pred - r_post) . unit(v_post); positive = the prediction is AHEAD of where the satellite was = actual drag
                less than predicted. e_radial_km, e_cross_km likewise. lead_h = hours from prediction file start to post file start; a_km_per_h2 = e_along / lead_h^2.
  reference   : the same gap for the earlier non-eclipse pair (Aug 12 06:08 morning file predicting the pre-eclipse file's start), e_ref_along_km, a_ref_km_per_h2, and DiD = a - a_ref.
Output: results/ephemeris_drift/along_track_gap_per_satellite.csv (new)
"""
import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import sys
sys.dont_write_bytecode = True
import importlib.util, os, warnings
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.interpolate import CubicSpline
warnings.filterwarnings("ignore")

OUT = SP.ED; SNAP = SP.SNAP_DIR
sys.path.insert(0, SP.SHADOW_SRC)
from eclipse_shadow import sun_moon_gcrs_km  # noqa: E402
dm = SP.load_dose_module()
R_E, R_SUN, R_MOON = 6378.137, 696000.0, 1737.4; GAP_S, MIN_PASS, THR = 600, 1.0, 0.01; E1 = pd.Timestamp("2026-08-12T18:38:00Z").timestamp()
F = {"pred": "20260812T155855Z_2026-08-12_preeclipse", "ref": "20260812T060837Z_2026-08-12_morning",
     "T-22min": "20260812T200617Z_2026-08-12_T-minus22min", "posteclipse_full": "20260813T014750Z_2026-08-13_posteclipse_full", "T+12h": "20260813T054149Z_2026-08-13_Tplus12h",
     "T+24h": "20260813T165333Z_2026-08-13_Tplus24h", "T+48h": "20260814T173942Z_2026-08-14_Tplus48h"}
POST = ["T-22min", "posteclipse_full", "T+12h", "T+24h", "T+48h"]
tg = pd.read_csv(os.path.join(SP.RESULTS, "satellite_dosage.csv"), dtype={"sat_key": str}); tg = tg[tg.has_ephemeris]
pl = pd.read_csv(SP.PLANES_CSV, dtype={"sat_key": str})[["sat_key", "shell", "plane_id"]]
# optional file from an earlier part of the project (not produced here); only fills the in_matched_cohort flag column
coh = set(pd.read_csv(SP.COHORT_CSV, dtype={"sat_key": str}).query("cohort == '43-483km'").sat_key) if os.path.exists(SP.COHORT_CSV) else set()
keys = list(tg.sat_key)

def load(label):
    d = pd.read_parquet(os.path.join(SNAP, F[label], "states.parquet"), columns=["norad", "epoch_utc", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"])
    d["norad"] = d.norad.astype(str); d = d[d.norad.isin(keys)].copy(); d["t"] = pd.to_datetime(d.epoch_utc, utc=True).astype("int64") / 1e9
    return {k: g.sort_values("t") for k, g in d.groupby("norad")}
files = {k: load(k) for k in F}; print("files loaded", flush=True)

T0, T1 = pd.Timestamp("2026-08-12T15:00:00Z"), pd.Timestamp("2026-08-12T20:30:00Z"); secs = pd.date_range(T0, T1, freq="1s"); ts = secs.astype("int64").values / 1e9
rs, rm = sun_moon_gcrs_km(Time(secs.to_pydatetime(), scale="utc")); shat = rs / np.linalg.norm(rs, axis=1)[:, None]
def passes(k):
    g = files["pred"][k]; xyz = CubicSpline(g.t.values, g[["x_km", "y_km", "z_km"]].values, axis=0)(ts); vs, vm = rs - xyz, rm - xyz
    dsv, dmv = np.linalg.norm(vs, axis=1), np.linalg.norm(vm, axis=1); th = np.arccos(np.clip((vs * vm).sum(1) / (dsv * dmv), -1, 1))
    f = dm.obscuration_fraction(th, np.arcsin(np.clip(R_SUN / dsv, -1, 1)), np.arcsin(np.clip(R_MOON / dmv, -1, 1)))
    proj = (xyz * shat).sum(1); perp = np.sqrt(np.maximum((xyz ** 2).sum(1) - proj ** 2, 0)); f = np.where((proj < 0) & (perp < R_E), 0.0, f)
    idx = np.flatnonzero(f > THR); out = []
    if len(idx):
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > GAP_S) + 1):
            if f[seg].sum() >= MIN_PASS: out.append((ts[seg[0]], ts[seg[-1]], float(f[seg].sum())))
    return out

def gap(Pl, Bl, k):
    P, B = files[Pl].get(k), files[Bl].get(k)
    if P is None or B is None: return None
    t0 = B.t.iloc[0]; Bw = B[B.t < t0 + 3600]
    if t0 < P.t.iloc[0] or Bw.t.iloc[-1] > P.t.iloc[-1]: return None
    rp = CubicSpline(P.t.values, P[["x_km", "y_km", "z_km"]].values, axis=0)(Bw.t.values); rb = Bw[["x_km", "y_km", "z_km"]].values
    u = Bw[["vx_kms", "vy_kms", "vz_kms"]].values; u = u / np.linalg.norm(u, axis=1)[:, None]; rad = rb / np.linalg.norm(rb, axis=1)[:, None]; cr = np.cross(u, rad); d = rp - rb
    return dict(e_along=(d * u).sum(1).mean(), e_radial=(d * rad).sum(1).mean(), e_cross=(d * cr).sum(1).mean(), t_post=t0, lead=(t0 - P.t.iloc[0]) / 3600, t_pred=P.t.iloc[0])
iso = lambda t: pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S") if t == t else ""
rows = []
for n, k in enumerate(keys):
    ps = passes(k); r = dict(sat_key=k); r["n_dose_passes"] = len(ps)
    for i in range(5):
        p = ps[i] if i < len(ps) else (np.nan, np.nan, np.nan); r[f"pass_{i + 1}_start_utc"], r[f"pass_{i + 1}_end_utc"], r[f"pass_{i + 1}_dose_s"] = iso(p[0]), iso(p[1]), p[2]
    r["dose_first_start_utc"] = iso(ps[0][0]) if ps else ""; r["dose_last_end_utc"] = iso(ps[-1][1]) if ps else ""; r["dose_windows_total_dose_s"] = sum(p[2] for p in ps) if ps else 0.0
    anchor = ps[-1][1] if ps else E1; r["anchor_used"] = "own last dose end" if ps else "global eclipse end 18:38 (no dose pass)"
    chosen = None
    for lab in POST:
        B = files[lab].get(k)
        if B is not None and B.t.iloc[0] >= anchor: chosen = lab; break
    r["post_state_file"] = chosen or ""
    if chosen:
        g = gap("pred", chosen, k)
        if g:
            r.update(pred_file_start_utc=iso(g["t_pred"]), post_state_start_utc=iso(g["t_post"]), hours_last_dose_end_to_post_state=(g["t_post"] - anchor) / 3600 if ps else np.nan,
                     hours_eclipse_end_to_post_state=(g["t_post"] - E1) / 3600, lead_h=g["lead"], e_along_km=g["e_along"], e_radial_km=g["e_radial"], e_cross_km=g["e_cross"], a_km_per_h2=g["e_along"] / g["lead"] ** 2)
    gr = gap("ref", "pred", k)
    if gr: r.update(e_ref_along_km=gr["e_along"], ref_lead_h=gr["lead"], a_ref_km_per_h2=gr["e_along"] / gr["lead"] ** 2)
    if "a_km_per_h2" in r and "a_ref_km_per_h2" in r: r["DiD_a_minus_a_ref"] = r["a_km_per_h2"] - r["a_ref_km_per_h2"]
    rows.append(r)
    if (n + 1) % 250 == 0: print(n + 1, "of", len(keys), flush=True)
D = pd.DataFrame(rows).merge(tg[["sat_key", "name", "group", "umbra_seconds"]], on="sat_key").merge(pl, on="sat_key", how="left")
D["in_matched_cohort"] = D.sat_key.isin(coh)
first = ["sat_key", "name", "group", "shell", "plane_id", "in_matched_cohort", "umbra_seconds", "dose_windows_total_dose_s", "n_dose_passes"] + [f"pass_{i}_{s}" for i in range(1, 6) for s in ("start_utc", "end_utc", "dose_s")] + \
        ["dose_first_start_utc", "dose_last_end_utc", "anchor_used", "pred_file_start_utc", "post_state_file", "post_state_start_utc", "hours_last_dose_end_to_post_state", "hours_eclipse_end_to_post_state", "lead_h",
         "e_along_km", "e_radial_km", "e_cross_km", "a_km_per_h2", "e_ref_along_km", "ref_lead_h", "a_ref_km_per_h2", "DiD_a_minus_a_ref"]
D[first].to_csv(os.path.join(OUT, "along_track_gap_per_satellite.csv"), index=False, float_format="%.6g")
print(f"\nrows {len(D)} | with a gap {int(D.e_along_km.notna().sum())} | with dose passes {int((D.n_dose_passes > 0).sum())} | no dose pass {int((D.n_dose_passes == 0).sum())}")
print("post state file:", D.post_state_file.value_counts().to_dict()); print("n passes:", D.n_dose_passes.value_counts().sort_index().to_dict())
x = D.dropna(subset=["e_along_km"]); print("e_along_km: median %.3f, IQR %.3f to %.3f, min %.2f, max %.2f" % (x.e_along_km.median(), x.e_along_km.quantile(.25), x.e_along_km.quantile(.75), x.e_along_km.min(), x.e_along_km.max()))
print("hours last dose end -> post state (dosed): median %.1f, range %.1f to %.1f" % tuple(x[x.n_dose_passes > 0].hours_last_dose_end_to_post_state.agg(["median", "min", "max"])))
print("matched cohort rows:", int(D.in_matched_cohort.sum()))
