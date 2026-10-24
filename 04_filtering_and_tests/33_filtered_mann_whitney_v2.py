#!/usr/bin/env python3
"""
33_filtered_mann_whitney_v2.py
==============================
Rerun with the filters of PROTOCOL_filtered_mann_whitney_addendum.md (altitude >= 440 km; burn check both directions; IQR backstop).
Outputs (new): results/ephemeris_drift/filtered_mw_v2_per_satellite.csv, filtered_mw_v2_results.md
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
MU = 398600.4418; rng = np.random.default_rng(20260924)
FILES = ["20260810T055748Z_baseline", "20260810T162156Z_2026-08-10_afternoon", "20260811T064712Z_2026-08-11_morning", "20260812T060837Z_2026-08-12_morning",
         "20260812T155855Z_2026-08-12_preeclipse", "20260812T200617Z_2026-08-12_T-minus22min", "20260813T014750Z_2026-08-13_posteclipse_full",
         "20260813T054149Z_2026-08-13_Tplus12h", "20260813T165333Z_2026-08-13_Tplus24h", "20260814T173942Z_2026-08-14_Tplus48h"]
ts = lambda s: pd.Timestamp(s, tz="UTC").timestamp()
W0, W1 = ts("2026-08-12 15:29:00"), ts("2026-08-12 20:03:00")
dose = pd.read_csv(os.path.join(SP.RESULTS, "satellite_dosage.csv"), dtype={"sat_key": str}).query("has_ephemeris")[["sat_key", "name", "umbra_seconds"]]
pl = pd.read_csv(SP.PLANES_CSV, dtype={"sat_key": str})[["sat_key", "shell"]].assign(shell=lambda x: x["shell"].astype(str))
ln = pd.read_csv(os.path.join(OUT, "dose_pre_post_predictions_v3.csv"), dtype={"sat_key": str}).groupby("sat_key").dose_delta_time.sum().rename("dose_length_min")
S = dose.merge(pl, on="sat_key", how="left").merge(ln.reset_index(), on="sat_key", how="left"); S["dose_length_min"] = S.dose_length_min.fillna(0.0)
keys = set(S.sat_key)
created = {}
for f in FILES:
    for p in glob.glob(os.path.join(SNAP, f, "raw", "MEME_*")):
        k = re.match(r"MEME_(\d+)_", os.path.basename(p)).group(1)
        if k in keys:
            with open(p) as fh: created[(f, k)] = pd.Timestamp(fh.readline().split("created:")[1].strip().replace(" UTC", ""), tz="UTC").timestamp()
data = {}
for f in FILES:
    d = pd.read_parquet(os.path.join(SNAP, f, "states.parquet"), columns=["norad", "epoch_utc", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms", "altitude_km"])
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
def rise(f, k, t_from, t_to):
    x = data[f][k]; x = x[(x.t >= t_from) & (x.t <= t_to)]
    if len(x) < 95 + 180: return np.nan
    r, v = x[["x_km", "y_km", "z_km"]].values, x[["vx_kms", "vy_kms", "vz_kms"]].values
    a = 1.0 / (2.0 / np.linalg.norm(r, axis=1) - (v ** 2).sum(1) / MU); sm = pd.Series(a).rolling(95, center=True).mean().dropna().values
    if len(sm) <= 180: return (np.nan, np.nan)
    dd = sm[180:] - sm[:-180]; return (float(dd.max()) * 1000, float(dd.min()) * 1000)      # metres: largest rise, largest drop
def measure(k, t_s, t_e):
    pre = [(created[(f, k)], f) for f in FILES if (f, k) in created and k in data[f] and created[(f, k)] < t_s]
    post = [(created[(f, k)], f) for f in FILES if (f, k) in created and k in data[f] and created[(f, k)] > t_e]
    if not pre or not post: return None
    fp, fq = max(pre)[1], min(post)[1]; tp, sp, _ = spl(fp, k); tq, sq, vq = spl(fq, k)
    if not (tp[0] <= t_e <= tp[-1]): return None
    if tq[0] <= t_e: al, meth = comps(sp(t_e), sq(t_e), vq(t_e))[0], "direct"
    else:
        sel = tq[(tq <= tq[0] + 8 * 3600) & (tq <= tp[-1])]
        if len(sel) < 60: return None
        D = comps(sp(sel), sq(sel), vq(sel))[:, 0]; tau = (sel - t_e) / 3600; n = np.mean(np.linalg.norm(vq(sel), axis=1) / np.linalg.norm(sq(sel), axis=1)) * 3600
        X = np.column_stack([np.ones_like(tau), tau, tau ** 2, np.cos(n * tau), np.sin(n * tau)]); c = np.linalg.lstsq(X, D, rcond=None)[0]; al, meth = c[0] + c[3], "extended"
    r1, r2 = rise(fp, k, tp[0], t_e), rise(fq, k, tq[0], tq[0] + 8 * 3600)
    x = data[fp][k]; alt = float(x[x.t < t_s].altitude_km.median()) if (x.t < t_s).any() else float(x.altitude_km.iloc[0])
    return al, meth, np.nanmax([r1[0], r2[0]]), np.nanmin([r1[1], r2[1]]), alt
def iqr_hi(v): q1, q3 = np.nanpercentile(v, [25, 75]); return q3 + 3 * (q3 - q1)
def iqr_ok(v): q1, q3 = np.nanpercentile(v, [25, 75]); return (v >= q1 - 3 * (q3 - q1)) & (v <= q3 + 3 * (q3 - q1))
def strat_mw(df, grp, a, b, nperm=5000):
    its = []
    for _, g in df.groupby("shell"):
        A, B = g[g[grp] == a].along_km.values, g[g[grp] == b].along_km.values
        if len(A) >= 5 and len(B) >= 5: its.append((np.r_[A, B], len(A), len(B), np.r_[np.ones(len(A), bool), np.zeros(len(B), bool)]))
    if not its: return None
    def st(it):
        num = den = 0.0
        for y, na, nb, lab in it: r = stats.rankdata(y); u = r[lab].sum() - na * (na + 1) / 2; w = 1 / (na + nb + 1); num += w * u; den += w * na * nb
        return num / den
    obs = st(its); perm = np.array([st([(y, na, nb, rng.permutation(lab)) for y, na, nb, lab in its]) for _ in range(nperm)])
    return obs, (np.sum(np.abs(perm - .5) >= abs(obs - .5)) + 1) / (nperm + 1), sum(x[1] for x in its), sum(x[2] for x in its), len(its)
def pooled_spear(df, col, nperm=5000):
    gs = [g for _, g in df.groupby("shell") if len(g) >= 15 and g[col].nunique() > 3]
    ys = [stats.rankdata(g.along_km) for g in gs]; xs = [g[col].values for g in gs]; n = sum(len(x) for x in xs)
    st = lambda xl: sum(len(x) * np.corrcoef(stats.rankdata(x), y)[0, 1] for x, y in zip(xl, ys)) / n
    obs = st(xs); perm = np.array([st([rng.permutation(x) for x in xs]) for _ in range(nperm)]); return obs, (np.sum(np.abs(perm) >= abs(obs)) + 1) / (nperm + 1), n

L = ["# Filtered Mann-Whitney tests v2 (filters of PROTOCOL_filtered_mann_whitney_addendum.md; exploratory)", "",
     "along_km at the window end; positive = behind the forecast = LESS drag than forecast (Li et al. direction for dosed / longer dose). P(A>B) > 0.5 = Li direction.", ""]
for SHIFT in (0, -24, 24):
    rows = []
    for _, s in S.iterrows():
        m = measure(s.sat_key, W0 + SHIFT * 3600, W1 + SHIFT * 3600)
        if m: rows.append(dict(sat_key=s.sat_key, name=s["name"], shell=s["shell"], umbra_seconds=s.umbra_seconds, dose_length_min=s.dose_length_min, along_km=m[0], post_method=m[1], burn_rise_m=m[2], burn_drop_m=m[3], altitude_km=m[4]))
    R = pd.DataFrame(rows); R["alt_ok"] = R.altitude_km >= 440; A_ = R[R.alt_ok]
    q1, q3 = np.nanpercentile(A_.burn_rise_m, [25, 75]); fence = q3 + 3 * (q3 - q1); d1, d3 = np.nanpercentile(A_.burn_drop_m, [25, 75]); fence_lo = d1 - 3 * (d3 - d1)
    R["burn_flag"] = R.alt_ok & ((R.burn_rise_m > fence) | (R.burn_drop_m < fence_lo)); R["fence_rise_m"], R["fence_drop_m"] = fence, fence_lo
    K = R[R.alt_ok & ~R.burn_flag].copy(); K["iqr_ok"] = iqr_ok(K.along_km.values)
    R = R.merge(K[["sat_key", "iqr_ok"]], on="sat_key", how="left"); R["iqr_ok"] = R.iqr_ok.astype("boolean").fillna(False).astype(bool)
    R["kept"] = R.alt_ok & ~R.burn_flag & R.iqr_ok; K = R[R.kept].copy(); K["dosed"] = np.where(K.umbra_seconds > 0, "dosed", "undosed")
    D = K[K.dosed == "dosed"].copy()
    for col, nm in (("dose_length_min", "len"), ("umbra_seconds", "dose")): D[nm] = np.where(D[col] > D[col].median(), "long", "short")
    if SHIFT == 0: R.to_csv(os.path.join(OUT, "filtered_mw_v2_per_satellite.csv"), index=False, float_format="%.6g")
    tag = "REAL (eclipse window)" if SHIFT == 0 else f"PLACEBO (window {SHIFT:+d} h)"
    L += [f"## {tag}", "", f"- measured {len(R)}; below 440 km {int((~R.alt_ok).sum())}; burn fences rise {fence:.0f} m / drop {fence_lo:.0f} m -> flagged {int(R.burn_flag.sum())} (dosed {int((R.burn_flag & (R.umbra_seconds > 0)).sum())}, undosed {int((R.burn_flag & (R.umbra_seconds == 0)).sum())}); "
          f"IQR backstop removed {int((R.alt_ok & ~R.burn_flag & ~R.iqr_ok).sum())}; kept {len(K)} (dosed {int((K.dosed == 'dosed').sum())}, undosed {int((K.dosed == 'undosed').sum())}); extended share {100 * (K.post_method == 'extended').mean():.0f}%", "",
          "| test | n A / n B | strata | stratified P(A>B) | perm p | plain P(A>B), p |", "|---|---|---|---|---|---|"]
    for lab, df, g, a, b in (("(a) dosed vs undosed", K, "dosed", "dosed", "undosed"), ("(b) long vs short dose LENGTH (dosed)", D, "len", "long", "short"), ("(b) high vs low umbra_seconds (dosed)", D, "dose", "long", "short")):
        r = strat_mw(df, g, a, b); A, B = df[df[g] == a].along_km, df[df[g] == b].along_km; u = stats.mannwhitneyu(A, B, alternative="two-sided")
        L.append(f"| {lab} | {len(A)} / {len(B)} | " + (f"{r[4]} | {r[0]:.3f} | {r[1]:.3g} |" if r else "0 | too few | |") + f" {u.statistic / (len(A) * len(B)):.3f}, {u.pvalue:.3g} |")
    for col in ("dose_length_min", "umbra_seconds"):
        r = pooled_spear(D, col); L.append(f"| (c) Spearman {col} vs along, dosed, within shell | n {r[2]} | | rho {r[0]:+.3f} | {r[1]:.3g} | |")
    L.append(""); print("\n".join(L[-10:]), flush=True)
open(os.path.join(OUT, "filtered_mw_v2_results.md"), "w").write("\n".join(L) + "\n")
