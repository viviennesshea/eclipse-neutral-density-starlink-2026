#!/usr/bin/env python3
"""Five views of the forecast gap as a function of dose (filtered satellites, filtered_mw_v2_per_satellite_{aug11,aug12,aug13}.csv, kept rows).
Outputs (new, docs/): gap_vs_dose_1_scatter.png, _2_boxes.png, _3_three_days.png, _4_boxes_by_shell.png, _5_cumulative.png"""
import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import sys
sys.dont_write_bytecode = True
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ED = os.path.join(SP.ED); DOCS = os.path.join(SP.DOCS)
rng = np.random.default_rng(1)
BANDS = [("zero", 0, 0), ("1-150 s\n(grazing)", 0, 150), ("150-500 s", 150, 500), ("500-700 s", 500, 700), ("700+ s", 700, 1e9)]
COLS = ["#2E6F9E", "#F2C29B", "#E8A26A", "#D9782D", "#A34F12"]
def load(day):
    R = pd.read_csv(os.path.join(ED, f"filtered_mw_v2_per_satellite_{day}.csv"), dtype={"sat_key": str}); F = R[R.kept].copy(); F["gap_m"] = F.along_km * 1000
    F["band"] = [0 if u == 0 else next(i for i, (_, lo, hi) in enumerate(BANDS) if i and lo < u <= hi) for u in F.umbra_seconds]; F["shell"] = F["shell"].astype(str); return F
D = {d: load(d) for d in ("aug11", "aug12", "aug13")}; A = D["aug12"]
lim = np.percentile(A.gap_m, [1, 99]); pad = (lim[1] - lim[0]) * .08; YL = (lim[0] - pad, lim[1] + pad)
YLAB = "forecast gap along the orbit (m)\nhigher = behind the forecast = LESS drag"
def med_ci(v, n=2000):
    v = np.asarray(v); bs = np.median(rng.choice(v, (n, len(v))), axis=1); return np.median(v), *np.percentile(bs, [2.5, 97.5])
def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#E5E5E5", linewidth=.6, zorder=0); ax.set_axisbelow(True); ax.axhline(0, color="#8C8C8C", linewidth=.8) if grid == "y" else None
NOTE = "Aug 12 eclipse window, 862 satellites after filters. View clipped to the 1st-99th percentile."

# 1 scatter + binned medians
fig, ax = plt.subplots(figsize=(11, 6.5)); z = A[A.umbra_seconds == 0]; p = A[A.umbra_seconds > 0]
ax.scatter(rng.uniform(-40, -10, len(z)), z.gap_m, s=8, color=COLS[0], alpha=.45, label=f"zero dose (n={len(z)})", edgecolors="none")
ax.scatter(p.umbra_seconds, p.gap_m, s=8, color=COLS[3], alpha=.4, label=f"dosed (n={len(p)})", edgecolors="none")
e = np.quantile(p.umbra_seconds, np.linspace(0, 1, 9)); pts = [(-25, *med_ci(z.gap_m))]
for lo, hi in zip(e[:-1], e[1:]):
    s = p[(p.umbra_seconds >= lo) & (p.umbra_seconds <= hi)]; pts.append((s.umbra_seconds.median(), *med_ci(s.gap_m)))
pts = np.array(pts); ax.errorbar(pts[:, 0], pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]], fmt="-o", color="black", capsize=3, linewidth=2, zorder=5, label="median (95% interval): zero dose, then each eighth of dosed")
ax.set_ylim(*YL); ax.set_xlabel("dose received (umbra-seconds)"); ax.set_ylabel(YLAB); ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("Forecast gap vs dose, every satellite", loc="left", fontsize=12); clean(ax); fig.text(.01, -.02, NOTE, fontsize=8, color="#555555")
fig.savefig(os.path.join(DOCS, "gap_vs_dose_1_scatter.png"), dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 2 boxes by band
def boxes(ax, F, title):
    data = [F[F.band == i].gap_m.values for i in range(len(BANDS))]; keep = [i for i, v in enumerate(data) if len(v) >= 3]
    bp = ax.boxplot([data[i] for i in keep], positions=range(len(keep)), widths=.6, patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=2))
    for b, i in zip(bp["boxes"], keep): b.set_facecolor(COLS[i]); b.set_alpha(.75)
    for j, i in enumerate(keep): ax.scatter(j + rng.uniform(-.25, .25, len(data[i])), data[i], s=5, color=COLS[i], alpha=.45, zorder=3, edgecolors="none")
    ax.set_xticks(range(len(keep))); ax.set_xticklabels([f"{BANDS[i][0]}\nn={len(data[i])}\nmedian {np.median(data[i]):.0f} m" for i in keep], fontsize=9)
    ax.set_ylim(*YL); ax.set_title(title, loc="left", fontsize=12); clean(ax)
fig, ax = plt.subplots(figsize=(11, 6.5)); boxes(ax, A, "Forecast gap by dose band"); ax.set_ylabel(YLAB); ax.set_xlabel("dose received")
fig.text(.01, -.03, NOTE, fontsize=8, color="#555555"); fig.savefig(os.path.join(DOCS, "gap_vs_dose_2_boxes.png"), dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 3 three days
fig, ax = plt.subplots(figsize=(11, 6.5)); style = {"aug11": ("Aug 11 (placebo)", "#8C8C8C", "--"), "aug12": ("Aug 12 (eclipse)", "#D9782D", "-"), "aug13": ("Aug 13 (placebo)", "#555555", ":")}
for k, (lab, c, ls) in style.items():
    F = D[k]; F = F.assign(g=F.gap_m - F[F.umbra_seconds == 0].gap_m.median())
    pts = np.array([(i, *med_ci(F[F.band == i].g)) for i in range(len(BANDS)) if (F.band == i).sum() >= 3]); off = {"aug11": -.08, "aug12": 0, "aug13": .08}[k]
    ax.errorbar(pts[:, 0] + off, pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]], fmt="o", linestyle=ls, color=c, capsize=3, linewidth=2 if k == "aug12" else 1.4, label=lab)
ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([b[0] for b in BANDS]); ax.set_xlabel("dose received (Aug 12 eclipse)")
ax.set_ylabel("median forecast gap relative to zero-dose satellites (m)\nhigher = LESS drag than zero-dose satellites"); ax.legend(frameon=False)
ax.set_title("Dose band vs forecast gap, eclipse day and the two placebo days\n(each day's zero-dose median set to 0; bars = 95% interval)", loc="left", fontsize=12); clean(ax)
fig.text(.01, -.02, "Same satellites and filters each day; only the time window moves. Dose bands always refer to the Aug 12 eclipse.", fontsize=8, color="#555555")
fig.savefig(os.path.join(DOCS, "gap_vs_dose_3_three_days.png"), dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 4 boxes by shell
fig, axs = plt.subplots(1, 2, figsize=(17, 6.5), sharey=True)
for ax, sh in zip(axs, ("43", "53")): boxes(ax, A[A["shell"] == sh], f"Shell {sh} only")
axs[0].set_ylabel(YLAB); fig.suptitle("Forecast gap by dose band, within each inclination shell (shells 70 and 97 have no zero-dose satellites)", fontsize=12, y=1.02)
fig.text(.01, -.03, NOTE, fontsize=8, color="#555555"); fig.savefig(os.path.join(DOCS, "gap_vs_dose_4_boxes_by_shell.png"), dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)

# 5 cumulative
fig, ax = plt.subplots(figsize=(11, 6.5))
for i, (lab, *_ ) in enumerate(BANDS):
    v = np.sort(A[A.band == i].gap_m.values)
    if len(v) >= 3: ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLS[i] if i else COLS[0], linewidth=2.2, label=f"{lab.replace(chr(10), ' ')} (n={len(v)})")
ax.set_xlim(*YL); ax.axvline(0, color="#8C8C8C", linewidth=.8); ax.set_xlabel("forecast gap along the orbit (m); further right = LESS drag"); ax.set_ylabel("share of satellites at or below this gap")
ax.set_title("Cumulative distribution of the forecast gap by dose band\n(a curve further to the right = that band met less drag)", loc="left", fontsize=12); ax.legend(frameon=False, loc="lower right"); clean(ax, "x")
fig.text(.01, -.02, NOTE, fontsize=8, color="#555555"); fig.savefig(os.path.join(DOCS, "gap_vs_dose_5_cumulative.png"), dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
for k in D: print(k, D[k].groupby("band").gap_m.agg(["size", "median"]).round(0).T.to_dict())
