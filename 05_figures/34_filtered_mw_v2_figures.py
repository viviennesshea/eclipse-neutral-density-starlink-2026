#!/usr/bin/env python3
"""
34_filtered_mw_v2_figures.py
============================
Figures for the v2 filters (results/ephemeris_drift/filtered_mw_v2_per_satellite.csv, eclipse window). New files in docs/:
  mw2_fig1_altitude_filter.png, mw2_fig2_burn_filter.png, mw2_fig3_iqr_filter.png, mw2_fig4_mann_whitney.png, mw2_fig5_dose_length.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DOCS = os.path.join(SP.DOCS); rng = np.random.default_rng(1)
R = pd.read_csv(os.path.join(SP.ED, "filtered_mw_v2_per_satellite.csv"), dtype={"sat_key": str})
BLUE, ORNG, RED, GREY = "#2E6F9E", "#D9782D", "#B03A2E", "#8C8C8C"
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=.6, zorder=0); ax.set_axisbelow(True)

# 1 altitude
fig, ax = plt.subplots(figsize=(11, 5.5)); bins = np.arange(250, 600, 5)
ax.hist(R[R.alt_ok].altitude_km, bins=bins, color=BLUE, label=f"kept (n={int(R.alt_ok.sum())})", zorder=3)
ax.hist(R[~R.alt_ok].altitude_km, bins=bins, color=RED, label=f"removed, below 440 km (n={int((~R.alt_ok).sum())})", zorder=3)
ax.axvline(440, color="black", linestyle="--"); ax.text(442, ax.get_ylim()[1] * .9, "cutoff 440 km", fontsize=10)
ax.set_xlabel("altitude before the window (km)"); ax.set_ylabel("number of satellites"); ax.set_title("Step 1: remove low satellites (orbit-raising new launches and others)", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper left"); clean(ax); fig.savefig(os.path.join(DOCS, "mw2_fig1_altitude_filter.png"), dpi=160, bbox_inches="tight", facecolor="white")

# 2 burn
A = R[R.alt_ok]; fr, fd = A.fence_rise_m.iloc[0], A.fence_drop_m.iloc[0]; f = A.burn_flag
fig, ax = plt.subplots(figsize=(9, 7.5)); ax.set_xscale("symlog", linthresh=50); ax.set_yscale("symlog", linthresh=50)
xmax, ymin = max(A.burn_drop_m.max(), 0) + 50, min(A.burn_rise_m.min(), 0) - 50
ax.add_patch(plt.Rectangle((fd, ymin), xmax - fd, fr - ymin, color="#E8F0F7", zorder=0))
ax.scatter(A[~f].burn_drop_m, A[~f].burn_rise_m, s=10, color=BLUE, label=f"normal, kept (n={int((~f).sum())})", zorder=3)
ax.scatter(A[f].burn_drop_m, A[f].burn_rise_m, s=24, color=RED, label=f"probable burn, removed (n={int(f.sum())})", zorder=4)
ax.axhline(fr, color="black", linestyle="--"); ax.axvline(fd, color="black", linestyle="--")
ax.text(xmax, fr * 1.15, f"rise limit +{fr:.0f} m", fontsize=9, ha="right", va="bottom"); ax.text(fd * 1.1, ax.get_ylim()[1] * .6, f"drop limit {fd:.0f} m", fontsize=9, ha="right", va="top", rotation=90)
ax.set_xlabel("largest DROP in orbit size within 3 hours (m)"); ax.set_ylabel("largest RISE in orbit size within 3 hours (m)")
ax.set_title("Step 2: remove probable thruster burns\nEach dot is a satellite. Drag shrinks the orbit slowly and steadily;\na sudden rise or drop means a burn. Shaded = normal range (IQR x3 limits). Log-scaled axes.", loc="left", fontsize=11)
ax.legend(frameon=False, loc="lower left"); clean(ax); fig.savefig(os.path.join(DOCS, "mw2_fig2_burn_filter.png"), dpi=160, bbox_inches="tight", facecolor="white")

# 3 IQR
K = R[R.alt_ok & ~R.burn_flag].sort_values("along_km").reset_index(drop=True); a1, a3 = K.along_km.quantile([.25, .75]); lo, hi = a1 - 3 * (a3 - a1), a3 + 3 * (a3 - a1)
fig, ax = plt.subplots(figsize=(12, 6.5)); x = np.arange(len(K)); ok = K.iqr_ok.values
ax.set_yscale("symlog", linthresh=0.05); ax.axhspan(lo, hi, color="#E8F0F7", zorder=0)
ax.scatter(x[ok], K.along_km[ok], s=7, color=BLUE, label=f"kept (n={ok.sum()})", zorder=3); ax.scatter(x[~ok], K.along_km[~ok], s=14, color=RED, label=f"removed (n={(~ok).sum()})", zorder=3)
ax.axhline(hi, color="black", linestyle="--"); ax.text(len(K) * .02, hi * 1.25, f"upper cutoff {hi:+.3f} km", fontsize=10)
ax.axhline(lo, color="black", linestyle="--"); ax.text(len(K) * .98, lo * 1.25, f"lower cutoff {lo:+.3f} km", fontsize=10, ha="right", va="top")
ax.axhline(0, color=GREY, linewidth=.8); ax.set_xlim(0, len(K))
ax.set_xlabel(f"the {len(K)} satellites left after steps 1 and 2, sorted by the gap"); ax.set_ylabel("along-track gap (km, symmetric-log scale)\npositive = behind the forecast = less drag")
ax.set_title("Step 3: IQR x3 outlier removal on the along-track gap", loc="left", fontsize=13); ax.legend(frameon=False, loc="upper left"); clean(ax)
fig.savefig(os.path.join(DOCS, "mw2_fig3_iqr_filter.png"), dpi=160, bbox_inches="tight", facecolor="white")

# 4 Mann-Whitney
F = R[R.kept].copy(); F["dosed"] = F.umbra_seconds > 0; D = F[F.dosed]; mlen, mdose = D.dose_length_min.median(), D.umbra_seconds.median()
P = [("Dosed vs undosed", F[~F.dosed].along_km, F[F.dosed].along_km, "undosed", "dosed", "P(dosed > undosed) = 0.746, p = 0.0002", "day before: 0.462 (p 0.25) | day after: 0.503 (p 0.93)"),
     (f"Dose length: short vs long (split at {mlen:.0f} min)", D[D.dose_length_min <= mlen].along_km, D[D.dose_length_min > mlen].along_km, "short", "long", "P(long > short) = 0.636, p = 0.0002", "day before: 0.513 (p 0.62) | day after: 0.663 (p 0.0002)"),
     (f"Umbra-seconds: low vs high (split at {mdose:.0f} s)", D[D.umbra_seconds <= mdose].along_km, D[D.umbra_seconds > mdose].along_km, "low", "high", "P(high > low) = 0.533, p = 0.15", "day before: 0.520 (p 0.41) | day after: 0.587 (p 0.0006)")]
fig, axs = plt.subplots(1, 3, figsize=(18, 6.3), sharey=True); lim = F.along_km.quantile([.01, .99]).values; pad = (lim[1] - lim[0]) * .1
for ax, (t, a, b, la, lb, res, plac) in zip(axs, P):
    bp = ax.boxplot([a, b], widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=2))
    for bx, c in zip(bp["boxes"], (BLUE, ORNG)): bx.set_facecolor(c); bx.set_alpha(.6)
    for i, (v, c) in enumerate(zip((a, b), (BLUE, ORNG)), 1): ax.scatter(i + rng.uniform(-.22, .22, len(v)), v, s=6, color=c, alpha=.4, zorder=3, edgecolors="none")
    ax.axhline(0, color=GREY, linewidth=.8); ax.set_ylim(lim[0] - pad, lim[1] + pad); ax.set_xticks([1, 2]); ax.set_xticklabels([f"{la}\nn={len(a)}", f"{lb}\nn={len(b)}"], fontsize=11)
    ax.set_title(f"{t}\n{res}\nplacebo {plac}", loc="left", fontsize=10); clean(ax)
axs[0].set_ylabel("along-track gap (km)\npositive = behind the forecast = less drag")
fig.suptitle(f"Mann-Whitney tests, eclipse window, after all filters (n={len(F)}). P > 0.5 = less drag for the right-hand group. p: shell-stratified permutation.", fontsize=12, y=1.04)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "mw2_fig4_mann_whitney.png"), dpi=150, bbox_inches="tight", facecolor="white")

# 5 dose length
fig, ax = plt.subplots(figsize=(10, 6)); ax.scatter(D.dose_length_min, D.along_km, s=9, color=ORNG, alpha=.45, edgecolors="none")
e = np.quantile(D.dose_length_min, np.linspace(0, 1, 9)); g = D.assign(b=pd.cut(D.dose_length_min, e, include_lowest=True)).groupby("b", observed=True).agg(x=("dose_length_min", "median"), y=("along_km", "median"))
ax.plot(g.x, g.y, "-o", color="black", linewidth=2, label="median per eighth of dose length"); ax.axhline(0, color=GREY, linewidth=.8); ax.set_ylim(lim[0] - pad, lim[1] + pad)
ax.set_xlabel("total dose length (minutes in shadow)"); ax.set_ylabel("along-track gap (km)\npositive = behind the forecast = less drag")
ax.set_title(f"Dose length vs along-track gap, dosed satellites (n={len(D)})\nwithin-shell Spearman +0.16 (p 0.0002); placebo: day before +0.01, day after +0.13", loc="left", fontsize=11)
ax.legend(frameon=False, loc="lower right"); clean(ax); fig.savefig(os.path.join(DOCS, "mw2_fig5_dose_length.png"), dpi=160, bbox_inches="tight", facecolor="white")
print("done")
