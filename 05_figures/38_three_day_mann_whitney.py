#!/usr/bin/env python3
"""Dosed vs no-dose Mann-Whitney, one chart per day (Aug 11 placebo, Aug 12 eclipse, Aug 13 placebo) and a combined chart on one scale.
Outputs (new): docs/mw_dosed_vs_undosed_aug11.png, _aug12.png, _aug13.png, docs/mw_dosed_vs_undosed_three_days.png"""
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

ED = os.path.join(SP.ED)
DAYS = [("aug11", "Aug 11 (placebo, no eclipse)", "0.462", "= 0.25"), ("aug12", "Aug 12 (eclipse day)", "0.746", "≤ 0.0002"), ("aug13", "Aug 13 (placebo, no eclipse)", "0.503", "= 0.93")]
data = {}
for k, *_ in DAYS:
    R = pd.read_csv(os.path.join(ED, f"filtered_mw_v2_per_satellite_{k}.csv"), dtype={"sat_key": str}); F = R[R.kept]
    data[k] = (F[F.umbra_seconds == 0].along_km.values * 1000, F[F.umbra_seconds > 0].along_km.values * 1000)
allv = np.concatenate([np.r_[a, b] for a, b in data.values()]); lim = np.percentile(allv, [1, 99]); pad = (lim[1] - lim[0]) * .08
def panel(ax, k, title, P, p):
    rng = np.random.default_rng(1); U, D = data[k]
    bp = ax.boxplot([U, D], widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=2.5))
    for b, c in zip(bp["boxes"], ("#2E6F9E", "#D9782D")): b.set_facecolor(c); b.set_alpha(.6)
    for i, (v, c) in enumerate(zip((U, D), ("#2E6F9E", "#D9782D")), 1): ax.scatter(i + rng.uniform(-.22, .22, len(v)), v, s=7, color=c, alpha=.4, zorder=3, edgecolors="none")
    ax.set_ylim(lim[0] - pad, lim[1] + pad); ax.axhline(0, color="#8C8C8C", linewidth=.8)
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"no dose\nn={len(U)}\nmedian {np.median(U):.0f} m", f"dosed\nn={len(D)}\nmedian {np.median(D):.0f} m"], fontsize=10.5)
    ax.set_title(f"{title}\neffect size A = {P}, p {p}", loc="left", fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=.6, zorder=0); ax.set_axisbelow(True)
YL = "forecast gap along the orbit (m)\nhigher = behind the forecast = LESS drag than forecast"
NOTE = "Same method and filters each day; only the time window moves (15:29-20:03 UTC), so slightly different numbers of satellites pass the filters each day. 'Dosed' = was in the eclipse shadow on Aug 12. A = chance a dosed satellite has the larger gap (0.5 = no difference); p = p-value (labels shuffled within each orbital tilt). Same y scale on all charts; view clipped to the 1st-99th percentile."
for k, t, P, p in DAYS:
    fig, ax = plt.subplots(figsize=(7.5, 7)); panel(ax, k, t, P, p); ax.set_ylabel(YL, fontsize=10)
    fig.text(.01, -.03, NOTE, fontsize=7.5, color="#555555", wrap=True); fig.savefig(os.path.join(SP.DOCS, f"mw_dosed_vs_undosed_{k}.png"), dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(19, 7), sharey=True)
for ax, (k, t, P, p) in zip(axs, DAYS): panel(ax, k, t, P, p)
axs[0].set_ylabel(YL, fontsize=10); fig.suptitle("Dosed vs no-dose satellites: a gap appears only on eclipse day", fontsize=14, y=1.02)
fig.text(.01, -.03, NOTE, fontsize=8.5, color="#555555"); fig.tight_layout()
fig.savefig(os.path.join(SP.DOCS, "mw_dosed_vs_undosed_three_days.png"), dpi=160, bbox_inches="tight", facecolor="white")
