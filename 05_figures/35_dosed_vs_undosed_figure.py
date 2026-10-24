#!/usr/bin/env python3
"""Single Mann-Whitney figure: dosed vs undosed, eclipse window, after all filters (filtered_mw_v2_per_satellite.csv). Output: docs/dosed_vs_undosed_mann_whitney.png"""
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
import textwrap

rng = np.random.default_rng(1)
R = pd.read_csv(os.path.join(SP.ED, "filtered_mw_v2_per_satellite.csv"), dtype={"sat_key": str}); F = R[R.kept]
U, D = F[F.umbra_seconds == 0].along_km * 1000, F[F.umbra_seconds > 0].along_km * 1000
fig, ax = plt.subplots(figsize=(9, 7))
bp = ax.boxplot([U, D], widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=2.5))
for b, c in zip(bp["boxes"], ("#2E6F9E", "#D9782D")): b.set_facecolor(c); b.set_alpha(.6)
for i, (v, c) in enumerate(zip((U, D), ("#2E6F9E", "#D9782D")), 1): ax.scatter(i + rng.uniform(-.22, .22, len(v)), v, s=8, color=c, alpha=.45, zorder=3, edgecolors="none")
lim = np.percentile(np.r_[U, D], [1, 99]); pad = (lim[1] - lim[0]) * .1; ax.set_ylim(lim[0] - pad, lim[1] + pad); ax.axhline(0, color="#8C8C8C", linewidth=.8)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"no dose\nn={len(U)}\nmedian {U.median():.0f} m", f"dosed\nn={len(D)}\nmedian {D.median():.0f} m"], fontsize=11)
ax.set_ylabel("forecast gap along the orbit (m)\npre-dose forecast minus post-dose model\nhigher = satellite behind the forecast = LESS drag than forecast", fontsize=10)
ax.set_title("Did dosed satellites feel less drag than forecast?\n"
             "Mann-Whitney (stratified by shell): effect size A = 0.746, p ≤ 0.0002\n"
             "Placebo, same test on non-eclipse days:\nday before A = 0.462 (p = 0.25), day after A = 0.503 (p = 0.93)", loc="left", fontsize=10.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#E5E5E5", linewidth=.6, zorder=0); ax.set_axisbelow(True)
NOTE = "Filters: altitude >= 440 km, probable burns removed, IQR x3. Window 15:29-20:03 UTC, Aug 12. A = chance a dosed satellite has the larger gap (0.5 = no difference); p = p-value. View clipped to the 1st-99th percentile; statistics use all."
ax.text(0, -.19, textwrap.fill(NOTE, 105), transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#555555")
fig.savefig(os.path.join(SP.DOCS, "dosed_vs_undosed_mann_whitney.png"), dpi=170, bbox_inches="tight", facecolor="white")
