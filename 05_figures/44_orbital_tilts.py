#!/usr/bin/env python3
"""Explainer for orbital tilts (inclination shells): side view of Earth with the four tilts, and dosed / no-dose counts per tilt in the final test.
Output (new): docs/orbital_tilts.png"""
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
from matplotlib.patches import Circle, Arc
import numpy as np
import pandas as pd


R = pd.read_csv(os.path.join(SP.ED, "filtered_mw_v2_per_satellite_aug12.csv"), dtype={"sat_key": str}); F = R[R.kept].copy()
F["shell"] = F["shell"].astype(str); F["dosed"] = F.umbra_seconds > 0
cnt = F.groupby(["shell", "dosed"]).size().unstack(fill_value=0).reindex(["43", "53", "70", "97"]).fillna(0).astype(int)
INK, GREY, BLUE, ORNG = "#1F2A44", "#9A9A9A", "#2E6F9E", "#D9782D"
COL = {"43": "#6A4C93", "53": "#1982C4", "70": "#8AC926", "97": "#FF595E"}
fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), facecolor="white", gridspec_kw={"width_ratios": [1.05, 1]})

ax = axs[0]; ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.7); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((0, 0), 1, color="#DCE8F2", zorder=1)); ax.add_patch(Circle((0, 0), 1, fill=False, ec="#8FB3D0", lw=1.5, zorder=2))
ax.plot([-1.55, 1.55], [0, 0], color=GREY, lw=1.2, ls="--", zorder=2); ax.text(-1.6, .04, "equator", fontsize=10, color=GREY, ha="right", va="bottom")
ax.plot([0, 0], [-1.25, 1.35], color=GREY, lw=1, zorder=2); ax.text(.04, 1.3, "North Pole", ha="left", fontsize=10, color=GREY)
r = 1.28
for sh, inc in (("43", 43), ("53", 53), ("70", 70), ("97", 97)):
    t = np.radians(inc); ax.plot([-r * np.cos(t), r * np.cos(t)], [-r * np.sin(t), r * np.sin(t)], color=COL[sh], lw=3.2, zorder=3, solid_capstyle="round")
    ax.plot(np.cos(t), np.sin(t), "o", ms=8, color=COL[sh], zorder=4)
    ax.text(r * np.cos(t) + (.06 if inc < 90 else -.06), r * np.sin(t) + .05, f"{inc}°", color=COL[sh], fontsize=14, fontweight="bold", ha="left" if inc < 90 else "right")
ax.add_patch(Arc((0, 0), .55, .55, theta1=0, theta2=43, color=COL["43"], lw=1.5)); ax.text(.33, .1, "tilt", fontsize=9.5, color=COL["43"])
ax.set_title("Each group of Starlink satellites orbits at its own tilt", loc="left", fontsize=14, color=INK, fontweight="bold")
ax.text(0, -1.42, "An orbit's tilt (inclination) is its angle to the equator, and it sets how far north and south the satellite goes.\n"
        "The dots mark the farthest latitude each tilt reaches: 43° N, 53° N, 70° N, and about 83° N for 97°\n"
        "(a near-polar orbit tilted just past 90°, so it circles slightly backwards). Drawn edge-on; not to scale.",
        ha="center", va="top", fontsize=10.5, color=INK)

ax = axs[1]; x = np.arange(4); w = .38
b1 = ax.bar(x - w / 2, cnt.get(False, 0), w, color=BLUE, label="no dose"); b2 = ax.bar(x + w / 2, cnt.get(True, 0), w, color=ORNG, label="dosed")
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 6, f"{int(b.get_height())}", ha="center", fontsize=11, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"{s}°" for s in cnt.index], fontsize=13)
for lab, s in zip(ax.get_xticklabels(), cnt.index): lab.set_color(COL[s]); lab.set_fontweight("bold")
ax.set_xlabel("orbital tilt", fontsize=12, color=INK); ax.set_ylabel("satellites in our test", fontsize=12, color=INK)
ax.legend(frameon=False, fontsize=11, loc="upper right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#E5E5E5", lw=.6); ax.set_axisbelow(True)
ax.set_title("Why we only compare satellites with the same tilt", loc="left", fontsize=14, color=INK, fontweight="bold")
ax.text(.02, -.1, "Satellites at different tilts pass through different parts of the sky and are forecast differently.\n"
        "Almost all no-dose satellites fly at 43°, and the 70° and 97° groups have none, so comparing\n"
        "across tilts would mix up the eclipse with the tilt. Our test compares dosed and no-dose\n"
        "satellites within each tilt, then combines the results.", transform=ax.transAxes, va="top", fontsize=10.5, color=INK)
fig.savefig(os.path.join(SP.DOCS, "orbital_tilts.png"), dpi=160, bbox_inches="tight", facecolor="white")
print(cnt)
