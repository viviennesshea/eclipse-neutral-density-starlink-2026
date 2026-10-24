#!/usr/bin/env python3
"""The finding in one picture: where typical dosed and no-dose satellites ended up relative to their own forecasts at 20:03 (medians from
filtered_mw_v2_per_satellite_aug12.csv, kept rows). Output (new): docs/finding_in_one_picture_v3.png
v3: title starts 'Visual summary:' (reviewer wording)."""
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
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


R = pd.read_csv(os.path.join(SP.ED, "filtered_mw_v2_per_satellite_aug12.csv"), dtype={"sat_key": str}); F = R[R.kept]
nd = -F[F.umbra_seconds == 0].along_km.median() * 1000; ds = -F[F.umbra_seconds > 0].along_km.median() * 1000   # metres AHEAD of forecast
diff = nd - ds; ms = diff / 7.6
INK, BLUE, ORNG, RED, GREY = "#1F2A44", "#2E6F9E", "#D9782D", "#B03A2E", "#9A9A9A"
fig, ax = plt.subplots(figsize=(13, 4.4), facecolor="white"); ax.set_xlim(-60, 200); ax.set_ylim(-1.6, 2.4); ax.axis("off")
xx = np.linspace(-55, 195, 300); f = lambda x: .9 - 3.5e-5 * (x - 70) ** 2; ax.plot(xx, f(xx), color=GREY, lw=4, zorder=1)
ax.add_patch(FancyArrowPatch((-45, f(-45) - .55), (-10, f(-10) - .55), arrowstyle="-|>", mutation_scale=22, lw=2.3, color=GREY)); ax.text(-27, f(-27) - 1.05, "direction of travel", fontsize=11, color=GREY, ha="center")
ax.plot(0, f(0), "o", ms=20, mfc="white", mec=INK, mew=2.5, zorder=4); ax.text(0, f(0) + .75, "where the pre-dose\nforecast said", ha="center", fontsize=12, color=INK, fontweight="bold")
ax.plot(ds, f(ds), "o", ms=18, color=ORNG, zorder=5); ax.text(ds, f(ds) + .75, f"typical DOSED\nsatellite: {ds:.0f} m ahead", ha="center", fontsize=12, color=ORNG, fontweight="bold")
ax.plot(nd, f(nd), "o", ms=18, color=BLUE, zorder=5); ax.text(nd, f(nd) + .75, f"typical NO-DOSE\nsatellite: {nd:.0f} m ahead", ha="center", fontsize=12, color=BLUE, fontweight="bold")
y = min(f(ds), f(nd)) - .55
ax.plot([ds, nd], [y, y], color=RED, lw=2.5); [ax.plot([x, x], [y - .15, y + .15], color=RED, lw=2.5) for x in (ds, nd)]
ax.text((ds + nd) / 2, y - .55, f"{diff:.0f} m", ha="center", fontsize=20, color=RED, fontweight="bold")
ax.text((ds + nd) / 2, y - 1.05, f"about {ms:.0f} milliseconds of flight at 7.6 km/s", ha="center", fontsize=11.5, color=RED)
ax.set_title("Visual summary: at 20:03 UTC, dosed satellites were about 80 m further back\nalong their path than satellites with no dose",
             loc="left", fontsize=13.5, color=INK, fontweight="bold")
fig.text(.5, .02, "Positions are measured against each satellite's own pre-dose forecast (medians of 701 dosed and 161 no-dose satellites). Both groups ran ahead of their\n"
         "forecasts, which means a little more drag than forecast for the whole fleet; the dosed satellites ran less far ahead, meaning they felt LESS drag. Not to scale vertically.",
         ha="center", fontsize=10, color="#555555", style="italic")
fig.savefig(os.path.join(SP.DOCS, "finding_in_one_picture_v3.png"), dpi=160, bbox_inches="tight", facecolor="white")
print(f"no dose {nd:.1f} m ahead | dosed {ds:.1f} m ahead | diff {diff:.1f} m | {ms:.1f} ms")
