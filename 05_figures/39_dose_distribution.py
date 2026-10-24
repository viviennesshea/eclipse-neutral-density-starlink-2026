#!/usr/bin/env python3
"""Distribution of dose (umbra-seconds) for the 862 satellites in the dosed-vs-undosed test (filtered_mw_v2_per_satellite_aug12.csv, kept).
Horizontal histogram: umbra-seconds on the y axis (zero dose as its own bar), number of satellites on the x axis. Output (new): docs/dose_distribution.png"""
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


R = pd.read_csv(os.path.join(SP.ED, "filtered_mw_v2_per_satellite_aug12.csv"), dtype={"sat_key": str}); d = R[R.kept].umbra_seconds.values
z, pos = int((d == 0).sum()), d[d > 0]; W = 50; edges = np.arange(0, pos.max() + W, W); cnt, _ = np.histogram(pos, edges)
fig, ax = plt.subplots(figsize=(9, 9))
ax.barh(edges[:-1] + W / 2, cnt, height=W * .9, color="#D9782D", label=f"dosed (n={len(pos)})")
ax.barh(-W / 2 - 10, z, height=W * .9, color="#2E6F9E", label=f"zero dose (n={z})")
ax.text(z + 3, -W / 2 - 10, f"{z}", va="center", fontsize=9)
med = np.median(pos); ax.axhline(med, color="black", linestyle="--", linewidth=1); ax.text(ax.get_xlim()[1] * .98, med + 8, f"median of dosed: {med:.0f} s", ha="right", fontsize=9)
ax.set_yticks(list(np.arange(0, pos.max() + 100, 100)) + [-W / 2 - 10]); ax.set_yticklabels([f"{int(v)}" for v in np.arange(0, pos.max() + 100, 100)] + ["0 (none)"])
ax.set_ylabel("dose received (umbra-seconds)"); ax.set_xlabel("number of satellites")
ax.set_title(f"Dose received by the {len(d)} satellites in the test\nbars = {W}-second bands; max {pos.max():.0f} s", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#E5E5E5", linewidth=.6); ax.set_axisbelow(True)
fig.savefig(os.path.join(SP.DOCS, "dose_distribution.png"), dpi=160, bbox_inches="tight", facecolor="white")
print(f"n {len(d)} | zero {z} | dosed {len(pos)} | median dosed {med:.0f} | max {pos.max():.0f} | under 10 s: {int((pos < 10).sum())}")
