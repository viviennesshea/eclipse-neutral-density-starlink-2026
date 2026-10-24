#!/usr/bin/env python3
"""Filter funnel for the dosed-vs-undosed ephemeris test. Outputs (new): docs/filter_funnel.png, docs/FILTER_PIPELINE.md"""
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
import pandas as pd


T = pd.read_csv(os.path.join(SP.RESULTS, "satellite_dosage.csv"), dtype={"sat_key": str})
R = pd.read_csv(os.path.join(SP.ED, "filtered_mw_v2_per_satellite.csv"), dtype={"sat_key": str})
dz = lambda d: (int((d.umbra_seconds > 0).sum()), int((d.umbra_seconds == 0).sum()))
s0 = T; s1 = T[T.has_ephemeris]; s2 = R; s3 = R[R.alt_ok]; s4 = s3[~s3.burn_flag]; s5 = s4[s4.iqr_ok]
steps0 = [("All target satellites", len(s0), None, "", "", "eclipse_targets.csv"),
         ("Have ephemeris (position data)", len(s1), dz(s1), "no ephemeris in any download", "", "results/satellite_dosage.csv"),
         ("Altitude >= 440 km", len(s3), dz(s3), "below 440 km (orbit-raising / low)", "mw2_fig1_altitude_filter.png", ""),
         ("No thruster burn detected", len(s4), dz(s4), "sudden rise or drop in orbit size", "mw2_fig2_burn_filter.png", ""),
         ("IQR x3 outlier check", len(s5), dz(s5), "gap far outside the fleet", "mw2_fig3_iqr_filter.png", "")]
steps = [x for x in steps0]
fig, ax = plt.subplots(figsize=(9, 7)); mx = steps[0][1]
for i, (lab, n, d, why, *_ ) in enumerate(steps):
    y = len(steps) - i; w = n / mx
    ax.barh(y, w, left=(1 - w) / 2, height=.7, color="#2E6F9E" if i < len(steps) - 1 else "#D9782D")
    ax.text(.5, y, f"{lab}\n{n:,}" + (f"  (dosed {d[0]:,} / no dose {d[1]:,})" if d else ""), ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(.4, len(steps) + .6); ax.axis("off")
ax.set_title("Filtering funnel: from all satellites to the ones tested (dosed vs no dose)", loc="left", fontsize=13)
fig.savefig(os.path.join(SP.DOCS, "filter_funnel.png"), dpi=160, bbox_inches="tight", facecolor="white")
L = ["# Filter pipeline: dosed vs undosed forecast-gap test", "", "| step | satellites left | removed | dosed / no dose | why removed | graph |", "|---|---|---|---|---|---|"]
for i, (lab, n, d, why, g, _) in enumerate(steps):
    L.append(f"| {i}. {lab} | {n:,} | {'' if i == 0 else f'{steps[i - 1][1] - n:,}'} | {'' if not d else f'{d[0]:,} / {d[1]:,}'} | {why} | {'`docs/' + g + '`' if g else ''} |")
L += ["", "Funnel image: `docs/filter_funnel.png`. Test result: `docs/dosed_vs_undosed_mann_whitney.png`."]
open(os.path.join(SP.DOCS, "FILTER_PIPELINE.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
