#!/usr/bin/env python3
"""Two-part whiteboard explainer (white background): 1) the two forecasts, forward and backward projection, with the formula;
2) what the gap means, with the sign stated plainly. Example times are from STARLINK-36880. Output (new): docs/forward_backward_explainer_v4.png
v4: same as v3 with the reviewer's wording ('the eclipse window', 'Backward extrapolation model', 'In the analyzed dataset')."""
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


INK, BLUE, ORNG, RED, GREY, GREEN = "#1F2A44", "#2E6F9E", "#D9782D", "#B03A2E", "#9A9A9A", "#2E7D32"
W0, W1, CP, CQ, SQ = 15 + 29 / 60, 20 + 3 / 60, 9 + 40 / 60, 22 + 13 / 60, 22 + 1 / 60
fig = plt.figure(figsize=(15, 13), facecolor="white"); gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=.25)

# ---------- Part 1
ax = fig.add_subplot(gs[0]); ax.set_xlim(8, 31); ax.set_ylim(-3.6, 10.3); ax.axis("off")
ax.set_title("1.  Two forecasts, compared at one moment (example satellite)", loc="left", fontsize=17, color=INK, fontweight="bold")
ax.annotate("", xy=(31, 1.2), xytext=(8, 1.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
for h in range(9, 31, 3): ax.plot([h, h], [1.0, 1.4], color=INK, lw=1.5); ax.text(h, .35, f"{h % 24:02d}:00" + (" Aug 13" if h >= 24 else ""), ha="center", fontsize=10, color=INK)
ax.axvspan(W0, W1, ymin=(1.2 + 3.6) / 13.9, ymax=1.0, color="#F2D6B8", alpha=.8)
ax.text((W0 + W1) / 2, 9.75, "the eclipse window, 12 Aug\n15:29 - 20:03\n(eclipse shadow passes)", ha="center", va="top", fontsize=11, color="#8A4B14")
ax.plot([W1, W1], [1.2, 8.3], color=RED, lw=2.5, ls="--"); ax.text(W1 + .15, 7.8, "compare the two forecasts here:\nend of window, 20:03", color=RED, fontsize=11.5, fontweight="bold", va="center")
ax.plot(CP, 6.2, "o", ms=13, color=BLUE); ax.text(CP, 6.9, "pre-dose forecast\nmade 09:40, before the eclipse", ha="center", fontsize=11, color=BLUE, fontweight="bold")
ax.add_patch(FancyArrowPatch((CP + .3, 6.2), (W1 - .05, 6.2), arrowstyle="-|>", mutation_scale=25, lw=3, color=BLUE))
ax.text((CP + W1) / 2, 4.9, "PROJECT FORWARD: read its position at 20:03\n(it knows nothing about the eclipse)", ha="center", fontsize=11, color=BLUE)
ax.plot(CQ, 3.8, "o", ms=13, color=ORNG); ax.text(CQ + .3, 4.35, "post-dose forecast made 22:13 (first point 22:01)", fontsize=11, color=ORNG, fontweight="bold")
ax.plot([SQ, CQ], [3.8, 3.8], color=ORNG, lw=3); ax.add_patch(FancyArrowPatch((CQ, 3.8), (31, 3.8), arrowstyle="-|>", mutation_scale=25, lw=3, color=ORNG))
ax.add_patch(FancyArrowPatch((SQ - .05, 3.8), (W1 + .05, 3.8), arrowstyle="-|>", mutation_scale=25, lw=3, color=ORNG, linestyle=(0, (4, 3))))
ax.text(W1 + .15, 2.85, "WORK BACKWARDS to 20:03 (this file has no data before 22:01)", fontsize=11, color=ORNG, fontweight="bold")
ax.text(19.6, -3.4,
        "Backward extrapolation model:  measure the gap between the two forecasts every minute over the post-dose file's first 8 hours, fit\n"
        r"$D(\tau) = c_0 + c_1\tau + c_2\tau^2 + c_3\cos(n\tau) + c_4\sin(n\tau)$" + "      ($\\tau$ = hours after 20:03,  $n$ = one orbit about every 94 minutes)\n"
        r"then set $\tau = 0$ to get the gap at 20:03:   $D(0) = c_0 + c_3$",
        ha="center", va="bottom", fontsize=12, color=INK, bbox=dict(boxstyle="round,pad=.6", fc="white", ec=GREY))

# ---------- Part 2
ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 20); ax.set_ylim(0, 9.6); ax.axis("off")
ax.set_title("2.  What the gap means at 20:03", loc="left", fontsize=17, color=INK, fontweight="bold")
ax.text(10, 9.3, "gap  =  (where the pre-dose forecast put the satellite)  minus  (where the post-dose forecast puts it),\nmeasured along the direction of travel",
        ha="center", va="top", fontsize=12.5, color=INK, fontweight="bold")
def case(x0, title, pre_x, post_x, sign, verdict, col, big):
    xx = np.linspace(x0, x0 + 8.6, 200); y0 = 4.4; f = lambda x: y0 - .03 * (x - (x0 + 4.3)) ** 2
    ax.plot(xx, f(xx), color=GREY, lw=4)
    ax.add_patch(FancyArrowPatch((x0 + .2, f(x0 + .2) - .5), (x0 + 1.7, f(x0 + 1.7) - .5), arrowstyle="-|>", mutation_scale=22, lw=2.3, color=GREY))
    ax.text(x0 + .2, f(x0 + .2) - 1.0, "direction of travel", fontsize=10, color=GREY)
    ax.plot(pre_x, f(pre_x), "s", ms=19, color=BLUE, zorder=5); ax.text(pre_x, f(pre_x) + .5, "pre-dose\nforecast\nsaid here", ha="center", fontsize=10.5, color=BLUE, fontweight="bold")
    ax.plot(post_x, f(post_x), "s", ms=19, color=ORNG, zorder=5); ax.text(post_x, f(post_x) + .5, "post-dose\nforecast:\nactually here", ha="center", fontsize=10.5, color=ORNG, fontweight="bold")
    ax.add_patch(FancyArrowPatch((min(pre_x, post_x), f(4.3 + x0) - .55), (max(pre_x, post_x), f(4.3 + x0) - .55), arrowstyle="<->", mutation_scale=18, lw=2.2, color=RED))
    ax.text((pre_x + post_x) / 2, f(4.3 + x0) - 1.05, f"gap = {sign}", ha="center", fontsize=12, color=RED, fontweight="bold")
    ax.text(x0 + 4.3, 7.4, title, ha="center", fontsize=13.5, color=col, fontweight="bold")
    ax.text(x0 + 4.3, 1.9, verdict, ha="center", va="top", fontsize=12, color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=.55", fc="#F1F8F1" if big else "#F7F7F7", ec=col, lw=2 if big else 1))
case(.5, "Satellite BEHIND its forecast", 5.9, 3.3, "+",
     "It did not get as far as forecast.\nGap is POSITIVE (+).\nIt felt LESS drag than forecast.\nThis is what an eclipse should cause.", GREEN, True)
case(10.9, "Satellite AHEAD of its forecast", 13.4, 16.0, "-",
     "It got further than forecast.\nGap is NEGATIVE (-).\nIt felt MORE drag than forecast.\n(Drag lowers the orbit; lower orbits are faster.)", RED, False)
fig.text(.5, .015, "In the analyzed dataset, the whole fleet ran slightly ahead of its forecasts (negative gaps), but dosed satellites were less far ahead (median -55 m) than satellites\n"
         "with no dose (median -138 m). Their gap was about 80 m more positive, so they felt LESS drag than the satellites that stayed out of the shadow.",
         ha="center", fontsize=11.5, color=INK, style="italic")
fig.savefig(os.path.join(SP.DOCS, "forward_backward_explainer_v4.png"), dpi=140, bbox_inches="tight", facecolor="white")
