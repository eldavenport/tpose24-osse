"""Presentation figure: the three array shapes, one symmetric 1° example each.

Draws a single 1° symmetric diamond / square / hexagon (the equator single-cell
configs from experiment_1) side by side, moorings = squares, gliders = circles.
Moorings are the *actual* TAO array locations (0, ±1, ±2 °N at 140 W, from
tao_moorings.json); every other array point is a glider — so a 1° equator cell
has a single mooring at its center (0 °N) and gliders at its vertices. There is
no physical x-axis unit — all cells sit at 140 W and the columns are only spread
out for visibility; the y axis is latitude. Shapes are colored by the results
palette (diamond blue, square red, hexagon green). Axes are equal aspect so the
shapes render true.

    python run_shape_demo.py  ->  summary_figs/shape_demo.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Circle
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
CFG_ROOT = HERE.parent / "experiment_1" / "configs"
CFG = CFG_ROOT / "equator"

# match the equator/heat-flux shape colors: diamond blue, square red, hexagon green
SHAPE_COLOR = {"diamond": "#1f77b4", "square": "#d62728", "hexagon": "#2ca02c"}
# columns left -> right, and the symmetric 1° single-cell config for each
SHAPES = ["diamond", "square", "hexagon"]
CONFIG = {
    "diamond": "equator_1deg_w0.5.json",
    "square": "equator_sq1deg_w0.5.json",
    "hexagon": "equator_hex1deg_w0.5.json",
}
COL_X = {"diamond": 0.0, "square": 2.0, "hexagon": 4.0}  # arbitrary column spacing
MOORING_LON = 220.0  # central meridian (140 W); only used to center the columns
# actual TAO array locations — an array point is a mooring only if it sits on one
TAO = {(round(la, 3), round(lo, 3))
       for la, lo in json.load(open(HERE.parent.parent / "tao_moorings.json"))["positions"]}


def is_mooring(lat, lon):
    return (round(lat, 3), round(lon, 3)) in TAO


def fig_shapes():
    """One symmetric 1° example of each shape, side by side."""
    fig, ax = plt.subplots(figsize=(11, 5))

    for shape in SHAPES:
        color = SHAPE_COLOR[shape]
        pos = [tuple(p) for p in json.load(open(CFG / CONFIG[shape]))["positions"]]
        # plot coords: longitude offset from the central meridian, shifted to this
        # shape's column; latitude is the real (physical) y value
        xy = np.array([((lon - MOORING_LON) + COL_X[shape], la) for la, lon in pos])
        # outline = convex-hull boundary (the interior center point stays inside)
        hull = ConvexHull(xy).vertices
        ring = np.append(hull, hull[0])
        ax.add_patch(Polygon(xy[hull], closed=True, facecolor=color,
                             edgecolor="none", alpha=0.15, zorder=1))
        ax.plot(xy[ring, 0], xy[ring, 1], color=color, lw=2.0, zorder=2)
        # TAO moorings = squares; every other array point = glider circle
        for (la, lon), (x, y) in zip(pos, xy):
            moored = is_mooring(la, lon)
            ax.scatter(x, y, s=130 if moored else 100, color=color,
                       marker="s" if moored else "o",
                       edgecolor="k", linewidth=0.9, zorder=3)
        # column header
        ax.text(COL_X[shape], 0.72, shape.capitalize(), ha="center", va="bottom",
                fontsize=18, fontweight="bold", color=color)

    ax.set_ylabel("Latitude (°N)", fontweight="bold", fontsize=16)
    ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax.set_ylim(-0.75, 0.9)
    ax.set_xlim(-0.9, 4.9)
    ax.set_aspect("equal")        # so the 1° shapes render true (square looks square)
    ax.set_xticks([])             # no physical x-axis unit (all cells are at 140 W)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    # marker legend (shape is conveyed by the column headers / color)
    handles = [
        Line2D([], [], marker="s", color="w", markerfacecolor="0.5",
               markeredgecolor="k", markersize=12, label="mooring"),
        Line2D([], [], marker="o", color="w", markerfacecolor="0.5",
               markeredgecolor="k", markersize=11, label="glider"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.02), fontsize=14)

    fig.tight_layout()
    out = HERE / "summary_figs" / "shape_demo.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


# alternating light/dark blue so adjacent (overlapping) diamonds are easy to tell apart
DIAMOND_SHADES = ["#1f77b4", "#9ecae1"]


def fig_diamond_latitudes():
    """Every diamond center we test, stacked by latitude, TAO moorings marked.

    Centers -1.5 ... +1.5 (from the base + `_mid` shift diamond configs). The
    1° cells overlap in latitude, so adjacent cells alternate light/dark blue.
    Moorings are the real TAO locations (0, ±1, ±2 °N) — they land on the N/S
    tips of the ±0.5-centered cells and at the centers of the integer-centered
    cells; every other vertex is a glider.
    """
    cells = {}
    for f in ["shift/shift_w0.5.json", "shift/shift_w0.5_mid.json"]:
        for c in json.load(open(CFG_ROOT / f))["cells"]:
            cells[round(c["center_lat"], 3)] = [tuple(p) for p in c["positions"]]
    cells = dict(sorted(cells.items()))

    fig, ax = plt.subplots(figsize=(4.6, 9))
    ax.axvline(0.0, color="0.7", lw=1.0, ls=":", zorder=0)  # the TAO meridian (140 W)

    # diamond outlines, alternating shade; collect all vertices for the markers
    all_pos = set()
    for i, (clat, pos) in enumerate(cells.items()):
        shade = DIAMOND_SHADES[i % 2]
        xy = np.array([(lon - MOORING_LON, la) for la, lon in pos])
        hull = ConvexHull(xy).vertices
        ax.add_patch(Polygon(xy[hull], closed=True, facecolor=shade,
                             edgecolor="none", alpha=0.45, zorder=1))
        ring = np.append(hull, hull[0])
        ax.plot(xy[ring, 0], xy[ring, 1], color=shade, lw=2.0, zorder=2)
        all_pos.update(pos)

    # markers: TAO moorings = squares, gliders = circles (deduped across cells)
    for la, lon in sorted(all_pos):
        moored = is_mooring(la, lon)
        ax.scatter(lon - MOORING_LON, la, s=150 if moored else 90, color="k",
                   marker="s" if moored else "o", zorder=4)

    ax.set_ylabel("Latitude (°N)", fontweight="bold", fontsize=16)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_ylim(-2.4, 2.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_aspect("equal")        # so the 1° diamonds render true
    ax.set_xticks([])             # no physical x-axis unit (all cells are at 140 W)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D([], [], marker="s", color="w", markerfacecolor="k",
               markeredgecolor="k", markersize=12, label="mooring"),
        Line2D([], [], marker="o", color="w", markerfacecolor="k",
               markeredgecolor="k", markersize=11, label="glider"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.0), fontsize=14)

    fig.tight_layout()
    out = HERE / "summary_figs" / "diamond_latitudes.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


# symmetric REGULAR-shape sweep geometry (rows = shape, cols = diameter, color = center)
SYM_FAMS = [("symdia", "diamond"), ("symsq", "square"), ("symhex", "hexagon")]
SYM_DIAMS = [0.3, 0.5, 0.75, 1.0]
SYM_CENTERS = [0.5, 0.0, -0.5]
CENTER_COLOR = {0.5: "#fb6a4a", 0.0: "#111111", -0.5: "#6baed6"}  # match summary LAT_COLORS


def fig_sym_layouts():
    """The whole symmetric REGULAR-shape sweep as a geometry diagram: rows = shape
    (diamond / square / hexagon), columns = E-W diameter (0.3 / 0.5 / 0.75 / 1.0°), and
    within each panel the three center latitudes (+0.5 / 0 / −0.5°N, colored) drawn from
    the actual config vertices. No moorings (these arrays are glider-only). Presentation
    aid: 'here are the 36 regular-array experiments we ran.'"""
    nrow, ncol = len(SYM_FAMS), len(SYM_DIAMS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for i, (fam, shape) in enumerate(SYM_FAMS):
        for j, d in enumerate(SYM_DIAMS):
            ax = axes[i][j]
            ax.axhline(0.0, color="0.85", lw=0.8, ls=":", zorder=0)
            for c in SYM_CENTERS:
                col = CENTER_COLOR[c]
                cfg = CFG_ROOT / fam / f"{fam}_d{d}_c{c:+.1f}.json"
                pos = [tuple(p) for p in json.load(open(cfg))["positions"]]
                xy = np.array([(lon - MOORING_LON, la) for la, lon in pos])
                hull = ConvexHull(xy).vertices
                ring = np.append(hull, hull[0])
                ax.add_patch(Polygon(xy[hull], closed=True, facecolor=col,
                                     edgecolor="none", alpha=0.12, zorder=1))
                ax.plot(xy[ring, 0], xy[ring, 1], color=col, lw=1.6, zorder=2)
                ax.scatter(xy[:, 0], xy[:, 1], s=24, color=col, edgecolor="k",
                           linewidth=0.4, zorder=3)
                # truth footprint = the disk of radius d/2 the vertices lie on (shared
                # by all three shapes at this diameter & centre)
                ax.add_patch(Circle((0.0, c), d / 2.0, fill=False, edgecolor=col,
                                    lw=1.0, ls="--", alpha=0.6, zorder=2))
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(f"diameter {d:g}°", fontsize=14, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{shape}\nLatitude (°N)", fontsize=13, fontweight="bold",
                              color=SHAPE_COLOR[shape])
            if i == nrow - 1:
                ax.set_xlabel("Longitude offset (°)", fontsize=11)
    axes[0][0].set_xlim(-0.7, 0.7)
    axes[0][0].set_ylim(-1.2, 1.2)

    handles = [Line2D([], [], marker="o", color=CENTER_COLOR[c], ls="",
                      markeredgecolor="k", markersize=10, label=f"center {c:+.1f}°")
               for c in SYM_CENTERS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=13, title="cell center latitude")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = HERE / "summary_figs" / "sym_sweep" / "sym_layouts.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 14})
    fig_shapes()
    fig_diamond_latitudes()
    fig_sym_layouts()
