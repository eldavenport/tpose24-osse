"""
summary_fig_tools.py — shared style, helpers, and figure builders for the OSSE
W-skill summary notebooks (experiment_*/summary.ipynb) and the exp1-vs-exp2
comparison (compare_exp_1_2). Each notebook does its own setup (load metrics.csv,
pick output dir) and then calls the make_* builders below.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import cmocean

import osse_tools as ot

W2DAY = 86400.0                                  # m/s -> m/day


def apply_style():
    """Bold, slightly larger axis labels and larger legends (call once per notebook)."""
    plt.rcParams.update({
        "axes.labelsize": 12.5, "axes.labelweight": "bold",
        "legend.fontsize": 12, "legend.title_fontsize": 14,
    })


# ---- color / style dictionaries --------------------------------------------
LAT_COLORS = {
    -1.5: "#08306b", -1.0: "#2171b5", -0.5: "#6baed6",
     0.0: "#111111",
     0.5: "#fb6a4a",  1.0: "#cb181d",  1.5: "#67000d",
}
GLIDER_COLORS = {2: "#440154", 4: "#21918c", 6: "#9fda3a"}
PAT_STYLE = {                                    # combined fig2: (linestyle, marker)
    "density_2g":      (":",  "o"), "density_4g":    ("--", "o"),
    "density_6g":      ("-",  "o"),
    "equator_1deg":    ("--", "s"), "equator_2deg":  (":",  "s"),
    "equator_3cell":   ("-",  "s"),
    "equator_hex1deg": ("--", "D"), "equator_hex2deg": (":", "D"),
    "equator_sq1deg":  ("--", "P"), "equator_sq2deg":  (":", "P"),
    "shift":           ("-",  "^"), "shift_hex":       ("-",  "H"),
}
# equator single-cell: color = SHAPE, height (1° vs 2°) encoded by LINE WIDTH
# (thin = 1°, thick = 2°). Linestyle is reserved for experiment (solid = exp1,
# dashed = exp2) in the compare figures; single-experiment summaries draw solid.
# Scatter panels (no line to thicken) carry height in marker size instead.
EQS_SHAPE_COLOR = {"diamond": "#1f77b4", "hexagon": "#2ca02c", "square": "#d62728"}
EQS_HEIGHT_LW   = {1.0: 1.0, 2.0: 2.25}          # 1° thin, 2° thick
EQS_HEIGHT_MS   = {1.0: 5.0, 2.0: 8.0}           # scatter marker size: small=1°, large=2°
EQS_META = {                                     # pattern -> (shape, height_deg)
    "equator_1deg":    ("diamond", 1.0), "equator_2deg":    ("diamond", 2.0),
    "equator_hex1deg": ("hexagon", 1.0), "equator_hex2deg": ("hexagon", 2.0),
    "equator_sq1deg":  ("square",  1.0), "equator_sq2deg":  ("square",  2.0),
}
EQS_STYLE = {pat: dict(color=EQS_SHAPE_COLOR[sh], lw=EQS_HEIGHT_LW[h],
                       ms=EQS_HEIGHT_MS[h], lbl=f"{h:g}° {sh}", shape=sh, height=h)
             for pat, (sh, h) in EQS_META.items()}
EQS_PATS = ["equator_1deg", "equator_2deg", "equator_hex1deg", "equator_hex2deg",
            "equator_sq1deg", "equator_sq2deg"]

WIDTHS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]        # glider lon offsets swept


def _lat_lbl(v):
    return f"{v:+.1f}"


# ---- shared metric axis limits ---------------------------------------------
# Every summary figure (all experiments + comparisons) shares the same
# axes and can be compared by eye. Curves are allowed to run off the plot for
# the rare outlier -- read exact values off the fig6 heatmaps instead. Chosen to
# comfortably hold the bulk of the data across every experiment (width-swept and
# depth-resolved). Keyed by metric id; import as sft.LIMITS in any notebook that
# does its own plotting.
LIMITS = {
    "w_mean":      (-1.0, 2.0),      # mean upwelling <w>  (m day^-1)
    "bias_sig":    (-0.3, 0.3),      # mean bias / sigma_y
    "frac_bias":   (-2.0, 2.0),      # mean bias / <w>  (blows up at depth -> off-plot)
    "sigma_ratio": (0.5, 1.5),       # sigma_x / sigma_y  (est / model)
    "corr":        (0.4, 1.0),       # correlation r
    "norm_rms":    (0.0, 1.25),      # relative error  RMS / sigma_y
    "rms_ms":      (0.0, 1.2e-4),    # depth RMS & signal sigma  (m s^-1, fig5 only)
}

# ---- six line-able metrics (figure-2 order) --------------------------------
METRIC6 = [                                      # (fn(df)->series, label, refline)
    (lambda d: d["w_model_mean"] * W2DAY,
     r"mean upwelling  $\langle w\rangle$  (m day$^{-1}$)", 0.0),
    (lambda d: d["mean_bias"] / d["w_model_std"], r"mean bias / $\sigma_y$", 0.0),
    (lambda d: ot.frac_mean_bias(d["mean_bias"], d["w_model_mean"]),
     r"mean bias / $\langle w\rangle$", 0.0),
    (lambda d: d["w_est_std"] / d["w_model_std"], r"$\sigma_x/\sigma_y$  (est / model)", 1.0),
    (lambda d: d["corr"], "correlation  r", None),
    (lambda d: d["norm_rms"], r"relative error  RMS/$\sigma_y$", None),
]
# metric id per METRIC6 panel (row-major, figure-2 order), for LIMITS lookup
METRIC6_KEYS = ["w_mean", "bias_sig", "frac_bias", "sigma_ratio", "corr", "norm_rms"]
ESTTRUE = [Line2D([0], [0], color="0.35", ls="-", lw=1.9, label="estimated"),
           Line2D([0], [0], color="0.35", ls=":", lw=1.4, label="true (model)")]


def plot6(ax, x, d, **kw):
    """Plot the six METRIC6 metrics of one sorted group into a 2x3 grid (row-major,
    figure-2 order). Panel 0 shows estimated (styled) and true (dotted, faint)."""
    true_kw = dict(ls=":", lw=1.2, alpha=0.55, color=kw.get("color", "0.35"))
    ax.flat[0].plot(x, d["w_model_mean"] * W2DAY, **true_kw)
    ax.flat[0].plot(x, d["w_est_mean"] * W2DAY, **kw)
    for k in range(1, 6):
        ax.flat[k].plot(x, METRIC6[k][0](d), **kw)


def label6(ax, xlabel):
    """Y-labels, reference lines, x-labels, grids, and the est/true legend."""
    for k, (_fn, ylab, ref) in enumerate(METRIC6):
        a = ax.flat[k]
        a.set_ylabel(ylab); a.set_xlabel(xlabel); a.grid(alpha=0.3)
        if ref is not None:
            a.axhline(ref, color="0.5", lw=0.8)
        a.set_ylim(LIMITS[METRIC6_KEYS[k]])
    ax.flat[0].legend(handles=ESTTRUE, loc="best", frameon=True)


# ---- 2x4 skill grid (fig2a-2d) ---------------------------------------------
def scatter_kw(kw):
    """Marker-only style for the mean-recovery scatter, derived from a line's kwargs."""
    sk = {k: v for k, v in kw.items()
          if k in ("color", "ms", "mec", "mew", "mfc", "alpha")}
    sk.update(ls="", marker=kw.get("marker", "o"))
    return sk


def fill_skill_panels(ax, series):
    """Fill the 7 metric panels of a 2x4 grid; geometry (ax[1,3]) is the caller's.
    series: iterable of (dd, line_kw[, scatter_kw]); dd sorted by width. ax[0,0]
    shows estimated + true mean w; ax[0,3] is the estimated-vs-true mean scatter."""
    for item in series:
        dd, kw = item[0], item[1]
        skw = item[2] if len(item) > 2 and item[2] else scatter_kw(kw)
        w = dd.width
        true_kw = dict(ls=":", lw=1.2, alpha=0.55, color=kw.get("color", "0.35"))
        ax[0, 0].plot(w, dd.w_model_mean * W2DAY, **true_kw)
        ax[0, 0].plot(w, dd.w_est_mean * W2DAY, **kw)
        ax[0, 1].plot(w, dd.mean_bias / dd.w_model_std, **kw)
        ax[0, 2].plot(w, ot.frac_mean_bias(dd.mean_bias, dd.w_model_mean), **kw)
        ax[0, 3].plot(dd.w_model_mean * W2DAY, dd.w_est_mean * W2DAY, **skw)
        ax[1, 0].plot(w, dd.w_est_std / dd.w_model_std, **kw)
        ax[1, 1].plot(w, dd["corr"], **kw)
        ax[1, 2].plot(w, dd.norm_rms, **kw)


def label_skill_panels(ax):
    """Axis labels, reference lines, the scatter 1:1 line, and the est/true legend."""
    ax[0, 0].set_ylabel(r"mean upwelling  $\langle w\rangle$  (m day$^{-1}$)")
    ax[0, 0].axhline(0, color="0.5", lw=0.8)
    ax[0, 0].legend(handles=ESTTRUE, loc="best", frameon=True)
    ax[0, 1].set_ylabel(r"mean bias / $\sigma_y$"); ax[0, 1].axhline(0, color="0.5", lw=0.8)
    ax[0, 2].set_ylabel(r"mean bias / $\langle w\rangle$"); ax[0, 2].axhline(0, color="0.5", lw=0.8)
    ax[1, 0].set_ylabel(r"$\sigma_x/\sigma_y$  (est / model)"); ax[1, 0].axhline(1, color="0.5", lw=0.8)
    ax[1, 1].set_ylabel("correlation  r")
    ax[1, 2].set_ylabel(r"relative error  RMS/$\sigma_y$")
    for a, key in ((ax[0, 0], "w_mean"), (ax[0, 1], "bias_sig"), (ax[0, 2], "frac_bias"),
                   (ax[1, 0], "sigma_ratio"), (ax[1, 1], "corr"), (ax[1, 2], "norm_rms")):
        a.set_xlabel("glider lon offset (deg)"); a.grid(alpha=0.3); a.set_ylim(LIMITS[key])
    sc = ax[0, 3]                                # est-vs-true scatter: fixed w_mean box
    lo, hi = LIMITS["w_mean"][0], 1.5            # cap scatter at 1.5 (tighter than line panels)
    sc.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, zorder=0)
    sc.set_xlim(lo, hi); sc.set_ylim(lo, hi); sc.set_aspect("equal", "box")
    sc.set_xlabel(r"true $\langle w\rangle$ (m day$^{-1}$)")
    sc.set_ylabel(r"estimated $\langle w\rangle$ (m day$^{-1}$)")
    sc.grid(alpha=0.3)


# ---- geometry panels (squares = moorings, circles = gliders) ---------------
def _hull_poly(pts):
    """Vertices of the convex outline of pts, angle-sorted (pts are convex)."""
    pts = np.array(pts, float)
    c = pts.mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    p = pts[order]
    return np.vstack([p, p[0]])


def _layout_legend(ax, loc="upper right"):
    ax.plot([], [], marker="s", ls="", color="0.55", ms=6, label="mooring")
    ax.plot([], [], marker="o", ls="", color="0.4", mec="k", mew=0.3, ms=6, label="glider")
    ax.legend(fontsize=7, loc=loc)


def stacked_cells_panel(ax, cells, half, title, off_demo=0.5):
    """Vertical stack of diamond cells (an actual multi-cell array). Color = center lat."""
    vlats = sorted({c + s * half for c in cells for s in (-1, 1)} | set(cells))
    moor = [v for v in vlats if abs(v - round(v)) < 1e-9]
    ax.plot([0, 0], [min(vlats), max(vlats)], color="0.8", lw=0.8, zorder=1)
    ax.plot([0] * len(moor), moor, marker="s", ls="", color="0.55", ms=6, zorder=3)
    for c in cells:
        col = LAT_COLORS[c]
        ax.plot([-off_demo, off_demo], [c, c], marker="o", ls="", color=col,
                ms=6, mec="k", mew=0.3, zorder=4)
        poly = _hull_poly([(0, c + half), (off_demo, c), (0, c - half), (-off_demo, c)])
        ax.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=1.8, zorder=2)
        ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.07, zorder=0)
        ax.text(off_demo + 0.16, c, f"{c:+.1f}", va="center", ha="left", fontsize=7, color=col)
    _layout_legend(ax, loc="upper right")
    ax.set_aspect("equal"); ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(min(vlats) - 0.4, max(vlats) + 0.4)
    ax.set_xticks([]); ax.set_yticks(moor); ax.set_ylabel("lat (deg)")
    ax.set_title(title); ax.grid(alpha=0.2, axis="y")


def stacked_hex_cells_panel(ax, cells, half, title, off_demo=0.5):
    """Vertical stack of hexagon cells (the shift-hex array). Color = center lat.
    N/S vertices sit on the mooring line at center +/-half; 4 side gliders at
    +/-off_demo lon and +/-half/2 lat (mirrors stacked_cells_panel for diamonds)."""
    mid = half / 2.0
    vlats = sorted({c + s * half for c in cells for s in (-1, 1)} | set(cells))
    moor = [v for v in vlats if abs(v - round(v)) < 1e-9]
    ax.plot([0, 0], [min(vlats), max(vlats)], color="0.8", lw=0.8, zorder=1)
    ax.plot([0] * len(moor), moor, marker="s", ls="", color="0.55", ms=6, zorder=3)
    for c in cells:
        col = LAT_COLORS[c]
        gl = [(s * off_demo, c + t * mid) for s in (-1, 1) for t in (-1, 1)]
        ax.plot([p[0] for p in gl], [p[1] for p in gl], marker="o", ls="", color=col,
                ms=6, mec="k", mew=0.3, zorder=4)
        poly = _hull_poly([(0, c + half), (off_demo, c + mid), (off_demo, c - mid),
                           (0, c - half), (-off_demo, c - mid), (-off_demo, c + mid)])
        ax.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=1.8, zorder=2)
        ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.07, zorder=0)
        ax.text(off_demo + 0.16, c, f"{c:+.1f}", va="center", ha="left", fontsize=7, color=col)
    _layout_legend(ax, loc="upper right")
    ax.set_aspect("equal"); ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(min(vlats) - 0.4, max(vlats) + 0.4)
    ax.set_xticks([]); ax.set_yticks(moor); ax.set_ylabel("lat (deg)")
    ax.set_title(title); ax.grid(alpha=0.2, axis="y")


def density_geometry_panel(ax, off_demo=0.5):
    """Layouts of the 2/4/6-glider cells (center +0.5). Hull colored to match lines."""
    rows_by_ng = {2: [0.5], 4: [0.25, 0.75], 6: [0.25, 0.5, 0.75]}
    for i, ng in enumerate((2, 4, 6)):
        xc, col = i * 2.0, GLIDER_COLORS[ng]
        ax.plot([xc, xc], [0, 1], color="0.8", lw=0.8, zorder=1)
        ax.plot([xc, xc], [0, 1], marker="s", ls="", color="0.55", ms=6, zorder=3)
        gl = [(xc + s * off_demo, r) for r in rows_by_ng[ng] for s in (-1, 1)]
        ax.plot([p[0] for p in gl], [p[1] for p in gl], marker="o", ls="",
                color=col, ms=6, mec="k", mew=0.3, zorder=4)
        poly = _hull_poly([(xc, 0), (xc, 1)] + gl)
        ax.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=1.8, zorder=2)
        ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.08, zorder=0)
        ax.text(xc, -0.4, f"{ng} gliders", ha="center", va="top", fontsize=8, color=col)
    _layout_legend(ax)
    ax.set_aspect("equal"); ax.set_xlim(-1.0, 5.1); ax.set_ylim(-0.8, 1.5)
    ax.set_xticks([]); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("lat (deg)")
    ax.set_title("glider layout (offset = 0.5°)")
    ax.grid(alpha=0.2, axis="y")


def _draw_equator_cell(ax, xc, shape, half, color, label, off=0.5):
    if shape == "diamond":
        verts = [(0, half), (off, 0), (0, -half), (-off, 0)]
    elif shape == "hexagon":
        mid = half / 2.0
        verts = [(0, half), (off, mid), (off, -mid), (0, -half), (-off, -mid), (-off, mid)]
    else:                                       # square (box): 4 corner gliders
        verts = [(off, half), (off, -half), (-off, -half), (-off, half)]
    line_moor = [(0.0, ml) for ml in (-2, -1, 0, 1, 2) if -half - 1e-9 <= ml <= half + 1e-9]
    pts = verts + [p for p in line_moor if p not in verts]
    is_moor = lambda x, y: abs(x) < 1e-9 and abs(y - round(y)) < 1e-9
    ax.plot([xc, xc], [-half, half], color="0.8", lw=0.8, zorder=1)
    mo = [(xc + x, y) for x, y in pts if is_moor(x, y)]
    gl = [(xc + x, y) for x, y in pts if not is_moor(x, y)]
    ax.plot([p[0] for p in mo], [p[1] for p in mo], marker="s", ls="", color="0.55", ms=6, zorder=3)
    ax.plot([p[0] for p in gl], [p[1] for p in gl], marker="o", ls="",
            color=color, ms=6, mec="k", mew=0.3, zorder=4)
    poly = _hull_poly([(xc + x, y) for x, y in verts])
    ax.plot(poly[:, 0], poly[:, 1], "-", color=color, lw=1.8, zorder=2)
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.08, zorder=0)
    ax.text(xc, -1.45, label, ha="center", va="top", fontsize=7.5, color=color)


def equator_single_geometry(ax):
    layout = [("diamond", 0.5), ("diamond", 1.0), ("hexagon", 0.5), ("hexagon", 1.0),
              ("square", 0.5), ("square", 1.0)]
    for i, (shape, half) in enumerate(layout):
        _draw_equator_cell(ax, i * 2.0, shape, half, EQS_SHAPE_COLOR[shape], f"{2 * half:g}° {shape}")
    _layout_legend(ax, loc="upper right")
    ax.set_aspect("equal"); ax.set_xlim(-1.0, 11.0); ax.set_ylim(-1.9, 1.5)
    ax.set_xticks([]); ax.set_yticks([-1, 0, 1]); ax.set_ylabel("lat (deg)")
    ax.set_title("cell geometry (offset = 0.5°)")
    ax.grid(alpha=0.2, axis="y")


# ---- heatmap helpers -------------------------------------------------------
PAT_LABEL = {
    "density_2g": "density 2g", "density_4g": "density 4g", "density_6g": "density 6g",
    "equator_1deg": "diamond 1°", "equator_2deg": "diamond 2°",
    "equator_hex1deg": "hexagon 1°", "equator_hex2deg": "hexagon 2°",
    "equator_sq1deg": "square 1°", "equator_sq2deg": "square 2°",
    "equator_3cell": "3-cell", "shift": "shift", "shift_hex": "shift hex",
}
ORDER = [
    ("density_2g", 0.5), ("density_4g", 0.5), ("density_6g", 0.5),
    ("equator_1deg", 0.0), ("equator_2deg", 0.0),
    ("equator_hex1deg", 0.0), ("equator_hex2deg", 0.0),
    ("equator_sq1deg", 0.0), ("equator_sq2deg", 0.0),
    ("equator_3cell", -1.0), ("equator_3cell", 0.0), ("equator_3cell", 1.0),
    ("shift", -1.5), ("shift", -0.5), ("shift", 0.5), ("shift", 1.5),
    ("shift_hex", -1.5), ("shift_hex", -0.5), ("shift_hex", 0.5), ("shift_hex", 1.5),
]
EQS_ROWS = ["equator_1deg", "equator_2deg", "equator_hex1deg",
            "equator_hex2deg", "equator_sq1deg", "equator_sq2deg"]
SUB_ORDERS = [                                   # (tag, order) one method per figure
    ("6a_shift",          [o for o in ORDER if o[0] == "shift"]),
    ("6b_equator3cell",   [o for o in ORDER if o[0] == "equator_3cell"]),
    ("6c_density",        [o for o in ORDER if o[0].startswith("density")]),
    ("6d_equator_single", [o for o in ORDER if o[0] in EQS_ROWS]),
    ("6e_shift_hex",      [o for o in ORDER if o[0] == "shift_hex"]),
]


def row_label(pat, lat):
    base = PAT_LABEL.get(pat, pat)
    return f"{base} {lat:+.1f}°" if pat in ("equator_3cell", "shift", "shift_hex") else base


def _fam(pat):
    if pat.startswith("density"):
        return "density"
    if pat in ("shift", "shift_hex"):
        return pat
    return "equator"


def grid_from_col(df, col, order):
    """Grid of df[col] over (order rows) x (WIDTHS cols), NaN where missing."""
    g = np.full((len(order), len(WIDTHS)), np.nan)
    for i, (pat, lat) in enumerate(order):
        for j, w in enumerate(WIDTHS):
            sel = df[(df["pattern"] == pat) & np.isclose(df["center_lat"], lat)
                     & np.isclose(df["width"], w)]
            if len(sel):
                g[i, j] = sel.iloc[0][col]
    return g


def heatmap_cell(ax, G, order, cmap, center, fmt, title, sep_color="white"):
    """Draw one metric heatmap: colored squares + printed values + family rules."""
    nrows = len(order)
    if center is not None:
        vmax = np.nanmax(np.abs(G - center))
        vmax = vmax if (np.isfinite(vmax) and vmax > 0) else 1.0
        im = ax.imshow(G, aspect="auto", cmap=cmap,
                       norm=TwoSlopeNorm(vmin=center - vmax, vcenter=center, vmax=center + vmax))
    else:
        im = ax.imshow(G, aspect="auto", cmap=cmap)
    rgba = im.cmap(im.norm(G))
    lum = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
    for i in range(nrows):
        for j in range(len(WIDTHS)):
            if not np.isnan(G[i, j]):
                ax.text(j, i, fmt.format(G[i, j]), ha="center", va="center",
                        fontsize=7.5, color="white" if lum[i, j] < 0.5 else "black")
    ax.set_xticks(range(len(WIDTHS))); ax.set_xticklabels([f"{w:g}" for w in WIDTHS])
    ax.set_xlabel("glider lon offset (°)"); ax.set_title(title, fontsize=9.5)
    for i in range(1, nrows):
        if _fam(order[i][0]) != _fam(order[i - 1][0]):
            ax.axhline(i - 0.5, color=sep_color, lw=2.5)
    ax.set_xticks(np.arange(-0.5, len(WIDTHS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    return im


def _set_row_labels(axes, order):
    for i in range(2):
        axes[i, 0].set_yticks(range(len(order)))
        axes[i, 0].set_yticklabels([row_label(p, l) for p, l in order], fontsize=8)


# ============================================================================
# Figure builders — single experiment
# ============================================================================
def make_fig1(m, sumdir):
    """Distribution of each metric by array family (2x3 strip plots)."""
    fams = ["density", "equator", "shift"]
    ypos = {f: i for i, f in enumerate(fams)}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    sc = None
    for k, (fn, xlab, ref) in enumerate(METRIC6):
        ax = axes.flat[k]
        for f in fams:
            s = m[m.family == f]
            jit = rng.uniform(-0.16, 0.16, len(s))
            if k == 0:                            # mean upwelling: est + true
                xt = np.asarray(s["w_model_mean"] * W2DAY, float)
                xe = np.asarray(s["w_est_mean"] * W2DAY, float)
                ax.scatter(xt, ypos[f] + jit + 0.18, c=s.width, cmap="viridis",
                           vmin=m.width.min(), vmax=m.width.max(), marker="o",
                           s=34, edgecolor="k", lw=0.3, zorder=3)
                sc = ax.scatter(xe, ypos[f] + jit - 0.18, c=s.width, cmap="viridis",
                                vmin=m.width.min(), vmax=m.width.max(), marker="D",
                                s=30, edgecolor="k", lw=0.3, zorder=3)
            else:
                x = np.asarray(fn(s), float)
                sc = ax.scatter(x, ypos[f] + jit, c=s.width, cmap="viridis",
                                vmin=m.width.min(), vmax=m.width.max(),
                                s=40, edgecolor="k", lw=0.3, zorder=3)
                med = np.nanmedian(x)
                ax.plot([med, med], [ypos[f] - 0.32, ypos[f] + 0.32], "k-", lw=2.0, zorder=4)
        if ref is not None:
            ax.axvline(ref, color="0.4", ls="--", lw=1, zorder=1)
        ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
        ax.set_ylim(-0.6, len(fams) - 0.4)
        ax.set_xlim(LIMITS[METRIC6_KEYS[k]])
        ax.set_xlabel(xlab); ax.grid(axis="x", alpha=0.3)
    axes.flat[0].legend(handles=[Line2D([0], [0], ls="", marker="o", mfc="0.6", mec="k", label="true (model)"),
                                 Line2D([0], [0], ls="", marker="D", mfc="0.6", mec="k", label="estimated")],
                        loc="best", frameon=True)
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cb.set_label("glider lon offset (deg)")
    fig.savefig(os.path.join(sumdir, "fig1_skill_overview.png"), dpi=150, bbox_inches="tight")
    plt.show()


def make_fig2_combined(m, sumdir):
    """Skill vs width, all methods overlaid: color = center lat, line/marker = method."""
    g = (m.groupby(["family", "pattern", "center_lat", "width"], as_index=False)
           [["norm_rms", "corr", "mean_bias", "w_model_mean", "w_est_mean",
             "w_model_std", "w_est_std"]].mean())
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for (pat, lat), d in g.groupby(["pattern", "center_lat"]):
        if pat not in PAT_STYLE:            # sym_sweep families have their own figures
            continue
        d = d.sort_values("width")
        ls, mk = PAT_STYLE[pat]
        plot6(axes, d.width, d, ls=ls, marker=mk, color=LAT_COLORS[lat], lw=1.6,
              ms=6, mec="k", mew=0.3)
    label6(axes, "glider lon offset (deg)")
    lat_handles = [Line2D([0], [0], marker="s", ls="", mfc=LAT_COLORS[l], mec="k",
                          mew=0.3, ms=8, label=f"{l:+.1f}") for l in sorted(LAT_COLORS)]
    pat_handles = [Line2D([0], [0], color="0.35", ls=PAT_STYLE[p][0],
                          marker=PAT_STYLE[p][1], mfc="0.35", mec="k", mew=0.3, label=p)
                   for p in PAT_STYLE]
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    lg1 = fig.legend(handles=lat_handles, title="color = center lat", loc="upper center",
                     bbox_to_anchor=(0.28, 1.0), ncol=7, frameon=False)
    fig.add_artist(lg1)
    fig.legend(handles=pat_handles, title="line / marker = method", loc="upper center",
               bbox_to_anchor=(0.76, 1.03), ncol=4, frameon=False)
    fig.savefig(os.path.join(sumdir, "fig2_skill_vs_width.png"), dpi=150, bbox_inches="tight")
    plt.show()


def make_fig2abcd(m, sumdir):
    """fig2a-2d: skill vs width, one method per figure, as a 2x4 mean/fluctuation grid."""
    panels = [
        ("2a_shift", m.pattern == "shift", "center_lat", LAT_COLORS, "cell center lat", _lat_lbl,
         lambda ax: stacked_cells_panel(ax, [-1.5, -0.5, 0.5, 1.5], 0.5, "array layout (offset = 0.5 deg)")),
        ("2e_shift_hex", m.pattern == "shift_hex", "center_lat", LAT_COLORS, "cell center lat", _lat_lbl,
         lambda ax: stacked_hex_cells_panel(ax, [-1.5, -0.5, 0.5, 1.5], 0.5, "array layout (offset = 0.5 deg)")),
        ("2b_equator3cell", m.pattern == "equator_3cell", "center_lat", LAT_COLORS, "cell center lat", _lat_lbl,
         lambda ax: stacked_cells_panel(ax, [-1.0, 0.0, 1.0], 1.0, "array layout (offset = 0.5 deg)")),
        ("2c_density", m.family == "density", "n_gliders_cell", GLIDER_COLORS, "gliders/cell",
         lambda v: f"{int(v)}", density_geometry_panel),
    ]
    for tag, mask, key, cmap, leg_title, lbl, geom in panels:
        d = m[mask]
        keys = sorted(d[key].unique())
        fig, axes = plt.subplots(2, 4, figsize=(21, 10))
        series = [(d[d[key] == kv].drop_duplicates("width").sort_values("width"),
                   dict(color=cmap[kv], marker="o", ls="-", lw=1.9, ms=6, mec="k", mew=0.3))
                  for kv in keys]
        fill_skill_panels(axes, series)
        label_skill_panels(axes)
        handles = [Line2D([0], [0], color=cmap[kv], marker="o", mec="k", mew=0.3,
                          lw=1.9, label=lbl(kv)) for kv in keys]
        axes[0, 3].legend(handles=handles, title=leg_title, loc="best", frameon=True,
                          ncol=2 if len(handles) > 4 else 1)
        geom(axes[1, 3])
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.legend(handles=handles, title=f"color = {leg_title}", loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
        fig.savefig(os.path.join(sumdir, f"fig{tag}.png"), dpi=150, bbox_inches="tight")
        plt.show()

    # 2d: equator single cell, color = shape, line width = height. The scatter
    # panel carries height in marker size (small=1°, large=2°) since thickness
    # can't apply to points.
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    series = []
    for pat in EQS_PATS:
        st = EQS_STYLE[pat]
        dd = m[m.pattern == pat].sort_values("width")
        if dd.empty:
            continue
        line_kw = dict(color=st["color"], ls="-", lw=st["lw"])
        scat_kw = dict(color=st["color"], ls="", marker="o", ms=st["ms"], mec="k", mew=0.3)
        series.append((dd, line_kw, scat_kw))
    fill_skill_panels(axes, series)
    label_skill_panels(axes)
    shape_handles = [Line2D([0], [0], color=c, lw=2.4, label=s) for s, c in EQS_SHAPE_COLOR.items()]
    height_handles = [Line2D([0], [0], color="0.35", ls="-", lw=lw, label=f"{h:g} deg tall")
                      for h, lw in EQS_HEIGHT_LW.items()]
    scat_height = [Line2D([0], [0], color="0.35", ls="", marker="o", mec="k", mew=0.3, ms=ms,
                          label=f"{h:g} deg tall") for h, ms in EQS_HEIGHT_MS.items()]
    axes[0, 3].legend(handles=scat_height, title="size = height", loc="best", frameon=True)
    equator_single_geometry(axes[1, 3])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    lg1 = fig.legend(handles=shape_handles, title="color = shape", loc="upper center",
                     bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.add_artist(lg1)
    fig.legend(handles=height_handles, title="line width = height", loc="upper center",
               bbox_to_anchor=(0.85, 1.0), ncol=2, frameon=False, handlelength=3.0)
    fig.savefig(os.path.join(sumdir, "fig2d_equator_single.png"), dpi=150, bbox_inches="tight")
    plt.show()


def make_fig3(m, sumdir):
    """Skill vs cell latitude for the two multi-cell arrays (6 metric rows x 2 arrays)."""
    lat_arrays = [("shift", "Shift array (4 cells)"),
                  ("shift_hex", "Shift-hex array (4 cells)"),
                  ("equator_3cell", "Equator 3-cell array")]
    fig, axes = plt.subplots(len(METRIC6), 3, figsize=(17, 20), sharex=True, sharey="row")
    for j, (pat, ptitle) in enumerate(lat_arrays):
        d = m[m.pattern == pat]
        for w, dd in d.groupby("width"):
            dd = dd.sort_values("center_lat")
            base = axes[0, j].plot(dd.center_lat, dd.w_est_mean * W2DAY, "o-", label=f"{w:g}")
            axes[0, j].plot(dd.center_lat, dd.w_model_mean * W2DAY, ls=":", lw=1.2,
                            alpha=0.55, color=base[0].get_color())
            for k in range(1, len(METRIC6)):
                axes[k, j].plot(dd.center_lat, METRIC6[k][0](dd), "o-", label=f"{w:g}")
        axes[0, j].set_title(ptitle)
        axes[-1, j].set_xlabel("cell center lat (deg)")
    for k, (fn, ylab, ref) in enumerate(METRIC6):
        axes[k, 0].set_ylabel(ylab)
        axes[k, 0].set_ylim(LIMITS[METRIC6_KEYS[k]])   # sharey="row" propagates
        for j in range(axes.shape[1]):
            axes[k, j].grid(alpha=0.3)
            if ref is not None:
                axes[k, j].axhline(ref, color="0.5", lw=0.8)
    axes[0, 0].legend(handles=ESTTRUE, loc="best", frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, title="glider lon offset", loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(l), frameon=False)
    fig.savefig(os.path.join(sumdir, "fig3_skill_vs_latitude.png"), dpi=150, bbox_inches="tight")
    plt.show()


def make_fig4(m, sumdir):
    """Skill vs gliders-per-cell at fixed center 0.5N (2x3 metric grid)."""
    d = m[m.family == "density"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for w, dd in d.groupby("width"):
        dd = dd.sort_values("n_gliders_cell")
        plot6(axes, dd.n_gliders_cell, dd, marker="o", ls="-", label=f"{w:g}")
    label6(axes, "gliders per cell")
    for a in axes.flat:
        a.set_xticks([2, 4, 6])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    h, l = axes.flat[1].get_legend_handles_labels()
    fig.legend(h, l, title="glider lon offset", loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(l), frameon=False)
    fig.savefig(os.path.join(sumdir, "fig4_skill_vs_gliders.png"), dpi=150, bbox_inches="tight")
    plt.show()


def make_fig5(m, sumdir, load_cell):
    """fig5a-5d: skill vs depth, one method per figure (2x4 grid, figure-2 order).
    load_cell(row) -> Dataset with w_est, w_model for that metrics row."""
    depth_width = 0.5
    W = np.isclose(m.width, depth_width)

    def _plot_depth(rows_styles, tag, leg_title, dw=depth_width):
        fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=True)
        handles = []
        for r, st in rows_styles:
            ds = load_cell(r)
            B = ot.w_skill_by_depth(ds.w_est, ds.w_model)
            z = B.depth
            c = st["color"]; ls = st.get("ls", "-"); lw = st.get("lw", 1.9)
            axes[0, 0].plot(B.w_model_mean * W2DAY, z, color=c, ls=":", lw=1.2, alpha=0.6)
            axes[0, 0].plot(B.w_est_mean * W2DAY, z, color=c, ls=ls, lw=lw)
            axes[0, 1].plot(B.mean_bias / B.w_model_std, z, color=c, ls=ls, lw=lw)
            axes[0, 2].plot(ot.frac_mean_bias(B.mean_bias, B.w_model_mean), z, color=c, ls=ls, lw=lw)
            axes[0, 3].plot(B.rms, z, color=c, ls=ls, lw=lw)
            axes[0, 3].plot(B.w_model_std, z, color=c, ls=":", lw=1.2, alpha=0.6)
            axes[1, 0].plot(B.w_est_std / B.w_model_std, z, color=c, ls=ls, lw=lw)
            axes[1, 1].plot(B["corr"], z, color=c, ls=ls, lw=lw)
            axes[1, 2].plot(B.norm_rms, z, color=c, ls=ls, lw=lw)
            handles.append(Line2D([0], [0], color=c, ls=ls, lw=lw, label=st["lbl"]))
        axes[0, 0].set_title(r"mean $\langle w\rangle$: est & true  [m day$^{-1}$]")
        axes[0, 0].axvline(0, color="0.5", lw=0.8, zorder=0)
        axes[0, 0].legend(handles=ESTTRUE, loc="best", frameon=True)
        axes[0, 1].set_title(r"mean bias / $\sigma_y$"); axes[0, 1].axvline(0, color="0.5", lw=0.8, zorder=0)
        axes[0, 2].set_title(r"mean bias / $\langle w\rangle$"); axes[0, 2].axvline(0, color="0.5", lw=0.8, zorder=0)
        axes[0, 3].set_title(r"RMS (solid) & signal $\sigma$ (dotted)  [m s$^{-1}$]")
        axes[1, 0].set_title(r"$\sigma_x/\sigma_y$  (est / model)"); axes[1, 0].axvline(1, color="0.5", lw=0.8, zorder=0)
        axes[1, 1].set_title("correlation  r")
        axes[1, 2].set_title(r"relative error  RMS/$\sigma_y$")
        axes[1, 3].axis("off")
        axes[0, 0].set_ylabel("depth (m)"); axes[1, 0].set_ylabel("depth (m)")
        for a, key in ((axes[0, 0], "w_mean"), (axes[0, 1], "bias_sig"),
                       (axes[0, 2], "frac_bias"), (axes[0, 3], "rms_ms"),
                       (axes[1, 0], "sigma_ratio"), (axes[1, 1], "corr"),
                       (axes[1, 2], "norm_rms")):
            a.set_xlim(LIMITS[key])
        for a in axes.flat:
            if a.axison:
                a.grid(alpha=0.3)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.legend(handles=handles, title=f"color = {leg_title}", loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
        fig.savefig(os.path.join(sumdir, f"fig{tag}.png"), dpi=150, bbox_inches="tight")
        plt.show()

    for tag, pat in (("5a_shift", "shift"), ("5e_shift_hex", "shift_hex"),
                     ("5b_equator3cell", "equator_3cell")):
        d = m[(m.pattern == pat) & W]
        rs = [(d[np.isclose(d.center_lat, lat)].iloc[0],
               dict(color=LAT_COLORS[lat], ls="-", lbl=f"{lat:+.1f}"))
              for lat in sorted(d.center_lat.unique())]
        _plot_depth(rs, tag, "cell center lat")
    d = m[(m.family == "density") & W]
    rs = [(d[d.n_gliders_cell == ng].iloc[0], dict(color=GLIDER_COLORS[ng], ls="-", lbl=f"{int(ng)}"))
          for ng in sorted(d.n_gliders_cell.unique())]
    _plot_depth(rs, "5c_density", "gliders/cell")
    d = m[W]
    rs = [(d[d.pattern == pat].iloc[0],
           dict(color=EQS_STYLE[pat]["color"], ls="-", lw=EQS_STYLE[pat]["lw"],
                lbl=EQS_STYLE[pat]["lbl"]))
          for pat in EQS_PATS if (d.pattern == pat).any()]
    _plot_depth(rs, "5d_equator_single", "shape (color) x height (line width)")


def make_fig6(m, sumdir):
    """fig6 (+6a-d): per-config metric heatmaps (2x3, figure-2 order)."""
    mb = m.copy()
    mb["frac_bias"]   = ot.frac_mean_bias(mb["mean_bias"].values, mb["w_model_mean"].values)
    mb["wmean_ratio"] = 1.0 + mb["frac_bias"]                # est <w> / model <w>
    mb["bias_sig"]    = mb["mean_bias"] / mb["w_model_std"]
    mb["sigma_ratio"] = mb["w_est_std"] / mb["w_model_std"]
    panels = [
        ("wmean_ratio", r"est $\langle w\rangle$ / model $\langle w\rangle$   (1 = matched)", cmocean.cm.balance, 1.0, "{:.2f}"),
        ("bias_sig",    r"mean bias / $\sigma_y$   (0 = unbiased)",      cmocean.cm.balance,  0.0, "{:+.2f}"),
        ("frac_bias",   r"mean bias / $\langle w\rangle$   (0 = unbiased)", cmocean.cm.balance, 0.0, "{:+.2f}"),
        ("sigma_ratio", r"$\sigma_x/\sigma_y$   (1 = matched amplitude)", cmocean.cm.balance, 1.0, "{:.2f}"),
        ("corr",        "correlation   (higher = better)",               cmocean.cm.matter_r, None, "{:.2f}"),
        ("norm_rms",    r"RMS/$\sigma_y$   (lower = better)",            cmocean.cm.matter,   None, "{:.2f}"),
    ]

    def _fig(order, fname, figsize):
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=True)
        for ax, (col, title, cmap, center, fmt) in zip(axes.flat, panels):
            im = heatmap_cell(ax, grid_from_col(mb, col, order), order, cmap, center, fmt, title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        _set_row_labels(axes, order)
        fig.tight_layout()
        fig.savefig(os.path.join(sumdir, fname), dpi=150, bbox_inches="tight")
        plt.show()

    _fig(ORDER, "fig6_metric_heatmaps.png", (19, 13))
    for tag, order in SUB_ORDERS:
        _fig(order, f"fig{tag}.png", (19, max(6.5, 0.7 * len(order) + 4.5)))


# ============================================================================
# Figure 7 — time-mean <w> with autocorrelation-aware confidence intervals
# ============================================================================
# Within a fig7 panel, color separates the two quantities being compared
# (estimate vs model truth); config identity is carried by the panel title
# (profiles) or the y-axis category (totals), so it needs no color.
F7_EST_C = "#1f77b4"      # array estimate
F7_MOD_C = "#d62728"      # model truth
F7_Z = 1.96               # 95% CI = mean +/- 1.96 * SE (normal approximation).
# Optional stricter version: since sigma is estimated from the data, the exact
# multiplier is Student's t, t.ppf(0.975, n_eff-1), not 1.96. With n_eff ~ 30-70
# here that is ~2.0 (~2-4% wider), so 1.96 is used for simplicity. To switch, make
# the multiplier per-config from each cell's n_eff instead of this constant.


def _fig7_profiles(rows_labels, load_cell, sumdir, tag, group_lbl):
    """Render one fig7-style figure: per-config depth profiles of estimated (solid) and
    true (dotted) time-mean <w> with +/-1.96*SE bands, plus a total-over-depth forest
    panel. `rows_labels` is a list of (metrics_row, column_label). Shared by make_fig7 and
    the regular-shape (sym) CI profiles."""
    ncfg = len(rows_labels)
    fig, axes = plt.subplots(1, ncfg + 1, figsize=(3.3 * (ncfg + 1), 6.3))
    axes = np.atleast_1d(axes)
    # per-config depth profiles (recompute per-depth SE from the saved arrays)
    profs, lo, hi = [], 0.0, 0.0
    for r, lbl in rows_labels:
        with load_cell(r) as ds:
            B = ot.w_skill_by_depth(ds.w_est, ds.w_model)
        profs.append((B, lbl))
        for mean, se in ((B.w_model_mean, B.w_model_mean_se),
                         (B.w_est_mean, B.w_est_mean_se)):
            lo = min(lo, float(((mean - F7_Z * se) * W2DAY).min()))
            hi = max(hi, float(((mean + F7_Z * se) * W2DAY).max()))
    pad = 0.08 * (hi - lo) or 0.1
    for ax, (B, lbl) in zip(axes[:ncfg], profs):
        z = B.depth.values
        for mean, se, c, ls in ((B.w_model_mean, B.w_model_mean_se, F7_MOD_C, ":"),
                                (B.w_est_mean, B.w_est_mean_se, F7_EST_C, "-")):
            mu = mean.values * W2DAY; hw = F7_Z * se.values * W2DAY
            ax.fill_betweenx(z, mu - hw, mu + hw, color=c, alpha=0.18, lw=0)
            ax.plot(mu, z, color=c, ls=ls, lw=1.9)
        ax.axvline(0, color="0.5", lw=0.8, zorder=0)
        ax.set_title(lbl); ax.set_xlim(lo - pad, hi + pad); ax.grid(alpha=0.3)
        ax.set_xlabel(r"$\langle w\rangle$  (m day$^{-1}$)")
    axes[0].set_ylabel("depth (m)")
    for ax in axes[1:ncfg]:
        ax.set_yticklabels([])

    # total-over-depth forest panel (column means from the metrics row)
    axT = axes[ncfg]
    yv = np.arange(ncfg)[::-1]
    for y, (r, lbl) in zip(yv, rows_labels):
        for off, c, mk, mkey, skey in (
                (+0.15, F7_MOD_C, "o", "w_model_mean", "w_model_mean_se"),
                (-0.15, F7_EST_C, "s", "w_est_mean",   "w_est_mean_se")):
            axT.errorbar(r[mkey] * W2DAY, y + off, xerr=F7_Z * r[skey] * W2DAY,
                         fmt=mk, color=c, capsize=3, ms=6, lw=1.6)
    axT.axvline(0, color="0.5", lw=0.8)
    axT.set_yticks(yv); axT.set_yticklabels([lbl for _, lbl in rows_labels])
    axT.set_ylim(-0.6, ncfg - 0.4)
    axT.set_xlabel(r"total $\langle w\rangle$  (m day$^{-1}$)")
    axT.set_title("total over depth"); axT.grid(alpha=0.3, axis="x")

    handles = [Line2D([0], [0], color=F7_EST_C, ls="-", lw=1.9,
                      marker="s", label=r"estimated ($\pm$95% CI)"),
               Line2D([0], [0], color=F7_MOD_C, ls=":", lw=1.9,
                      marker="o", label=r"true / model ($\pm$95% CI)")]
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend(handles=handles, title=group_lbl, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.savefig(os.path.join(sumdir, f"fig{tag}.png"), dpi=150, bbox_inches="tight")
    plt.show()


def make_fig7(m, sumdir, load_cell):
    """fig7a-7e: the time-mean vertical velocity <w> and how well it is resolved,
    shown with autocorrelation-aware 95% confidence intervals.

    One figure per method group (same groups as fig5). For each config: a depth
    profile of the estimated (solid) and true (dotted) time-mean <w>, each with a
    shaded +/-1.96*SE band (per depth); plus a final "total over depth" panel that
    collapses the column to one estimated and one true <w> +/- CI per config.

    The SE (osse_tools.mean_se_autocorr) treats the 81-day record as ONE
    autocorrelated sample of the process, deflating N to N_eff = N/tau: the CI is
    a statement about the expected/long-run mean, not the exact window average
    (which is known exactly). A band that straddles zero means this record does
    not resolve a nonzero mean; overlapping est/true bands mean the estimate is
    statistically consistent with truth."""
    W = np.isclose(m.width, 0.5)

    def _render(rows_labels, tag, group_lbl):
        _fig7_profiles(rows_labels, load_cell, sumdir, tag, group_lbl)

    for tag, pat, glbl in (("7a_shift", "shift", "shift array — cell center lat"),
                           ("7e_shift_hex", "shift_hex", "shift hex array — cell center lat"),
                           ("7b_equator3cell", "equator_3cell", "equator 3-cell — cell center lat")):
        d = m[(m.pattern == pat) & W]
        rs = [(d[np.isclose(d.center_lat, lat)].iloc[0], _lat_lbl(lat))
              for lat in sorted(d.center_lat.unique())]
        if rs:
            _render(rs, tag, glbl)
    d = m[(m.family == "density") & W]
    rs = [(d[d.n_gliders_cell == ng].iloc[0], f"{int(ng)} gliders")
          for ng in sorted(d.n_gliders_cell.unique())]
    if rs:
        _render(rs, "7c_density", "density array — gliders per cell")
    d = m[W]
    rs = [(d[d.pattern == pat].iloc[0], EQS_STYLE[pat]["lbl"])
          for pat in EQS_PATS if (d.pattern == pat).any()]
    if rs:
        _render(rs, "7d_equator_single", "equator single cell — shape & height")


# glider lon offsets shown as the rows of fig8 (one width per row, so the reader
# can see how the time-mean <w> recovery and its CI change with array width).
F8_OFFSETS = [0.5, 1.0, 1.5, 2.0]


def make_fig8(m, sumdir, load_cell):
    """fig8a-8e: the time-mean vertical velocity <w> and its confidence interval,
    exactly like fig7 but with ONE ROW PER glider lon offset (0.5, 1.0, 1.5, 2.0)
    instead of a single fixed width. Columns are the configs of the method group
    plus a final total-over-depth forest panel; rows sweep the array width. All
    panels share x-limits so the rows are directly comparable. See make_fig7 for
    the SE/CI method and the est (solid, blue) vs true (dotted, red) encoding."""

    def _render(keys, lookup, tag, group_lbl):
        # keys: ordered list of (keyval, label) fixing the column layout;
        # lookup(width, keyval) -> metrics row (Series) or None if absent.
        grid = [(w, [(kv, lbl, lookup(w, kv)) for kv, lbl in keys]) for w in F8_OFFSETS]
        grid = [(w, [(kv, lbl, r) for kv, lbl, r in present if r is not None])
                for w, present in grid]
        grid = [(w, present) for w, present in grid if present]
        if not grid:
            return
        ncol, nrow = len(keys), len(grid)
        fig, axes = plt.subplots(nrow, ncol + 1, figsize=(4.1 * (ncol + 1), 5.6 * nrow),
                                 squeeze=False)
        # first pass: per-depth profiles + shared x-limits across every row/config
        Bcache, lo, hi = {}, 0.0, 0.0
        for ri, (w, present) in enumerate(grid):
            for kv, lbl, r in present:
                with load_cell(r) as ds:
                    B = ot.w_skill_by_depth(ds.w_est, ds.w_model)
                Bcache[(ri, kv)] = B
                for mean, se in ((B.w_model_mean, B.w_model_mean_se),
                                 (B.w_est_mean, B.w_est_mean_se)):
                    lo = min(lo, float(((mean - F7_Z * se) * W2DAY).min()))
                    hi = max(hi, float(((mean + F7_Z * se) * W2DAY).max()))
        pad = 0.08 * (hi - lo) or 0.1
        for ri, (w, present) in enumerate(grid):
            for ci, (kv, lbl) in enumerate(keys):
                ax = axes[ri, ci]
                B = Bcache.get((ri, kv))
                if B is None:
                    ax.set_visible(False)
                    continue
                z = B.depth.values
                for mean, se, c, ls in ((B.w_model_mean, B.w_model_mean_se, F7_MOD_C, ":"),
                                        (B.w_est_mean, B.w_est_mean_se, F7_EST_C, "-")):
                    mu = mean.values * W2DAY; hw = F7_Z * se.values * W2DAY
                    ax.fill_betweenx(z, mu - hw, mu + hw, color=c, alpha=0.18, lw=0)
                    ax.plot(mu, z, color=c, ls=ls, lw=1.9)
                ax.axvline(0, color="0.5", lw=0.8, zorder=0)
                ax.set_xlim(lo - pad, hi + pad); ax.grid(alpha=0.3)
                if ri == 0:
                    ax.set_title(lbl)
                if ri == nrow - 1:
                    ax.set_xlabel(r"$\langle w\rangle$  (m day$^{-1}$)")
                if ci == 0:
                    ax.set_ylabel(f"offset = {w:g}°\ndepth (m)")
                else:
                    ax.set_yticklabels([])

            # total-over-depth forest panel for this offset
            axT = axes[ri, ncol]; yv = np.arange(len(present))[::-1]
            for y, (kv, lbl, r) in zip(yv, present):
                for off, c, mk, mkey, skey in (
                        (+0.15, F7_MOD_C, "o", "w_model_mean", "w_model_mean_se"),
                        (-0.15, F7_EST_C, "s", "w_est_mean",   "w_est_mean_se")):
                    axT.errorbar(r[mkey] * W2DAY, y + off, xerr=F7_Z * r[skey] * W2DAY,
                                 fmt=mk, color=c, capsize=3, ms=6, lw=1.6)
            axT.axvline(0, color="0.5", lw=0.8)
            axT.set_yticks(yv); axT.set_yticklabels([lbl for _, lbl, _ in present])
            axT.set_ylim(-0.6, len(present) - 0.4); axT.grid(alpha=0.3, axis="x")
            if ri == 0:
                axT.set_title("total over depth")
            if ri == nrow - 1:
                axT.set_xlabel(r"total $\langle w\rangle$  (m day$^{-1}$)")

        handles = [Line2D([0], [0], color=F7_EST_C, ls="-", lw=1.9,
                          marker="s", label=r"estimated ($\pm$95% CI)"),
                   Line2D([0], [0], color=F7_MOD_C, ls=":", lw=1.9,
                          marker="o", label=r"true / model ($\pm$95% CI)")]
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        fig.savefig(os.path.join(sumdir, f"fig{tag}.png"), dpi=150, bbox_inches="tight")
        plt.show()

    def _lat_lookup(w, lat, pat):
        d = m[(m.pattern == pat) & np.isclose(m.center_lat, lat) & np.isclose(m.width, w)]
        return d.iloc[0] if len(d) else None

    for tag, pat, glbl in (("8a_shift", "shift", "shift array — cell center lat"),
                           ("8e_shift_hex", "shift_hex", "shift hex array — cell center lat"),
                           ("8b_equator3cell", "equator_3cell", "equator 3-cell — cell center lat")):
        lats = sorted(m[m.pattern == pat].center_lat.unique())
        keys = [(lat, _lat_lbl(lat)) for lat in lats]
        if keys:
            _render(keys, (lambda p: lambda w, lat: _lat_lookup(w, lat, p))(pat), tag, glbl)

    ngs = sorted(m[m.family == "density"].n_gliders_cell.unique())
    keys = [(ng, f"{int(ng)} gliders") for ng in ngs]
    if keys:
        def _dens_lookup(w, ng):
            d = m[(m.family == "density") & (m.n_gliders_cell == ng) & np.isclose(m.width, w)]
            return d.iloc[0] if len(d) else None
        _render(keys, _dens_lookup, "8c_density", "density array — gliders per cell")

    keys = [(pat, EQS_STYLE[pat]["lbl"]) for pat in EQS_PATS if (m.pattern == pat).any()]
    if keys:
        def _eqs_lookup(w, pat):
            d = m[(m.pattern == pat) & np.isclose(m.width, w)]
            return d.iloc[0] if len(d) else None
        _render(keys, _eqs_lookup, "8d_equator_single", "equator single cell — shape & height")


# ============================================================================
# Figure builders — exp1 vs exp2 comparison
# ============================================================================
EXP_STYLE = {
    1: dict(ls="-",  fill=True,  lbl="no extrapolation"),
    2: dict(ls="--", fill=False, lbl="with extrapolation"),
}


def mfc(exp, color):
    """Marker face color: filled (exp1) or open/white (exp2)."""
    return color if EXP_STYLE[exp]["fill"] else "white"


def exp_handles():
    return [Line2D([0], [0], color="0.3", ls=EXP_STYLE[e]["ls"], lw=1.9, label=EXP_STYLE[e]["lbl"])
            for e in (1, 2)]


# scatter legend for the two experiment fill states (edge carries the key color)
_EXP_FILL = [Line2D([0], [0], ls="", marker="o", mfc="0.4", mec="0.4", ms=8, label="no extrapolation (filled)"),
             Line2D([0], [0], ls="", marker="o", mfc="white", mec="0.4", mew=1.4, ms=8, label="with extrapolation (open)")]


def make_fig2abcd_compare(M, outdir):
    """fig2a-2d comparison: exp1 (solid/filled) vs exp2 (dashed/open) on a 2x4 grid."""
    panels = [
        ("2a_shift", lambda mm: mm.pattern == "shift", "center_lat", LAT_COLORS, "cell center lat", _lat_lbl,
         lambda ax: stacked_cells_panel(ax, [-1.5, -0.5, 0.5, 1.5], 0.5, "array layout (offset = 0.5deg)")),
        ("2e_shift_hex", lambda mm: mm.pattern == "shift_hex", "center_lat", LAT_COLORS, "cell center lat", _lat_lbl,
         lambda ax: stacked_hex_cells_panel(ax, [-1.5, -0.5, 0.5, 1.5], 0.5, "array layout (offset = 0.5deg)")),
        ("2b_equator3cell", lambda mm: mm.pattern == "equator_3cell", "center_lat", LAT_COLORS, "cell center lat", _lat_lbl,
         lambda ax: stacked_cells_panel(ax, [-1.0, 0.0, 1.0], 1.0, "array layout (offset = 0.5deg)")),
        ("2c_density", lambda mm: mm.family == "density", "n_gliders_cell", GLIDER_COLORS, "gliders/cell",
         lambda v: f"{int(v)}", density_geometry_panel),
    ]
    for tag, maskfn, key, cmap, leg_title, lbl, geom in panels:
        keys = sorted(M[1][maskfn(M[1])][key].unique())
        fig, axes = plt.subplots(2, 4, figsize=(21, 10))
        series = []
        for exp in (1, 2):
            st = EXP_STYLE[exp]; d = M[exp][maskfn(M[exp])]
            for kv in keys:
                dd = d[d[key] == kv].drop_duplicates("width").sort_values("width")
                lkw = dict(color=cmap[kv], ls=st["ls"], lw=1.9)
                skw = dict(color=cmap[kv], ls="", marker="o", ms=6, mew=1.2,
                           mec=cmap[kv], mfc=mfc(exp, cmap[kv]))
                series.append((dd, lkw, skw))
        fill_skill_panels(axes, series)
        label_skill_panels(axes)
        col_handles = [Line2D([0], [0], color=cmap[kv], marker="o", mec="k", mew=0.3,
                              lw=1.9, label=lbl(kv)) for kv in keys]
        axes[0, 3].legend(handles=col_handles + _EXP_FILL, loc="best", frameon=True, ncol=2)
        geom(axes[1, 3])
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        lg1 = fig.legend(handles=col_handles, title=f"color = {leg_title}", loc="upper center",
                         bbox_to_anchor=(0.5, 1.0), ncol=len(col_handles), frameon=False)
        fig.add_artist(lg1)
        fig.legend(handles=exp_handles(), title="experiment", loc="upper center",
                   bbox_to_anchor=(0.85, 1.0), ncol=1, frameon=False, handlelength=3.0)
        fig.savefig(os.path.join(outdir, f"fig{tag}_compare.png"), dpi=150, bbox_inches="tight")
        plt.show()

    # 2d: color = shape, line width = height, linestyle/fill = experiment. The
    # scatter panel carries height in marker size (small=1°, large=2°).
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    series = []
    for exp in (1, 2):
        st = EXP_STYLE[exp]
        for pat in EQS_PATS:
            s = EQS_STYLE[pat]
            dd = M[exp][M[exp].pattern == pat].sort_values("width")
            if dd.empty:
                continue
            lkw = dict(ls=st["ls"], color=s["color"], lw=s["lw"])
            skw = dict(ls="", marker="o", color=s["color"], ms=s["ms"], mew=1.2,
                       mec=s["color"], mfc=mfc(exp, s["color"]))
            series.append((dd, lkw, skw))
    fill_skill_panels(axes, series)
    label_skill_panels(axes)
    shape_handles = [Line2D([0], [0], color=c, lw=2.4, label=s) for s, c in EQS_SHAPE_COLOR.items()]
    height_handles = [Line2D([0], [0], color="0.35", ls="-", lw=lw, label=f"{h:g} deg tall")
                      for h, lw in EQS_HEIGHT_LW.items()]
    scat_handles = [Line2D([0], [0], ls="", marker="o", color="0.35", mec="k", mew=0.3,
                           ms=ms, label=f"{h:g} deg tall") for h, ms in EQS_HEIGHT_MS.items()]
    scat_handles += _EXP_FILL
    axes[0, 3].legend(handles=scat_handles, title="size = height", loc="best", frameon=True, ncol=2)
    equator_single_geometry(axes[1, 3])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    lg1 = fig.legend(handles=shape_handles, title="color = shape", loc="upper center",
                     bbox_to_anchor=(0.38, 1.0), ncol=3, frameon=False)
    fig.add_artist(lg1)
    lg2 = fig.legend(handles=height_handles, title="line width = height", loc="upper center",
                     bbox_to_anchor=(0.7, 1.0), ncol=2, frameon=False, handlelength=3.0)
    fig.add_artist(lg2)
    fig.legend(handles=exp_handles(), title="experiment", loc="upper center",
               bbox_to_anchor=(0.92, 1.0), ncol=1, frameon=False, handlelength=3.0)
    fig.savefig(os.path.join(outdir, "fig2d_equator_single_compare.png"), dpi=150, bbox_inches="tight")
    plt.show()


def make_fig5_compare(M, outdir, load_cell):
    """fig5a-5d comparison: overlay depth profiles for both experiments (2x4 grid).
    load_cell(exp, row) -> Dataset with w_est, w_model."""
    depth_width = 0.5

    def _depth_compare(rows_by_exp, tag, leg_title, line_handles):
        fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=True)
        for exp in (1, 2):
            est = EXP_STYLE[exp]
            for r, s in rows_by_exp[exp]:
                ds = load_cell(exp, r)
                B = ot.w_skill_by_depth(ds.w_est, ds.w_model).sortby("depth")
                z = B.depth
                common = dict(color=s["color"], ls=est["ls"], lw=s.get("lw", 1.8))
                axes[0, 0].plot(B.w_model_mean * W2DAY, z, color="0.6", ls=":", lw=1.1, alpha=0.6)
                axes[0, 0].plot(B.w_est_mean * W2DAY, z, **common)
                axes[0, 1].plot(B.mean_bias / B.w_model_std, z, **common)
                axes[0, 2].plot(ot.frac_mean_bias(B.mean_bias, B.w_model_mean), z, **common)
                axes[0, 3].plot(B.rms, z, **common)
                axes[0, 3].plot(B.w_model_std, z, color="0.6", ls=est["ls"], lw=1.0, alpha=0.6)
                axes[1, 0].plot(B.w_est_std / B.w_model_std, z, **common)
                axes[1, 1].plot(B["corr"], z, **common)
                axes[1, 2].plot(B.norm_rms, z, **common)
        axes[0, 0].set_title(r"mean $\langle w\rangle$: est & true  [m day$^{-1}$]")
        axes[0, 0].axvline(0, color="0.5", lw=0.8, zorder=0)
        axes[0, 0].legend(handles=[Line2D([0], [0], color="0.35", ls="-", lw=1.9, label="estimated"),
                                   Line2D([0], [0], color="0.6", ls=":", lw=1.2, label="true (model)")],
                          loc="best", frameon=True)
        axes[0, 1].set_title(r"mean bias / $\sigma_y$"); axes[0, 1].axvline(0, color="0.5", lw=0.8, zorder=0)
        axes[0, 2].set_title(r"mean bias / $\langle w\rangle$"); axes[0, 2].axvline(0, color="0.5", lw=0.8, zorder=0)
        axes[0, 3].set_title(r"RMS (colored) & signal $\sigma$ (gray)  [m s$^{-1}$]")
        axes[1, 0].set_title(r"$\sigma_x/\sigma_y$  (est / model)"); axes[1, 0].axvline(1, color="0.5", lw=0.8, zorder=0)
        axes[1, 1].set_title("correlation  r")
        axes[1, 2].set_title(r"relative error  RMS/$\sigma_y$")
        axes[1, 3].axis("off")
        axes[0, 0].set_ylabel("depth (m)"); axes[1, 0].set_ylabel("depth (m)")
        for a, key in ((axes[0, 0], "w_mean"), (axes[0, 1], "bias_sig"),
                       (axes[0, 2], "frac_bias"), (axes[0, 3], "rms_ms"),
                       (axes[1, 0], "sigma_ratio"), (axes[1, 1], "corr"),
                       (axes[1, 2], "norm_rms")):
            a.set_xlim(LIMITS[key])
        for a in axes.flat:
            if a.axison:
                a.grid(alpha=0.3)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        lg1 = fig.legend(handles=line_handles, title=f"color = {leg_title}", loc="upper center",
                         bbox_to_anchor=(0.5, 1.0), ncol=len(line_handles), frameon=False)
        fig.add_artist(lg1)
        fig.legend(handles=exp_handles(), title="experiment", loc="upper center",
                   bbox_to_anchor=(0.88, 1.0), ncol=1, frameon=False, handlelength=3.0)
        fig.savefig(os.path.join(outdir, f"fig{tag}_compare.png"), dpi=150, bbox_inches="tight")
        plt.show()

    for tag, pat in (("5a_shift", "shift"), ("5e_shift_hex", "shift_hex"),
                     ("5b_equator3cell", "equator_3cell")):
        rows_by_exp = {}
        for exp in (1, 2):
            d = M[exp][(M[exp].pattern == pat) & np.isclose(M[exp].width, depth_width)]
            rows_by_exp[exp] = [(d[np.isclose(d.center_lat, lat)].iloc[0],
                                 dict(color=LAT_COLORS[lat], lbl=f"{lat:+.1f}"))
                                for lat in sorted(d.center_lat.unique())]
        lats = sorted(M[1][M[1].pattern == pat].center_lat.unique())
        line_handles = [Line2D([0], [0], color=LAT_COLORS[l], lw=1.9, label=f"{l:+.1f}") for l in lats]
        _depth_compare(rows_by_exp, tag, "cell center lat", line_handles)

    rows_by_exp = {}
    for exp in (1, 2):
        d = M[exp][(M[exp].family == "density") & np.isclose(M[exp].width, depth_width)]
        rows_by_exp[exp] = [(d[d.n_gliders_cell == ng].iloc[0], dict(color=GLIDER_COLORS[ng], lbl=f"{int(ng)}"))
                            for ng in sorted(d.n_gliders_cell.unique())]
    ngs = sorted(M[1][M[1].family == "density"].n_gliders_cell.unique())
    line_handles = [Line2D([0], [0], color=GLIDER_COLORS[ng], lw=1.9, label=f"{int(ng)}") for ng in ngs]
    _depth_compare(rows_by_exp, "5c_density", "gliders/cell", line_handles)

    rows_by_exp = {}
    for exp in (1, 2):
        d = M[exp][np.isclose(M[exp].width, depth_width)]
        rows_by_exp[exp] = [(d[d.pattern == pat].iloc[0],
                             dict(color=EQS_STYLE[pat]["color"], lw=EQS_STYLE[pat]["lw"],
                                  lbl=EQS_STYLE[pat]["lbl"]))
                            for pat in EQS_PATS if (d.pattern == pat).any()]
    line_handles = [Line2D([0], [0], color=EQS_STYLE[p]["color"], lw=EQS_STYLE[p]["lw"],
                           label=EQS_STYLE[p]["lbl"]) for p in EQS_PATS]
    _depth_compare(rows_by_exp, "5d_equator_single", "shape (color) x height (line width)", line_handles)


def make_fig6_compare(M, outdir):
    """fig6 (+6a-d) comparison: exp2 - exp1 improvement heatmaps (teal = exp2 better).
    The mean-recovery improvement equals |bias/<w>| under differencing, so the
    redundant bias/<w> slot is left blank."""
    def _lk(df, pat, lat, w, col):
        sel = df[(df["pattern"] == pat) & np.isclose(df["center_lat"], lat) & np.isclose(df["width"], w)]
        return sel.iloc[0][col] if len(sel) else np.nan

    def _delta(kind, order):
        g = np.full((len(order), len(WIDTHS)), np.nan)
        for i, (pat, lat) in enumerate(order):
            for j, w in enumerate(WIDTHS):
                if kind == "rms":
                    g[i, j] = _lk(M[1], pat, lat, w, "norm_rms") - _lk(M[2], pat, lat, w, "norm_rms")
                elif kind == "corr":
                    g[i, j] = _lk(M[2], pat, lat, w, "corr") - _lk(M[1], pat, lat, w, "corr")
                elif kind in ("bias", "wmean"):     # |bias/<w>| improvement (== mean recovery)
                    a = ot.frac_mean_bias(_lk(M[1], pat, lat, w, "mean_bias"), _lk(M[1], pat, lat, w, "w_model_mean"))
                    b = ot.frac_mean_bias(_lk(M[2], pat, lat, w, "mean_bias"), _lk(M[2], pat, lat, w, "w_model_mean"))
                    g[i, j] = abs(a) - abs(b)
                elif kind == "bias_sig":
                    a = _lk(M[1], pat, lat, w, "mean_bias") / _lk(M[1], pat, lat, w, "w_model_std")
                    b = _lk(M[2], pat, lat, w, "mean_bias") / _lk(M[2], pat, lat, w, "w_model_std")
                    g[i, j] = abs(a) - abs(b)
                else:                               # |sigma_x/sigma_y - 1|
                    a = _lk(M[1], pat, lat, w, "w_est_std") / _lk(M[1], pat, lat, w, "w_model_std")
                    b = _lk(M[2], pat, lat, w, "w_est_std") / _lk(M[2], pat, lat, w, "w_model_std")
                    g[i, j] = abs(a - 1.0) - abs(b - 1.0)
        return g

    # (flat slot, kind, title); flat index 2 (bias/<w>) left blank as redundant
    panels = [
        (0, "wmean",    r"$\Delta$ mean recovery |est/model $\langle w\rangle - 1$|  (teal = exp2 better)"),
        (1, "bias_sig", r"$\Delta$ |bias/$\sigma_y$|  (teal = exp2 less biased)"),
        (3, "sigma",    r"$\Delta$ |$\sigma_x/\sigma_y-1$|  (teal = exp2 better amplitude)"),
        (4, "corr",     r"$\Delta$ correlation  (teal = exp2 higher corr)"),
        (5, "rms",      r"$\Delta$ RMS/$\sigma_y$  (teal = exp2 lower error)"),
    ]

    def _fig(order, fname, figsize):
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=True)
        axes.flat[2].axis("off")
        for slot, kind, title in panels:
            im = heatmap_cell(axes.flat[slot], _delta(kind, order), order, "BrBG", 0.0,
                              "{:+.2f}", title, sep_color="0.5")
            fig.colorbar(im, ax=axes.flat[slot], fraction=0.046, pad=0.02)
        _set_row_labels(axes, order)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
        plt.show()

    _fig(ORDER, "fig6_metric_heatmaps_compare.png", (19, 13))
    for tag, order in SUB_ORDERS:
        _fig(order, f"fig{tag}_compare.png", (19, max(6.5, 0.7 * len(order) + 4.5)))


# color for the two experiments' estimates in fig7 compare (truth is gray).
# Linestyle still carries experiment (solid=exp1, dashed=exp2, per convention);
# color is added only so the two CI bands are distinguishable in a shared panel.
F7C_EXP_C = {1: "#1f77b4", 2: "#e6820a"}
F7C_TRUE_C = "0.35"


def make_fig7_compare(M, outdir, load_cell):
    """fig7a-7e comparison: the estimated time-mean <w> +/- autocorrelation-aware
    95% CI for exp1 (solid) vs exp2 (dashed), against the shared model truth
    (gray dotted), per config. Same groups as fig5/fig7. Each config gets a depth
    profile (est CI bands per experiment + the true band); a final panel shows the
    total-over-depth <w> +/- CI as a forest plot (true, exp1, exp2 per config).
    The truth is drawn from exp1 (identical to exp2 on shared depths; exp2 merely
    extends to the surface). load_cell(exp, row) -> Dataset with w_est, w_model."""
    W_ = 0.5

    def _match(cfgs, tag, group_lbl):
        # cfgs: list of (label, {1: row1, 2: row2})
        ncfg = len(cfgs)
        fig, axes = plt.subplots(1, ncfg + 1, figsize=(3.3 * (ncfg + 1), 6.3))
        axes = np.atleast_1d(axes)
        profs, lo, hi = [], 0.0, 0.0
        for lbl, rows in cfgs:
            B = {}
            for e in (1, 2):
                with load_cell(e, rows[e]) as ds:
                    B[e] = ot.w_skill_by_depth(ds.w_est, ds.w_model)
            profs.append((lbl, B))
            for mean, se in ((B[1].w_model_mean, B[1].w_model_mean_se),
                             (B[1].w_est_mean, B[1].w_est_mean_se),
                             (B[2].w_est_mean, B[2].w_est_mean_se)):
                lo = min(lo, float(((mean - F7_Z * se) * W2DAY).min()))
                hi = max(hi, float(((mean + F7_Z * se) * W2DAY).max()))
        pad = 0.08 * (hi - lo) or 0.1
        for ax, (lbl, B) in zip(axes[:ncfg], profs):
            zt = B[1].depth.values                       # shared model truth (exp1)
            mu = B[1].w_model_mean.values * W2DAY; hw = F7_Z * B[1].w_model_mean_se.values * W2DAY
            ax.fill_betweenx(zt, mu - hw, mu + hw, color=F7C_TRUE_C, alpha=0.13, lw=0)
            ax.plot(mu, zt, color=F7C_TRUE_C, ls=":", lw=1.5)
            for e in (1, 2):
                z = B[e].depth.values
                mu = B[e].w_est_mean.values * W2DAY; hw = F7_Z * B[e].w_est_mean_se.values * W2DAY
                ax.fill_betweenx(z, mu - hw, mu + hw, color=F7C_EXP_C[e], alpha=0.15, lw=0)
                ax.plot(mu, z, color=F7C_EXP_C[e], ls=EXP_STYLE[e]["ls"], lw=1.9)
            ax.axvline(0, color="0.5", lw=0.8, zorder=0)
            ax.set_title(lbl); ax.set_xlim(lo - pad, hi + pad); ax.grid(alpha=0.3)
            ax.set_xlabel(r"$\langle w\rangle$  (m day$^{-1}$)")
        axes[0].set_ylabel("depth (m)")
        for ax in axes[1:ncfg]:
            ax.set_yticklabels([])

        axT = axes[ncfg]; yv = np.arange(ncfg)[::-1]
        for y, (lbl, rows) in zip(yv, cfgs):
            r1 = rows[1]
            axT.errorbar(r1["w_model_mean"] * W2DAY, y + 0.22,
                         xerr=F7_Z * r1["w_model_mean_se"] * W2DAY,
                         fmt="o", color=F7C_TRUE_C, capsize=3, ms=6, lw=1.6)
            for e, off, mk in ((1, 0.0, "s"), (2, -0.22, "D")):
                r = rows[e]
                axT.errorbar(r["w_est_mean"] * W2DAY, y + off,
                             xerr=F7_Z * r["w_est_mean_se"] * W2DAY,
                             fmt=mk, color=F7C_EXP_C[e], capsize=3, ms=6, lw=1.6)
        axT.axvline(0, color="0.5", lw=0.8)
        axT.set_yticks(yv); axT.set_yticklabels([lbl for lbl, _ in cfgs])
        axT.set_ylim(-0.6, ncfg - 0.4)
        axT.set_xlabel(r"total $\langle w\rangle$  (m day$^{-1}$)")
        axT.set_title("total over depth"); axT.grid(alpha=0.3, axis="x")

        handles = [Line2D([0], [0], color=F7C_TRUE_C, ls=":", lw=1.9, marker="o",
                          label=r"true / model ($\pm$95% CI)"),
                   Line2D([0], [0], color=F7C_EXP_C[1], ls="-", lw=1.9, marker="s",
                          label=EXP_STYLE[1]["lbl"] + r" ($\pm$95% CI)"),
                   Line2D([0], [0], color=F7C_EXP_C[2], ls="--", lw=1.9, marker="D",
                          label=EXP_STYLE[2]["lbl"] + r" ($\pm$95% CI)")]
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.legend(handles=handles, title=group_lbl, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        fig.savefig(os.path.join(outdir, f"fig{tag}_compare.png"), dpi=150, bbox_inches="tight")
        plt.show()

    def _rows_for(key_mask, key_col, key_val):
        rows = {}
        for e in (1, 2):
            d = M[e][key_mask(M[e]) & (M[e][key_col] == key_val) & np.isclose(M[e].width, W_)] \
                if key_col != "center_lat" else \
                M[e][key_mask(M[e]) & np.isclose(M[e].center_lat, key_val) & np.isclose(M[e].width, W_)]
            if d.empty:
                return None
            rows[e] = d.iloc[0]
        return rows

    for tag, pat, glbl in (("7a_shift", "shift", "shift array — cell center lat"),
                           ("7e_shift_hex", "shift_hex", "shift hex — cell center lat"),
                           ("7b_equator3cell", "equator_3cell", "equator 3-cell — cell center lat")):
        mask = (lambda p: (lambda mm: mm.pattern == p))(pat)
        cfgs = []
        for lat in sorted(M[1][mask(M[1])].center_lat.unique()):
            rows = _rows_for(mask, "center_lat", lat)
            if rows:
                cfgs.append((_lat_lbl(lat), rows))
        if cfgs:
            _match(cfgs, tag, glbl)
    mask = lambda mm: mm.family == "density"
    cfgs = []
    for ng in sorted(M[1][mask(M[1])].n_gliders_cell.unique()):
        rows = _rows_for(mask, "n_gliders_cell", ng)
        if rows:
            cfgs.append((f"{int(ng)} gliders", rows))
    if cfgs:
        _match(cfgs, "7c_density", "density array — gliders per cell")
    mask = lambda mm: mm.width == mm.width      # all rows
    cfgs = []
    for pat in EQS_PATS:
        rows = _rows_for(lambda mm: mm.pattern == pat, "pattern", pat)
        if rows:
            cfgs.append((EQS_STYLE[pat]["lbl"], rows))
    if cfgs:
        _match(cfgs, "7d_equator_single", "equator single cell — shape & height")


def make_fig8_compare(M, outdir, load_cell):
    """fig8a-8e comparison: like make_fig7_compare but with ONE ROW PER glider lon
    offset (0.5, 1.0, 1.5, 2.0) instead of a single fixed width. Columns are the
    configs of the method group plus a total-over-depth forest panel; rows sweep
    the array width. Per config: the shared model truth (gray dotted) plus the
    estimated <w> +/- 95% CI for exp1 (solid blue) and exp2 (dashed orange). All
    panels share x-limits so rows are comparable. load_cell(exp, row) -> Dataset."""

    def _render(keys, lookup, tag, group_lbl):
        # keys: ordered list of (keyval, label); lookup(width, keyval) -> {1:row,2:row}
        # or None if either experiment lacks that config at that width.
        grid = [(w, [(kv, lbl, lookup(w, kv)) for kv, lbl in keys]) for w in F8_OFFSETS]
        grid = [(w, [(kv, lbl, rows) for kv, lbl, rows in present if rows is not None])
                for w, present in grid]
        grid = [(w, present) for w, present in grid if present]
        if not grid:
            return
        ncol, nrow = len(keys), len(grid)
        fig, axes = plt.subplots(nrow, ncol + 1, figsize=(4.1 * (ncol + 1), 5.6 * nrow),
                                 squeeze=False)
        Bcache, lo, hi = {}, 0.0, 0.0
        for ri, (w, present) in enumerate(grid):
            for kv, lbl, rows in present:
                B = {}
                for e in (1, 2):
                    with load_cell(e, rows[e]) as ds:
                        B[e] = ot.w_skill_by_depth(ds.w_est, ds.w_model)
                Bcache[(ri, kv)] = B
                for mean, se in ((B[1].w_model_mean, B[1].w_model_mean_se),
                                 (B[1].w_est_mean, B[1].w_est_mean_se),
                                 (B[2].w_est_mean, B[2].w_est_mean_se)):
                    lo = min(lo, float(((mean - F7_Z * se) * W2DAY).min()))
                    hi = max(hi, float(((mean + F7_Z * se) * W2DAY).max()))
        pad = 0.08 * (hi - lo) or 0.1
        for ri, (w, present) in enumerate(grid):
            for ci, (kv, lbl) in enumerate(keys):
                ax = axes[ri, ci]
                B = Bcache.get((ri, kv))
                if B is None:
                    ax.set_visible(False)
                    continue
                zt = B[1].depth.values                   # shared model truth (exp1)
                mu = B[1].w_model_mean.values * W2DAY
                hw = F7_Z * B[1].w_model_mean_se.values * W2DAY
                ax.fill_betweenx(zt, mu - hw, mu + hw, color=F7C_TRUE_C, alpha=0.13, lw=0)
                ax.plot(mu, zt, color=F7C_TRUE_C, ls=":", lw=1.5)
                for e in (1, 2):
                    z = B[e].depth.values
                    mu = B[e].w_est_mean.values * W2DAY
                    hw = F7_Z * B[e].w_est_mean_se.values * W2DAY
                    ax.fill_betweenx(z, mu - hw, mu + hw, color=F7C_EXP_C[e], alpha=0.15, lw=0)
                    ax.plot(mu, z, color=F7C_EXP_C[e], ls=EXP_STYLE[e]["ls"], lw=1.9)
                ax.axvline(0, color="0.5", lw=0.8, zorder=0)
                ax.set_xlim(lo - pad, hi + pad); ax.grid(alpha=0.3)
                if ri == 0:
                    ax.set_title(lbl)
                if ri == nrow - 1:
                    ax.set_xlabel(r"$\langle w\rangle$  (m day$^{-1}$)")
                if ci == 0:
                    ax.set_ylabel(f"offset = {w:g}°\ndepth (m)")
                else:
                    ax.set_yticklabels([])

            axT = axes[ri, ncol]; yv = np.arange(len(present))[::-1]
            for y, (kv, lbl, rows) in zip(yv, present):
                r1 = rows[1]
                axT.errorbar(r1["w_model_mean"] * W2DAY, y + 0.22,
                             xerr=F7_Z * r1["w_model_mean_se"] * W2DAY,
                             fmt="o", color=F7C_TRUE_C, capsize=3, ms=6, lw=1.6)
                for e, off, mk in ((1, 0.0, "s"), (2, -0.22, "D")):
                    r = rows[e]
                    axT.errorbar(r["w_est_mean"] * W2DAY, y + off,
                                 xerr=F7_Z * r["w_est_mean_se"] * W2DAY,
                                 fmt=mk, color=F7C_EXP_C[e], capsize=3, ms=6, lw=1.6)
            axT.axvline(0, color="0.5", lw=0.8)
            axT.set_yticks(yv); axT.set_yticklabels([lbl for _, lbl, _ in present])
            axT.set_ylim(-0.6, len(present) - 0.4); axT.grid(alpha=0.3, axis="x")
            if ri == 0:
                axT.set_title("total over depth")
            if ri == nrow - 1:
                axT.set_xlabel(r"total $\langle w\rangle$  (m day$^{-1}$)")

        handles = [Line2D([0], [0], color=F7C_TRUE_C, ls=":", lw=1.9, marker="o",
                          label=r"true / model ($\pm$95% CI)"),
                   Line2D([0], [0], color=F7C_EXP_C[1], ls="-", lw=1.9, marker="s",
                          label=EXP_STYLE[1]["lbl"] + r" ($\pm$95% CI)"),
                   Line2D([0], [0], color=F7C_EXP_C[2], ls="--", lw=1.9, marker="D",
                          label=EXP_STYLE[2]["lbl"] + r" ($\pm$95% CI)")]
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        fig.savefig(os.path.join(outdir, f"fig{tag}_compare.png"), dpi=150, bbox_inches="tight")
        plt.show()

    def _rows_for(key_mask, key_col, key_val, w):
        rows = {}
        for e in (1, 2):
            d = M[e][key_mask(M[e]) & (M[e][key_col] == key_val) & np.isclose(M[e].width, w)] \
                if key_col != "center_lat" else \
                M[e][key_mask(M[e]) & np.isclose(M[e].center_lat, key_val) & np.isclose(M[e].width, w)]
            if d.empty:
                return None
            rows[e] = d.iloc[0]
        return rows

    for tag, pat, glbl in (("8a_shift", "shift", "shift array — cell center lat"),
                           ("8e_shift_hex", "shift_hex", "shift hex — cell center lat"),
                           ("8b_equator3cell", "equator_3cell", "equator 3-cell — cell center lat")):
        mask = (lambda p: (lambda mm: mm.pattern == p))(pat)
        keys = [(lat, _lat_lbl(lat)) for lat in sorted(M[1][mask(M[1])].center_lat.unique())]
        if keys:
            _render(keys, (lambda mk: lambda w, lat: _rows_for(mk, "center_lat", lat, w))(mask),
                    tag, glbl)

    mask = lambda mm: mm.family == "density"
    keys = [(ng, f"{int(ng)} gliders") for ng in sorted(M[1][mask(M[1])].n_gliders_cell.unique())]
    if keys:
        _render(keys, lambda w, ng: _rows_for(mask, "n_gliders_cell", ng, w),
                "8c_density", "density array — gliders per cell")

    keys = [(pat, EQS_STYLE[pat]["lbl"]) for pat in EQS_PATS if (M[1].pattern == pat).any()]
    if keys:
        _render(keys,
                lambda w, pat: _rows_for((lambda p: lambda mm: mm.pattern == p)(pat), "pattern", pat, w),
                "8d_equator_single", "equator single cell — shape & height")


# ---- reg_hex: regular vs nearest-irregular hexagon (one-off test) ----
# `_equator_hex_cell` is regular only when off = sqrt(3)/2 * half_height, which no swept
# WIDTH hits -- so every equator_hex* config is a stretched, 2-fold hexagon. A regular
# hexagon's plane-fit aliasing is orientation-independent.
REGHEX_PAIRS = [                       # (reg_config, sweep_pattern, height, nearest_irreg_width)
    ("reg_hex1deg", "equator_hex1deg", 1.0, 0.5),
    ("reg_hex2deg", "equator_hex2deg", 2.0, 1.0),
]


SYMHEX_CENTERS = [0.0, 0.5, -0.5]                # cell centers for the regular-hex sweep
SYM_DIAMS = [0.3, 0.5, 0.75, 1.0]                # E-W diameters (°) for the shape sweeps
# the three regular-shape families (hexagon/diamond/square) run at the same diameters
SYM_FAMILIES = ["symhex", "symdia", "symsq"]
SYM_SHAPE = {"symhex": "hexagon", "symdia": "diamond", "symsq": "square"}
SYM_MARKER = {"symhex": "h", "symdia": "D", "symsq": "s"}


def make_sym_summary(m, sumdir, family="symhex"):
    """Six-panel w-skill comparison (figure-2 metric order) of a symmetric REGULAR shape
    family (symhex/symdia/symsq) vs its E-W diameter (0.3/0.5/0.75/1.0°), one line per
    center latitude (0.0/±0.5°N). x = diameter = 2 × glider lon offset. These are
    geometrically regular (isotropic) footprints with no moorings, so the sweep isolates
    how the plane-fit w skill degrades as the footprint shrinks, independent of shape."""
    shape = SYM_SHAPE[family]
    d = m[m.family == family].copy()
    if d.empty:
        raise ValueError(f"metrics.csv has no {family} configs — run run_experiment_2.py")
    d["diameter"] = 2.0 * d["width"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for c in SYMHEX_CENTERS:
        sub = d[np.isclose(d.center_lat, c)].sort_values("diameter")
        if sub.empty:
            continue
        plot6(axes, sub.diameter, sub, color=LAT_COLORS[c], lw=2.0, marker=SYM_MARKER[family],
              ms=9, mec="k", mew=0.4, label=f"{c:+.1f}°")
    label6(axes, f"{shape} diameter (deg)")
    for a in axes.flat:
        a.set_xticks([0.3, 0.5, 0.75, 1.0])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    h, l = axes.flat[1].get_legend_handles_labels()
    fig.legend(h, l, title="cell center latitude", loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(l), frameon=False)
    out = os.path.join(sumdir, f"fig_{family}_skill_vs_diameter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    return out


def make_symhex_summary(m, sumdir):
    """Back-compat wrapper — the regular-hexagon w-skill sweep."""
    return make_sym_summary(m, sumdir, "symhex")


def make_sym_shape_summary(m, sumdir, center=0.0):
    """Six-panel w-skill SHAPE comparison (figure-2 metric order) of the three regular
    shapes (hexagon/diamond/square) vs E-W diameter at a fixed center latitude, one line
    per shape. Isolates how the footprint SHAPE — not size — affects plane-fit w skill."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for fam in SYM_FAMILIES:
        d = m[(m.family == fam) & np.isclose(m.center_lat, center)].copy()
        if d.empty:
            continue
        d["diameter"] = 2.0 * d["width"]
        d = d.sort_values("diameter")
        plot6(axes, d.diameter, d, color=EQS_SHAPE_COLOR[SYM_SHAPE[fam]], lw=2.0,
              marker=SYM_MARKER[fam], ms=9, mec="k", mew=0.4, label=SYM_SHAPE[fam])
    label6(axes, "array diameter (deg)")
    for a in axes.flat:
        a.set_xticks([0.3, 0.5, 0.75, 1.0])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    h, l = axes.flat[1].get_legend_handles_labels()
    fig.legend(h, l, title=f"regular shape (center {center:+.1f}°)", loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(l), frameon=False)
    out = os.path.join(sumdir, f"fig_sym_shape_skill_vs_diameter_c{center:+.1f}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    return out


def _sym_family_rows(m, family, diam):
    """Metrics rows (one per center latitude, sorted S->N) for a sym family at diameter
    `diam`°, each paired with a 'center ±x°' column label."""
    d = m[(m.family == family) & np.isclose(m.width, diam / 2.0)]
    return [(d[np.isclose(d.center_lat, c)].iloc[0], f"center {c:+.1f}°")
            for c in sorted(d.center_lat.unique())]


def make_fig7_sym(m, sumdir, load_cell, family="symhex", diam=1.0):
    """fig7-style time-mean <w> +/-95% CI depth profiles for one regular-shape family at a
    representative diameter, one column per center latitude (+ total-over-depth forest) —
    shows how the LATITUDE affects the resolved <w> at each depth. Reuses `_fig7_profiles`."""
    rows = _sym_family_rows(m, family, diam)
    if not rows:
        raise ValueError(f"metrics.csv has no {family} cells at diameter {diam}°")
    _fig7_profiles(rows, load_cell, sumdir, f"7_{family}_d{diam:g}",
                   f"{SYM_SHAPE[family]}, {diam:g}° diameter — cell center latitude")


def make_fig7_sym_shapes(m, sumdir, load_cell, center=0.0, diam=1.0):
    """fig7-style time-mean <w> +/-95% CI depth profiles COMPARING the three regular shapes
    (hexagon/diamond/square) at one center latitude and diameter, one column per shape (+
    total-over-depth forest) — shows how the SHAPE affects the resolved <w> at each depth."""
    rows = []
    for fam in SYM_FAMILIES:
        d = m[(m.family == fam) & np.isclose(m.width, diam / 2.0)
              & np.isclose(m.center_lat, center)]
        if not d.empty:
            rows.append((d.iloc[0], SYM_SHAPE[fam]))
    if not rows:
        raise ValueError(f"metrics.csv has no sym cells at center {center}°, diameter {diam}°")
    _fig7_profiles(rows, load_cell, sumdir, f"7_sym_shapes_c{center:+.1f}_d{diam:g}",
                   f"regular shapes, {diam:g}° diameter @ center {center:+.1f}°")


def make_sym_w_scatter(data_dir, sumdir, family):
    """Estimated-vs-true w scatter for one regular-shape family: a grid with rows = center
    latitude and columns = E-W diameter. Each panel pools every (time, depth) sample of the
    array's plane-fit w (est) against the model-truth hull-mean w (true), a subsample plotted
    colored by depth, with the 1:1 line and the correlation. All panels share limits."""
    import xarray as xr
    rng = np.random.default_rng(0)
    nrow, ncol = len(SYMHEX_CENTERS), len(SYM_DIAMS)
    centers = sorted(SYMHEX_CENTERS)                         # S -> N down the rows
    # load once; gather points and a shared symmetric limit + depth range
    pts, lim, zmax = {}, 0.0, 0.0
    for c in centers:
        for d in SYM_DIAMS:
            cfg = f"{family}_d{d}_c{c:+.1f}"
            with xr.open_dataset(os.path.join(data_dir, f"{cfg}__cell_{c:+.2f}.nc")) as ds:
                zt = np.broadcast_to(np.abs(ds.depth.values)[None, :], ds.w_est.shape)
                x = (ds.w_model.values * W2DAY).ravel()
                y = (ds.w_est.values * W2DAY).ravel()
                z = zt.ravel()
            g = np.isfinite(x) & np.isfinite(y)
            x, y, z = x[g], y[g], z[g]
            if x.size > 8000:
                sel = rng.choice(x.size, 8000, replace=False); x, y, z = x[sel], y[sel], z[sel]
            pts[(c, d)] = (x, y, z)
            lim = max(lim, np.nanpercentile(np.abs(np.concatenate([x, y])), 99.5))
            zmax = max(zmax, float(z.max()) if z.size else 0.0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.9 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    sc = None
    for i, c in enumerate(centers):
        for j, d in enumerate(SYM_DIAMS):
            ax = axes[i, j]
            x, y, z = pts[(c, d)]
            ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=1.0, zorder=1)
            sc = ax.scatter(x, y, c=z, cmap="viridis", vmin=0, vmax=zmax, s=4, alpha=0.35,
                            linewidths=0, zorder=2, rasterized=True)
            r = np.corrcoef(x, y)[0, 1] if x.size > 2 else np.nan
            ax.text(0.04, 0.93, f"r = {r:.3f}", transform=ax.transAxes, fontsize=9,
                    va="top", bbox=dict(fc="w", ec="none", alpha=0.7))
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(f"diameter {d:g}°", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"center {c:+.1f}°\nest w  (m day$^{{-1}}$)", fontsize=10)
            if i == nrow - 1:
                ax.set_xlabel(r"true w  (m day$^{-1}$)", fontsize=10)
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("depth (m)")
    fig.suptitle(f"{SYM_SHAPE[family]}: estimated vs true w  (all depths & times)",
                 y=0.995, fontsize=14)
    out = os.path.join(sumdir, f"fig_{family}_w_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    return out


# diameter is encoded by shading (light -> dark with increasing diameter), not marker shape
SYM_DIAM_FRAC = {d: 0.30 + 0.70 * i / (len(SYM_DIAMS) - 1) for i, d in enumerate(SYM_DIAMS)}


def _shade(color, frac):
    """Blend `color` toward white; frac=1 -> full color, frac->0 -> white."""
    from matplotlib.colors import to_rgb
    return tuple(1.0 - frac + frac * np.array(to_rgb(color)))


def make_sym_w_scatter_deepest(data_dir, sumdir):
    """One summary figure, three subplots (hexagon / diamond / square). Each panel scatters
    the TIME-MEAN estimated vs true `w` at the DEEPEST observed level — one point per config,
    overlaying every center latitude (color hue) and E-W diameter (shading, light->dark) so
    the whole sweep sits on one plot, with the 1:1 line. Companion to the per-family
    full-depth grids (`make_sym_w_scatter`)."""
    import xarray as xr
    centers = sorted(SYMHEX_CENTERS)
    # one (time-mean true, time-mean est) point per (family, center, diameter)
    pts, vmin, vmax, zdeep = {}, np.inf, -np.inf, None
    for fam in SYM_FAMILIES:
        for c in centers:
            for d in SYM_DIAMS:
                cfg = f"{fam}_d{d}_c{c:+.1f}"
                with xr.open_dataset(os.path.join(data_dir, f"{cfg}__cell_{c:+.2f}.nc")) as ds:
                    kd = int(np.argmax(np.abs(ds.depth.values)))    # deepest obs midpoint
                    zdeep = float(np.abs(ds.depth.values)[kd])
                    x = float(ds.w_model.isel(depth=kd).mean("time")) * W2DAY
                    y = float(ds.w_est.isel(depth=kd).mean("time")) * W2DAY
                pts[(fam, c, d)] = (x, y)
                vmin, vmax = min(vmin, x, y), max(vmax, x, y)
    pad = 0.08 * (vmax - vmin)
    lo, hi = vmin - pad, vmax + pad                    # shared square limits over the data
    fig, axes = plt.subplots(1, len(SYM_FAMILIES), figsize=(5.3 * len(SYM_FAMILIES), 5.4),
                             sharex=True, sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), SYM_FAMILIES):
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.0, zorder=1)
        for c in centers:
            for d in SYM_DIAMS:
                x, y = pts[(fam, c, d)]
                ax.scatter(x, y, s=95, marker="o", color=_shade(LAT_COLORS[c], SYM_DIAM_FRAC[d]),
                           edgecolor="k", linewidths=0.6, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_title(SYM_SHAPE[fam], fontsize=13)
        ax.set_xlabel(r"true $\langle w\rangle$  (m day$^{-1}$)")
    np.atleast_1d(axes)[0].set_ylabel(r"est $\langle w\rangle$  (m day$^{-1}$)")
    c_handles = [Line2D([], [], marker="o", ls="", color=LAT_COLORS[c], ms=9,
                        label=f"center {c:+.1f}°") for c in centers]
    # diameter legend: the shading ramp (light -> dark) on a neutral gray base
    d_handles = [Line2D([], [], marker="o", ls="", color=_shade("#444444", SYM_DIAM_FRAC[d]),
                        markeredgecolor="k", markeredgewidth=0.5, ms=9,
                        label=f"diameter {d:g}°") for d in SYM_DIAMS]
    leg1 = fig.legend(handles=c_handles, loc="upper left", bbox_to_anchor=(0.10, 1.0),
                      ncol=len(centers), frameon=False, fontsize=10)
    fig.add_artist(leg1)
    fig.legend(handles=d_handles, loc="upper right", bbox_to_anchor=(0.92, 1.0),
               ncol=len(SYM_DIAMS), frameon=False, fontsize=10)
    fig.suptitle(f"time-mean estimated vs true w at the deepest observed level (~{zdeep:.0f} m)",
                 y=1.06, fontsize=14)
    out = os.path.join(sumdir, "fig_sym_w_scatter_deepest.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    return out


# ---------------------------------------------------------------------------
# sym_sweep exp1-vs-exp2 comparison builders. Experiment is encoded ONLY by line
# style (exp1 solid / exp2 dashed) and marker fill (exp1 filled / exp2 open) — never
# by a new marker shape, which stays reserved for the shape family (hexagon/diamond/
# square). M = {1: metrics_df, 2: metrics_df}; DATA = {1: data_dir, 2: data_dir}.
#
# Because experiment_1 samples a shallower column than experiment_2 (8–70 vs 0–80 m),
# the compare figures contrast the two at the DEEPEST SHARED observed depth rather than
# over each experiment's own column — see sym_deepest_shared_metrics (w-skill) and the
# heat-flux reducers. Pass its recomputed Mc (and zshared) into the builders below.
# ---------------------------------------------------------------------------
def sym_deepest_shared_metrics(M, DATA):
    """Recompute the sym-family w-skill metrics for BOTH experiments AT the deepest depth
    level observed in both (experiment_1's sampled column stops shallower than
    experiment_2's), so the compare figures contrast them at the same single level instead
    of over each experiment's own depth range. Returns ({1: df, 2: df}, z_shared_m)."""
    import pandas as pd
    import xarray as xr
    Mc = {1: [], 2: []}
    zshared = None
    for _, r in M[1][M[1].family.isin(SYM_FAMILIES)].iterrows():
        base = os.path.basename(r.nc_path)
        dss = {e: xr.open_dataset(os.path.join(DATA[e], base)).load() for e in (1, 2)}
        common_z = min(float(np.abs(dss[e].depth.values).max()) for e in (1, 2))
        zshared = common_z
        for e in (1, 2):
            kd = int(np.argmin(np.abs(np.abs(dss[e].depth.values) - common_z)))
            met = ot.w_skill_metrics(dss[e].w_est.isel(depth=[kd]),
                                     dss[e].w_model.isel(depth=[kd]))
            row = M[e][(M[e].config == r.config)
                       & np.isclose(M[e].center_lat, r.center_lat)].iloc[0].to_dict()
            row.update(met)
            Mc[e].append(row)
        for e in (1, 2):
            dss[e].close()
    return {e: pd.DataFrame(Mc[e]) for e in (1, 2)}, zshared


def _deepest_shared_suptitle(fig, zshared):
    if zshared is not None:
        fig.suptitle(f"w-skill at the deepest observed depth shared by both experiments "
                     f"(~{zshared:.0f} m)", y=1.06, fontsize=13)


def make_sym_summary_compare(M, sumdir, family="symhex", zshared=None):
    """exp1 (solid/filled) vs exp2 (dashed/open) overlay of `make_sym_summary`: a regular
    shape family's six-panel w-skill (figure-2 metric order) vs E-W diameter, one colour
    per centre latitude. The two experiments differ only in the surface w treatment, so
    the gap between the solid (exp1) and dashed (exp2) curves is the effect of that choice."""
    shape = SYM_SHAPE[family]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for exp in (1, 2):
        st = EXP_STYLE[exp]
        d = M[exp][M[exp].family == family].copy()
        if d.empty:
            continue
        d["diameter"] = 2.0 * d["width"]
        for c in SYMHEX_CENTERS:
            sub = d[np.isclose(d.center_lat, c)].sort_values("diameter")
            if sub.empty:
                continue
            plot6(axes, sub.diameter, sub, color=LAT_COLORS[c], ls=st["ls"], lw=2.0,
                  marker=SYM_MARKER[family], ms=9, mec="k", mew=0.4,
                  mfc=mfc(exp, LAT_COLORS[c]))
    label6(axes, f"{shape} diameter (deg)")
    for a in axes.flat:
        a.set_xticks([0.3, 0.5, 0.75, 1.0])
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    lat_handles = [Line2D([0], [0], color=LAT_COLORS[c], lw=2.0, marker=SYM_MARKER[family],
                          mec="k", mew=0.4, label=f"{c:+.1f}°") for c in SYMHEX_CENTERS]
    lg1 = fig.legend(lat_handles, [h.get_label() for h in lat_handles],
                     title="cell center latitude", loc="upper center",
                     bbox_to_anchor=(0.35, 1.0), ncol=len(lat_handles), frameon=False)
    fig.add_artist(lg1)
    fig.legend(handles=exp_handles(), title="experiment", loc="upper center",
               bbox_to_anchor=(0.8, 1.0), ncol=1, frameon=False, handlelength=3.0)
    _deepest_shared_suptitle(fig, zshared)
    out = os.path.join(sumdir, f"fig_{family}_skill_vs_diameter_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    return out


def make_sym_shape_summary_compare(M, sumdir, center=0.0, zshared=None):
    """exp1 (solid/filled) vs exp2 (dashed/open) overlay of `make_sym_shape_summary`: the
    three regular shapes (colour) vs E-W diameter at a fixed centre latitude, six-panel
    w-skill. Shape = colour + marker, experiment = line style + marker fill. When `zshared`
    is given (via sym_deepest_shared_metrics) the metrics are at the deepest shared depth."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for exp in (1, 2):
        st = EXP_STYLE[exp]
        for fam in SYM_FAMILIES:
            d = M[exp][(M[exp].family == fam) & np.isclose(M[exp].center_lat, center)].copy()
            if d.empty:
                continue
            d["diameter"] = 2.0 * d["width"]
            d = d.sort_values("diameter")
            col = EQS_SHAPE_COLOR[SYM_SHAPE[fam]]
            plot6(axes, d.diameter, d, color=col, ls=st["ls"], lw=2.0,
                  marker=SYM_MARKER[fam], ms=9, mec="k", mew=0.4, mfc=mfc(exp, col))
    label6(axes, "array diameter (deg)")
    for a in axes.flat:
        a.set_xticks([0.3, 0.5, 0.75, 1.0])
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    shape_handles = [Line2D([0], [0], color=EQS_SHAPE_COLOR[SYM_SHAPE[f]], lw=2.0,
                            marker=SYM_MARKER[f], mec="k", mew=0.4, label=SYM_SHAPE[f])
                     for f in SYM_FAMILIES]
    lg1 = fig.legend(shape_handles, [h.get_label() for h in shape_handles],
                     title=f"regular shape (center {center:+.1f}°)", loc="upper center",
                     bbox_to_anchor=(0.35, 1.0), ncol=3, frameon=False)
    fig.add_artist(lg1)
    fig.legend(handles=exp_handles(), title="experiment", loc="upper center",
               bbox_to_anchor=(0.8, 1.0), ncol=1, frameon=False, handlelength=3.0)
    _deepest_shared_suptitle(fig, zshared)
    out = os.path.join(sumdir, f"fig_sym_shape_skill_vs_diameter_c{center:+.1f}_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    return out


def make_sym_w_scatter_deepest_compare(DATA, sumdir):
    """No-extrapolation (filled) vs with-extrapolation (open) overlay of
    `make_sym_w_scatter_deepest`: time-mean estimated-vs-true ⟨w⟩ at the deepest level
    COMMON to both experiments (exp1's column stops shallower, so both are read at that
    same depth), one point per config (colour hue = centre latitude, shading light→dark =
    E-W diameter), three shape panels. Experiment is the marker fill only."""
    import xarray as xr
    centers = sorted(SYMHEX_CENTERS)
    # deepest observed level present in BOTH experiments (compare at the same depth)
    ref = f"{SYM_FAMILIES[0]}_d{SYM_DIAMS[0]}_c{centers[0]:+.1f}"
    zmax = {}
    for exp in (1, 2):
        with xr.open_dataset(os.path.join(DATA[exp], f"{ref}__cell_{centers[0]:+.2f}.nc")) as ds:
            zmax[exp] = float(np.abs(ds.depth.values).max())
    common_z = min(zmax.values())
    pts, vmin, vmax = {}, np.inf, -np.inf
    for exp in (1, 2):
        for fam in SYM_FAMILIES:
            for c in centers:
                for d in SYM_DIAMS:
                    cfg = f"{fam}_d{d}_c{c:+.1f}"
                    with xr.open_dataset(os.path.join(DATA[exp], f"{cfg}__cell_{c:+.2f}.nc")) as ds:
                        kd = int(np.argmin(np.abs(np.abs(ds.depth.values) - common_z)))
                        x = float(ds.w_model.isel(depth=kd).mean("time")) * W2DAY
                        y = float(ds.w_est.isel(depth=kd).mean("time")) * W2DAY
                    pts[(exp, fam, c, d)] = (x, y)
                    vmin, vmax = min(vmin, x, y), max(vmax, x, y)
    pad = 0.08 * (vmax - vmin)
    lo, hi = vmin - pad, vmax + pad
    fig, axes = plt.subplots(1, len(SYM_FAMILIES), figsize=(5.3 * len(SYM_FAMILIES), 5.4),
                             sharex=True, sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), SYM_FAMILIES):
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.0, zorder=1)
        for exp in (1, 2):
            for c in centers:
                for d in SYM_DIAMS:
                    x, y = pts[(exp, fam, c, d)]
                    col = _shade(LAT_COLORS[c], SYM_DIAM_FRAC[d])
                    ax.scatter(x, y, s=95, marker="o", color=mfc(exp, col),
                               edgecolor=(col if exp == 2 else "k"),
                               linewidths=(1.4 if exp == 2 else 0.6), zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_title(SYM_SHAPE[fam], fontsize=13)
        ax.set_xlabel(r"true $\langle w\rangle$  (m day$^{-1}$)")
    np.atleast_1d(axes)[0].set_ylabel(r"est $\langle w\rangle$  (m day$^{-1}$)")
    c_handles = [Line2D([], [], marker="o", ls="", color=LAT_COLORS[c], ms=9,
                        label=f"center {c:+.1f}°") for c in centers]
    d_handles = [Line2D([], [], marker="o", ls="", color=_shade("#444444", SYM_DIAM_FRAC[d]),
                        markeredgecolor="k", markeredgewidth=0.5, ms=9,
                        label=f"diameter {d:g}°") for d in SYM_DIAMS]
    # two legend rows so nothing overlaps: centre (left) + diameter (right) on top,
    # the experiment fill key centred just below.
    leg1 = fig.legend(handles=c_handles, loc="upper left", bbox_to_anchor=(0.09, 1.14),
                      ncol=len(centers), frameon=False, fontsize=10)
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=d_handles, loc="upper right", bbox_to_anchor=(0.93, 1.14),
                      ncol=len(SYM_DIAMS), frameon=False, fontsize=10)
    fig.add_artist(leg2)
    fig.legend(handles=_EXP_FILL, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"time-mean est vs true w at the deepest level common to both experiments "
                 f"(~{common_z:.0f} m)", y=1.22, fontsize=13)
    out = os.path.join(sumdir, "fig_sym_w_scatter_deepest_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    return out


def _reghex_get(m, cfg):
    d = m[m.config == cfg]
    if d.empty:
        raise ValueError(f"metrics.csv missing config {cfg!r} — rerun run_experiment*.py "
                         "— this is a one-off; the reg_hex configs are no longer generated.")
    return d


def _reghex_panels(ax, m, sweep_ls="-", reg_filled=True, sweep_alpha=0.45):
    """Draw, for each height: the irregular width sweep (faint line) + the regular
    hexagon (marker at its own width) + the nearest irregular width (open marker).

    The sweep is essential context: the regular hexagon is SMALLER than its nearest
    irregular neighbor (off = sqrt(3)/2 * half), and skill varies strongly with
    width, so comparing the two configs alone confounds regularity with size.
    """
    c = EQS_SHAPE_COLOR["hexagon"]
    for reg_cfg, pat, h, near_w in REGHEX_PAIRS:
        lw = EQS_HEIGHT_LW[h]
        ms = EQS_HEIGHT_MS[h]
        sw = m[m.pattern == pat].sort_values("width")
        reg = _reghex_get(m, reg_cfg)
        near = sw[np.isclose(sw.width, near_w)]

        skw = dict(color=c, lw=lw, ls=sweep_ls, alpha=sweep_alpha)
        rkw = dict(color=c, ls="", marker="*", ms=ms + 7, mew=1.6,
                   mfc=c if reg_filled else "none")
        nkw = dict(color=c, ls="", marker="o", ms=ms, mew=1.6, mfc="none")

        # truth is the dotted sweep line; the regular hexagon's own truth lies on it,
        # so a separate marker for it would only clutter the panel
        ax.flat[0].plot(sw.width, sw.w_model_mean * W2DAY, color=c, lw=lw, ls=":",
                        alpha=sweep_alpha * 0.8)
        ax.flat[0].plot(sw.width, sw.w_est_mean * W2DAY, **skw)
        ax.flat[0].plot(reg.width, reg.w_est_mean * W2DAY, **rkw)
        ax.flat[0].plot(near.width, near.w_est_mean * W2DAY, **nkw)
        for k in range(1, 6):
            ax.flat[k].plot(sw.width, METRIC6[k][0](sw), **skw)
            ax.flat[k].plot(reg.width, METRIC6[k][0](reg), **rkw)
            ax.flat[k].plot(near.width, METRIC6[k][0](near), **nkw)


def make_fig_reghex(m, sumdir):
    """
    Six-panel skill comparison (figure-2 metric order) of the two REGULAR equator
    hexagons (stars) against their nearest irregular neighbors (open circles),
    over the faint irregular width sweep that supplies the size context.
    """
    c = EQS_SHAPE_COLOR["hexagon"]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6))
    _reghex_panels(ax, m)
    label6(ax, "glider lon offset (deg)")
    ax.flat[0].legend(handles=ESTTRUE, loc="best", frameon=True)
    handles = [
        Line2D([0], [0], color=c, ls="", marker="*", ms=12, mfc=c, label="regular hexagon"),
        Line2D([0], [0], color=c, ls="", marker="o", ms=7, mfc="none",
               label="nearest irregular"),
        Line2D([0], [0], color=c, lw=EQS_HEIGHT_LW[1.0], alpha=0.45,
               label="1° irregular sweep"),
        Line2D([0], [0], color=c, lw=EQS_HEIGHT_LW[2.0], alpha=0.45,
               label="2° irregular sweep"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(sumdir, "fig_reghex_regular_vs_irregular.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


