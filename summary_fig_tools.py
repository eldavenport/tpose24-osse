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
    "shift":           ("-",  "^"),
}
# equator single-cell: color = SHAPE, height encoded by linestyle (summary, where
# experiment is fixed) or by marker (compare, where linestyle = experiment).
EQS_SHAPE_COLOR = {"diamond": "#1f77b4", "hexagon": "#2ca02c", "square": "#d62728"}
EQS_HEIGHT_LS   = {1.0: "-", 2.0: "--"}          # summary line height
EQS_HEIGHT_MK   = {1.0: "o", 2.0: "s"}           # single-exp fig2d scatter marker
_EQS_MK_CMP     = {1.0: "o", 2.0: "^"}           # compare marker (ls = experiment)
EQS_META = {                                     # pattern -> (shape, height_deg)
    "equator_1deg":    ("diamond", 1.0), "equator_2deg":    ("diamond", 2.0),
    "equator_hex1deg": ("hexagon", 1.0), "equator_hex2deg": ("hexagon", 2.0),
    "equator_sq1deg":  ("square",  1.0), "equator_sq2deg":  ("square",  2.0),
}
EQS_STYLE = {pat: dict(color=EQS_SHAPE_COLOR[sh], ls=EQS_HEIGHT_LS[h],
                       mk=_EQS_MK_CMP[h], lbl=f"{h:g}° {sh}", shape=sh, height=h)
             for pat, (sh, h) in EQS_META.items()}
EQS_PATS = ["equator_1deg", "equator_2deg", "equator_hex1deg", "equator_hex2deg",
            "equator_sq1deg", "equator_sq2deg"]

WIDTHS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]        # glider lon offsets swept


def _lat_lbl(v):
    return f"{v:+.1f}"


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
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]):
        a.set_xlabel("glider lon offset (deg)"); a.grid(alpha=0.3)
    sc = ax[0, 3]
    lims = [*sc.get_xlim(), *sc.get_ylim()]
    lo, hi = min(lims), max(lims)
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
    "equator_3cell": "3-cell", "shift": "shift",
}
ORDER = [
    ("density_2g", 0.5), ("density_4g", 0.5), ("density_6g", 0.5),
    ("equator_1deg", 0.0), ("equator_2deg", 0.0),
    ("equator_hex1deg", 0.0), ("equator_hex2deg", 0.0),
    ("equator_sq1deg", 0.0), ("equator_sq2deg", 0.0),
    ("equator_3cell", -1.0), ("equator_3cell", 0.0), ("equator_3cell", 1.0),
    ("shift", -1.5), ("shift", -0.5), ("shift", 0.5), ("shift", 1.5),
]
EQS_ROWS = ["equator_1deg", "equator_2deg", "equator_hex1deg",
            "equator_hex2deg", "equator_sq1deg", "equator_sq2deg"]
SUB_ORDERS = [                                   # (tag, order) one method per figure
    ("6a_shift",          [o for o in ORDER if o[0] == "shift"]),
    ("6b_equator3cell",   [o for o in ORDER if o[0] == "equator_3cell"]),
    ("6c_density",        [o for o in ORDER if o[0].startswith("density")]),
    ("6d_equator_single", [o for o in ORDER if o[0] in EQS_ROWS]),
]


def row_label(pat, lat):
    base = PAT_LABEL.get(pat, pat)
    return f"{base} {lat:+.1f}°" if pat in ("equator_3cell", "shift") else base


def _fam(pat):
    return "density" if pat.startswith("density") else "shift" if pat == "shift" else "equator"


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
        ("2a_shift", m.family == "shift", "center_lat", LAT_COLORS, "cell center lat", _lat_lbl,
         lambda ax: stacked_cells_panel(ax, [-1.5, -0.5, 0.5, 1.5], 0.5, "array layout (offset = 0.5 deg)")),
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
        axes[0, 3].legend(handles=handles, title=leg_title, loc="best", frameon=True)
        geom(axes[1, 3])
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.legend(handles=handles, title=f"color = {leg_title}", loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
        fig.savefig(os.path.join(sumdir, f"fig{tag}.png"), dpi=150, bbox_inches="tight")
        plt.show()

    # 2d: equator single cell, color = shape, linestyle = height, scatter marker = height
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    series = []
    for pat in EQS_PATS:
        st = EQS_STYLE[pat]
        dd = m[m.pattern == pat].sort_values("width")
        if dd.empty:
            continue
        line_kw = dict(color=st["color"], ls=st["ls"], lw=2.2)
        scat_kw = dict(color=st["color"], ls="", marker=EQS_HEIGHT_MK[st["height"]],
                       ms=6, mec="k", mew=0.3)
        series.append((dd, line_kw, scat_kw))
    fill_skill_panels(axes, series)
    label_skill_panels(axes)
    shape_handles = [Line2D([0], [0], color=c, lw=2.4, label=s) for s, c in EQS_SHAPE_COLOR.items()]
    height_handles = [Line2D([0], [0], color="0.35", ls=ls, lw=2.0, label=f"{h:g} deg tall")
                      for h, ls in EQS_HEIGHT_LS.items()]
    scat_height = [Line2D([0], [0], color="0.35", ls="", marker=mk, mec="k", mew=0.3, ms=7,
                          label=f"{h:g} deg tall") for h, mk in EQS_HEIGHT_MK.items()]
    axes[0, 3].legend(handles=scat_height, title="marker = height", loc="best", frameon=True)
    equator_single_geometry(axes[1, 3])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    lg1 = fig.legend(handles=shape_handles, title="color = shape", loc="upper center",
                     bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.add_artist(lg1)
    fig.legend(handles=height_handles, title="line = height", loc="upper center",
               bbox_to_anchor=(0.85, 1.0), ncol=2, frameon=False, handlelength=3.0)
    fig.savefig(os.path.join(sumdir, "fig2d_equator_single.png"), dpi=150, bbox_inches="tight")
    plt.show()


def make_fig3(m, sumdir):
    """Skill vs cell latitude for the two multi-cell arrays (6 metric rows x 2 arrays)."""
    lat_arrays = [("shift", "Shift array (4 cells)"), ("equator_3cell", "Equator 3-cell array")]
    fig, axes = plt.subplots(len(METRIC6), 2, figsize=(12, 20), sharex=True, sharey="row")
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
        for j in (0, 1):
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
            c, ls = st["color"], st["ls"]
            axes[0, 0].plot(B.w_model_mean * W2DAY, z, color=c, ls=":", lw=1.2, alpha=0.6)
            axes[0, 0].plot(B.w_est_mean * W2DAY, z, color=c, ls=ls, lw=1.9)
            axes[0, 1].plot(B.mean_bias / B.w_model_std, z, color=c, ls=ls, lw=1.9)
            axes[0, 2].plot(ot.frac_mean_bias(B.mean_bias, B.w_model_mean), z, color=c, ls=ls, lw=1.9)
            axes[0, 3].plot(B.rms, z, color=c, ls=ls, lw=1.9)
            axes[0, 3].plot(B.w_model_std, z, color=c, ls=":", lw=1.2, alpha=0.6)
            axes[1, 0].plot(B.w_est_std / B.w_model_std, z, color=c, ls=ls, lw=1.9)
            axes[1, 1].plot(B["corr"], z, color=c, ls=ls, lw=1.9)
            axes[1, 2].plot(B.norm_rms, z, color=c, ls=ls, lw=1.9)
            handles.append(Line2D([0], [0], color=c, ls=ls, lw=1.9, label=st["lbl"]))
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
        for a in axes.flat:
            if a.axison:
                a.grid(alpha=0.3)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.legend(handles=handles, title=f"color = {leg_title}", loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
        fig.savefig(os.path.join(sumdir, f"fig{tag}.png"), dpi=150, bbox_inches="tight")
        plt.show()

    for tag, pat in (("5a_shift", "shift"), ("5b_equator3cell", "equator_3cell")):
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
           dict(color=EQS_STYLE[pat]["color"], ls=EQS_STYLE[pat]["ls"], lbl=EQS_STYLE[pat]["lbl"]))
          for pat in EQS_PATS if (d.pattern == pat).any()]
    _plot_depth(rs, "5d_equator_single", "shape (color) x height (line)")


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
# Figure builders — exp1 vs exp2 comparison
# ============================================================================
EXP_STYLE = {
    1: dict(ls="-",  fill=True,  lbl="exp1 (no surface extrap)"),
    2: dict(ls="--", fill=False, lbl="exp2 (extrap to surface)"),
}


def mfc(exp, color):
    """Marker face color: filled (exp1) or open/white (exp2)."""
    return color if EXP_STYLE[exp]["fill"] else "white"


def exp_handles():
    return [Line2D([0], [0], color="0.3", ls=EXP_STYLE[e]["ls"], lw=1.9, label=EXP_STYLE[e]["lbl"])
            for e in (1, 2)]


# scatter legend for the two experiment fill states (edge carries the key color)
_EXP_FILL = [Line2D([0], [0], ls="", marker="o", mfc="0.4", mec="0.4", ms=8, label="exp1 (filled)"),
             Line2D([0], [0], ls="", marker="o", mfc="white", mec="0.4", mew=1.4, ms=8, label="exp2 (open)")]


def make_fig2abcd_compare(M, outdir):
    """fig2a-2d comparison: exp1 (solid/filled) vs exp2 (dashed/open) on a 2x4 grid."""
    panels = [
        ("2a_shift", lambda mm: mm.family == "shift", "center_lat", LAT_COLORS, "cell center lat", _lat_lbl,
         lambda ax: stacked_cells_panel(ax, [-1.5, -0.5, 0.5, 1.5], 0.5, "array layout (offset = 0.5deg)")),
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
                lkw = dict(color=cmap[kv], ls=st["ls"], marker="o", lw=1.9, ms=6,
                           mec="k", mew=0.3, mfc=mfc(exp, cmap[kv]))
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

    # 2d: color = shape, marker = height, linestyle/fill = experiment
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    series = []
    for exp in (1, 2):
        st = EXP_STYLE[exp]
        for pat in EQS_PATS:
            s = EQS_STYLE[pat]
            dd = M[exp][M[exp].pattern == pat].sort_values("width")
            if dd.empty:
                continue
            lkw = dict(ls=st["ls"], marker=s["mk"], color=s["color"], lw=1.9, ms=6,
                       mec="k", mew=0.3, mfc=mfc(exp, s["color"]))
            skw = dict(ls="", marker=s["mk"], color=s["color"], ms=6, mew=1.2,
                       mec=s["color"], mfc=mfc(exp, s["color"]))
            series.append((dd, lkw, skw))
    fill_skill_panels(axes, series)
    label_skill_panels(axes)
    shape_handles = [Line2D([0], [0], color=EQS_STYLE[p]["color"], marker=EQS_STYLE[p]["mk"],
                            mec="k", mew=0.3, lw=1.9, label=EQS_STYLE[p]["lbl"]) for p in EQS_PATS]
    scat_handles = [Line2D([0], [0], ls="", marker=_EQS_MK_CMP[h], color="0.35", mec="k",
                           mew=0.3, ms=7, label=f"{h:g} deg tall") for h in (1.0, 2.0)]
    scat_handles += [Line2D([0], [0], ls="", marker="o", mfc="0.4", mec="0.4", ms=8, label="exp1 (filled)"),
                     Line2D([0], [0], ls="", marker="o", mfc="white", mec="0.4", mew=1.4, ms=8, label="exp2 (open)")]
    axes[0, 3].legend(handles=scat_handles, loc="best", frameon=True, ncol=2)
    equator_single_geometry(axes[1, 3])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    lg1 = fig.legend(handles=shape_handles, title="color = shape  x  marker = height",
                     loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.add_artist(lg1)
    fig.legend(handles=exp_handles(), title="experiment", loc="upper center",
               bbox_to_anchor=(0.9, 1.0), ncol=1, frameon=False, handlelength=3.0)
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
                common = dict(color=s["color"], ls=est["ls"], lw=1.8, marker=s.get("mk"),
                              markevery=5, ms=6, mec="k", mew=0.3, mfc=mfc(exp, s["color"]))
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

    for tag, pat in (("5a_shift", "shift"), ("5b_equator3cell", "equator_3cell")):
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
                             dict(color=EQS_STYLE[pat]["color"], mk=EQS_STYLE[pat]["mk"], lbl=EQS_STYLE[pat]["lbl"]))
                            for pat in EQS_PATS if (d.pattern == pat).any()]
    line_handles = [Line2D([0], [0], color=EQS_STYLE[p]["color"], marker=EQS_STYLE[p]["mk"],
                           mec="k", mew=0.3, lw=1.9, label=EQS_STYLE[p]["lbl"]) for p in EQS_PATS]
    _depth_compare(rows_by_exp, "5d_equator_single", "shape (color) x height (marker)", line_handles)


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
