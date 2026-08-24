"""
heat_flux_fig_tools.py — figure builders for the experiment_2 vertical
advective-heating summary notebook (heat_flux_summary.ipynb).

The quantity throughout is the vertical advective heating  w * dT/dz  (a heating
rate, deg C/day). Notation, following the analysis instructions:
    [ . ]  = SPATIAL average over the array footprint — either the array's estimate
             (plane-fit w / glider-average T) or the model truth (hull area-average).
    < . >  = TEMPORAL average.
So [w][dT/dz] is the product of the two footprint averages, and [w dT/dz] is the
footprint average of the pointwise product; they differ by the within-footprint
spatial covariance.

Two complementary sets of figures (each with an -a equator-shape sweep and a -b
off-equator shift diamond/hexagon variant):

  1. AREA-AVERAGE quantities — the footprint-mean advective heating.
     * Part 1 (structural): truth <[w dT/dz]>, the resolved product <[w][dT/dz]>
       (both from the model area-means), and the irreducible error between them
       (the time-mean within-footprint spatial covariance). truth = resolved +
       irreducible.  ->  make_area_components_{a,b}
     * Part 2 (reducible): the plane-fit / sampling errors. [w] true vs plane-fit
       estimate, [dT/dz] true vs mooring/CTD estimate, and the reducible flux error
       <[w]_est [dT/dz]_est> - <[w]_true [dT/dz]_true>.  ->  make_area_reducible_{a,b}

  2. POINT quantities — the array's single area-mean w [w] used as a stand-in for the
     local w at each glider, applied to that glider's own dT/dz. Glider-mean profiles:
     truth <w_g dT/dz_g(native)>, estimate <[w] dT/dz_g(2 m)>, and the error split into
     a w-substitution part ([w]-w_g) and a gradient vertical-resolution part
     (2 m vs native).  ->  make_point_flux_{a,b}

  3. CONTEXT maps — where in the equatorial box the footprint average is least
     reliable (largest w / stratification variability and heating structure), with a
     representative footprint overlaid.  ->  make_context_maps

Style/colors are reused from summary_fig_tools (EQS_STYLE for equator shapes,
SHIFT_SHAPE_COLOR / LAT_COLORS for the shift figures).
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import osse_tools as ot
import summary_fig_tools as sft

DAY = 86400.0                                    # s/day; heating deg C/s -> deg C/day
DAY_W = 86400.0                                  # s/day; w m/s -> m/day (readable)
apply_style = sft.apply_style
LAT_COLORS = sft.LAT_COLORS
EQS_STYLE = sft.EQS_STYLE                        # pattern -> dict(color, lw, ms, lbl, shape, height)

# equator single-cell shapes (a-figures) and diameters (2 x lon offset)
EQ_SHAPES = ['equator_1deg', 'equator_2deg', 'equator_hex1deg', 'equator_hex2deg',
             'equator_sq1deg', 'equator_sq2deg']
# split by cell height — the a-figures are drawn once per height (3 shapes each)
EQ_SHAPES_1DEG = ['equator_1deg', 'equator_hex1deg', 'equator_sq1deg']
EQ_SHAPES_2DEG = ['equator_2deg', 'equator_hex2deg', 'equator_sq2deg']
EQ_WIDTHS = [0.25, 0.5, 0.75]                    # lon offsets -> diameters 0.5/1.0/1.5
EQ_REP_W = 0.5                                    # representative width (diameter 1.0°)

# off-equator latitude sweep (b-figures) — 1° diameter (0.5° lon offset) cells.
# ±0.5/±1.5 come from shift_{,hex_,sq_}w0.5; the interleaved -1/0/+1 from the *_mid configs.
SHIFT_SHAPES = ('diamond', 'hexagon', 'square')
SHIFT = {'diamond': 'shift_w0.5', 'hexagon': 'shift_hex_w0.5', 'square': 'shift_sq_w0.5'}
SHIFT_MID = {'diamond': 'shift_w0.5_mid', 'hexagon': 'shift_hex_w0.5_mid',
             'square': 'shift_sq_w0.5_mid'}
# match the equator shape colors: diamond blue, hexagon green, square red
SHIFT_SHAPE_COLOR = {'diamond': '#1f77b4', 'hexagon': '#2ca02c', 'square': '#d62728'}
SHIFT_LATS = [-1.5, -0.5, 0.5, 1.5]              # original sweep
SHIFT_ALL_LATS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]   # + interleaved -1/0/+1
B_REP_LAT = 0.5                                  # representative latitude for per-panel figs
B_REP_LATS = [0.0, 0.5, 1.5]                     # per-latitude reducible / point figures


def _shift_config(shape, lat):
    """Config name holding the shift cell for this shape at this latitude."""
    return SHIFT_MID[shape] if abs(lat) in (0.0, 1.0) else SHIFT[shape]


KEY_DEPTHS = (25, 50, 75)

# axis labels — plain, minimal jargon
ADV_LBL = "advective heating  (°C day$^{-1}$)"
W_LBL = "w  (m day$^{-1}$)"
DTDZ_LBL = "∂T/∂z  (°C m$^{-1}$)"
EST_ERR_LBL = "estimate error  (°C day$^{-1}$)"
SUBGRID_LBL = "sub-array  (°C day$^{-1}$)"
# depth-integrated heat-flux (W/m^2) counterparts of the heating-rate labels
FLUX_LBL = "advective heat flux  (W m$^{-2}$)"
FLUX_ERR_LBL = "heat flux estimate error  (W m$^{-2}$)"
FLUX_SUB_LBL = "sub-array heat flux  (W m$^{-2}$)"


# ---------------------------------------------------------------------------
# loading
def eq_config(shape, w):
    return f'{shape}_w{w:g}'


def load_cell(data_dir, config, center_lat):
    path = os.path.join(data_dir, f'{config}__cell_{center_lat:+.2f}.nc')
    return xr.open_dataset(path)


def load_glider(data_dir, config, center_lat):
    path = os.path.join(data_dir, f'{config}__cell_{center_lat:+.2f}__glider.nc')
    return xr.open_dataset(path)


# ---------------------------------------------------------------------------
# heating-rate profile  ->  vertically-integrated heat flux (W/m^2)
def heating_to_flux(H):
    """Cumulative vertical integral of a time-mean advective-heating profile into a heat
    flux, ρ₀cp∫ w∂ᵤT dz (W m⁻²).

    H is a heating-rate profile (°C/day) on the negative-down `depth` coord (top ≈ -9 m to
    bottom -79 m). The integral is accumulated from the top of the sampled column downward,
    so at each depth it is the heat flux carried through that level and the bottom point is
    the full-column integral. ρ₀, cp match the TPOSE24 MITgcm run (osse_tools.RHO0/CP).
    Integration is linear, so it commutes with the true−estimate / error differences taken
    on the heating profiles (the flux of a difference equals the difference of the fluxes)."""
    from scipy.integrate import cumulative_trapezoid
    z = np.asarray(H['depth'].values, float)             # negative, decreasing top->bottom
    Hs = np.asarray(H.values, float) / DAY               # °C/day -> °C/s
    F = ot.RHO0 * ot.CP * cumulative_trapezoid(Hs, x=-z, initial=0.0)
    return xr.DataArray(F, dims=H.dims, coords=H.coords)


def heating_column_flux(H):
    """Full-column (0–80 m sampled) depth integral of a heating profile -> heat flux
    (W m⁻²): the bottom endpoint of `heating_to_flux`."""
    return float(heating_to_flux(H).isel(depth=-1))


# ---------------------------------------------------------------------------
# computations
def area_components(ds):
    """AREA set, Part 1 (structural) — time-mean profiles (°C/day).

    true_total = <[w dT/dz]>                 footprint mean of the pointwise product
    true_mean  = <[w]_true [dT/dz]_true>     product of the TRUE footprint means
    est_mean   = <[w]_est  [dT/dz]_est>      product of the ESTIMATED footprint means
    est_err    = true_mean - est_mean        observable estimate error (reducible)
    subgrid    = true_total - true_mean      unobservable within-footprint covariance

    Identity: true_total = est_mean + est_err + subgrid.
    """
    dTh = ds.Tbar_hull.differentiate('depth')            # [dT/dz]_true, each instant
    dTg = ds.Tbar_glider.differentiate('depth')          # [dT/dz]_est (glider-mean gradient)
    true_total = ds.A_true_total.mean('time') * DAY
    true_mean = (ds.wbar_hull * dTh).mean('time') * DAY
    est_mean = (ds.w_est_mid * dTg).mean('time') * DAY
    return dict(z=ds.depth, true_total=true_total, true_mean=true_mean, est_mean=est_mean,
                est_err=true_mean - est_mean, subgrid=true_total - true_mean)


def area_reducible(ds):
    """AREA set, Part 2 (reducible) — time-mean profiles.

    w        : [w]_true (wbar_hull)         vs [w]_est (plane fit at moorings)
    dTdz     : [dT/dz]_true (∂z Tbar_hull)  vs [dT/dz]_est (∂z glider-mean T)
    flux     : <[w]_true [dT/dz]_true>      vs <[w]_est [dT/dz]_est>   (°C/day)
    The reducible flux error is flux_est - flux_true.
    """
    dTh = ds.Tbar_hull.differentiate('depth')
    dTg = ds.Tbar_glider.differentiate('depth')
    flux_true = (ds.wbar_hull * dTh).mean('time') * DAY
    flux_est = (ds.w_est_mid * dTg).mean('time') * DAY
    return dict(
        z=ds.depth,
        w_true=ds.wbar_hull.mean('time') * DAY_W, w_est=ds.w_est_mid.mean('time') * DAY_W,
        g_true=dTh.mean('time'), g_est=dTg.mean('time'),
        flux_true=flux_true, flux_est=flux_est,
        # depth-integrated heat flux (W/m^2) counterparts of the heating rate above
        fluxint_true=heating_to_flux(flux_true), fluxint_est=heating_to_flux(flux_est),
    )


def point_components(dsm, dsg):
    """POINT set — glider-mean profiles (°C/day). Area-mean w [w] stands in for the
    local w at each glider, applied to that glider's own dT/dz.

    truth    = <w_g · dT/dz_g(native)>            local model w, native-resolution gradient
    est      = <[w] · dT/dz_g(2 m)>               area-mean w, 2 m glider gradient
    err_w    = <(w_g-[w]) · dT/dz_g(2 m)>         w-substitution error
    err_dTdz = <w_g · (dT/dz_native - dT/dz_2m)>  gradient vertical-resolution error
    err_total = truth - est = err_w + err_dTdz    (exact; same sign as truth - estimate)
    """
    wA = dsm.w_est_mid                                    # [w] area-mean (depth); broadcasts
    wg = dsg.w_true_glider                               # local w (depth, glider)
    g2 = dsg.T_glider.differentiate('depth')             # 2 m obs gradient
    gN = dsg.T_glider_native.differentiate('Zc')         # native gradient
    gN = gN.interp(Zc=xr.DataArray(dsg.depth.values, dims='depth',
                                   coords={'depth': dsg.depth.values}))
    gm = lambda x: x.mean('glider').mean('time') * DAY
    return dict(
        z=dsg.depth,
        truth=gm(wg * gN), est=gm(wA * g2),
        err_w=gm((wg - wA) * g2), err_dTdz=gm(wg * (gN - g2)),
        err_total=gm(wg * gN - wA * g2),
    )


# ---------------------------------------------------------------------------
# shared plotting helpers
def _tidy_xticks(ax):
    """Keep x tick labels legible: few ticks + a common ×10ⁿ offset for small numbers,
    so labels like 0.000/0.002/…/0.010 don't crowd and overlap."""
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-1, 4))                 # factor out ×10ⁿ once |x| < 0.1
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.get_offset_text().set_fontsize(12)


def _prof(ax, xlim=None, sym=False):
    ax.axvline(0, color='0.7', lw=0.8, zorder=0)
    ax.set_ylim(-80, 0); ax.grid(alpha=0.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if sym:
        lo, hi = ax.get_xlim(); a = max(abs(lo), abs(hi)); ax.set_xlim(-a, a)
    _tidy_xticks(ax)


def _span(vals, pad=0.05):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(v.min()), float(v.max()); d = pad * (hi - lo or 1.0)
    return (lo - d, hi + d)


# symbols for the exact quantity each panel plots ([·]=footprint avg, ⟨·⟩=time avg)
S_TRUE_TOTAL = r"$\langle[w\,\partial_z T]\rangle$"               # true footprint mean of product
S_TRUE_MEAN = r"$\langle[w][\partial_z T]\rangle_{\mathrm{true}}$"   # product of TRUE footprint means
S_EST_MEAN = r"$\langle[w][\partial_z T]\rangle_{\mathrm{est}}$"     # product of ESTIMATED footprint means
S_EST_ERR = (r"$\langle[w][\partial_z T]\rangle_{\mathrm{true}}"
             r"-\langle[w][\partial_z T]\rangle_{\mathrm{est}}$")     # observable estimate error
S_SUBGRID = (r"$\langle[w\,\partial_z T]\rangle"
             r"-\langle[w][\partial_z T]\rangle_{\mathrm{true}}$")    # unobservable subgrid covariance
S_FLUX = r"$\langle[w][\partial_z T]\rangle$"
# POINT symbols: ⟨·⟩ = average over time AND the gliders (one profile); [·] = plane-fit
# footprint average, on the stand-in w only. The truth uses LOCAL w & gradient — no [·].
S_PT_TRUTH = r"$\langle w\,\partial_z T\rangle$"
S_PT_EST = r"$\langle[w]\,\partial_z T\rangle$"
S_PT_TE = r"$\langle w\,\partial_z T\rangle-\langle[w]\,\partial_z T\rangle$"
S_PT_W = r"$\langle(w-[w])\,\partial_z T\rangle$"
S_PT_G = r"$\langle w\,(\partial_z T_{\mathrm{nat}}-\partial_z T_{2m})\rangle$"


def _flux_sym(s):
    """Wrap a heating-rate term symbol r"$X$" into its depth-integrated heat flux
    ρ₀cp∫X dz, so the W/m² panels are labeled consistently with the °C/day ones."""
    return r"$\rho_0 c_p\!\int\!" + s.strip('$') + r"\,\mathrm{d}z$"


def _term_sym(ax, text, loc='upper right', fontsize=10.5):
    """Small corner inset naming the exact term plotted, out of the way of the data."""
    pos = {'upper right': (0.97, 0.96, 'right', 'top'),
           'upper left':  (0.03, 0.96, 'left', 'top'),
           'lower right': (0.97, 0.04, 'right', 'bottom'),
           'lower left':  (0.03, 0.04, 'left', 'bottom')}[loc]
    x, y, ha, va = pos
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            color='0.12', zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8', alpha=0.9))


def _shape_handles(shapes=EQ_SHAPES):
    return [Line2D([], [], color=EQS_STYLE[s]['color'], lw=EQS_STYLE[s]['lw'],
                   label=EQS_STYLE[s]['lbl']) for s in shapes]


def _shift_shape_handles():
    return [Line2D([], [], color=SHIFT_SHAPE_COLOR[k], lw=2.2, label=k) for k in SHIFT_SHAPES]


def _top_legend(fig, handles, ncol, y=1.005):
    fig.legend(handles, [h.get_label() for h in handles], loc='upper center',
               ncol=ncol, bbox_to_anchor=(0.5, y), frameon=False)


def _te_handles(true_lbl='true', est_lbl='estimate'):
    return [Line2D([], [], color='0.3', ls='-', lw=2, label=true_lbl),
            Line2D([], [], color='0.3', ls='--', lw=1.6, label=est_lbl)]


# config-column builders (a = equator shapes, b = shift) -----------------------
def _eq_cols_by_width(data_dir, shapes=EQ_SHAPES):
    """One column per diameter; each column overlays the given equator shapes."""
    cols = [[(load_cell(data_dir, eq_config(s, w), 0.0), EQS_STYLE[s]['color'], EQS_STYLE[s]['lw'])
             for s in shapes] for w in EQ_WIDTHS]
    return cols, [f'diameter {2 * w:g}°' for w in EQ_WIDTHS]


def _shift_cols_by_lat(data_dir, lats=SHIFT_ALL_LATS):
    """One column per latitude; each column overlays diamond & hexagon."""
    cols = [[(load_cell(data_dir, _shift_config(k, lat), lat), SHIFT_SHAPE_COLOR[k], 2.2)
             for k in SHIFT_SHAPES] for lat in lats]
    return cols, [f'{lat:+.1f}°' for lat in lats]


def _eq_rep_cells(data_dir, shapes=EQ_SHAPES):
    """Equator shapes at the representative diameter (1°)."""
    return [(load_cell(data_dir, eq_config(s, EQ_REP_W), 0.0),
             EQS_STYLE[s]['color'], EQS_STYLE[s]['lw']) for s in shapes]


def _shift_rep_cells(data_dir, lat=B_REP_LAT):
    """Diamond & hexagon at one latitude."""
    return [(load_cell(data_dir, _shift_config(k, lat), lat), SHIFT_SHAPE_COLOR[k], 2.2)
            for k in SHIFT_SHAPES]


def _eq_rep_gliders(data_dir, shapes=EQ_SHAPES):
    return [(load_cell(data_dir, eq_config(s, EQ_REP_W), 0.0),
             load_glider(data_dir, eq_config(s, EQ_REP_W), 0.0),
             EQS_STYLE[s]['color'], EQS_STYLE[s]['lw']) for s in shapes]


def _shift_rep_gliders(data_dir, lat=B_REP_LAT):
    return [(load_cell(data_dir, _shift_config(k, lat), lat),
             load_glider(data_dir, _shift_config(k, lat), lat),
             SHIFT_SHAPE_COLOR[k], 2.2) for k in SHIFT_SHAPES]


# ===========================================================================
# AREA set, Part 1 — structural decomposition
#   true_total = est_mean + est_err (observable) + subgrid (unobservable)
# ===========================================================================
def _area_components(cols_by_col, col_titles, flux=False, xlims=None):
    """Three rows, one column per diameter / latitude:
      1. true mean <[w][dT/dz]>_true (solid) vs estimated mean <[w][dT/dz]>_est (dashed)
      2. estimate error  true_mean - est_mean                              (reducible)
      3. subgrid  true_total - true_mean = <[w dT/dz]> - <[w][dT/dz]>_true (unobservable).

    `flux=True` renders the exact same decomposition but with every profile vertically
    integrated into a heat flux ρ₀cp∫·dz (W m⁻²) — a units-only duplicate of the °C/day
    figure (integration is linear, so the identity true total = est mean + est err + subgrid
    is preserved level by level and column by column). `xlims` (from `components_xlims`) is a
    (row-1, rows-2&3) pair of shared x-spans, overriding the per-figure spans so every figure
    in the group aligns."""
    conv = heating_to_flux if flux else (lambda x: x)     # profile transform per panel
    lbl1 = FLUX_LBL if flux else ADV_LBL
    lbl2 = FLUX_ERR_LBL if flux else EST_ERR_LBL
    lbl3 = FLUX_SUB_LBL if flux else SUBGRID_LBL
    sym = _flux_sym if flux else (lambda s: s)
    nc = len(cols_by_col)
    fig, axes = plt.subplots(3, nc, figsize=(4.7 * nc, 11.7), sharey=True, squeeze=False)
    store = [[({k: conv(v) if k != 'z' else v for k, v in area_components(ds).items()},
               color, lw) for ds, color, lw in cells]
             for cells in cols_by_col]
    if xlims is not None:
        x1, x23 = xlims
    else:
        m1, m2, m3 = [], [], []
        for col in store:
            for R, _, _ in col:
                m1 += [R['true_mean'].values, R['est_mean'].values]
                m2.append(R['est_err'].values)
                m3.append(R['subgrid'].values)
        x1 = _span(np.concatenate(m1))
        x23 = _span(np.concatenate(m2 + m3))             # rows 2 & 3 share limits
    for j, col in enumerate(store):
        a1, a2, a3 = axes[0, j], axes[1, j], axes[2, j]
        for R, color, lw in col:
            z = R['z']; lw2 = max(lw - 0.4, 1.0)
            a1.plot(R['true_mean'], z, '-', color=color, lw=lw)
            a1.plot(R['est_mean'], z, '--', color=color, lw=lw2)
            a2.plot(R['est_err'], z, '-', color=color, lw=lw)
            a3.plot(R['subgrid'], z, '-', color=color, lw=lw)
        _prof(a1, x1); _prof(a2, x23); _prof(a3, x23)
        a1.set_title(col_titles[j], fontsize=11)
        a1.set_xlabel(lbl1, fontsize=9.5)
        a2.set_xlabel(lbl2, fontsize=9.5)
        a3.set_xlabel(lbl3, fontsize=9.5)
        _term_sym(a1, f"true mean  {sym(S_TRUE_MEAN)}\nest mean  {sym(S_EST_MEAN)}", fontsize=8.5)
        _term_sym(a2, sym(S_EST_ERR), fontsize=8)
        _term_sym(a3, sym(S_SUBGRID), fontsize=8)
        if j == 0:                                        # plain-English row labels, col 0 only
            for ax, lbl in ((a1, "true and est. area avg."),
                            (a2, "(true − est.) area avg."),
                            (a3, "true total − true area avg.")):
                ax.text(0.03, 0.97, lbl, transform=ax.transAxes, ha='left', va='top',
                        fontsize=8, fontweight='bold', zorder=6)
    for r in range(3):
        axes[r, 0].set_ylabel("depth (m)")
    axes[0, 0].legend(handles=_te_handles('true mean', 'estimated mean'), fontsize=8, loc='lower left')
    return fig


def _all_components_cells(data_dir):
    """Every (ds, color, lw) cell that appears in ANY area_components figure — all 6 equator
    shapes × 3 diameters plus the 3 shift shapes × 7 latitudes — for shared x-limits."""
    eq_cols, _ = _eq_cols_by_width(data_dir, EQ_SHAPES)
    sh_cols, _ = _shift_cols_by_lat(data_dir, SHIFT_ALL_LATS)
    return [c for col in eq_cols for c in col] + [c for col in sh_cols for c in col]


def components_xlims(data_dir, flux=False):
    """Shared area_components x-limits over ALL of the group's figures: a (row-1, rows-2&3)
    pair of x-spans. `flux=True` gives the W/m² duplicate's spans."""
    conv = heating_to_flux if flux else (lambda x: x)
    m1, m23 = [], []
    for ds, _, _ in _all_components_cells(data_dir):
        R = area_components(ds)
        tm, em = conv(R['true_mean']), conv(R['est_mean'])
        ee, sg = conv(R['est_err']), conv(R['subgrid'])
        m1 += [tm.values, em.values]; m23 += [ee.values, sg.values]
    return _span(np.concatenate(m1)), _span(np.concatenate(m23))


def make_area_components_a(data_dir, shapes=EQ_SHAPES, flux=False, xlims=None):
    """Equator shape sweep — columns = diameter, one line per shape. `flux=True` for the
    heat-flux (W m⁻²) duplicate. `xlims` (components_xlims) shares limits across the group."""
    cols, titles = _eq_cols_by_width(data_dir, shapes)
    fig = _area_components(cols, titles, flux=flux, xlims=xlims)
    _top_legend(fig, _shape_handles(shapes), len(shapes))
    fig.tight_layout(rect=(0, 0, 1, 0.96)); return fig


def make_area_components_b(data_dir, lats=SHIFT_ALL_LATS, flux=False, xlims=None):
    """Off-equator shift sweep — columns = latitude, diamond vs hexagon."""
    cols, titles = _shift_cols_by_lat(data_dir, lats)
    fig = _area_components(cols, titles, flux=flux, xlims=xlims)
    _top_legend(fig, _shift_shape_handles(), 3)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); return fig


# ===========================================================================
# AREA set, Part 2 — reducible (plane-fit / sampling) errors
# ===========================================================================
_REDUCIBLE_ROWS = [
    ('w_true', 'w_est', W_LBL, "w estimate error  (m day$^{-1}$)",
     r"$\langle[w]\rangle$",
     r"$\langle[w]\rangle_{\mathrm{true}}-\langle[w]\rangle_{\mathrm{est}}$"),
    ('g_true', 'g_est', DTDZ_LBL, "∂T/∂z estimate error  (°C m$^{-1}$)",
     r"$\langle[\partial_z T]\rangle$",
     r"$\langle[\partial_z T]\rangle_{\mathrm{true}}-\langle[\partial_z T]\rangle_{\mathrm{est}}$"),
    ('flux_true', 'flux_est', ADV_LBL, EST_ERR_LBL, S_FLUX, S_EST_ERR),
    # 4th row: the heating profile above, integrated into a heat flux (W/m^2)
    ('fluxint_true', 'fluxint_est', FLUX_LBL, FLUX_ERR_LBL, _flux_sym(S_FLUX), _flux_sym(S_EST_ERR)),
]


def _reducible_row_xlims(cells):
    """Per-row (left, right) x-spans over a set of cells — for sharing limits across figures."""
    store = [area_reducible(ds) for ds, _, _ in cells]
    out = []
    for kt, ke, *_ in _REDUCIBLE_ROWS:
        tv, dv = [], []
        for R in store:
            tv += [R[kt].values, R[ke].values]; dv.append((R[kt] - R[ke]).values)
        out.append((_span(np.concatenate(tv)), _span(np.concatenate(dv))))
    return out


def shift_reducible_xlims(data_dir, lats=B_REP_LATS):
    """Shared reducible row x-limits across all shift latitudes (so the per-lat figures
    are directly comparable)."""
    cells = [c for lat in lats for c in _shift_rep_cells(data_dir, lat)]
    return _reducible_row_xlims(cells)


def reducible_xlims(data_dir):
    """Shared reducible row x-limits over ALL area_reducible figures in the group — the 6
    equator shapes (representative diameter) plus the 3 shift shapes at every rep latitude —
    so every figure of the type aligns row by row."""
    cells = _eq_rep_cells(data_dir, EQ_SHAPES) + \
        [c for lat in B_REP_LATS for c in _shift_rep_cells(data_dir, lat)]
    return _reducible_row_xlims(cells)


def _area_reducible(cells, legend_handles, legend_ncol, subtitle=None, row_xlims=None):
    """Four rows (w, dT/dz, advective heating, and that heating integrated into a heat
    flux ρ₀cp∫·dz in W m⁻²) x two columns (left: true solid vs estimate dashed; right:
    estimate error = true - estimate). The heating-row error matches the estimate-error row
    of the components figure exactly, and the flux row is its depth-integral. `cells` = list
    of (ds, color, lw). `subtitle` (e.g. a latitude) is appended to the column headers.
    `row_xlims` (from `_reducible_row_xlims`) overrides the per-figure spans for sharing."""
    store = [(area_reducible(ds), color, lw) for ds, color, lw in cells]
    nrow = len(_REDUCIBLE_ROWS)
    fig, axes = plt.subplots(nrow, 2, figsize=(10, 4 * nrow), sharey=True, squeeze=False)
    for r, (kt, ke, lbl, dlbl, sym, dsym) in enumerate(_REDUCIBLE_ROWS):
        aL, aR = axes[r, 0], axes[r, 1]
        if row_xlims is not None:
            _xt, _xd = row_xlims[r]
        else:
            tv, dv = [], []
            for R, _, _ in store:
                tv += [R[kt].values, R[ke].values]; dv.append((R[kt] - R[ke]).values)
            _xt = _span(np.concatenate(tv)); _xd = _span(np.concatenate(dv))
        for R, color, lw in store:
            z = R['z']; lw2 = max(lw - 0.4, 1.0)
            aL.plot(R[kt], z, '-', color=color, lw=lw)
            aL.plot(R[ke], z, '--', color=color, lw=lw2)
            aR.plot(R[kt] - R[ke], z, '-', color=color, lw=lw)
        _prof(aL, _xt); _prof(aR, _xd)
        aL.set_xlabel(lbl, fontsize=9.5); aR.set_xlabel(dlbl, fontsize=9)
        _term_sym(aL, sym); _term_sym(aR, dsym, fontsize=8)
    suf = f'  ({subtitle})' if subtitle else ''
    axes[0, 0].set_title('true vs estimate' + suf, fontsize=11)
    axes[0, 1].set_title('estimate error (true − estimate)' + suf, fontsize=11)
    for r in range(nrow):
        axes[r, 0].set_ylabel('depth (m)')
    axes[0, 0].legend(handles=_te_handles(), fontsize=8, loc='lower left')
    _top_legend(fig, legend_handles, legend_ncol)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); return fig


def make_area_reducible_a(data_dir, shapes=EQ_SHAPES, row_xlims=None):
    """Equator shapes at the representative diameter (1°). Pass `row_xlims` (reducible_xlims)
    to share limits across the whole figure group."""
    return _area_reducible(_eq_rep_cells(data_dir, shapes), _shape_handles(shapes), len(shapes),
                           row_xlims=row_xlims)


def make_area_reducible_b(data_dir, lat=B_REP_LAT, row_xlims=None):
    """Shift shapes at one latitude. Pass `row_xlims` (reducible_xlims) to share limits
    across the whole figure group."""
    return _area_reducible(_shift_rep_cells(data_dir, lat), _shift_shape_handles(), 3,
                           subtitle=f'{lat:+.1f}°', row_xlims=row_xlims)


# ===========================================================================
# POINT set — area-mean w as a stand-in at each glider (glider-mean profiles)
# ===========================================================================
def _point_cols(R, flux=False):
    """The four per-column profiles of a point_components result R, as (truth, est) for
    column 0 and the single curve for columns 1–3. `flux=True` integrates each into a heat
    flux (W/m²); integration is linear so truth−est stays the difference of the two."""
    conv = heating_to_flux if flux else (lambda x: x)
    truth, est = conv(R['truth']), conv(R['est'])
    return [(truth, est), truth - est, conv(R['err_w']), conv(R['err_dTdz'])]


def _point_xlims(cells):
    """Per-row ((truth/est, truth−est, error) heating spans, same in flux) x-spans over a
    set of (dsm,dsg,...) cells — for sharing limits across figures. Returns
    (heating_spans, flux_spans)."""
    store = [point_components(dsm, dsg) for dsm, dsg, *_ in cells]

    def _spans(flux):
        fv, tev, ev = [], [], []
        for R in store:
            (t, e), te, ew, eg = _point_cols(R, flux)
            fv += [t.values, e.values]; tev.append(te.values); ev += [ew.values, eg.values]
        return (_span(np.concatenate(fv)), _span(np.concatenate(tev)), _span(np.concatenate(ev)))

    return (_spans(False), _spans(True))


def shift_point_xlims(data_dir, lats=B_REP_LATS):
    """Shared point-flux x-limits across all shift latitudes."""
    return _point_xlims([c for lat in lats for c in _shift_rep_gliders(data_dir, lat)])


def point_xlims(data_dir):
    """Shared point-flux x-limits over ALL point_flux figures in the group — the 6 equator
    shapes (representative diameter), the 3 shift shapes at every rep latitude, and the
    0°N,140°W mooring variant — so every figure of the type aligns column by column."""
    cells = _eq_rep_gliders(data_dir, EQ_SHAPES) + \
        [c for lat in B_REP_LATS for c in _shift_rep_gliders(data_dir, lat)]
    dsg = _mooring_glider(data_dir)
    cells += [(load_cell(data_dir, _shift_config(k, 0.0), 0.0), dsg, SHIFT_SHAPE_COLOR[k], 2.2)
              for k in SHIFT_SHAPES]
    return _point_xlims(cells)


def _point_flux(cells, legend_handles, legend_ncol, subtitle=None, xlims=None):
    """Two rows × four columns. Columns: (1) truth solid vs estimate dashed; (2) truth −
    estimate (total point error); (3) w-substitution error; (4) gradient (2 m vs native)
    error. Row 1 is the advective heating (°C/day); row 2 is that heating integrated into a
    heat flux ρ₀cp∫·dz (W/m²). Columns 3–4 share x-limits within each row so their magnitudes
    compare. `cells` = list of (dsm, dsg, color, lw). `xlims` (from `_point_xlims`) overrides
    the per-figure spans for sharing."""
    store = [(point_components(dsm, dsg), color, lw) for dsm, dsg, color, lw in cells]
    fig, axes = plt.subplots(2, 4, figsize=(19, 10.4), sharey=True, squeeze=False)
    heat_x, flux_x = xlims if xlims is not None else (None, None)
    suf = f'  ({subtitle})' if subtitle else ''
    titles = ['truth vs estimate' + suf, 'truth − estimate',
              'w-substitution error', 'gradient error (2 m vs native)']
    syms = [f"truth  {S_PT_TRUTH}\nest  {S_PT_EST}", S_PT_TE, S_PT_W, S_PT_G]
    sym_fs = [9, 7.5, 9, 8]
    for row, (flux, lbl, spans) in enumerate([(False, ADV_LBL, heat_x),
                                              (True, FLUX_LBL, flux_x)]):
        cols = [(_point_cols(R, flux), color, lw) for R, color, lw in store]
        if spans is None:
            fv, tev, ev = [], [], []
            for (parts, _, _) in cols:
                (t, e), te, ew, eg = parts
                fv += [t.values, e.values]; tev.append(te.values); ev += [ew.values, eg.values]
            spans = (_span(np.concatenate(fv)), _span(np.concatenate(tev)),
                     _span(np.concatenate(ev)))
        xf, xte, xe = spans
        a0, a1, a2, a3 = axes[row]
        for (parts, color, lw), (R, _, _) in zip(cols, store):
            (t, e), te, ew, eg = parts
            z = R['z']; lw2 = max(lw - 0.4, 1.0)
            a0.plot(t, z, '-', color=color, lw=lw); a0.plot(e, z, '--', color=color, lw=lw2)
            a1.plot(te, z, '-', color=color, lw=lw)
            a2.plot(ew, z, '-', color=color, lw=lw)
            a3.plot(eg, z, '-', color=color, lw=lw)
        _prof(a0, xf); _prof(a1, xte); _prof(a2, xe); _prof(a3, xe)
        _fs = _flux_sym if flux else (lambda s: s)
        for ax, sym, fs in zip((a0, a1, a2, a3), syms, sym_fs):
            _term_sym(ax, _fs(sym) if ax is not a0 else
                      f"truth  {_fs(S_PT_TRUTH)}\nest  {_fs(S_PT_EST)}", fontsize=fs)
            ax.set_xlabel(lbl, fontsize=9.5)
        if row == 0:
            for ax, t in zip((a0, a1, a2, a3), titles):
                ax.set_title(t, fontsize=11)
        a0.set_ylabel('depth (m)')
        a0.legend(handles=_te_handles('truth (local w)', 'estimate (area-mean w)'),
                  fontsize=8, loc='lower left')
    _top_legend(fig, legend_handles, legend_ncol)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); return fig


def make_point_flux_a(data_dir, shapes=EQ_SHAPES, xlims=None):
    """Equator shapes at the representative diameter (1°), glider-mean point flux. Pass
    `xlims` (point_xlims) to share limits across the whole figure group."""
    return _point_flux(_eq_rep_gliders(data_dir, shapes), _shape_handles(shapes), len(shapes),
                       xlims=xlims)


def make_point_flux_b(data_dir, lat=B_REP_LAT, xlims=None):
    """Shift shapes at one latitude, glider-mean point flux. Pass `xlims`
    (point_xlims) to share limits across the whole figure group."""
    return _point_flux(_shift_rep_gliders(data_dir, lat), _shift_shape_handles(), 3,
                       subtitle=f'{lat:+.1f}°', xlims=xlims)


# the 0°N, 140°W (220°E) TAO mooring is carried as the center obs of the equator cells
MOORING_LAT, MOORING_LON = 0.0, 220.0


def _mooring_glider(data_dir, config='equator_1deg_w0.5', center_lat=0.0):
    """The single 0°N,140°W mooring glider (the equator cells' center obs), as a
    one-glider dataset — reused so the local flux need not be re-sampled per array."""
    g = load_glider(data_dir, config, center_lat)
    idx = int(np.argmin(np.abs(g.lat.values - MOORING_LAT) + np.abs(g.lon.values - MOORING_LON)))
    assert abs(float(g.lat[idx]) - MOORING_LAT) < 1e-6 and abs(float(g.lon[idx]) - MOORING_LON) < 1e-6, \
        f'no glider at ({MOORING_LAT},{MOORING_LON}) in {config}'
    return g.isel(glider=[idx])


def make_point_flux_mooring(data_dir, lat=0.0, xlims=None):
    """Point flux at JUST the 0°N,140°W mooring (the cell center), using each shift
    lat-0 cell's area-mean [w] against the single mooring point's local w & dT/dz — no
    glider averaging. Diamond vs hexagon; the truth (local mooring flux) is identical
    for both, only the estimate [w] differs. Pass `xlims` (point_xlims) to share limits
    across the whole figure group."""
    dsg = _mooring_glider(data_dir)
    cells = [(load_cell(data_dir, _shift_config(k, lat), lat), dsg, SHIFT_SHAPE_COLOR[k], 2.2)
             for k in SHIFT_SHAPES]
    return _point_flux(cells, _shift_shape_handles(), 3, subtitle='0°N, 140°W mooring',
                       xlims=xlims)


# ===========================================================================
# SHIFT summary — how each error term scales with latitude, per shape
# ===========================================================================
def make_error_scaling_shift(data_dir, lats=SHIFT_ALL_LATS):
    """Summary of the shift experiments: the AREA-average advective-heating error terms
    vs center latitude, one line per shape (diamond/hexagon/square). Three panels — the
    total area error and its two parts:
      total = true_total - est_mean = <[w·∂T/∂z]> - <[w][∂T/∂z]>_est
      estimate error (reducible) = true_mean - est_mean
      subgrid (unobservable)     = true_total - true_mean
    (total = estimate error + subgrid.) Two rows: the 0–80 m depth-MEAN heating error
    (°C/day, top) and the 0–80 m depth-INTEGRATED heat flux error (ρ₀cp∫·dz, W/m², bottom).
    All from the cell datasets."""
    terms = [('total error', lambda A: A['true_total'] - A['est_mean']),
             ('estimate error', lambda A: A['est_err']),
             ('sub-array error', lambda A: A['subgrid'])]
    # per row: (reduction of a term profile -> scalar, y-axis label)
    rows = [(lambda P: float(P.mean('depth')),
             "depth-mean advective heating error  (°C day$^{-1}$)"),
            (heating_column_flux,
             "depth-integrated heat flux error  (W m$^{-2}$)")]
    data = {r: {nm: {k: [] for k in SHIFT_SHAPES} for nm, _ in terms} for r in range(len(rows))}
    for k in SHIFT_SHAPES:
        for lat in lats:
            A = area_components(load_cell(data_dir, _shift_config(k, lat), lat))
            for nm, fn in terms:
                P = fn(A)
                for r, (reduce, _) in enumerate(rows):
                    data[r][nm][k].append(reduce(P))
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.6 * len(rows)),
                             sharex=True, sharey='row', squeeze=False)
    for r, (_, ylbl) in enumerate(rows):
        for ax, (nm, _) in zip(axes[r], terms):
            for k in SHIFT_SHAPES:
                ax.plot(lats, data[r][nm][k], '-o', color=SHIFT_SHAPE_COLOR[k], lw=2, ms=4, label=k)
            ax.axhline(0, color='0.7', lw=0.8, zorder=0)
            ax.axvline(0, color='0.85', lw=0.8, zorder=0)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(nm, fontsize=11)
            if r == len(rows) - 1:
                ax.set_xlabel("center latitude (°N)", fontsize=10)
        axes[r, 0].set_ylabel(ylbl, fontsize=9.5)
    _top_legend(fig, _shift_shape_handles(), 3)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); return fig


# ===========================================================================
# SYMHEX set — symmetric REGULAR hexagons: diameter sweep × center latitude
#   (geometrically regular, isotropic footprints, no moorings)
# ===========================================================================
SYMHEX_DIAMS = [0.3, 0.5, 0.75, 1.0]             # E-W diameter (°) == 2 × lon offset
SYMHEX_CENTERS = [0.0, 0.5, -0.5]                # cell center latitudes (°N)
SYMHEX_REP = 1.0                                 # representative diameter (single-fig)
SYMHEX_LW = 2.2

# the three regular-shape families run at the same diameters × centers
SYM_FAMILIES = ['symhex', 'symdia', 'symsq']
SYM_SHAPE = {'symhex': 'hexagon', 'symdia': 'diamond', 'symsq': 'square'}
SYM_MARKER = {'symhex': 'h', 'symdia': 'D', 'symsq': 's'}
# shape colors match the equator experiments (hexagon green, diamond blue, square red)
SYM_SHAPE_COLOR = {'hexagon': '#2ca02c', 'diamond': '#1f77b4', 'square': '#d62728'}


def sym_config(family, d, c):
    """Config name for the regular `family` (symhex/symdia/symsq) of E-W diameter `d`°
    centered at `c`°N."""
    return f'{family}_d{d}_c{c:+.1f}'


def symhex_config(d, c):
    """Back-compat: config name for the regular hexagon of E-W diameter `d`° at `c`°N."""
    return sym_config('symhex', d, c)


def _symhex_center_handles(centers=SYMHEX_CENTERS):
    return [Line2D([], [], color=sft.LAT_COLORS[c], lw=SYMHEX_LW, label=f'center {c:+.1f}°')
            for c in centers]


def _sym_shape_handles(families=SYM_FAMILIES):
    return [Line2D([], [], color=SYM_SHAPE_COLOR[SYM_SHAPE[f]], lw=SYMHEX_LW,
                   label=SYM_SHAPE[f]) for f in families]


def _symhex_cols_by_diam(data_dir, family='symhex', centers=SYMHEX_CENTERS):
    """One column per diameter; each column overlays the center latitudes (colored)."""
    cols = [[(load_cell(data_dir, sym_config(family, d, c), c), sft.LAT_COLORS[c], SYMHEX_LW)
             for c in centers] for d in SYMHEX_DIAMS]
    return cols, [f'diameter {d:g}°' for d in SYMHEX_DIAMS]


def _symhex_rep_cells(data_dir, diam=SYMHEX_REP, family='symhex', centers=SYMHEX_CENTERS):
    return [(load_cell(data_dir, sym_config(family, diam, c), c), sft.LAT_COLORS[c], SYMHEX_LW)
            for c in centers]


def _symhex_rep_gliders(data_dir, diam=SYMHEX_REP, family='symhex', centers=SYMHEX_CENTERS):
    return [(load_cell(data_dir, sym_config(family, diam, c), c),
             load_glider(data_dir, sym_config(family, diam, c), c),
             sft.LAT_COLORS[c], SYMHEX_LW) for c in centers]


def symhex_components_xlims(data_dir, family='symhex', flux=False):
    """Shared area_components x-limits over the sym figure (all diameters × centers)."""
    conv = heating_to_flux if flux else (lambda x: x)
    m1, m23 = [], []
    cols, _ = _symhex_cols_by_diam(data_dir, family)
    for col in cols:
        for ds, _, _ in col:
            R = area_components(ds)
            m1 += [conv(R['true_mean']).values, conv(R['est_mean']).values]
            m23 += [conv(R['est_err']).values, conv(R['subgrid']).values]
    return _span(np.concatenate(m1)), _span(np.concatenate(m23))


def make_area_components_symhex(data_dir, family='symhex', flux=False, xlims=None):
    """Structural decomposition — columns = diameter (0.3/0.5/0.75/1.0°), one line per
    center latitude. `flux=True` for the heat-flux (W m⁻²) duplicate."""
    cols, titles = _symhex_cols_by_diam(data_dir, family)
    fig = _area_components(cols, titles, flux=flux, xlims=xlims)
    _top_legend(fig, _symhex_center_handles(), len(SYMHEX_CENTERS))
    fig.tight_layout(rect=(0, 0, 1, 0.96)); return fig


def make_area_reducible_symhex(data_dir, diam=SYMHEX_REP, family='symhex', row_xlims=None):
    """Reducible plane-fit / sampling errors at diameter `diam`, one line per center
    latitude."""
    return _area_reducible(_symhex_rep_cells(data_dir, diam, family), _symhex_center_handles(),
                           len(SYMHEX_CENTERS), subtitle=f'diameter {diam:g}°',
                           row_xlims=row_xlims)


def make_point_flux_symhex(data_dir, diam=SYMHEX_REP, family='symhex', xlims=None):
    """Glider-mean point flux (area-mean w stand-in) at diameter `diam`, one line per
    center latitude."""
    return _point_flux(_symhex_rep_gliders(data_dir, diam, family), _symhex_center_handles(),
                       len(SYMHEX_CENTERS), subtitle=f'diameter {diam:g}°', xlims=xlims)


def _error_scaling_terms():
    """(term name, extractor) and (reducer, ylabel) rows shared by the sym error-scaling
    summaries. total = estimate error + sub-array."""
    terms = [('total error', lambda A: A['true_total'] - A['est_mean']),
             ('estimate error', lambda A: A['est_err']),
             ('sub-array error', lambda A: A['subgrid'])]
    rows = [(lambda P: float(P.mean('depth')),
             "depth-mean advective heating error  (°C day$^{-1}$)"),
            (heating_column_flux,
             "depth-integrated heat flux error  (W m$^{-2}$)")]
    return terms, rows


def make_error_scaling_symhex(data_dir, family='symhex', centers=SYMHEX_CENTERS):
    """Summary of one sym family's sweep: the AREA-average advective-heating error terms vs
    DIAMETER, one line per center latitude. Three panels — the total area error and its two
    parts:
      total = true_total − est_mean = <[w·∂T/∂z]> − <[w][∂T/∂z]>_est
      estimate error (reducible) = true_mean − est_mean
      sub-array (unobservable)   = true_total − true_mean
    (total = estimate error + sub-array.) Two rows: the 0–80 m depth-MEAN heating error
    (°C/day, top) and the 0–80 m depth-INTEGRATED heat flux error (ρ₀cp∫·dz, W/m², bottom)."""
    terms, rows = _error_scaling_terms()
    data = {r: {nm: {c: [] for c in centers} for nm, _ in terms} for r in range(len(rows))}
    for c in centers:
        for d in SYMHEX_DIAMS:
            A = area_components(load_cell(data_dir, sym_config(family, d, c), c))
            for nm, fn in terms:
                P = fn(A)
                for r, (reduce, _) in enumerate(rows):
                    data[r][nm][c].append(reduce(P))
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.6 * len(rows)),
                             sharex=True, sharey='row', squeeze=False)
    for r, (_, ylbl) in enumerate(rows):
        for ax, (nm, _) in zip(axes[r], terms):
            for c in centers:
                ax.plot(SYMHEX_DIAMS, data[r][nm][c], '-o', color=sft.LAT_COLORS[c],
                        lw=2, ms=5, label=f'{c:+.1f}°')
            ax.axhline(0, color='0.7', lw=0.8, zorder=0)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(nm, fontsize=11)
            if r == len(rows) - 1:
                ax.set_xlabel(f"{SYM_SHAPE[family]} diameter (°)", fontsize=10)
        axes[r, 0].set_ylabel(ylbl, fontsize=9.5)
    for ax in axes.flat:
        ax.set_xticks(SYMHEX_DIAMS)
    _top_legend(fig, _symhex_center_handles(), len(centers))
    fig.tight_layout(rect=(0, 0, 1, 0.95)); return fig


def make_error_scaling_sym_shapes(data_dir, center=0.0, families=SYM_FAMILIES):
    """SHAPE comparison: the AREA-average advective-heating error terms vs DIAMETER at a
    fixed center latitude, one line per regular shape (hexagon/diamond/square). Same three
    panels (total / estimate / sub-array) and two rows (°C/day depth-mean, W/m² depth-integral)
    as `make_error_scaling_symhex` — isolates how the footprint SHAPE affects each error."""
    terms, rows = _error_scaling_terms()
    data = {r: {nm: {f: [] for f in families} for nm, _ in terms} for r in range(len(rows))}
    for f in families:
        for d in SYMHEX_DIAMS:
            A = area_components(load_cell(data_dir, sym_config(f, d, center), center))
            for nm, fn in terms:
                P = fn(A)
                for r, (reduce, _) in enumerate(rows):
                    data[r][nm][f].append(reduce(P))
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.6 * len(rows)),
                             sharex=True, sharey='row', squeeze=False)
    for r, (_, ylbl) in enumerate(rows):
        for ax, (nm, _) in zip(axes[r], terms):
            for f in families:
                ax.plot(SYMHEX_DIAMS, data[r][nm][f], '-o', color=SYM_SHAPE_COLOR[SYM_SHAPE[f]],
                        lw=2, ms=5, label=SYM_SHAPE[f])
            ax.axhline(0, color='0.7', lw=0.8, zorder=0)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(nm, fontsize=11)
            if r == len(rows) - 1:
                ax.set_xlabel("array diameter (°)", fontsize=10)
        axes[r, 0].set_ylabel(ylbl, fontsize=9.5)
    for ax in axes.flat:
        ax.set_xticks(SYMHEX_DIAMS)
    handles = _sym_shape_handles(families)
    fig.legend(handles, [h.get_label() for h in handles], loc='upper center',
               ncol=len(families), bbox_to_anchor=(0.5, 1.02), frameon=False,
               title=f'regular-shape comparison — center {center:+.1f}°')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); return fig


# ---------------------------------------------------------------------------
# sym_sweep exp1-vs-exp2 comparison builders. Experiment is encoded ONLY by line
# style (no-extrap solid / with-extrap dashed) and marker fill (filled / open) — never
# by a new marker shape, which stays reserved for the regular shape family.
# DATA = {1: data_heat_dir, 2: data_heat_dir}. Because experiment_1 samples a shallower
# column than experiment_2 (8–70 vs 8–80 m), the compare figures evaluate BOTH on the
# depth range common to the two — the deepest level in both — so the depth-mean and
# depth-integral cover the same range for each experiment.
# ---------------------------------------------------------------------------
def _common_max_depth(DATA, family='symhex', d=SYMHEX_DIAMS[0], c=SYMHEX_CENTERS[0]):
    """Deepest |depth| (m) present in BOTH experiments' data_heat (the depth grid is the
    same across configs within an experiment, so one representative cell suffices)."""
    zmax = [float(np.abs(load_cell(DATA[e], sym_config(family, d, c), c).depth.values).max())
            for e in (1, 2)]
    return min(zmax)


def _clip_depth(ds, zmax):
    """Restrict a cell dataset to |depth| <= zmax (the range common to both experiments)."""
    return ds.isel(depth=np.where(np.abs(ds.depth.values) <= zmax + 1e-6)[0])


def _error_scaling_rows_deepest():
    """Reducers for the COMPARE error-scaling figures. The profiles are pre-clipped to the
    depth range common to both experiments, so the deepest shared observed level is the last
    index. We compare the VALUE at that level — the heating-rate error (°C/day) and the
    heat-flux error integrated down to that level (W/m²) — instead of a depth mean, so
    experiment_1 (8–70 m) and experiment_2 (0–80 m) are contrasted at the same depth."""
    return [
        (lambda P: float(P.isel(depth=-1)),
         "advective heating error @ deepest shared depth  (°C day$^{-1}$)"),
        (lambda P: float(heating_to_flux(P).isel(depth=-1)),
         "heat-flux error to deepest shared depth  (W m$^{-2}$)"),
    ]


def make_error_scaling_symhex_compare(DATA, family='symhex', centers=SYMHEX_CENTERS):
    """exp1 (solid/filled) vs exp2 (dashed/open) overlay of `make_error_scaling_symhex`:
    the area-average advective-heating error terms (total / estimate / sub-array) vs E-W
    diameter, one colour per centre latitude. The sub-array term is a property of the
    truth field (identical across experiments); the estimate term carries the surface-w
    difference, so the solid/dashed gap is concentrated there."""
    terms, _ = _error_scaling_terms()
    rows = _error_scaling_rows_deepest()
    zc = _common_max_depth(DATA, family=family, c=centers[0])
    data = {e: {r: {nm: {c: [] for c in centers} for nm, _ in terms}
                for r in range(len(rows))} for e in (1, 2)}
    for e in (1, 2):
        for c in centers:
            for d in SYMHEX_DIAMS:
                A = area_components(_clip_depth(load_cell(DATA[e], sym_config(family, d, c), c), zc))
                for nm, fn in terms:
                    P = fn(A)
                    for r, (reduce, _) in enumerate(rows):
                        data[e][r][nm][c].append(reduce(P))
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.6 * len(rows)),
                             sharex=True, sharey='row', squeeze=False)
    for e in (1, 2):
        st = sft.EXP_STYLE[e]
        for r in range(len(rows)):
            for ax, (nm, _) in zip(axes[r], terms):
                for c in centers:
                    col = sft.LAT_COLORS[c]
                    ax.plot(SYMHEX_DIAMS, data[e][r][nm][c], ls=st['ls'], marker='o',
                            color=col, lw=2, ms=5, mfc=sft.mfc(e, col), mec=col)
    for r, (_, ylbl) in enumerate(rows):
        for ax, (nm, _) in zip(axes[r], terms):
            ax.axhline(0, color='0.7', lw=0.8, zorder=0)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(nm, fontsize=11)
            if r == len(rows) - 1:
                ax.set_xlabel(f"{SYM_SHAPE[family]} diameter (°)", fontsize=10)
        axes[r, 0].set_ylabel(ylbl, fontsize=9.5)
    for ax in axes.flat:
        ax.set_xticks(SYMHEX_DIAMS)
    ch = _symhex_center_handles(centers)
    fig.legend(ch, [h.get_label() for h in ch], loc='upper center', ncol=len(centers),
               bbox_to_anchor=(0.4, 1.04), frameon=False, title='cell center latitude')
    fig.legend(handles=sft.exp_handles(), loc='upper center', ncol=1,
               bbox_to_anchor=(0.82, 1.04), frameon=False, title='experiment', handlelength=3.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def make_error_scaling_sym_shapes_compare(DATA, center=0.0, families=SYM_FAMILIES):
    """exp1 (solid/filled) vs exp2 (dashed/open) overlay of `make_error_scaling_sym_shapes`:
    area-average advective-heating error terms vs E-W diameter at a fixed centre latitude,
    one colour per regular shape. Shape = colour + marker, experiment = line style + fill."""
    terms, _ = _error_scaling_terms()
    rows = _error_scaling_rows_deepest()
    zc = _common_max_depth(DATA, family=families[0], c=center)
    data = {e: {r: {f: [] for f in families} for r in range(len(rows))} for e in (1, 2)}
    for e in (1, 2):
        for f in families:
            per_term = {nm: [] for nm, _ in terms}
            for d in SYMHEX_DIAMS:
                A = area_components(_clip_depth(load_cell(DATA[e], sym_config(f, d, center), center), zc))
                for nm, fn in terms:
                    per_term[nm].append(fn(A))
            for r, (reduce, _) in enumerate(rows):
                data[e][r][f] = {nm: [reduce(P) for P in per_term[nm]] for nm, _ in terms}
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.6 * len(rows)),
                             sharex=True, sharey='row', squeeze=False)
    for e in (1, 2):
        st = sft.EXP_STYLE[e]
        for r in range(len(rows)):
            for ax, (nm, _) in zip(axes[r], terms):
                for f in families:
                    col = SYM_SHAPE_COLOR[SYM_SHAPE[f]]
                    ax.plot(SYMHEX_DIAMS, data[e][r][f][nm], ls=st['ls'], marker='o',
                            color=col, lw=2, ms=5, mfc=sft.mfc(e, col), mec=col)
    for r, (_, ylbl) in enumerate(rows):
        for ax, (nm, _) in zip(axes[r], terms):
            ax.axhline(0, color='0.7', lw=0.8, zorder=0)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(nm, fontsize=11)
            if r == len(rows) - 1:
                ax.set_xlabel("array diameter (°)", fontsize=10)
        axes[r, 0].set_ylabel(ylbl, fontsize=9.5)
    for ax in axes.flat:
        ax.set_xticks(SYMHEX_DIAMS)
    sh = _sym_shape_handles(families)
    fig.legend(sh, [h.get_label() for h in sh], loc='upper center', ncol=len(families),
               bbox_to_anchor=(0.4, 1.04), frameon=False,
               title=f'regular shape — center {center:+.1f}°')
    fig.legend(handles=sft.exp_handles(), loc='upper center', ncol=1,
               bbox_to_anchor=(0.82, 1.04), frameon=False, title='experiment', handlelength=3.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ===========================================================================
# CONTEXT maps — where the footprint average is least reliable
# ===========================================================================
import pickle as _pickle                                                    # noqa: E402
import cmocean.cm as _cmo                                                   # noqa: E402

SUBCELL_MAPS = '/data/SO3/edavenport/tpose24/cache/subcell_maps.pkl'
_EXP1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'experiments', 'experiment_1')


def load_subcell_maps():
    with open(SUBCELL_MAPS, 'rb') as f:
        return _pickle.load(f)


def _cell_positions(shape='equator_1deg', w=0.5):
    path = os.path.join(_EXP1, 'configs', 'equator', f'{shape}_w{w:g}.json')
    (_, pos), = ot.load_cells(path)
    return np.array(pos)


# (key, label, cmap, divergent, scale, symbol) for each map column
_CTX_MAPS = [
    ('wbar', "mean w  (m day$^{-1}$)", _cmo.balance, True, DAY_W, r"$\langle w\rangle$"),
    ('adv_heating', "mean advective heating  (°C day$^{-1}$)", _cmo.balance, True, DAY,
     r"$\langle w\,\partial_z T\rangle$"),
    ('var_w', "w variability  (m day$^{-1}$)", _cmo.amp, False, DAY_W, r"std$(w')$"),  # sqrt below
    ('var_dTdz', "∂T/∂z variability  (°C m$^{-1}$)", _cmo.amp, False, 1.0, r"std$(\partial_z T')$"),
]


def _depth_label(d):
    return '0–80 m avg' if d == 'depthavg' else f'{d} m'


def make_context_maps(maps=None, depths=(25, 50, 75, 'depthavg'), overlay='equator_1deg'):
    """Rows = depth (plus a 0–80 m depth-average row); columns = time-mean w, time-mean
    advective heating, and the temporal std of w and of dT/dz, over the equatorial box,
    with a representative 1° footprint overlaid. Strong structure / variability inside the
    footprint is where the single footprint-average value is least representative."""
    import matplotlib.ticker as mticker
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize
    from scipy.spatial import ConvexHull
    maps = maps or load_subcell_maps()
    pos = _cell_positions(overlay)
    h = ConvexHull(pos[:, ::-1])                          # hull in (lon, lat)
    vv = np.append(h.vertices, h.vertices[0])
    nr, ncm = len(depths), len(_CTX_MAPS)
    fig, axes = plt.subplots(nr, ncm, figsize=(4.6 * ncm, 3.9 * nr), squeeze=False)
    for i, d in enumerate(depths):
        ds = maps[d]
        for jx, (key, lbl, cmap, div, sc, sym) in enumerate(_CTX_MAPS):
            ax = axes[i, jx]
            v = ds[key]
            vals = v.values * sc
            if key.startswith('var_'):
                vals = np.sqrt(np.clip(vals, 0, None))   # variance -> std
            if div:
                vmax = float(np.nanpercentile(np.abs(vals), 98)); vmin = -vmax
            else:
                vmin = float(np.nanpercentile(vals, 2)); vmax = float(np.nanpercentile(vals, 98))
            ax.contourf(v.XC, v.YC, vals, levels=np.linspace(vmin, vmax, 60),
                        cmap=cmap, extend='both')
            sm = mcm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
            plt.colorbar(sm, ax=ax, shrink=0.85, ticks=mticker.MaxNLocator(5, symmetric=div))
            ax.plot(pos[:, 1][vv], pos[:, 0][vv], 'k-', lw=1.4)
            ax.axhline(0, color='k', lw=0.4, ls=':')
            _term_sym(ax, sym, loc='upper left', fontsize=10)
            if i == 0:
                ax.set_title(lbl, fontsize=10)
            if i == nr - 1:
                ax.set_xlabel('Longitude (°E)')
            if jx == 0:
                ax.set_ylabel(f'{_depth_label(d)}\nLatitude (°N)')
    fig.tight_layout(); return fig
