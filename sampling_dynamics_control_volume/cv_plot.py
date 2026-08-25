"""
Shared plotting + statistics helpers for the control-volume figures.

Style follows the project convention: legends across the TOP (no suptitles), bold/large
axis labels, American spelling, diameters coloured light->dark with size, obs estimate
solid + model truth dotted. Every figure script loads the cache written by run_ocv.py
through load_ocv() and reuses these builders.
"""

import os
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

import common as C
import osse_tools as ot

SEC_PER_DAY = C.SEC_PER_DAY

mpl.rcParams.update({
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 110, 'savefig.dpi': 130,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.formatter.limits': (-3, 4), 'axes.formatter.use_mathtext': True,
})

TRUTH_KW = dict(ls=':', lw=2.4)          # model truth = dotted
OBS_KW = dict(ls='-', lw=2.0)            # array (obs) estimate = solid


def tidy_x(ax, n=5):
    ax.xaxis.set_major_locator(MaxNLocator(n))


def outdir(*parts):
    d = os.path.join(C.HERE, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def load_ocv(diam, shape='symhex'):
    return xr.open_dataset(os.path.join(C.CACHE_DIR, f'{C.config_name(diam, shape)}_ocv.nc'))


def have_cache(shape='symhex'):
    return all(os.path.exists(os.path.join(C.CACHE_DIR, f'{C.config_name(d, shape)}_ocv.nc'))
               for d in C.DIAMETERS)


# --------------------------------------------------------------------------- series
def at_depth(da, z, dim=None):
    """Interpolate a (time, <depth>) series to depth z -> (time,) numpy array."""
    dim = dim or ('depth' if 'depth' in da.dims else 'obs_depth')
    return np.asarray(da.interp({dim: z}).values).ravel()


def w_series(ds, which, z):
    """Area-averaged w [m/day] at depth z, which in {'true','obs'}."""
    return at_depth(ds[f'w_{which}'], z) * SEC_PER_DAY


def adv_heating_vert_da(ds, which):
    """Vertical advective heating w*dT/dz [degC/day], (time, obs_depth). A heating RATE
    (depends on the GRADIENT), so reference-INDEPENDENT -- matches osse_tools.advective_heating.
    The continuity w (interfaces) is co-located onto the OCV-mean-T obs-depth midpoints."""
    w = ds[f'w_{which}']                     # (time, depth) interfaces, m/s
    T = ds[f'Tbar_{which}']                  # (time, obs_depth) OCV volume-mean T
    zmid = T['obs_depth'].values
    w_mid = w.interp(depth=zmid).rename({'depth': 'obs_depth'}).assign_coords(obs_depth=zmid)
    dTdz = T.differentiate('obs_depth')      # degC/m (obs_depth is z in metres)
    return (w_mid * dTdz) * SEC_PER_DAY      # degC/day


def adv_heating_horiz_da(ds, which):
    """Horizontal advective heating u*dT/dx + v*dT/dy [degC/day], (time, obs_depth). By the
    divergence theorem the OCV mean is the face flux of the temperature ANOMALY,
    <u_h.grad_h T> = (1/A) sum_faces <u_n (T - <T>)> L  =  (1/A) sum (unT - Tbar*un) L.
    Reference-independent (the anomaly removes the T*div term that made a flux frame-dependent)."""
    A = ds.attrs['area']
    L = ds['face_len']                       # (face,), m
    un = ds[f'un_{which}']                    # (time, face, obs_depth), m/s
    unT = ds[f'unT_{which}']                  # (time, face, obs_depth), degC m/s
    T = ds[f'Tbar_{which}']                   # (time, obs_depth), degC
    return ((unT - T * un) * L).sum('face') / A * SEC_PER_DAY


def adv_heating_da(ds, which, comp='total'):
    """Advective heating [degC/day]; comp in {'vert','horiz','total'}."""
    if comp == 'vert':
        return adv_heating_vert_da(ds, which)
    if comp == 'horiz':
        return adv_heating_horiz_da(ds, which)
    return adv_heating_vert_da(ds, which) + adv_heating_horiz_da(ds, which)


def adv_heating_flux_da(ds, which, comp='total'):
    """Advective heating expressed as a depth-cumulative heat FLUX [W/m^2]. Per the repo
    convention (osse_tools.advective_heating) a heating RATE integrates vertically to a flux,
    F(z) = rho0*cp * int_{top}^{z} (u.grad T) dz'. We undo the per-day scaling (-> degC/s),
    multiply by HFLUX (rho0*cp) and cumulative-trapezoid down from the shallowest observed
    depth (8 m). Sign follows the heating rate (positive = net warming of the column above z);
    comp in {'vert','horiz','total'}."""
    h = adv_heating_da(ds, which, comp).transpose('time', 'obs_depth') / SEC_PER_DAY  # degC/s
    # keep the OBSERVED column only: the OCV-mean T (hence w.dT/dz, and its dT/dz stencil one
    # point deeper) is undefined in the top ~8 m, so drop those near-surface NaN levels and
    # integrate the flux cumulatively from the top of the observed column downward.
    h = h.dropna('obs_depth', how='all')
    z = h['obs_depth'].values                                # metres, negative downward
    order = np.argsort(-z)                                    # shallow (near-surface) -> deep
    zc, vals = z[order], h.values[:, order]                  # (time, nz) shallow->deep
    zeta = -zc                                               # depth below surface, ascending
    integ = np.zeros_like(vals)
    integ[:, 1:] = np.cumsum(0.5 * (vals[:, 1:] + vals[:, :-1]) * np.diff(zeta), axis=1)
    F = xr.DataArray(C.HFLUX * integ, dims=('time', 'obs_depth'),
                     coords={'time': h['time'].values, 'obs_depth': zc})
    return F.sortby('obs_depth')


def dTdt_da(ds, which):
    """OCV volume-mean temperature tendency d<T>/dt [degC/day] (centred time difference)."""
    T = ds[f'Tbar_{which}']
    dt = C.ITER_STEP * C.DELTA_T             # 3-hourly spacing (s)
    ax = T.get_axis_num('time')
    return T.copy(data=np.gradient(T.values, dt, axis=ax)) * SEC_PER_DAY


def heat_series(ds, which, z, comp='total'):
    """Advective heating [degC/day] at depth z; comp in {'vert','horiz','total'}."""
    return at_depth(adv_heating_da(ds, which, comp), z)


def face_anom_flux_da(ds, which):
    """Per-face outward heat flux of the temperature ANOMALY, u_n*(T - <T>) [degC m/s],
    (time, face, obs_depth). Reference-free (the anomaly removes the reference-heat carried
    by the net mass flux); (x face_len / A, summed over faces) = horizontal advective heating."""
    return ds[f'unT_{which}'] - ds[f'Tbar_{which}'] * ds[f'un_{which}']


# --------------------------------------------------------------------------- stats
def js_dist(obs, truth, bins=60):
    """Jensen-Shannon distance between two 1-D temporal samples (0=identical)."""
    o = np.asarray(obs); o = o[np.isfinite(o)]
    t = np.asarray(truth); t = t[np.isfinite(t)]
    if o.size < 5 or t.size < 5:
        return np.nan
    lo = min(o.min(), t.min()); hi = max(o.max(), t.max())
    edges = np.linspace(lo, hi, bins + 1)
    return float(ot._js_distance(o.reshape(-1, 1), t.reshape(-1, 1), [edges]))


def step_pdf(ax, x, edges, color, ls, lw=1.9):
    """Outline (unfilled) density histogram, so several diameters overlay legibly."""
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size < 5:
        return
    h, _ = np.histogram(x, bins=edges, density=True)
    c = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(c, h, color=color, ls=ls, lw=lw)


def common_edges(samples, nbins=34, pad=(0.1, 99.9), margin=0.03):
    """Shared bin edges spanning several 1-D samples. Clips only the extreme tails
    (default 0.1/99.9 pct, ~1 outlier of a few hundred) then adds a small margin so the
    histogram/fit tails don't run off the frame."""
    both = np.concatenate([np.asarray(s).ravel() for s in samples])
    both = both[np.isfinite(both)]
    lo, hi = np.nanpercentile(both, pad[0]), np.nanpercentile(both, pad[1])
    span = hi - lo
    return np.linspace(lo - margin * span, hi + margin * span, nbins)


def overlay_pdf_panel(ax, series, unit, color_fn=C.diam_color):
    """Overlay obs (solid) vs truth (dotted) temporal PDFs for every key, coloured by
    color_fn(key). series maps key -> (obs_1d, truth_1d) (key = diameter or shape)."""
    edges = common_edges([v for pair in series.values() for v in pair])
    for k, (o, t) in series.items():
        col = color_fn(k)
        step_pdf(ax, t, edges, col, ':')
        step_pdf(ax, o, edges, col, '-')
    ax.set_xlabel(unit)
    tidy_x(ax)


def _skewfit_params(x):
    """Skew-normal MLE parameters (a, loc, scale); None if too few points."""
    from scipy.stats import skewnorm
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size < 20:
        return None
    try:
        return skewnorm.fit(x)
    except Exception:
        return None


def _skew_moments(params):
    """(mean, std, skewness) of the fitted skew-normal, for the parameter box."""
    from scipy.stats import skewnorm
    m, v, s = skewnorm.stats(*params, moments='mvs')
    return float(m), float(np.sqrt(v)), float(s)


def _param_box(ax, lines, loc='upper left', fs=7.5):
    """A SINGLE framed box holding all colour-coded fit-parameter lines: (text, colour)."""
    children = [TextArea(txt, textprops=dict(color=color, fontsize=fs))
                for txt, color in lines]
    packed = VPacker(children=children, align='left', pad=0, sep=1.0)
    box = AnchoredOffsetbox(loc=loc, child=packed, pad=0.25, borderpad=0.3, frameon=True)
    box.patch.set(facecolor='white', edgecolor='0.8', alpha=0.85)
    ax.add_artist(box)


def series_pdf_panel(ax, series, unit, nbins=36, box=True):
    """Draw several distributions on one axis. `series` = list of (label, sample, colour).
    The 'truth' series is drawn as grey filled shading; every other (estimate) histogram is a
    SOLID line; each distribution's skew-normal fit is a DASHED line in the same colour. One
    upper-left box lists the fit moments (mean/std/skew) per series."""
    from scipy.stats import skewnorm
    edges = common_edges([s for _, s, _ in series], nbins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xg = np.linspace(edges[0], edges[-1], 200)
    box_lines = []
    for label, samp, color in series:
        s = np.asarray(samp); s = s[np.isfinite(s)]
        if label == 'truth':
            ax.hist(s, bins=edges, density=True, color=color, alpha=0.45)  # grey shading
        else:
            h, _ = np.histogram(s, bins=edges, density=True)
            ax.plot(centers, h, color=color, lw=1.0, ls='-', drawstyle='steps-mid')
        p = _skewfit_params(s)
        if p is not None:
            ax.plot(xg, skewnorm.pdf(xg, *p), color=color, lw=1.8, ls='--')
            mu, sd, sk = _skew_moments(p)
            box_lines.append((f'{label}: $\\mu$={mu:.2g}, $\\sigma$={sd:.2g}, '
                              f'$\\gamma$={sk:+.2f}', color))
    if box and box_lines:
        _param_box(ax, box_lines)
    ax.set_xlabel(unit)
    tidy_x(ax)


def raise_pdf_headroom(axes, n_series):
    """Add top headroom to PDF panels so the upper-left fit box clears the histogram peak.
    Call AFTER all panels are drawn (autoscale settled). Scales with the number of stacked
    text lines (one per series). Safe with sharey (all axes share the same top)."""
    factor = 1.12 + 0.13 * n_series
    axs = list(np.ravel(np.asarray(axes)))
    tops = [ax.get_ylim()[1] for ax in axs]   # read all BEFORE modifying (sharey compounds)
    for ax, top in zip(axs, tops):
        ax.set_ylim(top=top * factor)


def pdf_panel(ax, obs, truth, unit, nbins=36):
    """Truth (filled grey) vs obs (step) temporal histogram + JS annotation."""
    o = np.asarray(obs); o = o[np.isfinite(o)]
    t = np.asarray(truth); t = t[np.isfinite(t)]
    both = np.concatenate([o, t])
    lo, hi = np.nanpercentile(both, 0.5), np.nanpercentile(both, 99.5)
    bins = np.linspace(lo, hi, nbins)
    ax.hist(t, bins=bins, density=True, color='0.6', alpha=0.55, label='model truth')
    ax.hist(o, bins=bins, density=True, histtype='step', color='#08306b', lw=2.2,
            label='array estimate')
    ax.axvline(np.mean(t), color='0.35', **TRUTH_KW)
    ax.axvline(np.mean(o), color='#08306b', **OBS_KW)
    ax.text(0.03, 0.96, f'JS {js_dist(o, t):.2f}', transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
    ax.set_xlabel(unit)
    tidy_x(ax)


# --------------------------------------------------------------------------- legends
def top_legend(fig, diam=True, method=True, extra=None, ncol=None, y=1.04):
    handles = []
    if diam:
        handles += [Line2D([0], [0], color=C.diam_color(d), lw=2.6, label=f'{d:g}$^\\circ$')
                    for d in C.DIAMETERS]
    if method:
        handles += [Line2D([0], [0], color='0.25', **OBS_KW, label='array estimate'),
                    Line2D([0], [0], color='0.25', **TRUTH_KW, label='model truth')]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc='upper center', ncol=ncol or len(handles),
               frameon=False, bbox_to_anchor=(0.5, y))


# shape -> colour/label for cross-shape figures (diamond blue, square red, hexagon green)
def shape_color(shape):
    return C.SHAPE_COLOR[shape]


def shape_top_legend(fig, shapes=None, method=True, extra=None, ncol=None, y=1.04):
    """Top legend keyed by SHAPE colour (+ obs/truth line-style key)."""
    shapes = shapes or C.SHAPES
    handles = [Line2D([0], [0], color=C.SHAPE_COLOR[s], lw=2.6, label=C.SHAPE_LABEL[s])
               for s in shapes]
    if method:
        handles += [Line2D([0], [0], color='0.25', **OBS_KW, label='array estimate'),
                    Line2D([0], [0], color='0.25', **TRUTH_KW, label='model truth')]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc='upper center', ncol=ncol or len(handles),
               frameon=False, bbox_to_anchor=(0.5, y))


def finish(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path
