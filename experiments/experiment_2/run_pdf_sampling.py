"""
run_pdf_sampling.py — how well does an array sample the TRUE field distribution
inside its own footprint?

For the symmetric REGULAR shape sweep at 0N/140W — hexagon (`symhex`), square
(`symsq`) and diamond (`symdia`), each at E-W diameter 0.3/0.5/0.75/1.0deg — this
compares, at 25/50/75 m and a 0-80 m depth average:

  * the distribution the ARRAY sees: the model fields interpolated to the handful
    of glider positions (`sample_fields`), against
  * the TRUTH inside the footprint: the model fields at every grid point in the
    convex hull of the array (`model_region`).

For each field (T, S, U, V, W, sigma0) the 1-D marginals are compared with the
Jensen-Shannon distance (0 identical / 1 disjoint; the repo's PDF-similarity
metric) and the Wasserstein (earth-mover) distance in the field's own units, each
against a random-placement null, plus the first three moments of both populations.
For the correlated
pairs the array is supposed to capture — the horizontal Reynolds stress (u'v'),
the vertical fluxes of horizontal momentum (u'w', v'w'), the eddy heat fluxes
(u'T', v'T', w'T') and the T-S relation — the JOINT
distribution gets a 2-D JS distance, and the correlation coefficient / covariance
the array reports is compared against the truth (primes are temporal-eddy
anomalies, value minus its own time mean, pooled over time and space).

W is the model's true WVEL sampled at the glider points (spatial-sampling
adequacy of w'T'), NOT the plane-fit w estimate — that estimator error is studied
separately by the heat-flux figures.

Usage:  python run_pdf_sampling.py [compute|plot|all]   (default all)
  compute — read the model once, sample every config, cache metrics + flagship
            (1.0deg) scatter arrays to cache/pdf_sampling.pkl and
            data/pdf_sampling_metrics.csv
  plot    — render everything under pdf_figs/ from the cache (no model read)
"""

import os
import pickle
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXP1 = os.path.join(REPO, 'experiments', 'experiment_1')
sys.path.insert(0, REPO)
from osse_tools import (load_model, load_positions, sample_fields, model_region,  # noqa: E402
                        add_density, _js_distance, sym_disk)
RUN_DIR = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS = list(range(36, 26173, 36))

FIG_DIR = os.path.join(HERE, 'pdf_figs')
# heavy .pkl cache lives on /data (not /home); see project_sym_disk_truth memory
CACHE_ROOT = '/data/SO3/edavenport/tpose24-osse/cache'
CACHE = os.path.join(CACHE_ROOT, 'experiment_2', 'pdf_sampling.pkl')
CSV = os.path.join(HERE, 'data', 'pdf_sampling_metrics.csv')

# sampling: 8-80 m at 4 m resolution (fine enough for a depth average and to pick
# the 25/50/75 m levels), time subsampled to keep the hull populations tractable.
MIN_DEPTH, MAX_DEPTH, DZ = 8, 80, 4
TSTEP = 3
FLAG_DIAM = 1.0                           # flagship diameter for the detail figures

SHAPES = {'symhex': 'hexagon', 'symsq': 'square', 'symdia': 'diamond'}
FAM_OF = {shape: fam for fam, shape in SHAPES.items()}
SHAPE_COLOR = {'diamond': '#1f77b4', 'hexagon': '#2ca02c', 'square': '#d62728'}
DIAMS = [0.3, 0.5, 0.75, 1.0]
DEPTHS = [('25m', -25.0), ('50m', -50.0), ('75m', -75.0), ('depthavg', None)]
DEPTH_KEYS = [d[0] for d in DEPTHS]

FIELDS = ['T', 'S', 'U', 'V', 'W', 'sigma0']
FIELD_UNITS = {'T': '°C', 'S': 'g/kg', 'U': 'm/s', 'V': 'm/s',
               'W': 'm/s', 'sigma0': 'kg/m³'}
FIELD_LABEL = {'T': 'T', 'S': 'S', 'U': 'U', 'V': 'V', 'W': 'W', 'sigma0': 'σ₀'}

# correlated pairs the array is meant to capture. Each: (x, y, label, human tag).
PAIRS = [
    ('Up', 'Vp', ("u'", "v'"), "Reynolds stress u'v'"),
    ('Up', 'Wp', ("u'", "w'"), "vert. flux zonal mom. u'w'"),
    ('Vp', 'Wp', ("v'", "w'"), "vert. flux merid. mom. v'w'"),
    ('Up', 'Tp', ("u'", "T'"), "zonal heat flux u'T'"),
    ('Vp', 'Tp', ("v'", "T'"), "merid. heat flux v'T'"),
    ('Wp', 'Tp', ("w'", "T'"), "vertical heat flux w'T'"),
    ('T',  'S',  ('T', 'S'),   'T-S'),
]
PAIR_KEYS = [p[3] for p in PAIRS]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _finite(a):
    a = np.asarray(a).ravel()
    return a[np.isfinite(a)]


def _edges(a, b, n):
    lo, hi = np.percentile(np.concatenate([a, b]), [0.5, 99.5])
    return np.linspace(lo, hi, n + 1) if hi > lo else np.linspace(lo - 1, hi + 1, n + 1)


def field_stats(obs2d, true2d, n_g, rng, nboot=40, bins=60):
    """1-D distance metrics + moments between the array (obs2d, time x n_glider) and
    the hull truth (true2d, time x n_point), plus a RANDOM-PLACEMENT null for BOTH the
    Jensen-Shannon and the Wasserstein distance: the metric you would get from `n_g`
    gliders dropped at random points in the same hull (mean and 95th pct over `nboot`
    draws), so the array can be read against the noise floor that its glider COUNT alone
    imposes. The shapes carry different glider counts (hexagon 6 vs square/diamond 4),
    so metric-minus-null is the fair shape-to-shape comparison."""
    from scipy.stats import skew, wasserstein_distance
    o = _finite(obs2d); t = _finite(true2d)
    if o.size < 3 or t.size < 3:
        return None
    edges = _edges(o, t, bins)
    js = _js_distance(o.reshape(-1, 1), t.reshape(-1, 1), [edges])
    wass = float(wasserstein_distance(o, t))
    npts = true2d.shape[1]
    js_nulls, w_nulls = [], []
    for _ in range(nboot):
        cols = rng.choice(npts, min(n_g, npts), replace=False)
        sub = _finite(true2d[:, cols])
        if sub.size >= 3:
            js_nulls.append(_js_distance(sub.reshape(-1, 1), t.reshape(-1, 1), [edges]))
            w_nulls.append(float(wasserstein_distance(sub, t)))
    js_nulls = np.array(js_nulls) if js_nulls else np.array([np.nan])
    w_nulls = np.array(w_nulls) if w_nulls else np.array([np.nan])
    return {
        'js': js, 'js_null': float(np.nanmean(js_nulls)),
        'js_null_p95': float(np.nanpercentile(js_nulls, 95)),
        'wasserstein': wass, 'w_null': float(np.nanmean(w_nulls)),
        'w_null_p95': float(np.nanpercentile(w_nulls, 95)),
        'obs_mean': float(o.mean()),  'true_mean': float(t.mean()),
        'obs_std': float(o.std()),    'true_std': float(t.std()),
        'obs_skew': float(skew(o)),   'true_skew': float(skew(t)),
        'n_obs': int(o.size),         'n_true': int(t.size),
    }


def _finite_cols(x2d, y2d, cols=None):
    if cols is not None:
        x2d, y2d = x2d[:, cols], y2d[:, cols]
    x, y = x2d.ravel(), y2d.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def pair_stats(ox2d, oy2d, tx2d, ty2d, n_g, rng, nboot=40, bins=45):
    """2-D JS + correlation/covariance the array reports vs the truth, with the same
    random-placement null on the joint JS (n_g random gliders in the hull)."""
    ox, oy = _finite_cols(ox2d, oy2d)
    tx, ty = _finite_cols(tx2d, ty2d)
    if ox.size < 3 or tx.size < 3:
        return None
    xe = _edges(ox, tx, bins); ye = _edges(oy, ty, bins)
    js = _js_distance(np.column_stack([tx, ty]), np.column_stack([ox, oy]), [xe, ye])
    npts = tx2d.shape[1]
    nulls = []
    for _ in range(nboot):
        cols = rng.choice(npts, min(n_g, npts), replace=False)
        sx, sy = _finite_cols(tx2d, ty2d, cols)
        if sx.size >= 3:
            nulls.append(_js_distance(np.column_stack([tx, ty]),
                                      np.column_stack([sx, sy]), [xe, ye]))
    nulls = np.array(nulls) if nulls else np.array([np.nan])
    return {
        'js2d': js, 'js2d_null': float(np.nanmean(nulls)),
        'js2d_null_p95': float(np.nanpercentile(nulls, 95)),
        'corr_obs': float(np.corrcoef(ox, oy)[0, 1]),
        'corr_true': float(np.corrcoef(tx, ty)[0, 1]),
        'cov_obs': float(np.cov(ox, oy)[0, 1]),
        'cov_true': float(np.cov(tx, ty)[0, 1]),
        'n_obs': int(ox.size), 'n_true': int(tx.size),
    }


def _reduce_depth(samp, target, space_dim):
    """Return a dict of (time x space) field arrays (incl. eddy primes) at one depth
    selection, keeping the point structure so a random-placement null can subsample
    locations.

    target None -> 0-80 m depth average; else the nearest obs_depth level. Eddy
    primes are deviations from the time mean at each location after the depth
    reduction."""
    if target is None:
        red = samp[FIELDS].mean('obs_depth')
    else:
        red = samp[FIELDS].sel(obs_depth=target, method='nearest')
    out = {}
    for v in FIELDS:
        da = red[v].transpose('time', space_dim)
        out[v] = da.values
        if v in ('U', 'V', 'W', 'T', 'S'):
            out[v + 'p'] = (da - da.mean('time')).values
    return out


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def _configs():
    """(path, shape, diameter) for the 12 symmetric shape configs at 0N."""
    out = []
    for fam, shape in SHAPES.items():
        for d in DIAMS:
            p = os.path.join(EXP1, 'configs', fam, f'{fam}_d{d}_c+0.0.json')
            if os.path.exists(p):
                out.append((p, shape, d))
    return out


def compute():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    ds = load_model(RUN_DIR, ITERS).sel(time=slice('2012-10-11', None))
    ds = ds.isel(time=slice(None, None, TSTEP))
    vars_sample = ('UVEL', 'VVEL', 'WVEL', 'THETA', 'SALT')

    field_rows, pair_rows, detail = [], [], {}
    rng = np.random.default_rng(0)

    for path, shape, diam in _configs():
        name = os.path.basename(path)[:-5]
        positions = load_positions(path)
        with open(path) as f:
            disk = sym_disk(json.load(f))        # circular-disk truth for sym shapes
        sys.stderr.write(f'[compute] {name}  ({shape}, {diam}deg)\n'); sys.stderr.flush()

        obs = add_density(sample_fields(ds, positions, vars=vars_sample,
                          min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, dz_obs=DZ)).compute()
        true = add_density(model_region(ds, positions, vars=vars_sample,
                           min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, dz_obs=DZ, disk=disk)).compute()

        n_g = len(positions)
        for dkey, dtarget in DEPTHS:
            ored = _reduce_depth(obs, dtarget, 'glider')
            tred = _reduce_depth(true, dtarget, 'point')

            for v in FIELDS:
                s = field_stats(ored[v], tred[v], n_g, rng)
                if s:
                    field_rows.append(dict(config=name, shape=shape, diameter=diam,
                                           depth=dkey, field=v, **s))
            for xk, yk, _lab, tag in PAIRS:
                s = pair_stats(ored[xk], ored[yk], tred[xk], tred[yk], n_g, rng)
                if s:
                    pair_rows.append(dict(config=name, shape=shape, diameter=diam,
                                          depth=dkey, pair=tag, **s))

            if diam == FLAG_DIAM:                    # keep flagship scatter arrays
                keep = FIELDS + ['Up', 'Vp', 'Wp', 'Tp', 'Sp']
                o = {k: ored[k].ravel() for k in keep}
                # thin the dense hull population for storage / plotting legibility
                n = tred['T'].reshape(-1).size
                idx = (rng.choice(n, 40000, replace=False) if n > 40000
                       else np.arange(n))
                t = {k: tred[k].reshape(-1)[idx] for k in keep}
                detail.setdefault(shape, {})[dkey] = {'obs': o, 'true': t}

    fields_df = pd.DataFrame(field_rows)
    pairs_df = pd.DataFrame(pair_rows)
    fields_df.to_csv(CSV, index=False)
    pairs_df.to_csv(CSV.replace('.csv', '_pairs.csv'), index=False)
    with open(CACHE, 'wb') as f:
        pickle.dump({'fields': fields_df, 'pairs': pairs_df, 'detail': detail,
                     'meta': dict(min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, dz=DZ,
                                  tstep=TSTEP, flag_diam=FLAG_DIAM)}, f)
    sys.stderr.write(f'[compute] wrote {CACHE}\n  {len(fields_df)} field rows, '
                     f'{len(pairs_df)} pair rows\n')


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _style():
    plt.rcParams.update({
        'axes.labelsize': 12.5, 'axes.labelweight': 'bold',
        'axes.titlesize': 12, 'legend.fontsize': 11, 'figure.dpi': 120,
    })


def _shape_legend(fig, shapes, extra=None):
    handles = [Line2D([0], [0], color=SHAPE_COLOR[s], lw=2.5, marker='o', label=s)
               for s in shapes]
    for label, ls in (extra or []):
        handles.append(Line2D([0], [0], color='0.3', lw=1.6, ls=ls, label=label))
    fig.legend(handles=handles, ncol=len(handles), loc='upper center',
               frameon=False, bbox_to_anchor=(0.5, 1.0))


def _save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _js2d(ox, oy, tx, ty):
    ox, oy = _finite2(ox, oy); tx, ty = _finite2(tx, ty)
    if ox.size < 3 or tx.size < 3:
        return np.nan
    return _js_distance(np.column_stack([tx, ty]), np.column_stack([ox, oy]),
                        [_edges(ox, tx, 45), _edges(oy, ty, 45)])


def _hist(ax, o, t, units=''):
    o, t = _finite(o), _finite(t)
    lo, hi = np.percentile(np.concatenate([o, t]), [0.5, 99.5])
    edges = np.linspace(lo, hi, 51) if hi > lo else np.linspace(lo - 1, hi + 1, 51)
    ax.hist(t, edges, density=True, color='0.6', alpha=0.6, label='truth (disk)')
    ax.hist(o, edges, density=True, histtype='step', color='C3', lw=1.8, label='array')


def fig_field_pdfs(detail, fields_df, shape, dkey):
    """1-D marginal PDFs (array vs hull-truth) for one shape at one depth, annotated
    with JS and Wasserstein (W, field units), each against its random-placement null
    (looked up from the CSV — the null needs the hull point structure, not cached here)."""
    d = detail[shape][dkey]
    cfg = f'{FAM_OF[shape]}_d{FLAG_DIAM}_c+0.0'
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, v in zip(axes.ravel(), FIELDS):
        _hist(ax, d['obs'][v], d['true'][v], FIELD_UNITS[v])
        row = fields_df[(fields_df.config == cfg) & (fields_df.depth == dkey)
                        & (fields_df.field == v)]
        if len(row):
            r = row.iloc[0]
            txt = (f"JS {r.js:.2f} / null {r.js_null:.2f}\n"
                   f"W {r.wasserstein:.2g} / null {r.w_null:.2g}")
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, bbox=dict(boxstyle='round', fc='w', ec='0.7', alpha=0.85))
        ax.set_xlabel(f'{FIELD_LABEL[v]} ({FIELD_UNITS[v]})')
        ax.set_ylabel('pdf'); ax.grid(alpha=0.3)
    axes[0, 0].legend(loc='upper left', fontsize=10, frameon=False)
    fig.tight_layout()
    _save(fig, f'field_pdfs_{shape}_d{FLAG_DIAM}_{dkey}.png')


def fig_joint(detail, pairs_df, shape, dkey, max_pts=8000):
    """Joint scatter (array over hull-truth) for the correlated pairs, each axis
    standardized by the truth mean/std so the correlation shape is legible across the
    pairs' very different physical scales. Annotated with the 2-D JS against its
    random-placement null (from the CSV)."""
    rng = np.random.default_rng(0)
    d = detail[shape][dkey]
    cfg = f'{FAM_OF[shape]}_d{FLAG_DIAM}_c+0.0'
    col = SHAPE_COLOR[shape]
    ncol = 4
    nrow = int(np.ceil(len(PAIRS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.75 * ncol, 4.5 * nrow))
    axflat = axes.ravel()
    for ax in axflat[len(PAIRS):]:
        ax.axis('off')
    for ax, (xk, yk, lab, tag) in zip(axflat, PAIRS):
        tx, ty = _finite2(d['true'][xk], d['true'][yk])
        ox, oy = _finite2(d['obs'][xk], d['obs'][yk])
        r_obs = np.corrcoef(ox, oy)[0, 1] if ox.size > 2 else np.nan
        r_true = np.corrcoef(tx, ty)[0, 1] if tx.size > 2 else np.nan
        js = _js2d(ox, oy, tx, ty)
        prow = pairs_df[(pairs_df.config == cfg) & (pairs_df.depth == dkey)
                        & (pairs_df.pair == tag)]
        jn = float(prow.js2d_null.values[0]) if len(prow) else np.nan
        # standardize each axis by the TRUTH mean/std (a common transform for both
        # clouds): the very different physical scales (w'~1e-4 m/s vs T'~0.5 degC)
        # otherwise flatten the momentum-flux clouds to a line. r and JS are invariant
        # under this affine rescaling; the array cloud stays NARROWER than truth where
        # it under-samples the variance (normalised by truth sigma, not its own).
        mx, sx = tx.mean(), tx.std()
        my, sy = ty.mean(), ty.std()
        sx = sx if sx > 0 else 1.0
        sy = sy if sy > 0 else 1.0
        tx, ty = (tx - mx) / sx, (ty - my) / sy
        ox, oy = (ox - mx) / sx, (oy - my) / sy
        if tx.size > max_pts:
            i = rng.choice(tx.size, max_pts, replace=False); tx, ty = tx[i], ty[i]
        xlo, xhi = np.percentile(np.concatenate([ox, tx]), [0.5, 99.5])
        ylo, yhi = np.percentile(np.concatenate([oy, ty]), [0.5, 99.5])
        lim = max(abs(xlo), abs(xhi), abs(ylo), abs(yhi))   # symmetric, square panel
        ax.scatter(tx, ty, s=6, color='0.6', alpha=0.35, lw=0, label='truth (disk)')
        ax.scatter(ox, oy, s=10, color=col, alpha=0.7, lw=0, label='array')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axhline(0, color='0.5', lw=0.5, ls=':'); ax.axvline(0, color='0.5', lw=0.5, ls=':')
        ax.set_xlabel(f'{lab[0]} / σ'); ax.set_ylabel(f'{lab[1]} / σ')
        ax.set_title(f"{tag}\nr: array {r_obs:.2f} / truth {r_true:.2f}\n"
                     f"JS {js:.2f} / null {jn:.2f}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc='upper left', fontsize=9, frameon=False, markerscale=2)
    fig.tight_layout()
    _save(fig, f'joint_{shape}_d{FLAG_DIAM}_{dkey}.png')


def _finite2(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def fig_js_grid(df, value, keycol, keys, key_units, fname, ylabel, null_col=None,
                hline=None, sharey=True):
    """rows = depth, cols = quantity: `value` vs diameter, one line per shape. If
    `null_col` is given, its random-placement floor is drawn as a thin dashed line of
    the same colour (array metric at or below its own null = geometry samples as well as
    random placement of the same glider count). `sharey=False` for the Wasserstein grid,
    whose columns carry different physical units and cannot share a y-axis."""
    shapes = [SHAPES[f] for f in SHAPES]
    nrow, ncol = len(DEPTH_KEYS), len(keys)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow),
                             sharex=True, sharey=sharey, squeeze=False)
    for r, dkey in enumerate(DEPTH_KEYS):
        for c, k in enumerate(keys):
            ax = axes[r, c]
            for shape in shapes:
                sub = df[(df[keycol] == k) & (df.depth == dkey) & (df['shape'] == shape)]
                sub = sub.sort_values('diameter')
                if len(sub):
                    ax.plot(sub.diameter, sub[value], '-o', color=SHAPE_COLOR[shape],
                            lw=2, ms=5)
                    if null_col:
                        ax.plot(sub.diameter, sub[null_col], ':', color=SHAPE_COLOR[shape],
                                lw=1.3, alpha=0.9)
            if hline is not None:
                ax.axhline(hline, color='k', lw=0.8, ls='--', alpha=0.6)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(key_units.get(k, k), fontsize=11)
            if c == 0:
                ax.set_ylabel(f'{dkey}\n{ylabel}')
            if r == nrow - 1:
                ax.set_xlabel('E-W diameter (°)')
            ax.set_xticks(DIAMS)
    _shape_legend(fig, shapes, extra=[('random-placement null', ':')] if null_col else None)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save(fig, fname)


def fig_corr_recovery(pairs_df):
    """rows = depth, cols = pair: the (dimensionless) correlation coefficient r of
    each pair over the hull truth (solid) vs the array sample (dashed) per shape.
    The solid-dashed gap is the array's sampling error in r."""
    shapes = [SHAPES[f] for f in SHAPES]
    nrow, ncol = len(DEPTH_KEYS), len(PAIR_KEYS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow),
                             sharex=True, squeeze=False)
    for r, dkey in enumerate(DEPTH_KEYS):
        for c, k in enumerate(PAIR_KEYS):
            ax = axes[r, c]
            for shape in shapes:
                sub = pairs_df[(pairs_df.pair == k) & (pairs_df.depth == dkey)
                               & (pairs_df['shape'] == shape)].sort_values('diameter')
                if len(sub):
                    ax.plot(sub.diameter, sub.corr_true, '-', color=SHAPE_COLOR[shape], lw=2.4)
                    ax.plot(sub.diameter, sub.corr_obs, '--o', color=SHAPE_COLOR[shape],
                            lw=1.6, ms=4, mfc='w')
            ax.axhline(0, color='k', lw=0.5, ls=':')
            ax.grid(alpha=0.3); ax.set_xticks(DIAMS)
            if r == 0:
                ax.set_title(k, fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{dkey}\ncorrelation coeff. r')
            if r == nrow - 1:
                ax.set_xlabel('E-W diameter (°)')
    handles = [Line2D([0], [0], color=SHAPE_COLOR[s], lw=2.5, label=s) for s in shapes]
    handles += [Line2D([0], [0], color='0.3', lw=2.4, label='truth (disk)'),
                Line2D([0], [0], color='0.3', lw=1.6, ls='--', marker='o',
                       mfc='w', label='array')]
    fig.legend(handles=handles, ncol=len(handles), loc='upper center',
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    _save(fig, 'summary_corr_recovery.png')


def fig_heatmap(fields_df, pairs_df):
    """JS-vs-null overview: rows = quantity, cols = config, per depth. Cells show
    JS − JS_null (marginals, top block) and joint-JS − null (pairs, bottom block), so
    the metric is read against the random-placement floor its glider COUNT imposes and
    hexagon (6) vs square/diamond (4) is a fair comparison. Diverging scale about 0:
    blue = below the floor (geometry beats random placement), red = above it (worse
    than random of the same count). The two families sit on SEPARATE symmetric scales."""
    configs = [f'{f}_d{d}_c+0.0' for f in SHAPES for d in DIAMS]
    short = {f'{f}_d{d}_c+0.0': f'{SHAPES[f][:3]}\n{d}' for f in SHAPES for d in DIAMS}
    field_q = [(v, FIELD_LABEL[v]) for v in FIELDS]
    pair_q = [(k, k) for k in PAIR_KEYS]
    ncol = len(DEPTH_KEYS)

    def _matrix(src, kc, val, null, keys, dkey):
        M = np.full((len(keys), len(configs)), np.nan)
        for qi, (key, _lab) in enumerate(keys):
            for ci, cfg in enumerate(configs):
                row = src[(src.config == cfg) & (src.depth == dkey) & (src[kc] == key)]
                if len(row):
                    M[qi, ci] = row[val].values[0] - row[null].values[0]
        return M

    # symmetric limits about 0, per block, shared across depths so columns compare.
    def _lim(src, kc, val, null, keys):
        vals = np.concatenate([_matrix(src, kc, val, null, keys, d).ravel()
                               for d in DEPTH_KEYS])
        vals = vals[np.isfinite(vals)]
        m = float(np.nanpercentile(np.abs(vals), 98)) if vals.size else 1.0
        return -m, m

    flo, fhi = _lim(fields_df, 'field', 'js', 'js_null', field_q)
    plo, phi = _lim(pairs_df, 'pair', 'js2d', 'js2d_null', pair_q)

    fig = plt.figure(figsize=(4.4 * ncol, 8))
    gs = fig.add_gridspec(2, ncol, height_ratios=[len(field_q), len(pair_q)],
                          hspace=0.18, wspace=0.08)
    for j, dkey in enumerate(DEPTH_KEYS):
        for bi, (keys, src, kc, val, null, vlo, vhi) in enumerate(
                [(field_q, fields_df, 'field', 'js', 'js_null', flo, fhi),
                 (pair_q, pairs_df, 'pair', 'js2d', 'js2d_null', plo, phi)]):
            ax = fig.add_subplot(gs[bi, j])
            M = _matrix(src, kc, val, null, keys, dkey)
            im = ax.imshow(M, aspect='auto', cmap='cmo.balance', vmin=vlo, vmax=vhi)
            ax.set_yticks(range(len(keys)))
            ax.set_yticklabels([k[1] for k in keys], fontsize=9 if j == 0 else 0)
            if j != 0:
                ax.tick_params(axis='y', length=0)
            if bi == 0:
                ax.set_title(dkey, fontsize=12, fontweight='bold')
                ax.set_xticks([])
            else:
                ax.set_xticks(range(len(configs)))
                ax.set_xticklabels([short[c] for c in configs], fontsize=7)
            for ci in range(1, len(SHAPES)):    # divide the three shape blocks
                ax.axvline(ci * len(DIAMS) - 0.5, color='k', lw=2.5)
            if j == ncol - 1:
                lbl = ('marginal JS − null' if bi == 0 else 'joint JS − null')
                fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label=lbl,
                             extend='both')
    _save(fig, 'summary_js_heatmap.png')


def fig_corr_heatmap(pairs_df):
    """Complement to the JS heatmap: the SIGNED correlation-recovery error
    Δr = r_array − r_true per pair × config, per depth. JS is an unsigned distance;
    this shows whether the array OVER- or UNDER-states each true eddy-flux correlation.
    Diverging scale about 0 (blue = under-states, white = recovers r exactly, red =
    over-states), symmetric limits clipped to the robust |Δr| spread."""
    configs = [f'{f}_d{d}_c+0.0' for f in SHAPES for d in DIAMS]
    short = {f'{f}_d{d}_c+0.0': f'{SHAPES[f][:3]}\n{d}' for f in SHAPES for d in DIAMS}
    ncol = len(DEPTH_KEYS)

    def _matrix(dkey):
        M = np.full((len(PAIR_KEYS), len(configs)), np.nan)
        for qi, key in enumerate(PAIR_KEYS):
            for ci, cfg in enumerate(configs):
                row = pairs_df[(pairs_df.config == cfg) & (pairs_df.depth == dkey)
                               & (pairs_df.pair == key)]
                if len(row):
                    M[qi, ci] = row.corr_obs.values[0] - row.corr_true.values[0]
        return M

    allv = np.concatenate([_matrix(d).ravel() for d in DEPTH_KEYS])
    vmax = float(np.nanpercentile(np.abs(allv[np.isfinite(allv)]), 98))

    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 4.4), squeeze=False)
    for j, dkey in enumerate(DEPTH_KEYS):
        ax = axes[0, j]
        im = ax.imshow(_matrix(dkey), aspect='auto', cmap='cmo.balance',
                       vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(PAIR_KEYS)))
        ax.set_yticklabels(PAIR_KEYS if j == 0 else [''] * len(PAIR_KEYS), fontsize=9)
        if j != 0:
            ax.tick_params(axis='y', length=0)
        ax.set_title(dkey, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([short[c] for c in configs], fontsize=7)
        for ci in range(1, len(SHAPES)):        # divide the three shape blocks
            ax.axvline(ci * len(DIAMS) - 0.5, color='k', lw=2.5)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02,
                 label='Δr = r$_{array}$ − r$_{true}$', extend='both')
    _save(fig, 'summary_corr_recovery_heatmap.png')


def fig_w_heatmap(fields_df):
    """Per-variable Wasserstein-vs-null heatmap: one panel per field (units differ, so
    each field gets its OWN colourbar — can't share a scale across fields). Cells show
    W − W_null in the field's units, rows = the 12 shape×diameter configs, cols = depth.
    Diverging scale about 0: blue = below the random-placement floor (geometry beats
    random of the same glider count), red = above it. This is the 'which pattern samples
    each variable best' view."""
    configs = [f'{f}_d{d}_c+0.0' for f in SHAPES for d in DIAMS]
    ylabels = [f'{SHAPES[f][:3]} {d}' for f in SHAPES for d in DIAMS]
    ncol = 3
    nrow = int(np.ceil(len(FIELDS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 2.7 * nrow), squeeze=False)
    axflat = axes.ravel()
    for ax in axflat[len(FIELDS):]:
        ax.axis('off')
    for ax, v in zip(axflat, FIELDS):
        M = np.full((len(configs), len(DEPTH_KEYS)), np.nan)
        for ci, cfg in enumerate(configs):
            for dj, dk in enumerate(DEPTH_KEYS):
                row = fields_df[(fields_df.config == cfg) & (fields_df.depth == dk)
                                & (fields_df.field == v)]
                if len(row):
                    M[ci, dj] = row.wasserstein.values[0] - row.w_null.values[0]
        finite = M[np.isfinite(M)]
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        vmax = vmax or 1.0
        im = ax.imshow(M, aspect='auto', cmap='cmo.balance', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{FIELD_LABEL[v]} ({FIELD_UNITS[v]})', fontsize=11,
                     fontweight='bold')
        ax.set_xticks(range(len(DEPTH_KEYS)))
        ax.set_xticklabels(DEPTH_KEYS, fontsize=8)
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(ylabels, fontsize=7)
        for ci in range(1, len(SHAPES)):        # divide the three shape blocks
            ax.axhline(ci * len(DIAMS) - 0.5, color='k', lw=2)
        fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label='W − W$_{null}$',
                     extend='both')
    fig.tight_layout()
    _save(fig, 'summary_w_heatmap.png')


def plot():
    import cmocean  # registers cmo.* colormaps
    _style()
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(CACHE, 'rb') as f:
        C = pickle.load(f)
    fields_df, pairs_df, detail = C['fields'], C['pairs'], C['detail']

    # detail figures at the flagship diameter
    for shape in detail:
        for dkey in DEPTH_KEYS:
            if dkey in detail[shape]:
                fig_field_pdfs(detail, fields_df, shape, dkey)
                fig_joint(detail, pairs_df, shape, dkey)

    # summary grids: JS (dimensionless) + Wasserstein (field units)
    fig_js_grid(fields_df, 'js', 'field', FIELDS,
                {v: FIELD_LABEL[v] for v in FIELDS},
                'summary_js_fields.png', 'JS', null_col='js_null')
    fig_js_grid(pairs_df, 'js2d', 'pair', PAIR_KEYS, {k: k for k in PAIR_KEYS},
                'summary_js_pairs.png', 'JS (2-D)', null_col='js2d_null')
    fig_js_grid(fields_df, 'wasserstein', 'field', FIELDS,
                {v: f'{FIELD_LABEL[v]} ({FIELD_UNITS[v]})' for v in FIELDS},
                'summary_w_fields.png', 'Wasserstein dist.', null_col='w_null',
                sharey=False)
    fields_df = fields_df.assign(std_ratio=fields_df.obs_std / fields_df.true_std)
    fig_js_grid(fields_df, 'std_ratio', 'field', FIELDS,
                {v: FIELD_LABEL[v] for v in FIELDS},
                'summary_std_ratio.png', 'std. dev. ratio\n(array / truth)', hline=1.0)
    fig_corr_recovery(pairs_df)
    fig_heatmap(fields_df, pairs_df)
    fig_w_heatmap(fields_df)
    fig_corr_heatmap(pairs_df)
    sys.stderr.write(f'[plot] wrote figures under {FIG_DIR}\n')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('compute', 'all'):
        compute()
    if mode in ('plot', 'all'):
        plot()
