"""
Distribution sampling for the CIRCLING glider array (experiment_3), at the repo top
level (the sampling_dynamics "fit_distributions" analysis, applied to orbiting gliders).

How well does an orbiting array of N gliders on a circle of a given diameter reproduce
the TRUE field distribution inside its footprint?  For each quantity we pool the array's
platform / plane-fit samples over all times and compare against the fixed-disk truth
(every model grid point inside the disk x subsampled times).  Distributions are fit
(skew-normal, or a 2-Gaussian mixture for the oscillatory u'/v') to summarise shape.

Reads ONLY the experiment_3 caches (experiments/experiment_3/cache/), written by
run_experiment_3.py.  No model read.

Figures -> circling_gliders/distributions/
  fit_distributions_d{diam}.png     rows = w, T', w'T', w'u'; columns = N gliders (fixed
                                    diameter); array (coloured by N) vs disk truth (grey).
  fit_distributions_uv_d{diam}.png  rows = u', v', u'v', u'T', v'T'; columns = N gliders.
  fit_distributions_n{n}.png        the same w/T'/w'T'/w'u' fits with columns = diameter
                                    for a fixed glider count.
  js_summary.png                    JS - JS_null vs diameter, one line per N, for w/T/w'T'.

Usage:  python circling_gliders/run_distributions.py
"""
import os
import re
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from scipy.stats import skewnorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP3 = os.path.join(REPO, 'experiments', 'experiment_3')
sys.path.insert(0, EXP3)
sys.path.insert(0, REPO)
import circle_common as C          # noqa: E402  (geometry + cache paths)
import circle_plot as P            # noqa: E402  (loaders + N colours + style)
import osse_tools as ot            # noqa: E402  (_js_distance)

SUBDIR = os.path.join(HERE, 'distributions')
FLAGSHIP = 1.0
DEPTH = -30.0
SEC_PER_DAY = C.SEC_PER_DAY
HFLUX = C.HFLUX


# --------------------------------------------------------------------------- flux helpers
def _anom(da):
    return da - da.mean('time')


def array_vert_flux(arr, a, b):
    """Vertical eddy-flux series from the plane-fit w' and the array-mean tracer' ."""
    return _anom(arr[a]) * _anom(arr[b].mean('glider'))


def array_lat_flux(arr, a, b):
    """Lateral eddy-flux series: platform mean of a'b' (over the gliders)."""
    return (_anom(arr[a]) * _anom(arr[b])).mean('glider')


def _at(da, z):
    return np.asarray(da.interp(obs_depth=z).values).ravel()


def _fit_samples(n, diam, z=DEPTH):
    """(array, truth) 1-D samples of w, T', w'T', w'u' at depth z for one (N, diameter)."""
    arr = P.load_array(n, diam); disk = P.load_disk(diam); cloud = P.load_cloud(diam)
    w = (_at(arr['w_est_mid'], z) * SEC_PER_DAY, _at(cloud['W'], z) * SEC_PER_DAY)
    Tp = (_at(_anom(arr['T']), z), _at(_anom(cloud['T']), z))
    wT = (_at(array_vert_flux(arr, 'w_est_mid', 'T'), z) * HFLUX, _at(disk['wT'], z) * HFLUX)
    wU = (_at(array_vert_flux(arr, 'w_est_mid', 'U'), z), _at(disk['wU'], z))
    arr.close(); disk.close(); cloud.close()
    return {'w (m day$^{-1}$)': w, "$T'$ ($^\\circ$C)": Tp,
            "$w'T'$ (W m$^{-2}$)": wT, "$w'u'$ (m$^2$ s$^{-2}$)": wU}


def _fit_samples_uv(n, diam, z=DEPTH):
    """(array, truth) 1-D samples of u', v', u'v', u'T', v'T' at depth z."""
    arr = P.load_array(n, diam); disk = P.load_disk(diam); cloud = P.load_cloud(diam)

    def latflux(a, b, key, scale=1.0):
        return (_at(array_lat_flux(arr, a, b), z) * scale, _at(disk[key], z) * scale)

    out = {"$u'$ (m s$^{-1}$)": (_at(_anom(arr['U']), z), _at(_anom(cloud['U']), z)),
           "$v'$ (m s$^{-1}$)": (_at(_anom(arr['V']), z), _at(_anom(cloud['V']), z)),
           "$u'v'$ (m$^2$ s$^{-2}$)": latflux('U', 'V', 'uv'),
           "$u'T'$ (W m$^{-2}$)": latflux('U', 'T', 'uT', HFLUX),
           "$v'T'$ (W m$^{-2}$)": latflux('V', 'T', 'vT', HFLUX)}
    arr.close(); disk.close(); cloud.close()
    return out


# --------------------------------------------------------------------------- JS helper
def _js(o, t, bins=60):
    o = o[np.isfinite(o)]; t = t[np.isfinite(t)]
    if o.size < 5 or t.size < 5:
        return np.nan
    lo = min(o.min(), t.min()); hi = max(o.max(), t.max())
    edges = np.linspace(lo, hi, bins + 1)
    return float(ot._js_distance(o.reshape(-1, 1), t.reshape(-1, 1), [edges]))


def _js_null(t, n_obs, bins=60, ndraw=40, seed=0):
    rng = np.random.default_rng(seed)
    t = t[np.isfinite(t)]
    if t.size < max(5, n_obs):
        return np.nan
    edges = np.linspace(t.min(), t.max(), bins + 1)
    return float(np.mean([ot._js_distance(rng.choice(t, n_obs, replace=False).reshape(-1, 1),
                                          t.reshape(-1, 1), [edges]) for _ in range(ndraw)]))


# --------------------------------------------------------------------------- fit grid (from sampling_dynamics)
def _bimodal_fit(data, xx, n_iter=300):
    from scipy.stats import norm
    x = np.asarray(data)[np.isfinite(data)]
    mu = np.percentile(x, [25.0, 75.0]).astype(float)
    var = np.full(2, x.var() / 2.0 + 1e-12)
    w = np.array([0.5, 0.5])
    for _ in range(n_iter):
        p = np.stack([w[k] * norm.pdf(x, mu[k], np.sqrt(var[k])) for k in range(2)])
        r = p / (p.sum(0) + 1e-300)
        nk = r.sum(1) + 1e-12
        w = nk / nk.sum(); mu = (r * x).sum(1) / nk
        var = (r * (x - mu[:, None]) ** 2).sum(1) / nk + 1e-12
    pdf = sum(w[k] * norm.pdf(xx, mu[k], np.sqrt(var[k])) for k in range(2))
    order = np.argsort(mu)
    return pdf, [(float(mu[k]), float(np.sqrt(var[k]))) for k in order]


def _fit_grid(per_col, cols, col_title, fname, bimodal_labels=frozenset(), col_colors=None):
    labels = list(per_col[cols[0]])
    xlims = {}
    for lab in labels:
        allv = np.concatenate([np.concatenate(per_col[c][lab]) for c in cols])
        allv = allv[np.isfinite(allv)]
        xlims[lab] = np.percentile(allv, [0.5, 99.5])
    nc, nr = len(cols), len(labels)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.4 * nr), sharey='row', squeeze=False)
    for ri, lab in enumerate(labels):
        lo, hi = xlims[lab]
        bins = np.linspace(lo, hi, 45); xx = np.linspace(lo, hi, 200)
        bimodal = lab in bimodal_labels
        for ci, col in enumerate(cols):
            ax = axes[ri, ci]
            o, t = per_col[col][lab]
            arr_color = col_colors[col] if col_colors else '#08306b'
            notes = []
            for data, color, name in [(t, '0.5', 'truth'), (o, arr_color, 'array')]:
                data = data[np.isfinite(data)]
                kw = dict(alpha=0.5) if name == 'truth' else dict(histtype='step', lw=2.2)
                ax.hist(data, bins=bins, density=True, color=color, **kw)
                if bimodal:
                    pdf, comps = _bimodal_fit(data, xx)
                    (m1, s1), (m2, s2) = comps
                    note = (f'{name}: $\\mu_1$={m1:.2g}, $\\sigma_1$={s1:.2g}, '
                            f'$\\mu_2$={m2:.2g}, $\\sigma_2$={s2:.2g}')
                else:
                    a, loc, sc = skewnorm.fit(data)
                    pdf = skewnorm.pdf(xx, a, loc, sc)
                    note = (f'{name}: $\\mu$={np.mean(data):.2g}, '
                            f'$\\sigma$={np.std(data):.2g}, skew a={a:.2f}')
                ax.plot(xx, pdf, color=color, lw=1.6, ls='--')
                notes.append((note, color))
            box = AnchoredOffsetbox(
                loc='upper left', frameon=True, pad=0.3, borderpad=0.3,
                bbox_to_anchor=(0.0, 1.0), bbox_transform=ax.transAxes,
                child=VPacker(pad=0, sep=2, align='left', children=[
                    TextArea(nn, textprops=dict(color=cc, fontsize=8.0)) for nn, cc in notes]))
            box.patch.set(boxstyle='round,pad=0.25', fc='white', ec='0.6', alpha=0.85)
            ax.add_artist(box)
            ax.set_xlim(lo, hi)
            P.tidy_x(ax, 4)
            if ri == 0:
                ax.set_title(col_title(col))
            m = re.search(r'\(([^()]*)\)\s*$', lab)
            ax.set_xlabel(m.group(1) if m else lab)
            if ci == 0:
                nm = re.sub(r'\s*\([^()]*\)\s*$', '', lab)
                ax.set_ylabel(f'{nm}\ndensity')
        row_top = max(a.get_ylim()[1] for a in axes[ri, :])
        for a in axes[ri, :]:
            a.set_ylim(top=row_top * 1.42)
    fig_h = 3.4 * nr
    fig.tight_layout(rect=[0, 0, 1, 1.0 - 0.5 / fig_h])
    if col_colors:
        handles = [Line2D([0], [0], color='0.5', lw=8, alpha=0.5, label='truth hist'),
                   Line2D([0], [0], color='0.3', lw=1.6, ls='--', label='distribution fit')]
        handles += [Line2D([0], [0], color=col_colors[c], lw=2.4, label=col_title(c)) for c in cols]
    else:
        handles = [Line2D([0], [0], color='0.5', lw=8, alpha=0.5, label='truth hist'),
                   Line2D([0], [0], color='#08306b', lw=2.2, label='array hist'),
                   Line2D([0], [0], color='0.3', lw=1.6, ls='--', label='distribution fit')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    os.makedirs(SUBDIR, exist_ok=True)
    path = os.path.join(SUBDIR, fname)
    fig.savefig(path); plt.close(fig)
    return path


UV_BIMODAL = {"$u'$ (m s$^{-1}$)", "$v'$ (m s$^{-1}$)"}


def _n_title(n):
    return f'{n} gliders'


def _diam_title(d):
    return f'{d:g}$^\\circ$ circle'


# columns = glider count (fixed diameter) -------------------------------------------------
def fig_fit_by_n(diam=FLAGSHIP, z=DEPTH):
    per_col = {n: _fit_samples(n, diam, z) for n in C.N_GLIDERS}
    return _fit_grid(per_col, C.N_GLIDERS, _n_title,
                     f'fit_distributions_d{diam:g}.png', col_colors=P.N_COLOR)


def fig_fit_uv_by_n(diam=FLAGSHIP, z=DEPTH):
    per_col = {n: _fit_samples_uv(n, diam, z) for n in C.N_GLIDERS}
    return _fit_grid(per_col, C.N_GLIDERS, _n_title,
                     f'fit_distributions_uv_d{diam:g}.png', bimodal_labels=UV_BIMODAL,
                     col_colors=P.N_COLOR)


# columns = diameter (fixed glider count) -------------------------------------------------
def fig_fit_by_diam(n=6, z=DEPTH):
    per_col = {d: _fit_samples(n, d, z) for d in C.DIAMETERS}
    return _fit_grid(per_col, C.DIAMETERS, _diam_title,
                     f'fit_distributions_n{n}.png', col_colors=P.DIAM_COLOR)


# JS summary ------------------------------------------------------------------------------
def fig_js_summary(z=DEPTH):
    specs = [('w (m day$^{-1}$)', 0), ("$T'$ ($^\\circ$C)", 1), ("$w'T'$ (W m$^{-2}$)", 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (lab, _) in zip(axes, specs):
        for n in C.N_GLIDERS:
            vals = []
            for d in C.DIAMETERS:
                o, t = _fit_samples(n, d, z)[lab]
                n_obs = int(np.isfinite(o).sum())
                vals.append(_js(o, t) - _js_null(t[np.isfinite(t)], n_obs))
            ax.plot(C.DIAMETERS, vals, '-o', color=P.n_color(n), lw=2, label=f'{n} gliders')
        ax.axhline(0, color='0.5', lw=0.9)
        ax.set_title(lab)
        ax.set_xlabel('circle diameter ($^\\circ$)')
        P.tidy_x(ax, 4)
    axes[0].set_ylabel('JS $-$ JS$_{\\mathrm{null}}$  (30 m)')
    axes[0].legend(loc='upper left', frameon=False, ncol=2)
    fig.tight_layout()
    os.makedirs(SUBDIR, exist_ok=True)
    path = os.path.join(SUBDIR, 'js_summary.png')
    fig.savefig(path); plt.close(fig)
    return path


def main():
    print(fig_fit_by_n(FLAGSHIP))
    print(fig_fit_uv_by_n(FLAGSHIP))
    print(fig_fit_by_diam(6))
    print(fig_js_summary())


if __name__ == '__main__':
    main()
