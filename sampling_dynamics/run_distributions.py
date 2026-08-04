"""
Distribution sampling: how well does the array reproduce the TRUE distribution of each
quantity inside its own footprint?

For each variable we pool the array's platform samples (6 gliders + mooring x all times)
and compare against the full hull point cloud (every grid point x subsampled times).
Metrics: Jensen-Shannon distance and Wasserstein distance, both against a random-
placement null (draw n_platform points at random from the truth) so a small sample's
inherent penalty is separated from real geometry effects; plus the first three moments
(mean, std, skew).  Distributions are also FIT (skew-normal) to summarise shape.

Figures -> distributions/
  pdf_panels_d{diam}_{depth}.png   array (step) vs truth (filled) PDFs per variable,
                                   annotated JS / Wasserstein and their nulls.
  js_summary.png                   JS - JS_null vs diameter, one line per variable.
  moment_recovery.png              std ratio and skew, array vs truth, vs diameter.
  fit_distributions.png            skew-normal fits to w and w'T', array vs truth.

Usage:  python run_distributions.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import skewnorm

import common as C
import sd_plot as P

SUB = 'distributions'

# variables compared as spatial-sampling distributions (label, unit-scale)
VARS = [('T', 1.0, '$^\\circ$C'), ('S', 1.0, 'g kg$^{-1}$'),
        ('U', 1.0, 'm s$^{-1}$'), ('V', 1.0, 'm s$^{-1}$'),
        ('W', C.SEC_PER_DAY, 'm day$^{-1}$'),
        ('kappaT', 1.0, 'm$^2$ s$^{-1}$'), ('N2', 1.0, 's$^{-2}$')]
FLAGSHIP = 1.0
DEPTH = -30.0


def _pool(ds, var, space_dim, z):
    """Flatten a variable to a 1-D sample at depth z (or depth-avg if z is None)."""
    da = ds[var]
    da = da.mean('obs_depth') if z is None else da.interp(obs_depth=z)
    return np.asarray(da.values).ravel()


# --------------------------------------------------------------------------- fig 1
def fig_pdf_panels(out, diam=FLAGSHIP, z=DEPTH):
    arr = P.load_array(diam); cloud = P.load_cloud(diam)
    n_plat = arr.sizes['glider']
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, (var, scale, unit) in zip(axes.ravel(), VARS):
        o = _pool(arr, var, 'glider', z) * scale
        t = _pool(cloud, var, 'point', z) * scale
        lo = np.nanpercentile(np.concatenate([o, t]), 0.5)
        hi = np.nanpercentile(np.concatenate([o, t]), 99.5)
        bins = np.linspace(lo, hi, 40)
        ax.hist(t, bins=bins, density=True, color='0.6', alpha=0.55, label='truth')
        ax.hist(o, bins=bins, density=True, histtype='step', color='#08306b', lw=2.2,
                label='array')
        js, w = P.js_wass(o, t); jsn, wn = P.js_wass_null(o, t, n_plat)
        ax.set_title(f'{C.VAR_LABEL[var][0]}  ({unit})')
        P.tidy_x(ax, 4)
        ax.text(0.03, 0.96, f'JS {js:.2f} (null {jsn:.2f})\nW {w:.2g} (null {wn:.2g})',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(fc='white', ec='0.8', alpha=0.8))
    axes.ravel()[-1].axis('off')
    fig.legend(handles=[Line2D([0], [0], color='0.6', lw=8, alpha=0.55, label='model truth (hull)'),
                        Line2D([0], [0], color='#08306b', lw=2.2, label='array samples')],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    zlab = 'depth-avg' if z is None else f'{-z:g} m'
    fig.text(0.01, 0.5, f'density  ({diam:g}$^\\circ$ footprint, {zlab})', rotation=90,
             va='center', fontweight='bold')
    return P.finish(fig, f'{out}/pdf_panels_d{diam:g}_{"depthavg" if z is None else f"{-z:g}m"}.png')


# --------------------------------------------------------------------------- fig 2
def fig_js_summary(out, z=None):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.cm.tab10
    for vi, (var, scale, _) in enumerate(VARS):
        vals = []
        for d in C.DIAMETERS:
            arr = P.load_array(d); cloud = P.load_cloud(d)
            o = _pool(arr, var, 'glider', z) * scale
            t = _pool(cloud, var, 'point', z) * scale
            js, _ = P.js_wass(o, t); jsn, _ = P.js_wass_null(o, t, arr.sizes['glider'])
            vals.append(js - jsn)
        ax.plot(C.DIAMETERS, vals, '-o', color=cmap(vi), lw=2, label=C.VAR_LABEL[var][0])
    ax.axhline(0, color='0.5', lw=0.9)
    ax.set(xlabel='E-W diameter ($^\\circ$)',
           ylabel='JS $-$ JS$_{\\mathrm{null}}$  (depth-avg)')
    ax.legend(loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.16))
    return P.finish(fig, f'{out}/js_summary.png')


# --------------------------------------------------------------------------- fig 3
def fig_moment_recovery(out, z=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.tab10
    for vi, (var, scale, _) in enumerate(VARS):
        sr, ska, skt = [], [], []
        for d in C.DIAMETERS:
            arr = P.load_array(d); cloud = P.load_cloud(d)
            o = _pool(arr, var, 'glider', z) * scale
            t = _pool(cloud, var, 'point', z) * scale
            mo = P.moments(o); mt = P.moments(t)
            sr.append(mo[1] / mt[1] if mt[1] else np.nan)
            ska.append(mo[2]); skt.append(mt[2])
        axes[0].plot(C.DIAMETERS, sr, '-o', color=cmap(vi), label=C.VAR_LABEL[var][0])
        axes[1].plot(skt, ska, 'o', color=cmap(vi), ms=8)
    axes[0].axhline(1, color='0.5', lw=0.9)
    axes[0].set(xlabel='E-W diameter ($^\\circ$)', ylabel='std ratio  array / truth')
    lim = np.array(axes[1].get_xlim() + axes[1].get_ylim())
    m = [min(lim), max(lim)]
    axes[1].plot(m, m, color='0.5', lw=0.9)
    axes[1].set(xlabel='truth skewness', ylabel='array skewness', title='shape recovery')
    axes[0].legend(loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.16))
    return P.finish(fig, f'{out}/moment_recovery.png')


# --------------------------------------------------------------------------- fig 4
def _fit_samples(diam, z):
    """(array, truth) 1-D samples of w, w'T' and w'u' at depth z for one diameter."""
    import run_transport as T
    arr = P.load_array(diam); hull = P.load_hull(diam); cloud = P.load_cloud(diam)
    w = (np.asarray(arr['w_est_mid'].interp(obs_depth=z).values).ravel() * C.SEC_PER_DAY,
         _pool(cloud, 'W', 'point', z) * C.SEC_PER_DAY)
    wT = (np.asarray(T.array_vert_flux(arr, 'w_est_mid', 'T').interp(obs_depth=z).values).ravel() * C.HFLUX,
          np.asarray(hull['wT'].interp(obs_depth=z).values).ravel() * C.HFLUX)
    wU = (np.asarray(T.array_vert_flux(arr, 'w_est_mid', 'U').interp(obs_depth=z).values).ravel(),
          np.asarray(hull['wU'].interp(obs_depth=z).values).ravel())
    return {'w (m day$^{-1}$)': w, "$w'T'$ (W m$^{-2}$)": wT,
            "$w'u'$ (m$^2$ s$^{-2}$)": wU}


def _fit_samples_uv(diam, z):
    """(array, truth) 1-D samples of u', v' and the lateral fluxes u'v', u'T', v'T'."""
    import run_transport as T
    arr = P.load_array(diam); hull = P.load_hull(diam); cloud = P.load_cloud(diam)

    def anom(ds, var):                                  # pooled temporal fluctuation
        a = (ds[var] - ds[var].mean('time')).interp(obs_depth=z)
        return np.asarray(a.values).ravel()

    def latflux(a, b, key, scale=1.0):                  # array (platform a'b') vs hull truth
        ar = np.asarray(T.array_lat_flux(arr, a, b).interp(obs_depth=z).values).ravel()
        tr = np.asarray(hull[key].interp(obs_depth=z).values).ravel()
        return (ar * scale, tr * scale)

    return {"$u'$ (m s$^{-1}$)": (anom(arr, 'U'), anom(cloud, 'U')),
            "$v'$ (m s$^{-1}$)": (anom(arr, 'V'), anom(cloud, 'V')),
            "$u'v'$ (m$^2$ s$^{-2}$)": latflux('U', 'V', 'uv'),
            "$u'T'$ (W m$^{-2}$)": latflux('U', 'T', 'uT', C.HFLUX),
            "$v'T'$ (W m$^{-2}$)": latflux('V', 'T', 'vT', C.HFLUX)}


def _bimodal_fit(data, xx, n_iter=300, seed=0):
    """2-component Gaussian mixture via a compact 1-D EM (no sklearn dependency).

    Returns (mixture pdf on xx, peak separation |mu2 - mu1|).  Used for the u'/v'
    rows, whose oscillatory (TIW/eddy) dynamics give a genuinely bimodal PDF that a
    single skew-normal cannot represent.
    """
    from scipy.stats import norm
    x = np.asarray(data)[np.isfinite(data)]
    mu = np.percentile(x, [25.0, 75.0]).astype(float)   # init peaks either side of centre
    var = np.full(2, x.var() / 2.0 + 1e-12)
    w = np.array([0.5, 0.5])
    for _ in range(n_iter):
        p = np.stack([w[k] * norm.pdf(x, mu[k], np.sqrt(var[k])) for k in range(2)])
        r = p / (p.sum(0) + 1e-300)                      # responsibilities (2, N)
        nk = r.sum(1) + 1e-12
        w = nk / nk.sum()
        mu = (r * x).sum(1) / nk
        var = (r * (x - mu[:, None]) ** 2).sum(1) / nk + 1e-12
    pdf = sum(w[k] * norm.pdf(xx, mu[k], np.sqrt(var[k])) for k in range(2))
    return pdf, abs(mu[1] - mu[0])


def _fit_grid(out, per_diam, fname, bimodal_labels=frozenset()):
    """Render a distribution-fit grid: rows = quantities, cols = diameters.

    Rows in `bimodal_labels` get a 2-Gaussian mixture (report peak separation);
    all others get a skew-normal (report the shape parameter a).
    """
    labels = list(per_diam[C.DIAMETERS[0]])
    xlims = {}                                          # shared per-row x-limits
    for lab in labels:
        allv = np.concatenate([np.concatenate(per_diam[d][lab]) for d in C.DIAMETERS])
        allv = allv[np.isfinite(allv)]
        xlims[lab] = np.percentile(allv, [0.5, 99.5])
    nc = len(C.DIAMETERS); nr = len(labels)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.4 * nr), sharey='row',
                             squeeze=False)
    for ri, lab in enumerate(labels):
        lo, hi = xlims[lab]
        bins = np.linspace(lo, hi, 45); xx = np.linspace(lo, hi, 200)
        bimodal = lab in bimodal_labels
        for ci, d in enumerate(C.DIAMETERS):
            ax = axes[ri, ci]
            o, t = per_diam[d][lab]
            for data, color, name in [(t, '0.5', 'truth'), (o, '#08306b', 'array')]:
                data = data[np.isfinite(data)]
                kw = dict(alpha=0.5) if name == 'truth' else dict(histtype='step', lw=2.2)
                ax.hist(data, bins=bins, density=True, color=color, **kw)
                if bimodal:                              # 2-Gaussian mixture -> peak separation
                    pdf, sep = _bimodal_fit(data, xx)
                    note = f'{name}: $\\Delta$peak={sep:.2g}'
                else:                                    # skew-normal -> shape parameter a
                    a, loc, sc = skewnorm.fit(data)
                    pdf = skewnorm.pdf(xx, a, loc, sc)
                    note = f'{name}: skew a={a:.2f}'
                ax.plot(xx, pdf, color=color, lw=1.6, ls='--')
                # right-aligned in the (typically emptier) upper-right corner with a light
                # background so it never sits on the data
                ax.text(0.97, 0.97 - (0.14 if name == 'array' else 0), note,
                        transform=ax.transAxes, va='top', ha='right', color=color,
                        fontsize=8.5,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.65))
            ax.set_xlim(lo, hi)
            P.tidy_x(ax, 4)
            if ri == 0:
                ax.set_title(f'{d:g}$^\\circ$ footprint')
            if ri == nr - 1:
                ax.set_xlabel(lab)
            if ci == 0:
                ax.set_ylabel(f'{lab}\ndensity')
    # pack the axes up to the figure top, reserving only a thin strip for the legend so
    # tall (many-row) grids don't leave a big empty band under the legend
    fig_h = 3.4 * nr
    top = 1.0 - 0.5 / fig_h
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.legend(handles=[Line2D([0], [0], color='0.5', lw=8, alpha=0.5, label='truth hist'),
                        Line2D([0], [0], color='#08306b', lw=2.2, label='array hist'),
                        Line2D([0], [0], color='0.3', lw=1.6, ls='--', label='distribution fit')],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    return P.finish(fig, f'{out}/{fname}')


def fig_fit(out, z=DEPTH):
    """Skew-normal fits to w, w'T', w'u' (rows), one column per diameter (30 m)."""
    per_diam = {d: _fit_samples(d, z) for d in C.DIAMETERS}
    return _fit_grid(out, per_diam, 'fit_distributions.png')


def fig_fit_uv(out, z=DEPTH):
    """Fits to u', v', u'v', u'T', v'T' (rows), one column per diameter.

    The oscillatory u'/v' rows get a bimodal (2-Gaussian) fit; the skewed flux rows
    keep the skew-normal.
    """
    per_diam = {d: _fit_samples_uv(d, z) for d in C.DIAMETERS}
    bimodal = {"$u'$ (m s$^{-1}$)", "$v'$ (m s$^{-1}$)"}
    return _fit_grid(out, per_diam, 'fit_distributions_uv.png', bimodal_labels=bimodal)


def main():
    out = P.outdir(SUB)
    print(fig_pdf_panels(out, FLAGSHIP, DEPTH))
    print(fig_pdf_panels(out, FLAGSHIP, None))
    print(fig_js_summary(out))
    print(fig_moment_recovery(out))
    print(fig_fit(out))
    print(fig_fit_uv(out))


if __name__ == '__main__':
    main()
