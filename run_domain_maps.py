"""
Domain maps of current statistics: spatial decorrelation scales of the currents,
horizontal gradients of the mean currents, and decorrelation scales of those
gradients. Companion to demo_domain (mean U/V/W); saves the same full-domain +
equatorial-crop pair for every figure into domain/.

For each depth cutoff (0-70/120/250 m) and averaging window (3-month, 1-month) it
writes four figure pairs:
    current_decorr    isotropic 1/e decorrelation length of U, V, W          (km)
    gradient_mag      |grad| of the time-mean U, V, W                        (1/s)
    gradient_shear    signed dU/dx, dU/dy, dV/dx, dV/dy, dW/dx, dW/dy        (1/s)
    gradient_decorr   decorrelation length of the |grad U|, |grad V|, |grad W| fields (km)

The depth-mean *time series* for U, V, W are read once (all three cutoffs share a
single pass over the 0-250 m column); the 1-month window is a time-slice of the
3-month series, and every map is computed on the full domain and then cropped for
the zoom view. Usage:
    python run_domain_maps.py [depths_csv] [periods_csv]
    e.g. python run_domain_maps.py 70,120,250 3mo,1mo   (default; full sweep)
         python run_domain_maps.py 70 1mo               (quick single config)
"""

import os
import sys
import time

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import cmocean.cm as cmo

import osse_tools as ot
from plotting_tools import plot_domain_grid

RUN_DIR = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS = list(range(36, 26173, 36))
OUTDIR = os.path.join(os.path.dirname(__file__), 'domain')

VARS = ('UVEL', 'VVEL', 'WVEL')          # -> U, V, W
LABEL = {'UVEL': 'U', 'VVEL': 'V', 'WVEL': 'W'}
CROP_LON = slice(217, 223)
CROP_LAT = slice(-3, 3)
MAX_LAG_KM = 600.0

# period key -> (time slice on the 3-month series, filename suffix, title label)
PERIODS = {
    '3mo': (slice('2012-10-11', None), '', 'Oct 11–Dec 2012'),
    '1mo': (slice('2012-10-11', '2012-11-11'), '_1mo', 'Oct 11–Nov 11 2012'),
}


def _xy(da):
    """(x-dim name, y-dim name) of a 2-D/3-D velocity DataArray on its own grid."""
    ydim, xdim = da.dims[-2], da.dims[-1]
    return xdim, ydim


def _as_da(arr2d, ref):
    """Wrap a (y, x) numpy result on the same horizontal coords as `ref`."""
    xdim, ydim = _xy(ref)
    return xr.DataArray(arr2d, dims=(ydim, xdim),
                        coords={ydim: ref[ydim], xdim: ref[xdim]})


def _panels(das, titles):
    """Build (values, lon, lat, title) panel tuples for full and cropped views."""
    full, crop = [], []
    for da, t in zip(das, titles):
        xdim, ydim = _xy(da)
        full.append((da.values, da[xdim].values, da[ydim].values, t))
        c = da.sel({xdim: CROP_LON, ydim: CROP_LAT})
        crop.append((c.values, c[xdim].values, c[ydim].values, t))
    return full, crop


def _save_pair(base, panels_full, panels_crop, **kw):
    """Save the full-domain and crop_ figure for one diagnostic."""
    plot_domain_grid(panels_full, fname=os.path.join(OUTDIR, f'{base}.png'), **kw)
    plot_domain_grid(panels_crop, fname=os.path.join(OUTDIR, f'crop_{base}.png'), **kw)


def main(depths, periods):
    os.makedirs(OUTDIR, exist_ok=True)
    ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice('2012-10-11', None))

    # Depth-mean time series for U, V, W at every cutoff (one read of the column).
    t0 = time.time()
    series = {}
    for v in VARS:
        series[v] = ot.depth_mean_series(ds, v, depths)
        sys.stderr.write(f"  loaded depth-mean series {LABEL[v]}  "
                         f"({time.time() - t0:.0f}s)\n")

    for pkey in periods:
        tsel, suf, plabel = PERIODS[pkey]
        for d in depths:
            t1 = time.time()
            # depth-mean series for this depth/window, per variable
            S = {v: series[v][d].sel(time=tsel) for v in VARS}
            head = f'TPOSE24 {plabel}, 0–{d} m'

            # --- 1. spatial decorrelation of the currents --------------------
            Ld = []
            for v in VARS:
                da = S[v]
                xdim, ydim = _xy(da)
                L = ot.decorrelation_scale(da.values, da[xdim].values,
                                           da[ydim].values, max_lag_km=MAX_LAG_KM)[0]
                Ld.append(_as_da(L, da))
            titles = [f'{LABEL[v]} decorrelation' for v in VARS]
            full, crop = _panels(Ld, titles)
            _save_pair(f'domain_current_decorr_{d}m{suf}', full, crop,
                       cbar_label='length scale (km)', cmap=cmo.thermal,
                       suptitle=f'{head} — current decorrelation scale (1/e)')

            # --- 2. gradient magnitude of the mean currents ------------------
            means = {v: S[v].mean('time') for v in VARS}
            Gmag = []
            for v in VARS:
                m = means[v]
                xdim, ydim = _xy(m)
                g = ot.gradient_magnitude(m.values, m[xdim].values, m[ydim].values)
                Gmag.append(_as_da(g, m))
            titles = [f'|∇{LABEL[v]}|' for v in VARS]
            full, crop = _panels(Gmag, titles)
            _save_pair(f'domain_gradient_mag_{d}m{suf}', full, crop,
                       cbar_label='s$^{-1}$', cmap=cmo.amp,
                       suptitle=f'{head} — mean-current gradient magnitude')

            # --- 3. signed shear components of the mean currents -------------
            Shear, stitles = [], []
            for v in VARS:
                m = means[v]
                xdim, ydim = _xy(m)
                fx, fy = ot.gradient_components(m.values, m[xdim].values, m[ydim].values)
                Shear += [_as_da(fx, m), _as_da(fy, m)]
                stitles += [f'∂{LABEL[v]}/∂x', f'∂{LABEL[v]}/∂y']
            full, crop = _panels(Shear, stitles)
            _save_pair(f'domain_gradient_shear_{d}m{suf}', full, crop,
                       cbar_label='s$^{-1}$', cmap=cmo.balance, diverging=True, ncols=2,
                       suptitle=f'{head} — mean-current shear components')

            # --- 4. spatial decorrelation of the gradient fields -------------
            Lg = []
            for v in VARS:
                da = S[v]
                xdim, ydim = _xy(da)
                gmag_series = ot.gradient_magnitude(da.values, da[xdim].values,
                                                    da[ydim].values)
                L = ot.decorrelation_scale(gmag_series, da[xdim].values,
                                           da[ydim].values, max_lag_km=MAX_LAG_KM)[0]
                Lg.append(_as_da(L, da))
            titles = [f'|∇{LABEL[v]}| decorrelation' for v in VARS]
            full, crop = _panels(Lg, titles)
            _save_pair(f'domain_gradient_decorr_{d}m{suf}', full, crop,
                       cbar_label='length scale (km)', cmap=cmo.thermal,
                       suptitle=f'{head} — gradient decorrelation scale (1/e)')

            sys.stderr.write(f"DONE {pkey} {d}m  ({time.time() - t1:.0f}s)\n")


if __name__ == '__main__':
    depths = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [70, 120, 250]
    periods = sys.argv[2].split(',') if len(sys.argv) > 2 else ['3mo', '1mo']
    main(depths, periods)
