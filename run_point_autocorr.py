"""
Raw spatial autocorrelation function of the currents anchored at a single point
(default 0°N, 140°W = 220°E) -- one panel per component, zonal (solid) and meridional
(dashed) correlation vs separation.

Unlike domain_current_autocorr_curves (band-averaged, with 1/e and 0.7 reference lines
and the array-span window shaded), this shows NO chosen cutoff scale: just the curve, so
the decay and the negative lobe (the dominant eddy/wave wavelength) can be read directly.

Uses the same temporal-anomaly Pearson correlation as the decorrelation family, on the
depth-mean time series, but evaluated only along the anchor's row/column -- so once the
model is loaded the curves are essentially free.

Writes domain/full_domain/velocities/domain_point_autocorr_{tag}_{d}m{suf}.png

Split into two phases like run_domain_maps, because the model load dominates (~11 min)
while the curves themselves are instant: COMPUTE pickles the tiny per-config curve
bundle to CACHE_DIR; PLOT re-renders from cache in seconds. After any figure-STYLE
tweak just run `plot` -- no model reload.

Usage:
    python run_point_autocorr.py [depths_csv] [periods_csv] [mode]
    mode in {all (default), compute, plot}
    e.g. python run_point_autocorr.py 70,120,250 3mo,1mo
         python run_point_autocorr.py 70,120,250 3mo,1mo plot   # restyle, fast
"""

import os
import sys
import time
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')

import cmocean.cm as cmo

import osse_tools as ot
from plotting_tools import plot_point_autocorr, plot_domain_grid
from run_domain_maps import (RUN_DIR, ITERS, VARS, LABEL, PERIODS, OUTDIR,
                             MAX_LAG_KM, CACHE_DIR)

ANCHOR_LAT, ANCHOR_LON = 0.0, 220.0      # 0°N, 140°W
ANCHOR_TAG = '0N140W'
WIN_LON, WIN_LAT = 6.0, 5.0              # 2-D corr-map window (deg) about the anchor


def _lonlat(da):
    """1-D lon/lat coordinate arrays of a (time, y, x) field on its own stagger."""
    xdim, ydim = da.dims[-1], da.dims[-2]
    return da[xdim].values, da[ydim].values


def _cache_path(d, pkey):
    return os.path.join(CACHE_DIR, f'point_autocorr_{ANCHOR_TAG}_{d}m_{pkey}.pkl')


def _window(r2d, lon, lat):
    """Crop a correlation map to the window about the anchor, for legibility."""
    jx = np.abs(lon - ANCHOR_LON) <= WIN_LON
    iy = np.abs(lat - ANCHOR_LAT) <= WIN_LAT
    return r2d[np.ix_(iy, jx)], lon[jx], lat[iy]


def main(depths, periods, mode='all'):
    outdir = os.path.join(OUTDIR, 'full_domain', 'velocities')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if mode in ('all', 'compute'):
        ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice('2012-10-11', None))
        t0 = time.time()
        series = {}
        for v in VARS:                    # one read of the 0-250 m column per var
            series[v] = ot.depth_mean_series(ds, v, depths)
            sys.stderr.write(f'  loaded depth-mean series {LABEL[v]}  ({time.time()-t0:.0f}s)\n')

        for pkey in periods:
            tsel = PERIODS[pkey][0]
            for d in depths:
                curves, maps = [], []
                for v in VARS:
                    S = series[v][d].sel(time=tsel)
                    lon, lat = _lonlat(S)
                    pac = ot.point_autocorr(S.values, lon, lat, ANCHOR_LON, ANCHOR_LAT,
                                            max_lag_km=MAX_LAG_KM)
                    curves.append((LABEL[v], pac))
                    r2d, _, _ = ot.point_corr_map(S.values, lon, lat,
                                                  ANCHOR_LON, ANCHOR_LAT)
                    maps.append((LABEL[v],) + _window(r2d, lon, lat))
                with open(_cache_path(d, pkey), 'wb') as f:
                    pickle.dump({'curves': curves, 'maps': maps}, f)
                sys.stderr.write(f'COMPUTED {pkey} {d}m\n')

    if mode in ('all', 'plot'):
        for pkey in periods:
            _, suf, plabel = PERIODS[pkey]
            for d in depths:
                with open(_cache_path(d, pkey), 'rb') as f:
                    cache = pickle.load(f)
                curves, maps = cache['curves'], cache['maps']
                lat0, lon0 = curves[0][1]['lat0'], curves[0][1]['lon0']
                head = (f'TPOSE24 {plabel}, 0–{d} m — spatial autocorrelation at '
                        f'{lat0:.2f}°N, {360-lon0:.2f}°W')

                plot_point_autocorr(
                    curves, suptitle=head,
                    fname=os.path.join(
                        outdir, f'domain_point_autocorr_{ANCHOR_TAG}_{d}m{suf}.png'))

                panels = [(r, lo, la, lab) for lab, r, lo, la in maps]
                plot_domain_grid(
                    panels, cbar_label='autocorrelation', cmap=cmo.balance,
                    vlim=(-1, 1), ncols=len(VARS), diverging=True,
                    suptitle=head + '  (2-D)',
                    fname=os.path.join(
                        outdir, f'domain_point_corr_map_{ANCHOR_TAG}_{d}m{suf}.png'))
                sys.stderr.write(f'PLOTTED {pkey} {d}m (curves + 2-D map)\n')


if __name__ == '__main__':
    depths = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [70, 120, 250]
    periods = sys.argv[2].split(',') if len(sys.argv) > 2 else ['3mo', '1mo']
    mode = sys.argv[3] if len(sys.argv) > 3 else 'all'
    main(depths, periods, mode)
