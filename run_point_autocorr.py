"""
Mean-field spatial autocorrelation curve of the currents anchored at a single point
(default 0°N, 140°W = 220°E) -- one panel per component, zonal (solid) and meridional
(dashed) correlation vs separation, with the 1/e crossing marked.

This is the single-point analogue of the domain_current_decorr maps: a windowed spatial
autocorrelation of the TIME-MEAN field (osse_tools.mean_field_point_autocorr), so the
decorrelation scale of the mean current at the anchor can be read directly. It replaces
the earlier eddy-field version (temporal-anomaly correlation of the anchor's time series
and its 2-D corr map), which is superseded by the mean-field decorrelation maps.

Reads the time-mean fields straight from the run_domain_maps cache -- no model access --
so it is essentially instant. Run run_domain_maps.py first to build that cache.

Writes domain/full_domain/velocities/domain_point_autocorr_{tag}_{d}m{suf}.png

Usage:
    python run_point_autocorr.py [depths_csv] [periods_csv]
    e.g. python run_point_autocorr.py 70,120,250 3mo,1mo
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')

import osse_tools as ot
from plotting_tools import plot_point_autocorr
from run_domain_maps import (VARS, LABEL, PERIODS, OUTDIR, CACHE_DIR,
                             DECORR_HALF_WIN_DEG)

ANCHOR_LAT, ANCHOR_LON = 0.0, 220.0      # 0°N, 140°W
ANCHOR_TAG = '0N140W'


def _lonlat(da):
    """1-D lon/lat coordinate arrays of a (y, x) mean field on its own stagger."""
    xdim, ydim = da.dims[-1], da.dims[-2]
    return da[xdim].values, da[ydim].values


def _cache_path(d, pkey):                # shared run_domain_maps cache
    return os.path.join(CACHE_DIR, f'domain_maps_{d}m_{pkey}.pkl')


def main(depths, periods):
    outdir = os.path.join(OUTDIR, 'full_domain', 'velocities')
    os.makedirs(outdir, exist_ok=True)

    for pkey in periods:
        _, suf, plabel = PERIODS[pkey]
        for d in depths:
            with open(_cache_path(d, pkey), 'rb') as f:
                means = pickle.load(f)['means']
            curves = []
            for v in VARS:
                m = means[v]
                lon, lat = _lonlat(m)
                pac = ot.mean_field_point_autocorr(
                    m.values, lon, lat, ANCHOR_LON, ANCHOR_LAT,
                    half_window_deg=DECORR_HALF_WIN_DEG)
                curves.append((LABEL[v], pac))
            lat0, lon0 = curves[0][1]['lat0'], curves[0][1]['lon0']
            head = (f'TPOSE24 {plabel}, 0–{d} m — mean-field spatial autocorrelation '
                    f'at {lat0:.2f}°N, {360 - lon0:.2f}°W '
                    f'({DECORR_HALF_WIN_DEG:g}° window)')
            plot_point_autocorr(
                curves, suptitle=head, thresh=1.0 / np.e,
                fname=os.path.join(
                    outdir, f'domain_point_autocorr_{ANCHOR_TAG}_{d}m{suf}.png'))
            sys.stderr.write(f'PLOTTED {pkey} {d}m\n')


if __name__ == '__main__':
    depths = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [70, 120, 250]
    periods = sys.argv[2].split(',') if len(sys.argv) > 2 else ['3mo', '1mo']
    main(depths, periods)
