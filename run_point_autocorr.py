"""
Spatial autocorrelation of the currents anchored at a single point (default 0°N, 140°W
= 220°E), three complementary views per component (U, V, W):

  * domain_point_autocorr_{tag}      -- MEAN-field autocorrelation vs separation, 1-D
    zonal (solid) / meridional (dashed) slices (mean_field_point_autocorr). Curve shape,
    no 1/e marker; shows how far the mean current stays coherent at the anchor.
  * domain_point_meanfield_corrmap_{tag} -- MEAN-field 2-D autocorrelation pattern
    r(dlon,dlat) around the anchor (mean_field_point_corr_map): the full anisotropy/tilt
    of the mean current's coherent footprint, overlay-able with a candidate array.
  * domain_point_corr_map_{tag}      -- TEMPORAL one-point correlation map: the anchor's
    velocity time series correlated with every grid point over the 3-month record
    (point_corr_map). Shows what an array would SEE over the run -- the eddy/TIW footprint
    (a different, generally shorter L than the mean-field one).

The temporal map needs the full (time, y, x) field, so this script READS THE MODEL once
(depth-mean series for U, V, W), unlike the cache-only mean-field diagnostics. It reuses
RUN_DIR / ITERS / PERIODS from run_domain_maps.

Writes into domain/full_domain/velocities/.

Usage:
    python run_point_autocorr.py [depths_csv] [periods_csv]
    e.g. python run_point_autocorr.py 70,120,250 3mo,1mo
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import osse_tools as ot
from plotting_tools import plot_point_autocorr, plot_anchor_corr_map
from run_domain_maps import VARS, LABEL, PERIODS, OUTDIR, RUN_DIR, ITERS

ANCHOR_LAT, ANCHOR_LON = 0.0, 220.0      # 0°N, 140°W
ANCHOR_TAG = '0N140W'
MAP_HALF_WIN_DEG = 6.0                    # ±deg sample box for the mean-field 2-D pattern
MAP_MAX_SEP_DEG = 3.0                     # ±deg cropped display extent (box stays 6° for stats)


def _lonlat(da):
    """1-D lon/lat coordinate arrays of a (y, x) field on its own stagger."""
    xdim, ydim = da.dims[-1], da.dims[-2]
    return da[xdim].values, da[ydim].values


def main(depths, periods):
    outdir = os.path.join(OUTDIR, 'full_domain', 'velocities')
    os.makedirs(outdir, exist_ok=True)

    # one model read: the temporal one-point map needs the time series (the mean-field
    # views are derived from the time-mean of the same series, so no cache is needed).
    ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice('2012-10-11', None))
    series = {v: ot.depth_mean_series(ds, v, depths) for v in VARS}
    sys.stderr.write('  loaded depth-mean series U, V, W\n')

    for pkey in periods:
        tsel, suf, plabel = PERIODS[pkey]
        for d in depths:
            S = {v: series[v][d].sel(time=tsel) for v in VARS}
            means = {v: S[v].mean('time') for v in VARS}

            curves, mf_map, t_map = [], [], []
            lon0 = lat0 = None
            for v in VARS:
                m = means[v]
                lon, lat = _lonlat(m)
                # same ±MAP_HALF_WIN_DEG box + FFT autocovariance as the 2-D map below,
                # so the curve's and map's zero crossings coincide by construction
                curves.append((LABEL[v], ot.mean_field_point_autocorr(
                    m.values, lon, lat, ANCHOR_LON, ANCHOR_LAT,
                    half_window_deg=MAP_HALF_WIN_DEG, max_sep_deg=MAP_HALF_WIN_DEG)))
                r, mlon, mlat, lon0, lat0 = ot.mean_field_point_corr_map(
                    m.values, lon, lat, ANCHOR_LON, ANCHOR_LAT,
                    half_window_deg=MAP_HALF_WIN_DEG, max_sep_deg=MAP_MAX_SEP_DEG)
                mf_map.append((r, mlon, mlat, LABEL[v]))
                rt, _, _ = ot.point_corr_map(
                    S[v].values, lon, lat, ANCHOR_LON, ANCHOR_LAT)
                t_map.append((rt, lon, lat, LABEL[v]))

            wdeg = 360 - lon0
            base = f'TPOSE24 {plabel}, 0–{d} m'
            loc = f'{lat0:.2f}°N, {wdeg:.2f}°W'

            plot_point_autocorr(
                curves,
                suptitle=(f'{base} — mean-field spatial autocorrelation at {loc} '
                          f'(±{MAP_HALF_WIN_DEG:g}° sample box)'),
                fname=os.path.join(
                    outdir, f'domain_point_autocorr_{ANCHOR_TAG}_{d}m{suf}.png'))
            plot_anchor_corr_map(
                mf_map, (lon0, lat0), ref_levels=(0.5,),
                suptitle=f'{base} — MEAN-field 2-D autocorrelation footprint at {loc}',
                fname=os.path.join(
                    outdir, f'domain_point_meanfield_corrmap_{ANCHOR_TAG}_{d}m{suf}.png'))
            # 0.5 & 0 contours on U, V; drop the 0 contour on W (clutter over its
            # small-scale temporally-incoherent field)
            t_ref = [(0.5,) if v == 'WVEL' else (0.5, 0.0) for v in VARS]
            plot_anchor_corr_map(
                t_map, (lon0, lat0),
                suptitle=(f'{base} — temporal one-point correlation over the 3-month '
                          f'record at {loc}'),
                ref_levels=t_ref,
                fname=os.path.join(
                    outdir, f'domain_point_corr_map_{ANCHOR_TAG}_{d}m{suf}.png'))
            sys.stderr.write(f'PLOTTED {pkey} {d}m\n')


if __name__ == '__main__':
    depths = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [70, 120, 250]
    periods = sys.argv[2].split(',') if len(sys.argv) > 2 else ['3mo', '1mo']
    main(depths, periods)
