"""
Temporal one-point correlation diagnostics of the currents (U, V, W), anchored at
0°N,140°W (=220°E) and its neighbors. Every map is the Pearson correlation of an anchor
point's velocity time series with every other point over the 3-month record (what a fixed
array would SEE) -- unlike run_domain_maps, nothing here is a time mean. Every figure is
rendered TWICE: the full model domain to domain/full_domain/autocorr_point/ and the ±5°
crop of the anchor to domain/crop_140/autocorr_point/.

Figure set (VARS = U, V, W panels; correlation maps use 101 filled levels, cmo.balance,
-1..1, star at the anchor, r=0.5/0 reference contours):

  (1)(2) point_corr_map_{lat}_{depth}   -- horizontal temporal one-point map on three
      depth treatments (0-70 m depth-avg, the 25 m level, the 75 m level) x five anchor
      latitudes along 140°W (-2,-1,0,1,2 °N).
  (3)  efold_timescale_{depth}          -- map of the e-folding temporal decorrelation
      scale (days) at every grid point, per depth treatment.
  (4)  hovmoller_{depth}                -- longitude-time hovmoller of the fluctuations
      along 0°N with the spatial x temporal decorrelation box drawn at the anchor.
  (5)  trans_corr_dlat_{lat}_{d}m / trans_corr_dlon_{lat}_{d}m -- depth-section one-point
      maps: correlation to the anchor at (lat,140°W, 25|75 m) across a depth-latitude
      plane (140°W) and a depth-longitude plane (fixed lat), per anchor latitude.
  (6)  corr_cuts_{lat}_{d}m             -- four-panel autocorrelation cuts (vs lon, lat,
      depth, time) through the maps at each anchor, U/V/W overlaid.

Two phases (mode ∈ {all, compute, plot}): COMPUTE reads the model once and pickles the
full-domain result arrays to CACHE; PLOT renders full + crop from the cache with no model
access. Reuses RUN_DIR / ITERS / PERIODS from run_domain_maps.

An optional averaging arg (avg ∈ {native, 5day}) controls the sampling used for the
correlations. 'native' (default) uses the raw 3-hourly diag_state records and writes to
.../autocorr_point/ (the original figures). '5day' first resamples the record to
non-overlapping 5-day means and writes to .../autocorr_point/5_day/ with its own cache,
so the native outputs are left untouched -- letting you compare decorrelation structure
across averaging intervals.

Usage:
    python run_point_autocorr.py [mode] [avg]
    e.g. python run_point_autocorr.py all 5day     # 5-day means -> autocorr_point/5_day/
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')

import osse_tools as ot
from plotting_tools import (plot_anchor_corr_map, plot_efolding_map,
                            plot_hovmoller, plot_corr_cuts)
from run_domain_maps import VARS, LABEL, PERIODS, OUTDIR, RUN_DIR, ITERS, CACHE_DIR

ANCHOR_LON = 220.0                          # 140°W
ANCHOR_LATS = [-2.0, -1.0, 0.0, 1.0, 2.0]   # anchors along 140°W
ANCHOR_DEPTHS = [25, 75]                     # anchor depths (m) for the depth sections
# depth treatments for the horizontal maps: key -> (kind, depth_m, label)
DEPTH_OPTS = [('depthavg', 70, '0–70 m'), ('25m', 25, '25 m'), ('75m', 75, '75 m')]
HALF_DEG = 5.0                              # crop ±5° lon/lat around the anchor
PLOT_MAXDEPTH = 200                         # depth-section display depth (m)
READ_MAXDEPTH = 250                         # water column read for the sections
PERIOD = '3mo'                              # 3-month window only
UVW_COLORS = ['#1b6ca8', '#c0392b', '#2e7d32']

# Time-averaging variants. The native run uses the raw 3-hourly diag_state records;
# the '5day' variant resamples the record to non-overlapping 5-day means BEFORE any
# correlation is computed (so dt_days, the ACFs, e-folding scales, one-point maps and
# sections all reflect the coarser sampling) to test how much the decorrelation
# structure depends on the averaging interval.  tag -> (pandas resample rule, subdir).
AVG_OPTS = {
    'native': (None, 'autocorr_point'),
    '5day':   ('5D', os.path.join('autocorr_point', '5_day')),
}
RESAMPLE = None                             # pandas resample rule or None (set in main)
SUBDIR = 'autocorr_point'                   # output subfolder under each domain view
CACHE = os.path.join(CACHE_DIR, 'point_autocorr_crop140.pkl')


# --------------------------------------------------------------------------- helpers
def _lat_tag(lat):
    if lat == 0:
        return 'eq'
    return f'{abs(lat):g}{"N" if lat > 0 else "S"}'


def _loc(lat):
    return 'equator, 140°W' if lat == 0 else f'{abs(lat):g}°{"N" if lat > 0 else "S"}, 140°W'


def _lonlat(da):
    """1-D lon/lat coordinate arrays of a (…, y, x) field on its own stagger."""
    return da[da.dims[-1]].values, da[da.dims[-2]].values


def _crop2d(r, xx, yy, xlim, ylim):
    """Crop a 2-D field + its 1-D axes to [xlim]/[ylim]."""
    xm = (xx >= xlim[0]) & (xx <= xlim[1])
    ym = (yy >= ylim[0]) & (yy <= ylim[1])
    return r[np.ix_(ym, xm)], xx[xm], yy[ym]


def _maybe_crop(r, xx, yy, box):
    """Crop (r, xx, yy) to box=(xlim, ylim), or pass through unchanged if box is None."""
    return (r, xx, yy) if box is None else _crop2d(r, xx, yy, *box)


def _clip1d(x, r, lim):
    """Restrict a 1-D curve (x, r) to lim=(lo, hi), or pass through if lim is None."""
    if lim is None:
        return x, r
    m = (x >= lim[0]) & (x <= lim[1])
    return x[m], r[m]


def _crop_trans(r, coord, z, lim):
    """Crop a depth-section (z, coord) map along its horizontal coord only."""
    if lim is None:
        return r, coord, z
    m = (coord >= lim[0]) & (coord <= lim[1])
    return r[:, m], coord[m], z


# --------------------------------------------------------------------------- compute
def compute():
    tsel = PERIODS[PERIOD][0]
    ds = ot.load_model(RUN_DIR, ITERS).sel(time=tsel)
    if RESAMPLE is not None:
        # non-overlapping time means (e.g. 5-day) before any correlation is computed
        ds = ds.resample(time=RESAMPLE).mean()
        sys.stderr.write(f'  resampled to {RESAMPLE} means\n')
    times = ds.time.values
    t_days = (times - times[0]) / np.timedelta64(1, 'D')
    dt_days = float(np.mean(np.diff(t_days)))
    sys.stderr.write(f'  loaded model: {len(times)} records, dt={dt_days:.3f} d\n')

    R = {'anchor_lon': ANCHOR_LON, 'anchor_lats': ANCHOR_LATS, 'anchor_depths': ANCHOR_DEPTHS,
         'depth_opts': DEPTH_OPTS, 'dt_days': dt_days, 'horiz': {}, 'trans': {}}

    # Full-domain arrays are stored; plot() renders the full extent AND the ±5° crop.
    # ---- horizontal one-point maps, e-folding maps, hovmollers, ACFs ----
    for dkey, dval, dlabel in DEPTH_OPTS:
        H = {'label': dlabel, 'cormap': {}, 'efold': {}, 'hov': {}, 'acf': {}}
        for v in VARS:
            if dkey == 'depthavg':
                da = ot.depth_mean_series(ds, v, [dval])[dval]
            else:
                da = ot.depth_level_series(ds, v, [dval])[dval]
            lon, lat = _lonlat(da)
            fld = np.asarray(da.values)                       # (time, y, x)

            # e-folding temporal decorrelation-scale map (full domain)
            tau = ot.efolding_timescale_map(fld, dt_days)
            H['efold'][v] = (tau, lon, lat)

            # one-point correlation maps at each anchor latitude (full domain)
            for alat in ANCHOR_LATS:
                r, lon0, lat0 = ot.point_corr_map(fld, lon, lat, ANCHOR_LON, alat)
                H['cormap'][(alat, v)] = (r, lon, lat)
                ix = int(np.argmin(np.abs(lon - ANCHOR_LON)))
                iy = int(np.argmin(np.abs(lat - alat)))
                H['acf'][(alat, v)] = ot.point_temporal_acf(fld[:, iy, ix], dt_days)

            # hovmoller of fluctuations along 0°N + its decorrelation box (full lon)
            iy0 = int(np.argmin(np.abs(lat - 0.0)))
            row = fld[:, iy0, :]
            anom = row - np.nanmean(row, axis=0)
            r0, lc0, la0 = H['cormap'][(0.0, v)]
            iyc = int(np.argmin(np.abs(la0 - 0.0)))
            Lx = ot.first_efold(lc0, r0[iyc], ANCHOR_LON)
            lags, racf = H['acf'][(0.0, v)]
            tau_t = ot.first_efold(lags, racf, 0.0)
            H['hov'][v] = (anom, lon, t_days, Lx, tau_t)
            sys.stderr.write(f'  computed horizontal {dkey} {LABEL[v]}\n')
        R['horiz'][dkey] = H

    # ---- depth-section one-point maps (full horizontal extent, depth to PLOT_MAXDEPTH) ----
    R['trans'] = {'dlat': {}, 'dlon': {}}
    for v in VARS:
        # depth-latitude plane at 140°W (one plane, anchored at each lat x depth)
        arr, z, yv, _ = ot.transect_plane_series(ds, v, READ_MAXDEPTH, lon=ANCHOR_LON)
        zk = z >= -PLOT_MAXDEPTH
        for alat in ANCHOR_LATS:
            for ad in ANCHOR_DEPTHS:
                r, y0, z0 = ot.point_corr_map(arr, yv, z, alat, -float(ad))
                R['trans']['dlat'][(alat, ad, v)] = (r[zk], yv, z[zk], y0, z0)
        del arr
        # depth-longitude plane at each anchor latitude
        for alat in ANCHOR_LATS:
            arr, z, xv, _ = ot.transect_plane_series(ds, v, READ_MAXDEPTH, lat=alat)
            zk = z >= -PLOT_MAXDEPTH
            for ad in ANCHOR_DEPTHS:
                r, x0, z0 = ot.point_corr_map(arr, xv, z, ANCHOR_LON, -float(ad))
                R['trans']['dlon'][(alat, ad, v)] = (r[zk], xv, z[zk], x0, z0)
            del arr
        sys.stderr.write(f'  computed sections {LABEL[v]}\n')

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(R, f)
    sys.stderr.write(f'CACHED -> {CACHE}\n')
    return R


# --------------------------------------------------------------------------- plot
def plot(R):
    base = f'TPOSE24 {PERIODS[PERIOD][2]}'
    lonlim = (R['anchor_lon'] - HALF_DEG, R['anchor_lon'] + HALF_DEG)
    latlim = (-HALF_DEG, HALF_DEG)
    # each variant renders the SAME figure set: full domain first, then the ±5° crop.
    # (subdir, horizontal crop box, lon clip, lat clip)
    variants = [('full_domain', None, None, None),
                ('crop_140', (lonlim, latlim), lonlim, latlim)]
    for sub, box, lonclip, latclip in variants:
        outdir = os.path.join(OUTDIR, sub, SUBDIR)
        os.makedirs(outdir, exist_ok=True)
        _render(R, base, outdir, box, lonclip, latclip)
        sys.stderr.write(f'  plotted {sub}\n')
    sys.stderr.write('PLOTTED all figures (full_domain + crop_140)\n')


def _render(R, base, outdir, box, lonclip, latclip):
    anchor = (R['anchor_lon'], 0.0)
    units = {'UVEL': "u' (m s⁻¹)", 'VVEL': "v' (m s⁻¹)", 'WVEL': "w' (m s⁻¹)"}

    for dkey, dval, dlabel in R['depth_opts']:
        H = R['horiz'][dkey]

        # (1)(2) horizontal one-point maps, one figure per anchor latitude
        for alat in R['anchor_lats']:
            panels, refs = [], []
            for v in VARS:
                r, lon, lat = _maybe_crop(*H['cormap'][(alat, v)], box)
                panels.append((r, lon, lat, LABEL[v]))
                refs.append((0.5,) if v == 'WVEL' else (0.5, 0.0))
            plot_anchor_corr_map(
                panels, (R['anchor_lon'], alat),
                suptitle=f'{base}, {dlabel} — temporal one-point correlation at {_loc(alat)}',
                ref_levels=refs,
                fname=os.path.join(outdir, f'point_corr_map_{_lat_tag(alat)}_{dkey}.png'))

        # (3) e-folding temporal decorrelation-scale map
        epanels = []
        for v in VARS:
            tau, lon, lat = _maybe_crop(*H['efold'][v], box)
            epanels.append((tau, lon, lat, LABEL[v]))
        plot_efolding_map(
            epanels, anchor, groups=['uv', 'uv', 'w'],   # U & V share a scale, W its own
            suptitle=f'{base}, {dlabel} — e-folding temporal decorrelation scale',
            fname=os.path.join(outdir, f'efold_timescale_{dkey}.png'))

        # (4) hovmoller of fluctuations along 0°N with the decorrelation box
        hpanels = []
        for v in VARS:
            anom, lon, t_days, Lx, tau = H['hov'][v]
            m = slice(None) if lonclip is None else (lon >= lonclip[0]) & (lon <= lonclip[1])
            hpanels.append((anom[:, m], lon[m], t_days, LABEL[v], Lx, tau, units[v]))
        plot_hovmoller(
            hpanels, R['anchor_lon'],
            suptitle=f'{base}, {dlabel} — fluctuation hovmoller along 0°N (box = e-folding L×τ)',
            fname=os.path.join(outdir, f'hovmoller_{dkey}.png'))

    # (5) depth-section one-point maps
    for orient, key, xlabel, clip in [
            ('depth–latitude section (140°W)', 'dlat', 'Latitude (°N)', latclip),
            ('depth–longitude section', 'dlon', 'Longitude (°E)', lonclip)]:
        for alat in R['anchor_lats']:
            for ad in R['anchor_depths']:
                panels, refs = [], []
                x0 = z0 = None
                for v in VARS:
                    r, coord, z, c0, z0 = R['trans'][key][(alat, ad, v)]
                    rc, cc, zc = _crop_trans(r, coord, z, clip)
                    x0 = c0
                    panels.append((rc, cc, zc, LABEL[v]))
                    refs.append((0.5,) if v == 'WVEL' else (0.5, 0.0))
                plot_anchor_corr_map(
                    panels, (x0, z0), ncols=3, ref_levels=refs,
                    xlabel=xlabel, ylabel='Depth (m)', hline=None,
                    suptitle=(f'{base} — {orient}: correlation to {_loc(alat)}, {ad} m'),
                    fname=os.path.join(outdir, f'trans_corr_{key}_{_lat_tag(alat)}_{ad}m.png'))

    # (6) autocorrelation cuts (vs lon, lat, depth, time), U/V/W overlaid. The 25/75 m
    # anchors include the depth panel (from their section); the depth-averaged cut drops
    # the depth panel (no single anchor depth to profile against).
    dlabels = {k: lbl for k, _, lbl in R['depth_opts']}
    cut_configs = [('depthavg', None)] + [(f'{ad}m', ad) for ad in R['anchor_depths']]
    for dkey, ad in cut_configs:
        H = R['horiz'][dkey]
        for alat in R['anchor_lats']:
            cuts = {'lon': [], 'lat': [], 'time': []}
            if ad is not None:
                cuts['depth'] = []
            for v in VARS:
                r, lon, lat = H['cormap'][(alat, v)]
                iy = int(np.argmin(np.abs(lat - alat)))
                ix = int(np.argmin(np.abs(lon - R['anchor_lon'])))
                cuts['lon'].append((LABEL[v], *_clip1d(lon, r[iy, :], lonclip)))
                cuts['lat'].append((LABEL[v], *_clip1d(lat, r[:, ix], latclip)))
                lags, racf = H['acf'][(alat, v)]
                cuts['time'].append((LABEL[v], lags, racf))
                if ad is not None:
                    rt, coord, z, _, _ = R['trans']['dlat'][(alat, ad, v)]
                    iyt = int(np.argmin(np.abs(coord - alat)))
                    cuts['depth'].append((LABEL[v], z, rt[:, iyt]))
            plot_corr_cuts(
                cuts,
                suptitle=f'{base}, {dlabels[dkey]} — autocorrelation cuts at {_loc(alat)}',
                fname=os.path.join(outdir, f'corr_cuts_{_lat_tag(alat)}_{dkey}.png'),
                colors=UVW_COLORS)


def main(mode, avg='native'):
    global RESAMPLE, SUBDIR, CACHE
    if avg not in AVG_OPTS:
        raise SystemExit(f'unknown avg {avg!r}; choose from {list(AVG_OPTS)}')
    RESAMPLE, SUBDIR = AVG_OPTS[avg]
    if avg != 'native':                     # keep the native cache/outputs untouched
        CACHE = os.path.join(CACHE_DIR, f'point_autocorr_crop140_{avg}.pkl')
    if mode in ('all', 'compute'):
        R = compute()
    else:
        with open(CACHE, 'rb') as f:
            R = pickle.load(f)
    if mode in ('all', 'plot'):
        plot(R)


if __name__ == '__main__':
    args = sys.argv[1:]
    _mode = args[0] if len(args) > 0 else 'all'
    _avg = args[1] if len(args) > 1 else 'native'
    main(_mode, _avg)
