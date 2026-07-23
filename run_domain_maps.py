"""
Domain maps of TIME-MEAN structure: time/depth-mean velocity, the mean horizontal
divergence, the mean-current gradient magnitude and shear components, the
time-mean temperature (depth-averaged and at 25/50/70 m), and depth-latitude
transects at 140°W (time-mean temperature + velocity). Every diagnostic is of the
time-mean field. Figures are saved under
domain/{full_domain,crop_140}/{velocities,gradients,temperature}/ — full-domain vs
equatorial-crop views, split by diagnostic type. The 140°W transects share the same
full/crop split (full latitude range vs the equatorial band).

Split into two phases so figure styling can be re-tuned without re-reading the model:
  * COMPUTE — reads the model once and caches only the time/depth-mean U, V, W fields
    per config to CACHE_DIR (a few small 2-D fields each).
  * PLOT    — reads the cache and derives+renders all figures; no model access.

For each depth cutoff (0-70/120/250 m) and window (3-month '', 1-month '_1mo') it
writes, under velocities/: domain_mean_velocity, domain_divergence; under
gradients/: domain_gradient_mag, domain_gradient_shear.

Usage:
    python run_domain_maps.py [depths_csv] [periods_csv] [mode]
    mode ∈ {all (default), compute, plot, transect}
    e.g. python run_domain_maps.py 70,120,250 3mo,1mo          # compute + plot
         python run_domain_maps.py 70,120,250 3mo,1mo plot     # re-plot from cache (fast)
         python run_domain_maps.py 70 1mo compute              # (re)compute one config
         python run_domain_maps.py 70 3mo,1mo transect         # only the 140°W transects
"""

import os
import sys
import time
import pickle

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import cmocean.cm as cmo

import osse_tools as ot
from plotting_tools import plot_domain_grid, plot_transect

# dt60 run: deltaT=60 s (auto-read from the run's `data` namelist by load_model),
# 3-hourly diag_state output -> iter step 180 (180*60 s = 3 h), 718 records.
RUN_DIR = '/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3'
ITERS = list(range(180, 129240 + 180, 180))
OUTDIR = os.path.join(os.path.dirname(__file__), 'domain')
CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'

VARS = ('UVEL', 'VVEL', 'WVEL')          # -> U, V, W
LABEL = {'UVEL': 'U', 'VVEL': 'V', 'WVEL': 'W'}
T_LEVELS = (25, 50, 70)                  # fixed depths (m) for the time-mean T maps
CROP_LON = slice(217, 223)
CROP_LAT = slice(-3, 3)
# meridional (depth-latitude) transect at a fixed longitude (140°W = 220°E). Unlike the
# depth-mean maps this keeps the full water column, so it is computed per period only
# (independent of the map depth cutoffs) and cached separately.
TRANSECT_LON = 220.0                      # 140°W
TRANSECT_MAXDEPTH = 250                   # m; deep enough for the thermocline/EUC core
T_ISOTHERMS = list(range(12, 30, 2))      # °C isotherms overlaid on the T transect
# shared color-limit groups: U & V share one scale (directly comparable), W its own
GROUP_UVW = ['uv', 'uv', 'w']                              # for U/V/W panels
# shear figure: the U row is flipped (∂U/∂y, ∂U/∂x) so each column holds one shared
# pair — left col = shear strain {∂U/∂y, ∂V/∂x}, right col = normal strain
# {∂U/∂x, ∂V/∂y} — stacked vertically for comparison; W's two share their own scale.
# Panel order: ∂U/∂y,∂U/∂x, ∂V/∂x,∂V/∂y, ∂W/∂x,∂W/∂y
GROUP_SHEAR = ['shear', 'normal', 'shear', 'normal', 'w', 'w']

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


def _save_pair(base, panels_full, panels_crop, subdir, plotter=plot_domain_grid, **kw):
    """
    Save the full-domain and crop_ figure for one diagnostic into the folder tree
    domain/{full_domain,crop_140}/{subdir}/, where subdir is 'velocities',
    'gradients' or 'temperature'. `plotter` picks the renderer (map vs transect).
    """
    full_dir = os.path.join(OUTDIR, 'full_domain', subdir)
    crop_dir = os.path.join(OUTDIR, 'crop_140', subdir)
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    plotter(panels_full, fname=os.path.join(full_dir, f'{base}.png'), **kw)
    plotter(panels_crop, fname=os.path.join(crop_dir, f'crop_{base}.png'), **kw)


def _cache_path(d, pkey):
    return os.path.join(CACHE_DIR, f'domain_maps_{d}m_{pkey}.pkl')


def _transect_cache_path(pkey):
    return os.path.join(CACHE_DIR, f'domain_transect_{pkey}.pkl')


# --------------------------------------------------------------------------- #
# Meridional (depth-latitude) transect at a fixed longitude                    #
# --------------------------------------------------------------------------- #
def compute_transect(ds, tsel, lon=TRANSECT_LON, maxdepth=TRANSECT_MAXDEPTH):
    """Time-mean depth-latitude section at fixed longitude `lon` for THETA and each
    velocity component, on each field's native grid (horizontal interp to `lon`, native
    depth to `maxdepth`). Returns {var -> (values[depth, lat], lat, depth)}."""
    out = {}
    for v in ('THETA',) + VARS:
        gx, _ = ot._GRID[v]
        zc = ot._ZCOORD.get(v, 'Z')
        da = ds[v].sel(time=tsel).mean('time').interp({gx: lon})
        da = da.sel({zc: slice(0.0, -maxdepth)}).compute()
        ydim = [d for d in da.dims if d != zc][0]      # the latitude dim (YC/YG)
        da = da.transpose(zc, ydim)
        out[v] = (da.values, da[ydim].values, da[zc].values)
    return out


def _transect_panels(cache, keys):
    """(values, lat, depth, title) tuples for full and crop (equatorial) latitude views."""
    full, crop = [], []
    for v, title in keys:
        vals, lat, dep = cache[v]
        full.append((vals, lat, dep, title))
        m = (lat >= CROP_LAT.start) & (lat <= CROP_LAT.stop)
        crop.append((vals[:, m], lat[m], dep, title))
    return full, crop


def plot_transect_config(cache, suf, plabel):
    """Temperature and velocity depth-latitude transects at 140°W (full + crop)."""
    head = f'TPOSE24 {plabel}, 140°W section'
    # temperature (with isotherms)
    full, crop = _transect_panels(cache, [('THETA', 'temperature')])
    _save_pair(f'domain_transect_temperature_140W{suf}', full, crop, 'temperature',
               cbar_label='°C', cmap=cmo.thermal, diverging=False, ncols=1,
               suptitle=f'{head} — time-mean temperature', contour=T_ISOTHERMS,
               plotter=plot_transect)
    # velocity components (each on its own scale, like the depth-mean velocity map)
    full, crop = _transect_panels(cache, [(v, LABEL[v]) for v in VARS])
    _save_pair(f'domain_transect_velocity_140W{suf}', full, crop, 'velocities',
               cbar_label='m s$^{-1}$', cmap=cmo.balance, diverging=True, ncols=3,
               suptitle=f'{head} — time-mean velocity', plotter=plot_transect)


# --------------------------------------------------------------------------- #
# COMPUTE: model -> per-config cache of map arrays                            #
# --------------------------------------------------------------------------- #
def compute_config(S):
    """
    Time/depth-mean U, V, W for one depth/window from the depth-mean series `S`
    (dict var -> DataArray(time, y, x)). Every mean-field diagnostic (divergence,
    gradients) derives from these means at plot time, so the cache stays a few small
    2-D fields and re-plotting never touches the model.
    """
    return {'means': {v: S[v].mean('time') for v in VARS}}


# --------------------------------------------------------------------------- #
# PLOT: per-config cache -> figures                                           #
# --------------------------------------------------------------------------- #
def plot_config(cache, d, suf, plabel):
    """Render all figures for one depth/window from a compute_config() cache dict."""
    head = f'TPOSE24 {plabel}, 0–{d} m'
    means = cache['means']

    # --- 0. time/depth-mean velocity --------------------------------------
    # Each component on its OWN scale: V is much weaker than U, so a shared scale
    # would wash V out. (Derivatives below DO share — see GROUP_*.)
    full, crop = _panels([means[v] for v in VARS], [LABEL[v] for v in VARS])
    _save_pair(f'domain_mean_velocity_{d}m{suf}', full, crop, 'velocities',
               cbar_label='m s$^{-1}$', cmap=cmo.balance, diverging=True,
               suptitle=f'{head} — time/depth-mean velocity')

    # --- 1. mean horizontal divergence ------------------------------------
    # δ = ∂ū/∂x + ∂v̄/∂y is exactly what a plane fit integrates for the mean w. U and V
    # sit on staggered grids; the half-cell offset is negligible against these scales, so
    # the components are added index-aligned onto the tracer (W) grid.
    ubar, vbar, wc = means['UVEL'], means['VVEL'], means['WVEL']
    ux = ot.gradient_components(ubar.values, ubar[_xy(ubar)[0]].values,
                                ubar[_xy(ubar)[1]].values)[0]
    vy = ot.gradient_components(vbar.values, vbar[_xy(vbar)[0]].values,
                                vbar[_xy(vbar)[1]].values)[1]
    div = _as_da(ux + vy, wc)
    full, crop = _panels([div], ['mean divergence δ'])
    _save_pair(f'domain_divergence_{d}m{suf}', full, crop, 'velocities',
               cbar_label='s$^{-1}$', cmap=cmo.balance, diverging=True, ncols=1,
               suptitle=f'{head} — mean-current horizontal divergence')

    # --- 2. gradient magnitude of the mean currents (recompute from means) -
    Gmag = []
    for v in VARS:
        m = means[v]
        xdim, ydim = _xy(m)
        Gmag.append(_as_da(ot.gradient_magnitude(m.values, m[xdim].values,
                                                 m[ydim].values), m))
    full, crop = _panels(Gmag, [f'|∇{LABEL[v]}|' for v in VARS])
    _save_pair(f'domain_gradient_mag_{d}m{suf}', full, crop, 'gradients',
               cbar_label='s$^{-1}$', cmap=cmo.amp, groups=GROUP_UVW,
               suptitle=f'{head} — mean-current gradient magnitude')

    # --- 3. signed shear components (recompute; U row flipped) -------------
    Shear, stitles = [], []
    for v in VARS:
        m = means[v]
        xdim, ydim = _xy(m)
        fx, fy = ot.gradient_components(m.values, m[xdim].values, m[ydim].values)
        dax, day = _as_da(fx, m), _as_da(fy, m)
        if v == 'UVEL':   # flip so shared pairs stack in the same column
            Shear += [day, dax]
            stitles += [f'∂{LABEL[v]}/∂y', f'∂{LABEL[v]}/∂x']
        else:
            Shear += [dax, day]
            stitles += [f'∂{LABEL[v]}/∂x', f'∂{LABEL[v]}/∂y']
    full, crop = _panels(Shear, stitles)
    _save_pair(f'domain_gradient_shear_{d}m{suf}', full, crop, 'gradients',
               cbar_label='s$^{-1}$', cmap=cmo.balance, diverging=True, ncols=2,
               groups=GROUP_SHEAR,
               suptitle=f'{head} — mean-current shear components')

    # --- 4. time-mean temperature structure -------------------------------
    # Depth-averaged over 0-d plus fixed levels 25/50/70 m, so the horizontal
    # temperature structure (equatorial cold tongue, fronts) is visible at each
    # depth. Panels scale independently (T falls ~10 °C from surface to 70 m, so a
    # shared scale would wash out the deeper structure).
    if 'T' in cache:
        T = cache['T']
        das = [T['depthavg']] + [T['levels'][l] for l in T_LEVELS]
        titles = [f'0–{d} m mean'] + [f'{l} m' for l in T_LEVELS]
        full, crop = _panels(das, titles)
        _save_pair(f'domain_mean_temperature_{d}m{suf}', full, crop, 'temperature',
                   cbar_label='°C', cmap=cmo.thermal, diverging=False, ncols=2,
                   suptitle=f'{head} — time-mean temperature')


def main(depths, periods, mode='all'):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 'transect' mode: (re)compute + plot only the 140°W transects, reusing the existing
    # depth-mean map caches (much cheaper than a full recompute).
    if mode == 'transect':
        ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice('2012-10-11', None))
        for pkey in periods:
            tsel, suf, plabel = PERIODS[pkey]
            t1 = time.time()
            cache = compute_transect(ds, tsel)
            with open(_transect_cache_path(pkey), 'wb') as f:
                pickle.dump(cache, f)
            plot_transect_config(cache, suf, plabel)
            sys.stderr.write(f"TRANSECT {pkey}  ({time.time() - t1:.0f}s)\n")
        return

    if mode in ('all', 'compute'):
        ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice('2012-10-11', None))
        t0 = time.time()
        series = {}
        for v in VARS:                       # one read of the 0-250 m column per var
            series[v] = ot.depth_mean_series(ds, v, depths)
            sys.stderr.write(f"  loaded depth-mean series {LABEL[v]}  "
                             f"({time.time() - t0:.0f}s)\n")
        # temperature: depth-averaged series (per cutoff) + fixed levels (25/50/70 m).
        # NB select time BEFORE interp(Z): xarray's interp drops the time index.
        t_series = ot.depth_mean_series(ds, 'THETA', depths)
        t_z = [-float(l) for l in T_LEVELS]
        sys.stderr.write(f"  loaded temperature fields  ({time.time() - t0:.0f}s)\n")
        for pkey in periods:
            tsel, _, _ = PERIODS[pkey]
            # meridional transect at 140°W — full column, independent of the map cutoffs
            t1 = time.time()
            with open(_transect_cache_path(pkey), 'wb') as f:
                pickle.dump(compute_transect(ds, tsel), f)
            sys.stderr.write(f"COMPUTED transect {pkey}  ({time.time() - t1:.0f}s)\n")
            Tlev_mean = ds.THETA.sel(time=tsel).interp(Z=t_z).mean('time').compute()
            Tlev = {l: Tlev_mean.sel(Z=-float(l), method='nearest') for l in T_LEVELS}
            for d in depths:
                t1 = time.time()
                S = {v: series[v][d].sel(time=tsel) for v in VARS}
                cache = compute_config(S)
                cache['T'] = {'depthavg': t_series[d].sel(time=tsel).mean('time').compute(),
                              'levels': Tlev}
                with open(_cache_path(d, pkey), 'wb') as f:
                    pickle.dump(cache, f)
                sys.stderr.write(f"COMPUTED {pkey} {d}m  ({time.time() - t1:.0f}s)\n")

    if mode in ('all', 'plot'):
        for pkey in periods:
            _, suf, plabel = PERIODS[pkey]
            if os.path.exists(_transect_cache_path(pkey)):
                with open(_transect_cache_path(pkey), 'rb') as f:
                    plot_transect_config(pickle.load(f), suf, plabel)
            sys.stderr.write(f"PLOTTED transect {pkey}\n")
            for d in depths:
                with open(_cache_path(d, pkey), 'rb') as f:
                    cache = pickle.load(f)
                plot_config(cache, d, suf, plabel)
                sys.stderr.write(f"PLOTTED {pkey} {d}m\n")


if __name__ == '__main__':
    depths = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [70, 120, 250]
    periods = sys.argv[2].split(',') if len(sys.argv) > 2 else ['3mo', '1mo']
    mode = sys.argv[3] if len(sys.argv) > 3 else 'all'
    main(depths, periods, mode)
