"""
COMPUTE phase of the sampling_dynamics "virtual mooring" study.

Reads the dt60 run ONCE and, for every symhex diameter (centre 0degN,140degW, with the
TAO mooring added at the centre), writes three cache files that every diagnostic
script reads without touching the model again:

  cache/<name>_array.nc     what the ARRAY samples: fields at the 6 gliders + mooring
                            (time, glider, obs_depth) + hbl(time,glider) + the plane-fit
                            w estimate (w_est interfaces, w_est_mid on the obs axis).
  cache/<name>_hull.nc      the model TRUTH inside the hexagon hull, as hull-mean
                            (time, obs_depth) profiles of every field, hbl(time), and
                            the instantaneous hull-mean eddy-flux series (vertical
                            w'T'/w'U'/w'V' and lateral U'T'/V'T'/U'V').
  cache/<name>_cloud.nc     the full hull point cloud, time-subsampled x3, for the
                            distribution comparisons (array vs every truth grid point).

The co-located bbox interpolation (region_bbox) is done once for the largest footprint
and reused for every diameter via select_hull, so only the hull's columns are read.

Usage:  python run_sample.py [d0.3 d0.5 ...]   (default: all diameters)
"""

import os
import sys

import numpy as np
import xarray as xr

import common as C          # inserts the repo root on sys.path for osse_tools
import osse_tools as ot

TIME_SUB = 3                        # time-subsample factor for the distribution cloud
FLUX_PAIRS_VERT = [('W', 'T', 'wT'), ('W', 'U', 'wU'), ('W', 'V', 'wV')]
FLUX_PAIRS_LAT = [('U', 'T', 'uT'), ('V', 'T', 'vT'), ('U', 'V', 'uv')]


def _plane_fit_w(arr):
    """Plane-fit w from the array U,V (compute_w_planefit defaults, like experiment_2)."""
    w_est = ot.compute_w_planefit(arr[['U', 'V']])['w_est']       # (time, depth) interfaces
    z = arr.obs_depth.values
    w_est_mid = w_est.interp(depth=xr.DataArray(z, dims='obs_depth',
                                                coords={'obs_depth': z}))
    w_est_mid = w_est_mid.drop_vars([c for c in ('depth',) if c in w_est_mid.coords])
    return w_est, w_est_mid


def _eddy_flux_series(sel, pairs):
    """Instantaneous hull-point-mean eddy-flux series <a'b'>_hull(time, obs_depth).

    Primes are deviations from the per-point time mean; the product is then averaged
    over the hull points, leaving a (time, obs_depth) series whose time mean is the
    true eddy flux and whose distribution/ spectrum we compare against the array.
    """
    out = {}
    for a, b, name in pairs:
        ap = sel[a] - sel[a].mean('time')
        bp = sel[b] - sel[b].mean('time')
        out[name] = (ap * bp).mean('point')
    return xr.Dataset(out)


def compute_config(ds, region, diam):
    name = C.config_name(diam)
    gliders = C.glider_positions(diam)
    positions = C.array_positions(diam)
    sys.stderr.write(f'[{name}] sampling array ({len(positions)} pts)\n'); sys.stderr.flush()

    # ---- array: fields at the 6 gliders + centre mooring, plus plane-fit w ----
    arr = C.sample_array(ds, positions).compute()
    is_moor = np.zeros(len(positions), bool)
    is_moor[C.mooring_index(diam)] = True
    arr = arr.assign_coords(is_mooring=('glider', is_moor))
    w_est, w_est_mid = _plane_fit_w(arr)
    arr['w_est'] = w_est
    arr['w_est_mid'] = w_est_mid
    arr.attrs.update(config=name, diameter=diam, mooring_index=C.mooring_index(diam))
    arr.to_netcdf(os.path.join(C.CACHE_DIR, f'{name}_array.nc'))
    sys.stderr.write(f'[{name}] wrote array.nc\n'); sys.stderr.flush()

    # ---- truth: model grid points inside the hexagon hull ----
    sel = C.select_hull(region, gliders)
    npts = sel.sizes['point']
    prof_vars = [v for v in ['U', 'V', 'T', 'S', 'W', 'nu', 'kappaT', 'N2', 'sigma0']
                 if v in sel]
    hull_mean = sel[prof_vars].mean('point')
    hull_mean['hbl'] = sel['hbl'].mean('point')
    flux = xr.merge([_eddy_flux_series(sel, FLUX_PAIRS_VERT),
                     _eddy_flux_series(sel, FLUX_PAIRS_LAT)])
    hull = xr.merge([hull_mean, flux], compat='override').compute()
    hull.attrs.update(config=name, diameter=diam, n_hull_points=npts)
    hull.to_netcdf(os.path.join(C.CACHE_DIR, f'{name}_hull.nc'))
    sys.stderr.write(f'[{name}] wrote hull.nc ({npts} hull points)\n'); sys.stderr.flush()

    # ---- distribution cloud: every hull point, time-subsampled ----
    # reset the stacked 'point' MultiIndex (YC/XC/lat/lon stay as plain coords) so
    # the cloud can be written to NetCDF.
    cloud = sel.isel(time=slice(None, None, TIME_SUB)).reset_index('point').compute()
    cloud.attrs.update(config=name, diameter=diam, n_hull_points=npts, time_sub=TIME_SUB)
    cloud.to_netcdf(os.path.join(C.CACHE_DIR, f'{name}_cloud.nc'))
    sys.stderr.write(f'[{name}] wrote cloud.nc ({cloud.sizes["time"]} times)\n')
    sys.stderr.flush()


def main(diams):
    half = max(C.DIAMETERS) / 2
    ds = C.load_bbox_memory(half)          # read the bbox once, into memory
    sys.stderr.write(f'bbox loaded into memory: {dict(ds.sizes)}\n'); sys.stderr.flush()
    region = C.region_bbox(ds, half_deg=half)
    sys.stderr.write(f'region bbox built: {region.sizes["point"]} points\n')
    sys.stderr.flush()
    for d in diams:
        compute_config(ds, region, d)
    sys.stderr.write('DONE\n')


if __name__ == '__main__':
    args = sys.argv[1:]
    diams = [float(a.lstrip('d')) for a in args] if args else C.DIAMETERS
    main(diams)
