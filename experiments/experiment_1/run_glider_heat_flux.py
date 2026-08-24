#!/usr/bin/env python
"""
run_glider_heat_flux.py — per-glider advective-heating truth for experiment_1.

Companion to run_heat_flux.py. That script forms the ARRAY estimate of vertical
heat transport (plane-fit area-mean w paired with the array-mean T). Here we set up
a different question: how well is the vertical advective heating  w * dT/dz
estimated AT EACH GLIDER, when we take the array's area-mean w (the only w the array
can produce) and apply it to that glider's OWN measured vertical temperature
gradient — instead of collapsing T to the array mean.

So per glider g:
  estimate  A_est_g  = w_area(t,z) * dT_g/dz(t,z)     (area-mean plane-fit w)
  truth     A_true_g = w_g(t,z)    * dT_g/dz(t,z)     (model w at the glider point)
Because the glider measures T (hence dT_g/dz) directly, the estimate/truth
difference isolates the error of substituting the area-mean w for the local w.

This needs only cheap glider-POINT samples (model THETA and WVEL interpolated to the
glider positions and obs midpoints), so it does not redo run_heat_flux's expensive
hull truth. The area-mean w estimate (w_est_mid) is read back from run_heat_flux's
per-cell .nc at plot time.

Outputs (experiment_1/data_heat/):
  <config>__cell_<center>__glider.nc   T_glider, w_true_glider (dims time, depth,
                                       glider) on the 2 m obs axis; plus
                                       T_glider_native, the SAME columns of THETA at
                                       the model's NATIVE vertical resolution (dims
                                       time, Zc, glider). The obs-axis T_glider gives
                                       the dT/dz a 2 m glider profile would estimate;
                                       the native column gives the true model dT/dz at
                                       that point, so the two isolate the vertical
                                       resolution difference. Glider lat/lon as coords.

Re-running skips cells whose __glider.nc already exists AND already carries the native
column (older files without it are regenerated).

Run (from the tpose env):  python experiment_1/run_glider_heat_flux.py
"""
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import osse_tools as ot            # noqa: E402
import run_heat_flux as rhf        # noqa: E402  (reuse SCOPE, params, model handle)


def _to_depth(da):
    """Rename the obs-midpoint axis to 'depth' to match the array-estimate files."""
    return da.rename({'obs_depth': 'depth'}) if 'obs_depth' in da.dims else da


def _native_T(ds, pos, max_depth):
    """THETA at the glider columns on the model's NATIVE vertical grid (horizontal
    interp only, no depth interp). A little deeper than max_depth so a centred dT/dz
    is defined down to the deepest obs level. Dims (time, Zc, glider)."""
    g = np.arange(len(pos))
    lat = xr.DataArray([p[0] for p in pos], dims='glider', coords={'glider': g})
    lon = xr.DataArray([p[1] for p in pos], dims='glider', coords={'glider': g})
    T = ds.THETA.sel(Z=slice(0.0, -(max_depth + 12))).interp(XC=lon, YC=lat)
    T = T.drop_vars([c for c in T.coords if c not in ('time', 'Z', 'glider')],
                    errors='ignore')
    return T.rename({'Z': 'Zc'}).transpose('time', 'Zc', 'glider')


def _glider_dataset(ds, pos):
    """Per-glider T and true model w at the glider points (obs midpoints), plus the
    native-resolution T column for the resolution comparison."""
    samp = ot.sample_fields(ds, pos, vars=('THETA', 'WVEL'),
                            max_depth=rhf.MAX_DEPTH, dz_obs=rhf.DZ_OBS,
                            min_depth=rhf.MIN_DEPTH).compute()
    out = xr.Dataset(dict(
        T_glider=_to_depth(samp['T']),
        w_true_glider=_to_depth(samp['W']),
        T_glider_native=_native_T(ds, pos, rhf.MAX_DEPTH),
    ))
    return out.compute()


def main():
    os.makedirs(rhf.DATA_DIR, exist_ok=True)
    ds = None
    for path in rhf.SCOPE:
        import json
        with open(path) as f:
            cfg = json.load(f)
        for center_lat, pos in ot.load_cells(path):
            nc_path = os.path.join(rhf.DATA_DIR,
                                   f'{cfg["name"]}__cell_{center_lat:+.2f}__glider.nc')
            if os.path.exists(nc_path):
                with xr.open_dataset(nc_path) as chk:
                    if 'T_glider_native' in chk.data_vars:
                        continue          # up to date; older files fall through to redo
            if ds is None:
                ds = ot.load_model(rhf.RUN_DIR, rhf.ITERS).sel(
                    time=slice(rhf.SPINUP_END, None))
                print(f'Model loaded: {ds.sizes["time"]} timesteps after spin-up')
            print(f'sampling gliders {cfg["name"]} cell {center_lat:+.2f}')
            _glider_dataset(ds, pos).to_netcdf(nc_path)
    print('done')


if __name__ == '__main__':
    main()
