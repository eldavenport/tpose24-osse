#!/usr/bin/env python
"""
run_meanw_field.py — a lightweight background mean-W field for the demonstration
figure (glider tracks over the mean vertical velocity).

The main compute (run_experiment_3.py) reads the whole 3-hourly record to sample the
moving gliders; the time-MEAN w field, by contrast, is a smooth large-scale pattern
(equatorial upwelling) that does not need every timestep.  So this script does a much
cheaper SUBSAMPLED read of just WVEL over the array bounding box, time-averages it onto
the obs-depth axis, and caches a 2-D field (depth, YC, XC) in metres/day.

Cache (experiment_3/cache/):
  meanw_field.nc   Wmean(depth, YC, XC) in m day^-1 — the post-spin-up time mean of the
                   model vertical velocity over the array bbox (with a small buffer so
                   the widest orbit has margin).  attrs: n_iters, iter_step.

Run (from the tpose env):  python experiment_3/run_meanw_field.py
Skips the read if the cache already exists (delete it to redo).
"""
import os
import sys
import warnings

import numpy as np
import xarray as xr

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circle_common as C            # noqa: E402
sys.path.insert(0, C.REPO)
import osse_tools as ot              # noqa: E402

ITER_STEP = 8                        # subsample the 3-hourly iters (every ~24 h)
BUF = 0.25                           # extra bbox half-width (deg) beyond the widest radius


def main():
    out_path = os.path.join(C.CACHE_DIR, 'meanw_field.nc')
    if os.path.exists(out_path):
        print(f'{out_path} exists — skipping (delete to redo)')
        return
    os.makedirs(C.CACHE_DIR, exist_ok=True)

    iters = C.ITERS[::ITER_STEP]
    print(f'reading WVEL for {len(iters)} subsampled iters ...')
    ds = ot.load_model(C.RUN_DIR, iters).sel(time=slice(C.SPINUP_END, None))

    half = max(C.DIAMETERS) / 2.0 + BUF
    lon0, lon1 = C.CENTER[1] - half, C.CENTER[1] + half
    lat0, lat1 = C.CENTER[0] - half, C.CENTER[0] + half
    W = ds['WVEL'].sel(XC=slice(lon0, lon1), YC=slice(lat0, lat1))

    obs_z = ot._obs_z(C.MAX_DEPTH, C.DZ_OBS, C.MIN_DEPTH)
    obs_z_da = xr.DataArray(obs_z, dims='obs_depth', coords={'obs_depth': obs_z})
    Wmean = (W.interp(Zl=obs_z_da).mean('time') * C.SEC_PER_DAY).compute()   # m/day
    Wmean = Wmean.rename('Wmean').drop_vars(
        [c for c in Wmean.coords if c not in ('obs_depth', 'YC', 'XC')])
    Wmean.attrs.update(units='m day-1', n_iters=len(iters), iter_step=ITER_STEP,
                       description='post-spin-up time-mean model WVEL over the array bbox')
    Wmean.to_dataset().to_netcdf(out_path)
    print(f'wrote {out_path}  {dict(Wmean.sizes)}')


if __name__ == '__main__':
    main()
