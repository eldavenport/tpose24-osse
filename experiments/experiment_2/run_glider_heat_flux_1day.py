#!/usr/bin/env python
"""
run_glider_heat_flux_1day.py — per-glider advective-heating truth for the sym* configs
at 1-day resolution. The 1-day companion to run_glider_heat_flux.py (needed by the
point-flux sym figures), restricted to the same sym* SCOPE as run_heat_flux_1day.py.

Same daily-averaging as run_heat_flux_1day: the model is resampled to daily means
before the glider-point samples are taken, so the per-glider T, dT/dz and true model w
match the daily fields used by the array estimate.

Outputs (experiment_2/data_heat_1day/):
  <config>__cell_<center>__glider.nc   T_glider, w_true_glider, T_glider_native
                                       (as in run_glider_heat_flux, dims time=days)

Run (from the tpose env):  python experiment_2/run_glider_heat_flux_1day.py
"""
import json
import os
import sys

import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import osse_tools as ot                    # noqa: E402
import run_heat_flux as rhf                # noqa: E402
import run_glider_heat_flux as rghf        # noqa: E402  (reuse _glider_dataset)
import run_heat_flux_1day as rhf1          # noqa: E402  (reuse sym SCOPE, DATA_DIR, RESAMPLE)


def main():
    os.makedirs(rhf1.DATA_DIR, exist_ok=True)
    ds = None
    for path in rhf1.SCOPE:
        with open(path) as f:
            cfg = json.load(f)
        for center_lat, pos in ot.load_cells(path):
            nc_path = os.path.join(rhf1.DATA_DIR,
                                   f'{cfg["name"]}__cell_{center_lat:+.2f}__glider.nc')
            if os.path.exists(nc_path):
                with xr.open_dataset(nc_path) as chk:
                    if 'T_glider_native' in chk.data_vars:
                        continue
            if ds is None:
                ds = ot.load_model(rhf.RUN_DIR, rhf.ITERS).sel(
                    time=slice(rhf.SPINUP_END, None))
                ds = ds.resample(time=rhf1.RESAMPLE).mean()
                print(f'Model loaded and averaged to {ds.sizes["time"]} daily means')
            print(f'sampling gliders {cfg["name"]} cell {center_lat:+.2f}')
            rghf._glider_dataset(ds, pos).to_netcdf(nc_path)
    print('done')


if __name__ == '__main__':
    main()
