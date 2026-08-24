#!/usr/bin/env python
"""
run_heat_flux_1day.py — vertical heat-flux skill for the symmetric regular-shape
(symhex/symdia/symsq) configs at **1-day** temporal resolution.

The 1-day companion to run_heat_flux.py, restricted to the sym* families (per
request; W-based advective heating + eddy flux, same quantities as run_heat_flux).
Advective heating w·∂T/∂z and eddy flux w'T' are NONLINEAR in the sampled fields, so
"1-day averages" cannot be applied by resampling the output series the way it can for
the linear w. Instead we resample the MODEL FIELDS to daily means up front
(`ds.resample(time='1D').mean()`) and run the unchanged run_heat_flux sampling on
that daily model: the estimate ([w]_est·[∂T/∂z]_est), the resolved hull components
([w]_hull, [∂T/∂z]_hull) and the truth pointwise products ([w·∂T/∂z], [w'T']) are all
formed from the SAME daily fields, so estimate and truth live in one self-consistent
"daily-sampled world" and the component decomposition identity still holds exactly.
(Sub-daily covariance is thus excluded from the truth too — the daily analogue of the
w rerun, which daily-averaged both the estimate inputs and the hull-mean truth.)

Outputs (experiment_2/data_heat_1day/):
  <config>__cell_<center>.nc   same variables as run_heat_flux (dims time=days, depth)
  heat_flux_metrics.csv        one row per cell: flux/heating skill at key depths

Re-running skips cells whose .nc already exists (delete data_heat_1day/ to recompute).

Run (from the tpose env):  python experiment_2/run_heat_flux_1day.py
"""
import json
import os
import sys

import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXP1 = os.path.join(REPO, 'experiments', 'experiment_1')
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import osse_tools as ot        # noqa: E402
import run_heat_flux as rhf    # noqa: E402  (reuse params, _cell_dataset, _metrics_row)

RESAMPLE = '1D'                                     # one estimate per day
# heavy .nc caches live on /data (not /home); see project_sym_disk_truth memory
CACHE_ROOT = '/data/SO3/edavenport/tpose24-osse/cache'
DATA_DIR = os.path.join(CACHE_ROOT, 'experiment_2', 'data_heat_1day')

# sym* families only: symhex/symdia/symsq at diameter 0.3/0.5/0.75/1.0°, centres 0.0/±0.5°N
SYM_DIAMS   = ['0.3', '0.5', '0.75', '1.0']
SYM_CENTERS = ['+0.0', '+0.5', '-0.5']
SCOPE = [os.path.join(EXP1, 'configs', fam, f'{fam}_d{d}_c{c}.json')
         for fam in ('symhex', 'symdia', 'symsq')
         for d in SYM_DIAMS for c in SYM_CENTERS]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = None
    rows = []
    for path in SCOPE:
        with open(path) as f:
            cfg = json.load(f)
        for center_lat, pos in ot.load_cells(path):
            nc_path = os.path.join(DATA_DIR, f'{cfg["name"]}__cell_{center_lat:+.2f}.nc')
            if not os.path.exists(nc_path):
                if ds is None:
                    ds = ot.load_model(rhf.RUN_DIR, rhf.ITERS).sel(
                        time=slice(rhf.SPINUP_END, None))
                    # 1-day averaging: collapse the 3-hourly model to daily means BEFORE
                    # sampling, so every nonlinear product is formed on daily fields.
                    ds = ds.resample(time=RESAMPLE).mean()
                    print(f'Model loaded and averaged to {ds.sizes["time"]} daily means')
                print(f'computing {cfg["name"]} cell {center_lat:+.2f}')
                rhf._cell_dataset(ds, pos, ot.sym_disk(cfg)).to_netcdf(nc_path)
            rows.append(rhf._metrics_row(cfg, center_lat, nc_path))

    metrics = pd.DataFrame(rows).sort_values(
        ['family', 'pattern', 'width', 'center_lat']).reset_index(drop=True)
    metrics_path = os.path.join(DATA_DIR, 'heat_flux_metrics.csv')
    metrics.to_csv(metrics_path, index=False)
    print(f'\nWrote {len(metrics)} rows -> {metrics_path}')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
