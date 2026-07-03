#!/usr/bin/env python
"""
run_experiment.py — sample the model once, estimate w for every experiment_1
configuration/cell, and save arrays plus a master metrics table.

Design:
  * The model is opened once and U,V are sampled at the union of all glider
    positions in a single pass (the only heavy read).
  * Model-truth w (hull-mean WVEL) is computed once per unique cell footprint
    and cached, since many configs share footprints.
  * Nothing here decides which configuration is "best" — it only records
    descriptive skill statistics for later inspection.

Outputs (experiment_1/data/):
  <config>__cell_<center>.nc   w_est, w_model, bias for one cell (dims time, depth)
  metrics.csv                  one row per cell with skill statistics + config metadata

Re-running skips cells whose .nc already exists; delete experiment_1/data/ to recompute.

Run (from the tpose env):  python experiment_1/run_experiment.py
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP1 = "/home/edavenport/analysis/tpose24-osse/experiment_1"
sys.path.insert(0, REPO)
import osse_tools as ot  # noqa: E402

# --- experiment parameters (shared by every config; keep in sync w/ notebooks) ---
RUN_DIR    = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS      = list(range(36, 26173, 36))     # 3-hourly diag_state steps
SPINUP_END = '2012-10-11'                   # drop model spin-up before this date
MIN_DEPTH  = 8                              # shallowest sampled depth (m); w=0 assumed here
MAX_DEPTH  = 70
DZ_OBS     = 2
MAX_WORKERS = 8

DATA_DIR = os.path.join(HERE, 'data')
CFG_GLOB = os.path.join(EXP1, 'configs', '**', '*.json')

def _footprint_key(positions):
    """Order-independent key for a cell's point set, so shared footprints compute once."""
    return tuple(sorted((round(p[0], 6), round(p[1], 6)) for p in positions))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    config_paths = sorted(glob.glob(CFG_GLOB, recursive=True))
    print(f'Found {len(config_paths)} configs')

    # Parse configs; collect every cell and the union of all glider positions.
    configs = []            # (meta dict, [(center_lat, positions), ...])
    all_positions = set()
    for path in config_paths:
        with open(path) as f:
            cfg = json.load(f)
        cells = ot.load_cells(path)
        configs.append((cfg, cells))
        for _, pos in cells:
            all_positions.update((round(p[0], 6), round(p[1], 6)) for p in pos)
    all_positions = sorted(all_positions)
    pos_idx = {p: i for i, p in enumerate(all_positions)}
    print(f'{len(all_positions)} unique glider positions across all cells')

    # Open the model and sample U,V once at every position (the heavy read).
    ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice(SPINUP_END, None))
    print(f'Model loaded: {ds.sizes["time"]} timesteps after spin-up')
    uv_all = ot.sample_fields(ds, all_positions, vars=('UVEL', 'VVEL'),
                              max_depth=MAX_DEPTH, dz_obs=DZ_OBS, min_depth=MIN_DEPTH).compute()
    print('Sampled U,V at all positions')

    # Model-truth w is expensive; compute one per unique cell footprint, in parallel.
    unique = {}
    for _, cells in configs:
        for _, pos in cells:
            unique.setdefault(_footprint_key(pos), pos)
    print(f'{len(unique)} unique cell footprints -> computing model-truth w')

    def _w_model(pos):
        return ot.sample_model_w(ds, pos, max_depth=MAX_DEPTH, dz_obs=DZ_OBS,
                                 min_depth=MIN_DEPTH, spatial_mean=True)

    keys = list(unique)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        w_model_cache = dict(zip(keys, ex.map(lambda k: _w_model(unique[k]), keys)))
    print('Model-truth w computed for all footprints')

    # Estimate w per cell, save arrays, and record skill statistics.
    rows = []
    for cfg, cells in configs:
        name = cfg['name']
        for center_lat, pos in cells:
            nc_path = os.path.join(DATA_DIR, f'{name}__cell_{center_lat:+.2f}.nc')
            key = _footprint_key(pos)
            w_model = w_model_cache[key]
            uv = uv_all.isel(glider=[pos_idx[(round(p[0], 6), round(p[1], 6))] for p in pos])
            # experiment_2: extrapolate U,V to the surface and integrate w from w=0
            # at 0 m (compute_w_planefit default), so w_est carries extra shallow
            # interfaces above MIN_DEPTH. w_skill_metrics aligns to the shared grid.
            w_est = ot.compute_w_planefit(uv)['w_est']
            bias = w_est - w_model

            xr.Dataset(dict(w_est=w_est, w_model=w_model, bias=bias)).to_netcdf(nc_path)
            m = ot.w_skill_metrics(w_est, w_model)
            rows.append(dict(
                config=name, family=cfg['family'], pattern=cfg['pattern'],
                width=cfg['width'], center_lat=center_lat,
                n_gliders_cell=cfg['n_gliders_per_cell'],
                n_gliders_total=cfg['n_gliders_total'],
                cell_height_deg=cfg['cell_height_deg'],
                min_depth=MIN_DEPTH, max_depth=MAX_DEPTH,
                nc_path=os.path.relpath(nc_path, HERE), **m))

    metrics = pd.DataFrame(rows).sort_values(
        ['family', 'pattern', 'width', 'center_lat']).reset_index(drop=True)
    metrics_path = os.path.join(DATA_DIR, 'metrics.csv')
    metrics.to_csv(metrics_path, index=False)
    print(f'\nWrote {len(metrics)} rows -> {metrics_path}')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
