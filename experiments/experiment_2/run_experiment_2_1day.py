#!/usr/bin/env python
"""
run_experiment_2_1day.py — experiment_2 w-skill for the symmetric regular-shape
(symhex/symdia/symsq) configs, but with **1-day-averaged** sampling.

Same physics as run_experiment_2.py (extrapolate shear to the surface, integrate
w from w=0 at 0 m) — the only change is temporal resolution: instead of an estimate
every 3 hours, the sampled U,V and the model-truth WVEL are averaged into 1-day means
first, so we make one w estimate per day. Because the plane-fit estimator and the
hull-mean truth are both linear, this daily mean commutes through them (see README),
i.e. daily-averaging the inputs equals daily-averaging the per-timestep w. Here we
average the sampled fields up front so it reads as "1 estimate/day".

Only the sym* families are run (per request; W only, no heat flux).

Outputs (experiment_2/data_1day/):
  <config>__cell_<center>.nc   w_est, w_model, bias (dims time=days, depth)
  metrics.csv                  one row per cell with skill statistics + config metadata

Re-running skips cells whose .nc already exists; delete experiment_2/data_1day/ to recompute.

Run (from the tpose env):  python experiment_2/run_experiment_2_1day.py
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
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root (dir now under experiments/)
EXP1 = "/home/edavenport/analysis/tpose24-osse/experiments/experiment_1"
sys.path.insert(0, REPO)
import osse_tools as ot  # noqa: E402

# --- experiment parameters (kept in sync with run_experiment_2.py) ---
RUN_DIR    = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS      = list(range(36, 26173, 36))     # 3-hourly diag_state steps
SPINUP_END = '2012-10-11'                   # drop model spin-up before this date
MIN_DEPTH  = 8                              # shallowest sampled depth (m); w=0 assumed here
MAX_DEPTH  = 80
DZ_OBS     = 2
MAX_WORKERS = 8
RESAMPLE   = '1D'                           # temporal averaging window (1-day estimates)

DATA_DIR = os.path.join(HERE, 'data_1day')
# sym* families only (regular hexagon / diamond / square sweeps)
CFG_GLOB = os.path.join(EXP1, 'configs', 'sym*', '*.json')


def _daily(da):
    """Average a (time, ...) DataArray/Dataset into 1-day means (1 estimate/day)."""
    return da.resample(time=RESAMPLE).mean()


def _footprint_key(positions):
    """Order-independent key for a cell's point set, so shared footprints compute once."""
    return tuple(sorted((round(p[0], 6), round(p[1], 6)) for p in positions))


def _metrics_row(cfg, center_lat, nc_path):
    """Skill row for one cell, computed from its saved w_est/w_model arrays."""
    with xr.open_dataset(nc_path) as ds:
        m = ot.w_skill_metrics(ds.w_est, ds.w_model)
    return dict(
        config=cfg['name'], family=cfg['family'], pattern=cfg['pattern'],
        width=cfg['width'], center_lat=center_lat,
        n_gliders_cell=cfg['n_gliders_per_cell'],
        n_gliders_total=cfg['n_gliders_total'],
        cell_height_deg=cfg['cell_height_deg'],
        min_depth=MIN_DEPTH, max_depth=MAX_DEPTH,
        nc_path=os.path.relpath(nc_path, HERE), **m)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    config_paths = sorted(glob.glob(CFG_GLOB, recursive=True))
    print(f'Found {len(config_paths)} sym* configs')

    cached = []             # (cfg, center_lat, nc_path)
    pending = []            # (cfg, center_lat, positions, nc_path)
    for path in config_paths:
        with open(path) as f:
            cfg = json.load(f)
        for center_lat, pos in ot.load_cells(path):
            nc_path = os.path.join(DATA_DIR, f'{cfg["name"]}__cell_{center_lat:+.2f}.nc')
            if os.path.exists(nc_path):
                cached.append((cfg, center_lat, nc_path))
            else:
                pending.append((cfg, center_lat, pos, nc_path))
    print(f'{len(cached)} cells cached (skipped), {len(pending)} to compute')

    rows = [_metrics_row(cfg, cl, p) for cfg, cl, p in cached]

    if pending:
        all_positions = sorted({(round(p[0], 6), round(p[1], 6))
                                 for _, _, pos, _ in pending for p in pos})
        pos_idx = {p: i for i, p in enumerate(all_positions)}
        print(f'{len(all_positions)} unique glider positions across pending cells')

        ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice(SPINUP_END, None))
        print(f'Model loaded: {ds.sizes["time"]} timesteps after spin-up')
        uv_all = ot.sample_fields(ds, all_positions, vars=('UVEL', 'VVEL'),
                                  max_depth=MAX_DEPTH, dz_obs=DZ_OBS, min_depth=MIN_DEPTH).compute()
        # 1-day averaging: average the sampled currents into daily means before the
        # plane fit (linear estimator, so this == daily-averaging w_est).
        uv_all = _daily(uv_all)
        print(f'Sampled U,V and averaged to {uv_all.sizes["time"]} daily means')

        # Symmetric-shape configs use the circular-disk truth (ot.sym_disk); others
        # fall back to the convex-hull truth (disk=None).
        unique = {}
        for cfg, _, pos, _ in pending:
            unique.setdefault(_footprint_key(pos), (pos, ot.sym_disk(cfg)))
        print(f'{len(unique)} unique pending footprints -> computing model-truth w')

        def _w_model(pos, disk):
            # daily-average the footprint-mean model-truth w on the same 1-day windows
            return _daily(ot.sample_model_w(ds, pos, max_depth=MAX_DEPTH, dz_obs=DZ_OBS,
                                            min_depth=MIN_DEPTH, spatial_mean=True, disk=disk))

        keys = list(unique)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            w_model_cache = dict(zip(keys, ex.map(lambda k: _w_model(*unique[k]), keys)))
        print('Model-truth w computed for pending footprints')

        for cfg, center_lat, pos, nc_path in pending:
            w_model = w_model_cache[_footprint_key(pos)]
            uv = uv_all.isel(glider=[pos_idx[(round(p[0], 6), round(p[1], 6))] for p in pos])
            w_est = ot.compute_w_planefit(uv)['w_est']
            bias = w_est - w_model
            xr.Dataset(dict(w_est=w_est, w_model=w_model, bias=bias)).to_netcdf(nc_path)
            rows.append(_metrics_row(cfg, center_lat, nc_path))

    metrics = pd.DataFrame(rows).sort_values(
        ['family', 'pattern', 'width', 'center_lat']).reset_index(drop=True)
    metrics_path = os.path.join(DATA_DIR, 'metrics.csv')
    metrics.to_csv(metrics_path, index=False)
    print(f'\nWrote {len(metrics)} rows -> {metrics_path}')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
