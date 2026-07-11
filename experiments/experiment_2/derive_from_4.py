#!/usr/bin/env python
"""
derive_from_4.py — build experiment_2's data by subsetting experiment_4's.

experiment_2 and experiment_4 use the IDENTICAL w method (extrapolate_to_surface
=True, w=0 at 0 m); they differ only in MAX_DEPTH (70 vs 120 m). compute_w_planefit
integrates w downward from the surface, so w_est over 8-70 m does not depend on how
deep the profile was sampled — the 8-70 m slice of a 4 profile is bit-identical to a
native 2 profile (same for w_model, a per-depth WVEL interpolation). So we can avoid
re-sampling the Ri5 model for 2 entirely: just copy each experiment_4/data/*.nc,
truncated to depth >= -MAX_DEPTH (70 m), into experiment_2/data/.

The truncation is REQUIRED, not cosmetic: run_experiment_2.py / w_skill_metrics pool
skill over the whole depth range stored in the file, so an un-truncated 4 file would
yield 8-120 m metrics.

After this runs, `python run_experiment_2.py` finds every cell cached and rebuilds
metrics.csv (stamped MIN_DEPTH=8, MAX_DEPTH=70) without opening the model.

Run (from the tpose env):  python experiment_2/derive_from_4.py
"""
import glob
import os

import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root (dir now under experiments/)
SRC_DIR = os.path.join(REPO, 'experiments', 'experiment_4', 'data')
DST_DIR = os.path.join(HERE, 'data')
MAX_DEPTH = 70   # keep in sync with run_experiment_2.py


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    src_files = sorted(glob.glob(os.path.join(SRC_DIR, '*.nc')))
    if not src_files:
        raise SystemExit(f'No source .nc files in {SRC_DIR} — run experiment_4 first.')
    print(f'Subsetting {len(src_files)} files from {SRC_DIR} to depth >= -{MAX_DEPTH} m')
    for src in src_files:
        dst = os.path.join(DST_DIR, os.path.basename(src))
        with xr.open_dataset(src) as ds:
            # depth is ascending after the exp4 save (-120, ..., -2, 0); keep depth
            # >= -MAX_DEPTH, i.e. the -70 .. 0 slice (36 interfaces), matching a
            # native 8-70 m experiment_2 file.
            sub = ds.sel(depth=slice(-MAX_DEPTH, 0)).load()
        sub.to_netcdf(dst)
    print(f'Wrote {len(src_files)} truncated files -> {DST_DIR}')
    print('Now run:  python run_experiment_2.py   (builds metrics.csv from cached files)')


if __name__ == '__main__':
    main()
