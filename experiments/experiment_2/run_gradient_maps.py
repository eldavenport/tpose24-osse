#!/usr/bin/env python
"""
run_gradient_maps.py — per-array maps of the plane-fit horizontal gradients vs the
true model field, for the experiment_2 heat-flux config set.

For each array cell it writes four 3x3 figures (depth-average 0-80 m, and single
levels 25/50/70 m). Every figure's rows are dU/dx, dV/dy and their sum (horizontal
divergence delta); its columns are:
  * True     — the spatially varying model gradient over the array footprint;
  * Estimate — the single plane-fit slope the array reports (constant over the cell,
               the same slope compute_w_planefit integrates to get w);
  * Difference (True - Estimate) — where the linear plane fit departs from the truth.
All panels are filled contours with 101 levels on a zero-centred balance colormap, so
the Difference column diverges about 0 by construction (see plotting_tools.plot_gradient_comparison).

Because the plane fit and time/depth-averaging are all LINEAR in the velocity field,
the time/depth-mean gradients equal the gradients of the time/depth-mean U, V (same
argument as the footprint-error maps). So the model is read only to cache the small
time-mean U, V fields; every figure is then derived from that cache.

Split into two phases, like run_domain_maps.py:
  * COMPUTE — read the model once, cache time-mean U, V over the array region for the
    depth-average (0-80 m) and each level (25/50/70 m) to CACHE_FILE.
  * PLOT    — read the cache and render all figures into experiment_figs/<config>/;
    no model access. Skips figures already on disk.

Usage:
    python experiment_2/run_gradient_maps.py [mode]
    mode in {all (default), compute, plot}
Re-running PLOT skips existing figures; delete them (or the cache) to redo.
"""
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXP1 = os.path.join(REPO, 'experiments', 'experiment_1')
sys.path.insert(0, REPO)
import osse_tools as ot            # noqa: E402
import plotting_tools as pt        # noqa: E402

# --- experiment parameters (keep in sync with run_heat_flux.py) ---
RUN_DIR    = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS      = list(range(36, 26173, 36))     # 3-hourly diag_state steps
SPINUP_END = '2012-10-11'                   # drop model spin-up before this date
DEPTH_AVG  = 80                             # depth-average layer (m); matches MAX_DEPTH
KEY_DEPTHS = [25, 50, 70]                   # single levels (m) for the fixed-depth maps

# Region cached / differentiated. Big enough to cover every heat-flux cell plus a
# display buffer, so np.gradient edge effects stay outside the shown footprints.
REGION_LON = (217.5, 222.5)
REGION_LAT = (-4.0, 4.0)
DISPLAY_BUF = 0.6                           # deg of context shown around each cell

CACHE_DIR  = os.path.join(HERE, 'data_grad')
CACHE_FILE = os.path.join(CACHE_DIR, 'gradient_means.pkl')
EFIG       = os.path.join(HERE, 'experiment_figs')

# Same focused set as run_heat_flux.py.
EQ_SHAPES = ['equator_1deg', 'equator_2deg', 'equator_hex1deg', 'equator_hex2deg',
             'equator_sq1deg', 'equator_sq2deg']
EQ_WIDTHS = ['0.25', '0.5', '0.75']
SCOPE = [os.path.join(EXP1, 'configs', 'equator', f'{s}_w{w}.json')
         for s in EQ_SHAPES for w in EQ_WIDTHS]
SCOPE += [
    os.path.join(EXP1, 'configs', 'shift', 'shift_w0.5.json'),      # 1° diameter (0.5° lon offset)
    os.path.join(EXP1, 'configs', 'shift', 'shift_hex_w0.5.json'),
]
# symmetric REGULAR-hexagon sweep (diameters 0.3/0.5/0.75/1.0° centred at 0.0/0.5/-0.5°N)
SCOPE += [os.path.join(EXP1, 'configs', 'symhex', f'symhex_d{d}_c{c}.json')
          for d in ('0.3', '0.5', '0.75', '1.0') for c in ('+0.0', '+0.5', '-0.5')]

# depth key -> (filename tag, plot label)
DEPTH_KEYS = [('depthavg', 'depthavg', f'0–{DEPTH_AVG} m mean')]
DEPTH_KEYS += [(d, f'{d}m', f'{d} m') for d in KEY_DEPTHS]


def _crop(da):
    """Crop a velocity DataArray to REGION regardless of its (U vs V) grid stagger."""
    xdim = 'XG' if 'XG' in da.dims else 'XC'
    ydim = 'YG' if 'YG' in da.dims else 'YC'
    return da.sel({xdim: slice(*REGION_LON), ydim: slice(*REGION_LAT)})


def compute():
    """Read the model once; cache region-cropped time-mean U, V per depth key."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice(SPINUP_END, None))
    sys.stderr.write(f'Model loaded: {ds.sizes["time"]} timesteps after spin-up\n')
    means = {}
    t0 = time.time()

    # depth-average over 0..DEPTH_AVG m (one column read per component)
    for var in ('UVEL', 'VVEL'):
        da = ot.depth_mean_series(ds, var, [DEPTH_AVG])[DEPTH_AVG].mean('time')
        means.setdefault('depthavg', {})[ot._RENAME[var]] = _crop(da.compute())
    sys.stderr.write(f'  depth-average done ({time.time() - t0:.0f}s)\n')

    # single levels 25/50/70 m
    for d in KEY_DEPTHS:
        for var in ('UVEL', 'VVEL'):
            da = ds[var].sel(time=slice(SPINUP_END, None)).interp(Z=-float(d)).mean('time')
            means.setdefault(d, {})[ot._RENAME[var]] = _crop(da.compute())
        sys.stderr.write(f'  {d} m level done ({time.time() - t0:.0f}s)\n')

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(means, f)
    sys.stderr.write(f'CACHED time-mean U,V -> {CACHE_FILE}  ({time.time() - t0:.0f}s)\n')
    return means


def plot(means=None):
    """Render the per-cell gradient-comparison figures from the cache."""
    if means is None:
        with open(CACHE_FILE, 'rb') as f:
            means = pickle.load(f)
    # gradient_map_fields (co-locate + differentiate) is config-independent, so build
    # it once per depth key and reuse across every cell.
    fields_by_key = {k: ot.gradient_map_fields(means[k]['U'], means[k]['V'])
                     for k, _, _ in DEPTH_KEYS}

    n = 0
    for path in SCOPE:
        with open(path) as f:
            cfg = json.load(f)
        outdir = os.path.join(EFIG, cfg['name'])
        os.makedirs(outdir, exist_ok=True)
        for center_lat, pos in ot.load_cells(path):
            bbox = ot._hull_bbox(pos, buf=DISPLAY_BUF)
            for key, tag, label in DEPTH_KEYS:
                fname = os.path.join(outdir, f'gradient_maps_{tag}_cell_{center_lat:+.2f}.png')
                if os.path.exists(fname):
                    continue
                pt.plot_gradient_comparison(fields_by_key[key], pos, bbox, label, fname)
                n += 1
                sys.stderr.write(f'  wrote {os.path.relpath(fname, HERE)}\n')
    sys.stderr.write(f'PLOTTED {n} new gradient-map figures '
                     f'(skipped existing)\n')


def main(mode='all'):
    means = compute() if mode in ('all', 'compute') else None
    if mode in ('all', 'plot'):
        plot(means)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
