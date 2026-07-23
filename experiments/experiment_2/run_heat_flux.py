#!/usr/bin/env python
"""
run_heat_flux.py — vertical heat-flux (w'T') skill for the experiment_2 plane-fit w.

Augments experiment_2 (same configs, same w method: extrapolate shear to the
surface, w=0 at 0 m) to study vertical heat transport when w comes from the plane
fit. Two physically meaningful, reference-independent quantities are analysed (the
raw product <w T> is deliberately excluded — it depends on T's zero point):
  * eddy vertical heat flux  w'T'      (a proper flux, W/m^2 via HFLUX);
  * vertical advective heating w dT/dz (a heating rate deg C/day; integrates
    vertically to a flux, rho0 cp int w dT/dz dz).
The w-skill results in run_experiment_2.py are unchanged; this is separate.

For each cell we form
  * estimate : plane-fit w_est (interpolated to obs midpoints) paired with the
               array-mean glider temperature T_bar_glider;
  * truth    : the full hull-area eddy flux <w'T'>_hull and advective heating
               <w dT/dz>_hull (point-wise over the cell, so they carry the sub-cell
               covariance the array cannot see), plus the hull-mean components
               w_bar_hull, T_bar_hull as an intermediate area-mean reference.
These component/flux time series are enough to rebuild every diagnostic
(mean, variability, temporal PDFs, and the w-vs-T error attribution) offline.

Scope:
  (a) equator single-cell SHAPE sweep: diamond/hexagon/square at 1 and 2 deg cell
      height, each at diameters 0.5/1.0/1.5 deg (lon offsets 0.25/0.5/0.75);
  (b) off-equator LATITUDE sweep: 1 deg diamond & hexagon cells at +/-0.5, +/-1.5.

Outputs (experiment_2/data_heat/):
  <config>__cell_<center>.nc   w_est_mid, Tbar_glider, wbar_hull, Tbar_hull,
                               F_true_hull_eddy, A_true_total, A_true_eddy
                               (dims time, depth); depth = obs midpoints (m).
                               Eddy flux deg C m/s; advective heating deg C/s.
  heat_flux_metrics.csv        one row per cell: flux/heating skill at key depths.

Re-running skips cells whose .nc already exists (delete data_heat/ to recompute).

Run (from the tpose env):  python experiment_2/run_heat_flux.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXP1 = os.path.join(REPO, 'experiments', 'experiment_1')
sys.path.insert(0, REPO)
import osse_tools as ot  # noqa: E402

# --- experiment parameters (keep in sync with run_experiment_2.py) ---
RUN_DIR    = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS      = list(range(36, 26173, 36))     # 3-hourly diag_state steps
SPINUP_END = '2012-10-11'                   # drop model spin-up before this date
MIN_DEPTH  = 8
MAX_DEPTH  = 80
DZ_OBS     = 2

DATA_DIR = os.path.join(HERE, 'data_heat')
# Focused set of configs (paths under experiment_1/configs/).
#  (a) equator single-cell SHAPE sweep: diamond/hexagon/square at 1° and 2° cell
#      height, each at diameters 0.5/1.0/1.5° (lon offsets 0.25/0.5/0.75).
#  (b) off-equator LATITUDE sweep: 1° diamond & hexagon cells at ±0.5, ±1.5.
EQ_SHAPES = ['equator_1deg', 'equator_2deg', 'equator_hex1deg', 'equator_hex2deg',
             'equator_sq1deg', 'equator_sq2deg']
EQ_WIDTHS = ['0.25', '0.5', '0.75']
SCOPE = [os.path.join(EXP1, 'configs', 'equator', f'{s}_w{w}.json')
         for s in EQ_SHAPES for w in EQ_WIDTHS]
SCOPE += [
    os.path.join(EXP1, 'configs', 'shift', 'shift_w0.5.json'),      # 1° diameter (0.5° lon offset)
    os.path.join(EXP1, 'configs', 'shift', 'shift_hex_w0.5.json'),
    # interleaved latitudes -1/0/+1 filling the gaps in the ±0.5/±1.5 shift sweep
    os.path.join(EXP1, 'configs', 'shift', 'shift_w0.5_mid.json'),
    os.path.join(EXP1, 'configs', 'shift', 'shift_hex_w0.5_mid.json'),
    # square shift cells at the same latitudes (diamond/hex/square shape comparison off-equator)
    os.path.join(EXP1, 'configs', 'shift', 'shift_sq_w0.5.json'),
    os.path.join(EXP1, 'configs', 'shift', 'shift_sq_w0.5_mid.json'),
]
KEY_DEPTHS = [25, 50, 70]                    # m; temporal-PDF / summary depths


def _to_depth(da):
    """Rename the obs-midpoint axis to 'depth' so it matches heat_flux_skill_by_depth.
    Drops any leftover non-dim 'depth' coord (w_est carried one before interpolation)."""
    if 'depth' in da.coords and 'depth' not in da.dims:
        da = da.drop_vars('depth')
    return da.rename({'obs_depth': 'depth'}) if 'obs_depth' in da.dims else da


def _cell_dataset(ds, pos):
    """Compute the estimate/truth component & flux series for one cell."""
    # --- estimate: plane-fit w paired with array-mean glider T ---
    samp = ot.sample_fields(ds, pos, vars=('UVEL', 'VVEL', 'THETA'),
                            max_depth=MAX_DEPTH, dz_obs=DZ_OBS, min_depth=MIN_DEPTH).compute()
    z = samp.obs_depth.values
    w_est = ot.compute_w_planefit(samp[['U', 'V']])['w_est']
    # interpolate w_est (interfaces) onto the tracer obs midpoints, as array_vertical_flux does
    w_est_mid = w_est.interp(depth=xr.DataArray(z, dims='obs_depth', coords={'obs_depth': z}))
    Tbar_glider = samp['T'].mean('glider')

    # --- truth: full hull-area flux + hull-mean components (native points, depth
    #     interp only — cheap, like sample_model_w) ---
    truth = ot.sample_model_heat_flux(ds, pos, max_depth=MAX_DEPTH, dz_obs=DZ_OBS,
                                      min_depth=MIN_DEPTH)

    out = xr.Dataset(dict(
        w_est_mid=_to_depth(w_est_mid),
        Tbar_glider=_to_depth(Tbar_glider),
        wbar_hull=_to_depth(truth.wbar),
        Tbar_hull=_to_depth(truth.Tbar),
        F_true_hull_eddy=_to_depth(truth.eddy_flux),
        A_true_total=_to_depth(truth.adv_total),
        A_true_eddy=_to_depth(truth.adv_eddy),
    ))
    return out.compute()


def _metrics_row(cfg, center_lat, nc_path):
    """Flux-skill row for one cell, from its saved component/flux series."""
    with xr.open_dataset(nc_path) as d:
        d = d.load()
    # eddy heat flux w'T' (a proper flux) — est vs full-hull truth
    est_f = ot.flux_total_and_eddy(d.w_est_mid, d.Tbar_glider)
    F_est_eddy = est_f['eddy']
    F_true_eddy = d.F_true_hull_eddy
    B = ot.heat_flux_skill_by_depth(F_est_eddy, F_true_eddy)   # per-depth eddy-flux skill
    # advective heating w dT/dz (heating rate) — est vs full-hull truth
    est_a = ot.advective_heating(d.w_est_mid, d.Tbar_glider)
    A_est_total = est_a['total']
    A_true_total = d.A_true_total
    BA = ot.heat_flux_skill_by_depth(A_est_total, A_true_total)  # per-depth heating skill

    row = dict(
        config=cfg['name'], family=cfg['family'], pattern=cfg['pattern'],
        width=cfg['width'], center_lat=center_lat,
        n_gliders_cell=cfg['n_gliders_per_cell'],
        n_gliders_total=cfg['n_gliders_total'],
        cell_height_deg=cfg['cell_height_deg'],
        min_depth=MIN_DEPTH, max_depth=MAX_DEPTH,
        nc_path=os.path.relpath(nc_path, HERE),
    )
    # eddy-flux (wpTp) and advective-heating (w dT/dz) skill + temporal-PDF distance
    # at each key depth (raw deg C m/s and deg C/s respectively)
    for zk in KEY_DEPTHS:
        b = B.sel(depth=-zk, method='nearest')
        s = ot.dist_stats(F_est_eddy.sel(depth=-zk, method='nearest').values,
                          F_true_eddy.sel(depth=-zk, method='nearest').values)
        a = BA.sel(depth=-zk, method='nearest')
        sa = ot.dist_stats(A_est_total.sel(depth=-zk, method='nearest').values,
                           A_true_total.sel(depth=-zk, method='nearest').values)
        row.update({
            f'mean_true_{zk}': float(b.mean_true),
            f'mean_est_{zk}':  float(b.mean_est),
            f'mean_bias_{zk}': float(b.mean_bias),
            f'std_true_{zk}':  float(b.true_std),
            f'std_est_{zk}':   float(b.est_std),
            f'std_ratio_{zk}': float(b.std_ratio),
            f'corr_{zk}':      float(b.corr),
            f'js_{zk}':        s['js'],
            f'ks_{zk}':        s['ks'],
            f'adv_mean_true_{zk}': float(a.mean_true),
            f'adv_mean_est_{zk}':  float(a.mean_est),
            f'adv_mean_bias_{zk}': float(a.mean_bias),
            f'adv_std_true_{zk}':  float(a.true_std),
            f'adv_std_est_{zk}':   float(a.est_std),
            f'adv_std_ratio_{zk}': float(a.std_ratio),
            f'adv_corr_{zk}':      float(a.corr),
            f'adv_js_{zk}':        sa['js'],
            f'adv_ks_{zk}':        sa['ks'],
        })
    return row


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
                    ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice(SPINUP_END, None))
                    print(f'Model loaded: {ds.sizes["time"]} timesteps after spin-up')
                print(f'computing {cfg["name"]} cell {center_lat:+.2f}')
                _cell_dataset(ds, pos).to_netcdf(nc_path)
            rows.append(_metrics_row(cfg, center_lat, nc_path))

    metrics = pd.DataFrame(rows).sort_values(
        ['family', 'pattern', 'width', 'center_lat']).reset_index(drop=True)
    metrics_path = os.path.join(DATA_DIR, 'heat_flux_metrics.csv')
    metrics.to_csv(metrics_path, index=False)
    print(f'\nWrote {len(metrics)} rows -> {metrics_path}')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
