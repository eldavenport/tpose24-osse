#!/usr/bin/env python
"""
run_heat_flux.py — vertical heat-flux (w'T') skill for the experiment_1 plane-fit w.

experiment_1 counterpart of experiment_2/run_heat_flux.py, restricted to the
symmetric REGULAR-shape (sym_sweep) config families and using experiment_1's w
method: NO extrapolation of the shear to the surface (w=0 at the shallowest
sampled depth, 8 m) and an 8–70 m sampled column (MAX_DEPTH=70). Studies vertical
heat transport when w comes from the plane fit. Two physically meaningful,
reference-independent quantities are analysed (the raw product <w T> is
deliberately excluded — it depends on T's zero point):
  * eddy vertical heat flux  w'T'      (a proper flux, W/m^2 via HFLUX);
  * vertical advective heating w dT/dz (a heating rate deg C/day; integrates
    vertically to a flux, rho0 cp int w dT/dz dz).
The w-skill results in run_experiment.py are unchanged; this is separate.

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
MAX_DEPTH  = 70
DZ_OBS     = 2

# heavy .nc caches live on /data (not /home); see project_sym_disk_truth memory
CACHE_ROOT = '/data/SO3/edavenport/tpose24-osse/cache'
DATA_DIR = os.path.join(CACHE_ROOT, 'experiment_1', 'data_heat')
# experiment_1 replicates ONLY the symmetric REGULAR-shape (sym_sweep) heat-flux
# results, so the exp2 equator/shift heat-flux scope is intentionally omitted here.
SCOPE = []
# (c) symmetric REGULAR-hexagon sweep: diameters 0.3/0.5/0.75/1.0° (E-W width) centred
#     at 0.0/0.5/-0.5°N, no moorings — isotropic footprints, diameter-vs-latitude compare.
SYMHEX_DIAMS   = ['0.3', '0.5', '0.75', '1.0']
SYMHEX_CENTERS = ['+0.0', '+0.5', '-0.5']
SCOPE += [os.path.join(EXP1, 'configs', 'symhex', f'symhex_d{d}_c{c}.json')
          for d in SYMHEX_DIAMS for c in SYMHEX_CENTERS]
# (d) symmetric REGULAR diamond & square sweeps — the shape analogues of symhex at the
#     same diameters and centre latitudes (0.0/0.5/-0.5°N), no moorings — for the
#     hexagon/diamond/square shape comparison.
for _fam in ('symdia', 'symsq'):
    SCOPE += [os.path.join(EXP1, 'configs', _fam, f'{_fam}_d{d}_c{c}.json')
              for d in SYMHEX_DIAMS for c in SYMHEX_CENTERS]
KEY_DEPTHS = [25, 50, 70]                    # m; temporal-PDF / summary depths


def _to_depth(da):
    """Rename the obs-midpoint axis to 'depth' so it matches heat_flux_skill_by_depth.
    Drops any leftover non-dim 'depth' coord (w_est carried one before interpolation)."""
    if 'depth' in da.coords and 'depth' not in da.dims:
        da = da.drop_vars('depth')
    return da.rename({'obs_depth': 'depth'}) if 'obs_depth' in da.dims else da


def _cell_dataset(ds, pos, disk=None):
    """Compute the estimate/truth component & flux series for one cell.

    `disk`=(center_lat, center_lon, radius_deg) selects the circular-disk truth for a
    symmetric-shape config; None falls back to the convex-hull truth."""
    # --- estimate: plane-fit w paired with array-mean glider T ---
    samp = ot.sample_fields(ds, pos, vars=('UVEL', 'VVEL', 'THETA'),
                            max_depth=MAX_DEPTH, dz_obs=DZ_OBS, min_depth=MIN_DEPTH).compute()
    z = samp.obs_depth.values
    w_est = ot.compute_w_planefit(samp[['U', 'V']], extrapolate_to_surface=False)['w_est']
    # interpolate w_est (interfaces) onto the tracer obs midpoints, as array_vertical_flux does
    w_est_mid = w_est.interp(depth=xr.DataArray(z, dims='obs_depth', coords={'obs_depth': z}))
    Tbar_glider = samp['T'].mean('glider')

    # --- truth: full hull-area flux + hull-mean components (native points, depth
    #     interp only — cheap, like sample_model_w) ---
    truth = ot.sample_model_heat_flux(ds, pos, max_depth=MAX_DEPTH, dz_obs=DZ_OBS,
                                      min_depth=MIN_DEPTH, disk=disk)

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
    # Optional round-robin partition for parallel workers: `python run_heat_flux.py K N`
    # processes SCOPE[K::N] (disjoint per K, so no write races) and SKIPS the metrics.csv
    # write. Run once more with no args after every partition finishes to assemble
    # heat_flux_metrics.csv from the completed .nc (cheap — no model read once all exist).
    part = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else None
    scope = SCOPE[part[0]::part[1]] if part else SCOPE
    ds = None
    rows = []
    for path in scope:
        with open(path) as f:
            cfg = json.load(f)
        for center_lat, pos in ot.load_cells(path):
            nc_path = os.path.join(DATA_DIR, f'{cfg["name"]}__cell_{center_lat:+.2f}.nc')
            if not os.path.exists(nc_path):
                if ds is None:
                    ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice(SPINUP_END, None))
                    print(f'Model loaded: {ds.sizes["time"]} timesteps after spin-up', flush=True)
                print(f'computing {cfg["name"]} cell {center_lat:+.2f}', flush=True)
                _cell_dataset(ds, pos, ot.sym_disk(cfg)).to_netcdf(nc_path)
            if part is None:
                rows.append(_metrics_row(cfg, center_lat, nc_path))

    if part is not None:
        print(f'partition {part[0]}/{part[1]} done ({len(scope)} configs)', flush=True)
        return

    metrics = pd.DataFrame(rows).sort_values(
        ['family', 'pattern', 'width', 'center_lat']).reset_index(drop=True)
    metrics_path = os.path.join(DATA_DIR, 'heat_flux_metrics.csv')
    metrics.to_csv(metrics_path, index=False)
    print(f'\nWrote {len(metrics)} rows -> {metrics_path}')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
