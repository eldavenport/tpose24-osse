#!/usr/bin/env python
"""
run_experiment_3.py — COMPUTE phase of the circling-glider experiment.

Reads the transp_cons bounding box ONCE into memory, then for every circle diameter
builds the fixed-disk TRUTH (and a distribution cloud), and for every (N gliders x
diameter) circling array samples the moving gliders and forms the plane-fit w and the
vertical heat-flux estimate.  Everything downstream (figures) reads the caches only.

Caches (experiment_3/cache/):
  circle_d{diam}_disk.nc     fixed-disk truth for a diameter (shared by every N):
                             w_true (interfaces), and on the obs axis eddy_flux <w'T'>,
                             adv_total <w dT/dz>, adv_eddy, wbar, Tbar, plus the
                             instantaneous disk-point-mean eddy-flux series
                             wT/wU/wV/uT/vT/uv for the distribution fits.
  circle_d{diam}_cloud.nc    the disk point cloud (U,V,T,W), time-subsampled x3, for
                             the array-vs-truth distribution comparison.
  circle_n{n}_d{diam}_array.nc   what the circling array measures: U,V,T at the moving
                             gliders (time,glider,obs_depth), the plane-fit w_est
                             (interfaces) + w_est_mid (obs axis), and the array-mean
                             glider T (Tbar_glider).

Metrics (experiment_3/data/):
  metrics.csv       one row per (N, diameter): w-skill (w_est vs disk-mean w) and
                    heat-flux skill (eddy flux w'T' and advective heating w dT/dz vs
                    the disk truth) at the KEY_DEPTHS.

Re-running skips caches already on disk; delete cache/ (and data/metrics.csv) to redo.

Run (from the tpose env):  python experiment_3/run_experiment_3.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')            # silence xmitgcm's available_diagnostics.log notice

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circle_common as C            # noqa: E402
sys.path.insert(0, C.REPO)
import osse_tools as ot              # noqa: E402

TIME_SUB = 3                          # time-subsample factor for the distribution cloud
FLUX_PAIRS = [('W', 'T', 'wT'), ('W', 'U', 'wU'), ('W', 'V', 'wV'),
              ('U', 'T', 'uT'), ('V', 'T', 'vT'), ('U', 'V', 'uv')]


def _eddy_flux_series(sel, pairs):
    """Instantaneous disk-point-mean eddy-flux series <a'b'>_disk(time, obs_depth)."""
    out = {}
    for a, b, name in pairs:
        ap = sel[a] - sel[a].mean('time')
        bp = sel[b] - sel[b].mean('time')
        out[name] = (ap * bp).mean('point')
    return xr.Dataset(out)


def build_disk_truth(ds, diam):
    """Fixed-disk truth + distribution cloud for one circle diameter."""
    disk_path = os.path.join(C.CACHE_DIR, f'circle_d{diam}_disk.nc')
    cloud_path = os.path.join(C.CACHE_DIR, f'circle_d{diam}_cloud.nc')
    if os.path.exists(disk_path) and os.path.exists(cloud_path):
        return
    region = C.region_bbox(ds, half_deg=diam / 2.0)
    sel = C.disk_select(region, diam)
    npts = sel.sizes['point']
    print(f'  disk d{diam}: {npts} grid points inside the disk')

    dTdz = sel['T'].differentiate('obs_depth')
    Wp = sel['W'] - sel['W'].mean('time')
    Tp = sel['T'] - sel['T'].mean('time')
    dTdzp = dTdz - dTdz.mean('time')
    truth = xr.Dataset(dict(
        eddy_flux=(Wp * Tp).mean('point'),
        adv_total=(sel['W'] * dTdz).mean('point'),
        adv_eddy=(Wp * dTdzp).mean('point'),
        wbar=sel['W'].mean('point'),
        Tbar=sel['T'].mean('point'),
    ))
    truth = xr.merge([truth, _eddy_flux_series(sel, FLUX_PAIRS)])
    truth['w_true'] = C.disk_mean_w(ds, diam)          # interface-depth disk-mean w (w-skill target)
    truth.attrs.update(diameter=diam, n_disk_points=int(npts))
    truth.compute().to_netcdf(disk_path)

    cloud = sel[['U', 'V', 'T', 'W']].isel(time=slice(None, None, TIME_SUB))
    cloud = cloud.reset_index('point').compute()
    cloud.attrs.update(diameter=diam, n_disk_points=int(npts), time_sub=TIME_SUB)
    cloud.to_netcdf(cloud_path)
    print(f'  wrote disk + cloud (d{diam})')


def build_array(ds, n, diam):
    """Moving-glider samples + plane-fit w + heat-flux inputs for one (N, diameter)."""
    arr_path = os.path.join(C.CACHE_DIR, f'{C.config_name(n, diam)}_array.nc')
    if os.path.exists(arr_path):
        return
    lat_da, lon_da = C.glider_positions(n, diam, ds.time.values)
    samp = C.sample_moving(ds, lat_da, lon_da,
                           vars=('UVEL', 'VVEL', 'THETA')).compute()
    wf = C.compute_w_planefit_moving(samp[['U', 'V']])
    w_est = wf['w_est']
    z = samp.obs_depth.values
    w_est_mid = w_est.interp(depth=xr.DataArray(z, dims='obs_depth',
                                                coords={'obs_depth': z}))
    w_est_mid = w_est_mid.drop_vars([c for c in ('depth',) if c in w_est_mid.coords])
    out = xr.Dataset(dict(
        U=samp['U'], V=samp['V'], T=samp['T'],
        w_est=w_est, w_est_mid=w_est_mid,
        Tbar_glider=samp['T'].mean('glider'),
    ))
    out.attrs.update(n_gliders=n, diameter=diam)
    out.compute().to_netcdf(arr_path)
    print(f'  wrote array {C.config_name(n, diam)}')


def _metrics_row(n, diam):
    """w-skill + heat-flux-skill row for one (N, diameter) from the caches."""
    arr = xr.open_dataset(os.path.join(C.CACHE_DIR, f'{C.config_name(n, diam)}_array.nc'))
    disk = xr.open_dataset(os.path.join(C.CACHE_DIR, f'circle_d{diam}_disk.nc'))

    row = dict(n_gliders=n, diameter=diam, config=C.config_name(n, diam),
               n_disk_points=int(disk.attrs.get('n_disk_points', -1)),
               min_depth=C.MIN_DEPTH, max_depth=C.MAX_DEPTH)

    # --- vertical velocity: plane-fit w_est vs disk-mean truth ---
    wm = ot.w_skill_metrics(arr.w_est, disk.w_true)
    # prefix with 'w_', but leave keys that already carry it (w_est_std, ...) unchanged
    row.update({(k if k.startswith('w_') else f'w_{k}'): v for k, v in wm.items()})
    row['w_std_ratio'] = (wm['w_est_std'] / wm['w_model_std']
                          if wm['w_model_std'] else np.nan)
    row['w_frac_mean_bias'] = (wm['mean_bias'] / wm['w_model_mean']
                               if wm['w_model_mean'] else np.nan)

    # --- eddy heat flux w'T' and advective heating w dT/dz (array vs disk truth) ---
    # heat_flux_skill_by_depth wants a 'depth' dim; caches keep 'obs_depth' (for the
    # distribution code), so rename just for the skill calls.
    def _d(da):
        return da.rename({'obs_depth': 'depth'}) if 'obs_depth' in da.dims else da
    F_est = _d(ot.flux_total_and_eddy(arr.w_est_mid, arr.Tbar_glider)['eddy'])
    A_est = _d(ot.advective_heating(arr.w_est_mid, arr.Tbar_glider)['total'])
    F_true = _d(disk.eddy_flux)
    A_true = _d(disk.adv_total)
    B = ot.heat_flux_skill_by_depth(F_est, F_true)
    BA = ot.heat_flux_skill_by_depth(A_est, A_true)
    for zk in C.KEY_DEPTHS:
        b = B.sel(depth=-zk, method='nearest')
        a = BA.sel(depth=-zk, method='nearest')
        s = ot.dist_stats(F_est.sel(depth=-zk, method='nearest').values,
                          F_true.sel(depth=-zk, method='nearest').values)
        sa = ot.dist_stats(A_est.sel(depth=-zk, method='nearest').values,
                           A_true.sel(depth=-zk, method='nearest').values)
        row.update({
            f'flux_corr_{zk}': float(b.corr), f'flux_std_ratio_{zk}': float(b.std_ratio),
            f'flux_mean_bias_{zk}': float(b.mean_bias), f'flux_js_{zk}': s['js'],
            f'adv_corr_{zk}': float(a.corr), f'adv_std_ratio_{zk}': float(a.std_ratio),
            f'adv_mean_bias_{zk}': float(a.mean_bias), f'adv_js_{zk}': sa['js'],
        })
    arr.close(); disk.close()
    return row


def main():
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    os.makedirs(C.DATA_DIR, exist_ok=True)

    need = [(n, d, nm) for n, d, nm in C.all_configs()
            if not os.path.exists(os.path.join(C.CACHE_DIR, f'{nm}_array.nc'))]
    need_disk = [d for d in C.DIAMETERS
                 if not (os.path.exists(os.path.join(C.CACHE_DIR, f'circle_d{d}_disk.nc'))
                         and os.path.exists(os.path.join(C.CACHE_DIR, f'circle_d{d}_cloud.nc')))]

    if need or need_disk:
        print('loading bbox into memory ...')
        ds = C.load_bbox_memory()
        print(f'bbox loaded: {dict(ds.sizes)}')
        for d in C.DIAMETERS:
            print(f'diameter {d}:')
            build_disk_truth(ds, d)
            for n in C.N_GLIDERS:
                build_array(ds, n, d)

    rows = [_metrics_row(n, d) for n, d, _ in C.all_configs()]
    metrics = pd.DataFrame(rows).sort_values(['diameter', 'n_gliders']).reset_index(drop=True)
    metrics_path = os.path.join(C.DATA_DIR, 'metrics.csv')
    metrics.to_csv(metrics_path, index=False)
    print(f'\nWrote {len(metrics)} rows -> {metrics_path}')
    cols = ['n_gliders', 'diameter', 'w_corr', 'w_std_ratio', 'w_frac_mean_bias',
            'flux_corr_50', 'adv_corr_50']
    cols = [c for c in cols if c in metrics.columns]
    print(metrics[cols].to_string(index=False))


if __name__ == '__main__':
    main()
