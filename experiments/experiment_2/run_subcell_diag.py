#!/usr/bin/env python
"""
run_subcell_diag.py — diagnostics for the UNRESOLVED sub-cell covariance that the
array can never measure.

The true vertical heat flux splits exactly into a resolved (cell-scale) part the array
can build from area-means, plus a sub-cell part it cannot:

    <w'T'>_cell,t  =  <w'>_x <T'>_x            +  <w'' T''>_x,t
       total           resolved <W'><T'>          sub-cell covariance (unresolved)

where '' is the deviation from the cell (spatial) mean of the temporal anomaly. The
same holds for the advective heating w*dz T. This script produces two things about
that sub-cell part, both on the SAME model as the heat-flux run (transp_cons):

  1. 2-D maps over the equatorial box of the temporal VARIANCE of each flux component
     (w', T', dz T') and of the mean fluxes (w'T', w*dz T), at 25/50/70 m — so we can
     see WHERE the variance in w vs T lives relative to the array footprint.
     -> cache/subcell_maps.pkl

  2. For the representative 1 deg cells (diamond/hexagon/square), the pooled sub-cell
     deviations w''(x,t), T''(x,t), (dz T)''(x,t) at 50 m — for a scatter showing
     whether the sub-cell covariance is driven by variations in w or in T.
     -> cache/subcell_scatter.pkl

Run (tpose env):  python experiments/experiment_2/run_subcell_diag.py [mode]
    mode in {all (default), maps, scatter}
"""
import os
import sys
import pickle

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import osse_tools as ot            # noqa: E402
import run_heat_flux as rhf        # noqa: E402  (RUN_DIR, ITERS, SPINUP_END, params)

CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'
BOX_LON = slice(217, 223)
BOX_LAT = slice(-3, 3)
DEPTHS = (25, 50, 75)
DZ = 2.0                                   # m, half-stencil for dz T
# depth stack (m) averaged for the 0–80 m depth-average context maps
DEPTHAVG_LEVELS = tuple(float(z) for z in np.arange(4.0, 80.1, 4.0))
SCAT_DEPTH = 50                            # m, depth for the sub-cell scatter
SCAT_SHAPES = ['equator_1deg', 'equator_hex1deg', 'equator_sq1deg']
SCAT_WIDTH = 0.5                           # representative diameter (1.0 deg)
SCAT_TSTRIDE = 3                           # subsample time for the scatter sample


def _box(ds):
    return ds.sel(XC=BOX_LON, YC=BOX_LAT)


def _fields_at(ds, d):
    """w, T and dz T over the box at depth d (dims time, YC, XC)."""
    b = _box(ds)
    W = b.WVEL.interp(Zl=-float(d))
    T = b.THETA.interp(Z=-float(d))
    Tsh = b.THETA.interp(Z=-float(d - DZ))     # shallower
    Tdp = b.THETA.interp(Z=-float(d + DZ))     # deeper
    dTdz = (Tsh - Tdp) / (2 * DZ)              # dT/dz, z upward positive
    return W, T, dTdz


def _maps_at(ds, d):
    """Time-mean fields and temporal variances over the box at depth d (deg C, m/s)."""
    W, T, g = _fields_at(ds, d)
    Wp, Tp, gp = W - W.mean('time'), T - T.mean('time'), g - g.mean('time')
    return xr.Dataset(dict(
        var_w=(Wp ** 2).mean('time'),          # (m/s)^2
        var_T=(Tp ** 2).mean('time'),          # (deg C)^2
        var_dTdz=(gp ** 2).mean('time'),       # (deg C/m)^2
        eddy_flux=(Wp * Tp).mean('time'),      # deg C m/s
        adv_heating=(W * g).mean('time'),      # deg C/s
        wbar=W.mean('time'), Tbar=T.mean('time'),
    ))


def compute_maps(ds):
    """Temporal variance of the flux components and mean fluxes, per depth, plus a
    0–80 m depth-average ('depthavg', the mean of the per-depth maps over DEPTHAVG_LEVELS)."""
    out = {d: _maps_at(ds, d).compute() for d in DEPTHS}
    acc = None
    for d in DEPTHAVG_LEVELS:
        m = _maps_at(ds, d)
        acc = m if acc is None else acc + m
    out['depthavg'] = (acc / len(DEPTHAVG_LEVELS)).compute()
    return out


def compute_scatter(ds):
    """Pooled sub-cell deviations w'', T'', (dz T)'' at SCAT_DEPTH for each shape."""
    W, T, g = (x.compute() for x in _fields_at(ds, SCAT_DEPTH))
    Wp, Tp, gp = W - W.mean('time'), T - T.mean('time'), g - g.mean('time')
    xc, yc = W.XC.values, W.YC.values
    out = {}
    for s in SCAT_SHAPES:
        path = os.path.join(rhf.EXP1, 'configs', 'equator', f'{s}_w{SCAT_WIDTH:g}.json')
        (_, pos), = ot.load_cells(path)                       # single equator cell
        mask = ot._convex_hull_mask(xc, yc, pos)
        m = xr.DataArray(mask, dims=('YC', 'XC'), coords={'YC': yc, 'XC': xc})
        # sub-cell deviation = temporal anomaly minus its cell (spatial) mean each time
        def subcell(a):
            am = a.where(m)
            return (am - am.mean(('YC', 'XC'))).where(m)
        wpp = subcell(Wp).stack(pt=('YC', 'XC')).isel(time=slice(None, None, SCAT_TSTRIDE))
        Tpp = subcell(Tp).stack(pt=('YC', 'XC')).isel(time=slice(None, None, SCAT_TSTRIDE))
        gpp = subcell(gp).stack(pt=('YC', 'XC')).isel(time=slice(None, None, SCAT_TSTRIDE))
        wv, Tv, gv = (a.values.ravel() for a in (wpp, Tpp, gpp))
        ok = np.isfinite(wv) & np.isfinite(Tv) & np.isfinite(gv)
        wv, Tv, gv = wv[ok], Tv[ok], gv[ok]
        out[s] = dict(
            wpp=wv, Tpp=Tv, gpp=gv,
            sig_w=float(wv.std()), sig_T=float(Tv.std()), sig_g=float(gv.std()),
            cov_wT=float(np.mean(wv * Tv)), corr_wT=float(np.corrcoef(wv, Tv)[0, 1]),
            cov_wg=float(np.mean(wv * gv)), corr_wg=float(np.corrcoef(wv, gv)[0, 1]),
            npts=int(wv.size),
        )
        print(f'  {s}: n={wv.size}  cov(w,T)={out[s]["cov_wT"]:.3e}  '
              f'corr={out[s]["corr_wT"]:+.2f}')
    return out


def main(mode='all'):
    os.makedirs(CACHE_DIR, exist_ok=True)
    ds = ot.load_model(rhf.RUN_DIR, rhf.ITERS).sel(time=slice(rhf.SPINUP_END, None))
    print(f'Model loaded: {ds.sizes["time"]} timesteps after spin-up')
    if mode in ('all', 'maps'):
        with open(os.path.join(CACHE_DIR, 'subcell_maps.pkl'), 'wb') as f:
            pickle.dump(compute_maps(ds), f)
        print('wrote subcell_maps.pkl')
    if mode in ('all', 'scatter'):
        with open(os.path.join(CACHE_DIR, 'subcell_scatter.pkl'), 'wb') as f:
            pickle.dump(compute_scatter(ds), f)
        print('wrote subcell_scatter.pkl')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
