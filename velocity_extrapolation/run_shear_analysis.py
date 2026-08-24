"""
Does the constant-shear surface extrapolation in ``compute_w_planefit`` make sense?

The plane-fit w estimator samples U, V from ``min_depth`` (8 m) downward and, before
fitting, extrapolates into 0-8 m with ``extrapolate_currents_to_surface``: it measures the vertical
shear dU/dz between the two shallowest sampled levels (~9-11 m) and assumes that shear
is CONSTANT all the way to the surface. This script tests that assumption against the
TPOSE24 truth in the equatorial box 2°S-2°N, 142-138°W (218-222°E).

Vertical shear here = dU/dz, dV/dz of the horizontal currents (per meter), and the
shear magnitude |S| = sqrt((dU/dz)^2 + (dV/dz)^2). U (on XG) and V (on YG) are first
interpolated to the tracer centers (XC, YC); shear is a centered difference between
adjacent Z cell-centers, placed at the interface midpoint. The top ~20 m has 1 m layers,
so the near-surface shear is well resolved.

Two phases (so figures can be restyled without re-reading the model):
  * COMPUTE - reads the model once, streams time/space reductions, caches small arrays
    (band-mean +/- std profiles; time-mean/max/min shear maps; select-depth fields).
  * PLOT    - reads the cache and renders every figure; no model access.

Figures (velocity_extrapolation/figs/):
  1. shear_profiles_latbands   - |S|, dU/dz, dV/dz profiles (0-20 m) by 0.5° lat band,
                                  mean +/- std; the ~10 m "sampled shear" the estimator
                                  uses is marked, so you can see whether it equals the
                                  0-8 m shear it is extrapolated over.
  2. velocity_profiles_latbands- U, V profiles by lat band: the curvature the constant-
                                  shear extrapolation ignores.
  3. shear_profiles_lonbands   - same as (1) but banded by longitude (checks zonal
                                  uniformity of the shear structure).
  4. maps_depthstats           - depth-mean / depth-max / depth-min of time-mean |S|
                                  over 0-20 m (|S| range within the column).
  5. maps_8m_timestats         - time-mean / time-max / time-min |S| at 8 m.
  6. maps_shear_differences    - time-mean shear CHANGE with depth: |S|(8m)-|S|(surf),
                                  |S|(20m)-|S|(8m), and the dU/dz, dV/dz versions.
  7. maps_extrap_error         - surface-velocity error (m/s) of the constant shear
                                  vs. a longer-range (8-20 m) shear average, in the mean
                                  state - i.e. whether a deeper shear average is better.

Time evolution + a statistical extrapolation rule (do snapshots look like the mean, and
is there a rule that maps the 8 m current to the shallower current at each observation?):
  8. shear_hovmoller           - box-mean |S|, dU/dz, dV/dz vs (time, depth): whether the
                                  surface intensification is persistent or intermittent.
  9. profiles_snapshots        - each daily box-mean profile (thin) vs the time mean: does
                                  every snapshot share the mean's curved shape?
 10. regression_rule           - coefficients of u(z),v(z) regressed on u(8 m),v(8 m)
                                  across time (self gain, cross gain / veering) and skill.
 11. regression_skill          - held-out RMS reconstruction error vs depth: persistence,
                                  constant shear, and the regression onto u(8 m),v(8 m) only,
                                  then + the measured 8 m shear, then + u,v,shear at 4 depths.
 12. regression_matrix_maps    - surface map of every component of the shear-augmented rule's
                                  2x4 transfer matrix: targets (u,v) x predictors (u(8 m),
                                  v(8 m), ∂u/∂z, ∂v/∂z); gain ≈ 1 on the diagonal, ≈ 0 cross.
 13. regression_r2_maps        - reconstruction skill (vector R² of U,V, averaged over 0-8 m)
                                  per method, all on one shared colorscale.
 14. regression_cumerr_maps    - CUMULATIVE (0-8 m depth-integrated) velocity error per method
                                  — the transport error whose divergence sets the w error;
                                  errors integrate, so a coherent bias is penalised here.

Usage:
    python run_shear_analysis.py [mode]      mode in {all (default), compute, plot}
"""

import os
import sys
import time
import pickle

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean.cm as cmo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import osse_tools as ot

# dt60 run (same as run_domain_maps): deltaT=60 s, 3-hourly diag_state -> iter step 180.
RUN_DIR = '/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3'
ITERS = list(range(180, 129240 + 180, 180))
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figs')
CACHE = os.path.join(HERE, 'shear_cache.pkl')

# analysis box (142-138°W, 2°S-2°N) and the depth range examined
LON0, LON1 = 218.0, 222.0
LAT0, LAT1 = -2.0, 2.0
BUF = 0.2                     # halo for the staggered->center interpolation
ZMAX = -22.0                 # examine the top ~20 m (a couple of m of headroom)

# The estimator nominally starts sampling at min_depth = 8 m. We evaluate the constant-shear
# extrapolation on the NATIVE model levels (1 m thick here), not the 2 m glider spacing: it
# covers the levels shallower than 8 m and is anchored at the shallowest level at/below 8 m (-8.5 m),
# with the "top shear" taken between that level and the next one down (-8.5, -9.5 -> -9 m).
MIN_DEPTH = 8.0
SAMPLED_SHEAR_Z = -9.0     # native interface between the -8.5 and -9.5 m model levels

# representative depths for the single-level maps / differences
Z_SURF = -1.0                # shallowest resolved interface (0-1 m shear ~ "surface")
Z_8M = -8.0
Z_20M = -20.0

LAT_BAND_EDGES = np.arange(LAT0, LAT1 + 1e-6, 0.5)     # 0.5° lat bands
LON_BAND_EDGES = np.arange(LON0, LON1 + 1e-6, 1.0)     # 1° lon bands


# --------------------------------------------------------------------------- compute
def _load_centered_uv():
    """U, V interpolated to tracer centers (XC, YC), top ~22 m, over the box+halo.

    Returns a lazy Dataset with U, V on dims (time, Z, YC, XC).
    """
    ds = ot.load_model(RUN_DIR, ITERS)
    zsel = slice(0.0, ZMAX)
    u = ds.UVEL.sel(Z=zsel, XG=slice(LON0 - BUF, LON1 + BUF),
                    YC=slice(LAT0 - BUF, LAT1 + BUF))
    v = ds.VVEL.sel(Z=zsel, YG=slice(LAT0 - BUF, LAT1 + BUF),
                    XC=slice(LON0 - BUF, LON1 + BUF))
    xc = ds.XC.sel(XC=slice(LON0, LON1)).values
    yc = ds.YC.sel(YC=slice(LAT0, LAT1)).values
    # staggered -> center by linear interpolation onto the tracer coordinates
    U = u.interp(XG=xc, YC=yc).rename({'XG': 'XC'})
    V = v.interp(YG=yc, XC=xc).rename({'YG': 'YC'})
    return xr.Dataset({'U': U, 'V': V})


def _shear(uv):
    """dU/dz, dV/dz, |S| at interface midpoints (dims time, Zi, YC, XC)."""
    z = uv.Z.values
    zi = 0.5 * (z[:-1] + z[1:])
    dz = xr.DataArray(np.diff(z), dims='Zi', coords={'Zi': zi})
    su = (uv.U.diff('Z').rename({'Z': 'Zi'}).assign_coords(Zi=zi)) / dz
    sv = (uv.V.diff('Z').rename({'Z': 'Zi'}).assign_coords(Zi=zi)) / dz
    smag = np.sqrt(su ** 2 + sv ** 2)
    return xr.Dataset({'SU': su, 'SV': sv, 'Smag': smag})


def _divergence(U, V):
    """Horizontal divergence dU/dx + dV/dy (1/s) for center-grid DataArrays (..,YC,XC).

    U and V must share the (YC, XC) grid (both already interpolated to tracer centers).
    Longitude spacing is scaled by cos(lat) of the box center; the box is small so the
    meridional variation of that factor is negligible.
    """
    deg = np.pi / 180 * 6371000.0
    lat_c = float(U.YC.mean())
    x_m = (U.XC.values - U.XC.values.mean()) * np.cos(np.radians(lat_c)) * deg
    y_m = (U.YC.values - U.YC.values.mean()) * deg
    dudx = np.gradient(U.values, x_m, axis=U.dims.index('XC'))
    dvdy = np.gradient(V.values, y_m, axis=V.dims.index('YC'))
    return xr.DataArray(dudx + dvdy, dims=U.dims, coords=U.coords)


def _band_profiles(field, coord, edges, zdim='Zi'):
    """Per-band mean and std profiles over (time + the two horizontal dims).

    field: (time, zdim, YC, XC); coord in {'YC','XC'}; edges: band boundaries.
    Returns dict: centers (nb,), z (nz,), mean (nb, nz), std (nb, nz).
    """
    centers, means, stds = [], [], []
    other = 'XC' if coord == 'YC' else 'YC'
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = field.sel({coord: slice(lo, hi)})
        centers.append(0.5 * (lo + hi))
        means.append(sub.mean(['time', coord, other]).values)
        stds.append(sub.std(['time', coord, other]).values)
    return dict(centers=np.array(centers), z=field[zdim].values,
                mean=np.array(means), std=np.array(stds))


def _time_evolution_stats(uv):
    """Daily box-mean profiles vs time, for the Hovmoller and snapshot figures.

    The rest of the script collapses time to a single mean; here we KEEP time so we can
    ask whether every snapshot looks like the mean (a fixed, curved, surface-intensified
    shape) or whether the mean averages over very different profiles. Uses DAILY means
    (removes sub-daily tidal/inertial noise) of the box-mean profile.
    """
    uv_d = uv.resample(time='1D').mean()
    sh_d = _shear(uv_d)                          # shear is linear in U,V -> shear of daily mean
    box = ['YC', 'XC']
    t_days = ((uv_d.time - uv_d.time[0]) / np.timedelta64(1, 'D')).values
    out = {'t_days': t_days, 'z_uv': uv_d.Z.values, 'zi': sh_d.Zi.values}
    # (nz, nday) box-mean profile time series
    for name, fld, zdim in [('U', uv_d.U, 'Z'), ('V', uv_d.V, 'Z'),
                            ('SU', sh_d.SU, 'Zi'), ('SV', sh_d.SV, 'Zi'),
                            ('Smag', sh_d.Smag, 'Zi')]:
        bm = fld.mean(box).transpose(zdim, 'time').values
        out['day_' + name] = bm                  # (nz, nday) each column a daily snapshot
        out['tm_' + name] = bm.mean(axis=1)      # time-mean profile
    return out


def _fit_predict(P, y, Pe, ridge=0.0):
    """Per-point OLS: fit beta from design P (nf,Tf,npts) & target y (Tf,npts); predict on
    Pe (nf,Te,npts). Returns beta (npts,nf) and yhat (Te,npts).

    ``ridge`` (relative) inflates each normal-matrix diagonal by (1+ridge) — scale-invariant
    Tikhonov regularization that stabilises the many collinear predictors of the multi-depth
    fit without materially touching the well-posed low-order fits.
    """
    XtX = np.einsum('itp,jtp->pij', P, P)
    Xty = np.einsum('itp,tp->pi', P, y)
    if ridge:
        d = np.arange(P.shape[0])
        XtX[:, d, d] *= (1.0 + ridge)
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]     # (npts,nf)
    yhat = np.einsum('itp,pi->tp', Pe, beta)
    return beta, yhat


def _regression_extrap(uv):
    """Statistical extrapolation rule: at every lat/lon regress u(z), v(z) across TIME onto
    the 8 m currents u(8 m), v(8 m).

    For each fill level z (0-8 m) and each point we fit the linear transfer
        u(z,t) = a_u + b_uu·u8(t) + b_uv·v8(t)
        v(z,t) = a_v + b_vu·u8(t) + b_vv·v8(t)
    i.e. a 2x2 matrix + offset that lets the surface current be amplified (b_..>1) and
    rotated (cross terms) relative to 8 m. This is the "rule you could apply at each
    observation": measure u8,v8 and predict the shallower current.

    Every rule is LOCAL (per lat/lon: each column's curve is predicted from its own 8 m time
    series). Coefficients are reported from a full-record fit; skill (R², RMS) is scored on
    HELD-OUT halves of the record via BLOCK cross-validation (train one contiguous half, score
    the other, both ways) so it tests generalization to an independent period. Baselines
    compared per depth: persistence (hold the 8 m value) and the instantaneous constant-shear
    extrapolation; richer regressions add the measured 8 m shear, then u,v,shear at 4 depths.
    """
    z = uv.Z.values
    ia = int(np.argmin(np.abs(z + 8.5)))         # anchor level -8.5 m (the "8 m" obs)
    ib = ia + 1                                   # -9.5 m: level below, for constant shear
    fill_idx = np.where(z > -8.0)[0]             # 0-8 m fill levels (-0.5 .. -7.5)
    zf = z[fill_idx]
    U = uv.U.values
    V = uv.V.values
    T, _, ny, nx = U.shape
    npts = ny * nx
    Ur = U.reshape(T, U.shape[1], npts)
    Vr = V.reshape(T, V.shape[1], npts)
    u8 = Ur[:, ia, :]                            # (T,npts)
    v8 = Vr[:, ia, :]
    su = (Ur[:, ia, :] - Ur[:, ib, :]) / (z[ia] - z[ib])   # instantaneous dU/dz at -9 m
    sv = (Vr[:, ia, :] - Vr[:, ib, :]) / (z[ia] - z[ib])

    # Held-out scheme: 2-fold BLOCK cross-validation in time — train on one contiguous half
    # of the record and score the other (both ways), so the skill tests generalization to an
    # independent period rather than interpolation between autocorrelated 3-hourly neighbours.
    tA = np.arange(0, T // 2)          # first ~6 weeks
    tB = np.arange(T // 2, T)          # last  ~6 weeks
    ones = np.ones((T, npts))
    P_all = np.stack([ones, u8, v8], axis=0)                       # (3,T,npts): u8,v8 only
    P_A, P_B = P_all[:, tA, :], P_all[:, tB, :]
    # augmented design: also give the regression the MEASURED 8 m shear (the same
    # instantaneous ∂u/∂z the constant-shear extrapolation uses). This lets a statistical
    # rule both project the instantaneous shear AND learn to amplify it toward the surface.
    Ps_all = np.stack([ones, u8, v8, su, sv], axis=0)              # (5,T,npts)
    Ps_A, Ps_B = Ps_all[:, tA, :], Ps_all[:, tB, :]
    # multi-depth design: u, v AND their local shear at four sampled depths spanning ~8-14 m,
    # so the fit sees the SHAPE of the observed profile below 8 m, not just the anchor level.
    dep_idx = [j for j in (ia, ia + 2, ia + 4, ia + 5) if j + 1 < U.shape[1]]  # -8.5,-10.5,-12.75,-14.25
    feats = [ones]
    for j in dep_idx:
        sj_u = (Ur[:, j, :] - Ur[:, j + 1, :]) / (z[j] - z[j + 1])
        sj_v = (Vr[:, j, :] - Vr[:, j + 1, :]) / (z[j] - z[j + 1])
        feats += [Ur[:, j, :], Vr[:, j, :], sj_u, sj_v]
    Pm_all = np.stack(feats, axis=0)                               # (1+4*ndep, T, npts)
    Pm_A, Pm_B = Pm_all[:, tA, :], Pm_all[:, tB, :]
    dep_z = [float(z[j]) for j in dep_idx]

    nzf = len(fill_idx)
    # box-mean coefficient / skill profiles
    prof = {k: np.zeros(nzf) for k in
            ('b_uu', 'b_uv', 'b_vu', 'b_vv', 'a_u', 'a_v',
             'r2u_loc', 'r2v_loc',
             'r2u_shear', 'r2v_shear', 'c_su', 'c_sv', 'geom_dz')}
    # RMS(depth) per method, m/s, for U and V. Every regression is LOCAL (per lat/lon: each
    # column's 0-8 m curve is predicted from its OWN 8 m time series).
    methods = ('persist', 'cshear', 'reg_loc', 'reg_shear', 'reg_multi')
    rmse = {f'{m}_{c}': np.zeros(nzf) for m in methods for c in ('u', 'v')}
    surf = {}                                    # surface-level (-0.5 m) maps
    # per-point vector (U,V-combined) R² accumulated over the 0-8 m fill layers, per method
    r2v_acc = {m: np.zeros(npts) for m in methods}
    # running VERTICAL INTEGRAL of the signed velocity error over the 0-8 m layers (m²/s),
    # per method — for w the layer errors integrate (a sign-coherent bias accumulates, a
    # zero-mean wiggle cancels), so this differs from the depth-averaged skill above.
    Iu = {m: np.zeros((T, npts), np.float32) for m in methods}
    Iv = {m: np.zeros((T, npts), np.float32) for m in methods}
    DZ_FILL = 1.0                                # native fill layers are 1 m thick

    def _r2_pt(y, yhat):
        res = ((y - yhat) ** 2).sum(0)
        tot = ((y - y.mean(0)) ** 2).sum(0)
        return 1 - res / np.where(tot == 0, np.nan, tot)

    def _r2_vec(uz, vz, uh, vh):
        """Per-point R² for the velocity VECTOR (U and V residual variance combined)."""
        res = ((uz - uh) ** 2 + (vz - vh) ** 2).sum(0)
        tot = ((uz - uz.mean(0)) ** 2 + (vz - vz.mean(0)) ** 2).sum(0)
        return 1 - res / np.where(tot == 0, np.nan, tot)

    for k, iz in enumerate(fill_idx):
        uz = Ur[:, iz, :]
        vz = Vr[:, iz, :]
        dz = z[iz] - z[ia]                        # >0 (shallower than anchor)
        # --- LOCAL per-point rule: coefficients from the full record ---
        beta_u, _ = _fit_predict(P_all, uz, P_all[:, :1, :])   # coeffs only
        beta_v, _ = _fit_predict(P_all, vz, P_all[:, :1, :])
        prof['a_u'][k] = beta_u[:, 0].mean(); prof['b_uu'][k] = beta_u[:, 1].mean()
        prof['b_uv'][k] = beta_u[:, 2].mean()
        prof['a_v'][k] = beta_v[:, 0].mean(); prof['b_vu'][k] = beta_v[:, 1].mean()
        prof['b_vv'][k] = beta_v[:, 2].mean()
        # --- LOCAL skill: 2-fold CV, held-out predictions cover the whole record ---
        _, uhatB = _fit_predict(P_A, uz[tA], P_B)
        _, uhatA = _fit_predict(P_B, uz[tB], P_A)
        _, vhatB = _fit_predict(P_A, vz[tA], P_B)
        _, vhatA = _fit_predict(P_B, vz[tB], P_A)
        uho = np.empty_like(uz); vho = np.empty_like(vz)
        uho[tA] = uhatA; uho[tB] = uhatB
        vho[tA] = vhatA; vho[tB] = vhatB
        prof['r2u_loc'][k] = np.nanmean(_r2_pt(uz, uho))
        prof['r2v_loc'][k] = np.nanmean(_r2_pt(vz, vho))
        # --- AUGMENTED rule: u8,v8 AND the measured 8 m shear (per-point, CV) ---
        bsu, _ = _fit_predict(Ps_all, uz, Ps_all[:, :1, :])   # full-record coeffs
        bsv, _ = _fit_predict(Ps_all, vz, Ps_all[:, :1, :])
        prof['c_su'][k] = bsu[:, 3].mean()       # learned gain on ∂u/∂z (u-prediction)
        prof['c_sv'][k] = bsv[:, 4].mean()       # learned gain on ∂v/∂z (v-prediction)
        prof['geom_dz'][k] = dz                  # geometric distance constant shear uses
        _, ushB = _fit_predict(Ps_A, uz[tA], Ps_B)
        _, ushA = _fit_predict(Ps_B, uz[tB], Ps_A)
        _, vshB = _fit_predict(Ps_A, vz[tA], Ps_B)
        _, vshA = _fit_predict(Ps_B, vz[tB], Ps_A)
        usho = np.empty_like(uz); vsho = np.empty_like(vz)
        usho[tA] = ushA; usho[tB] = ushB
        vsho[tA] = vshA; vsho[tB] = vshB
        prof['r2u_shear'][k] = np.nanmean(_r2_pt(uz, usho))
        prof['r2v_shear'][k] = np.nanmean(_r2_pt(vz, vsho))
        # --- MULTI-DEPTH rule: u,v + shear at four depths below 8 m (ridge-stabilised, CV) ---
        _, umB = _fit_predict(Pm_A, uz[tA], Pm_B, ridge=1e-6)
        _, umA = _fit_predict(Pm_B, uz[tB], Pm_A, ridge=1e-6)
        _, vmB = _fit_predict(Pm_A, vz[tA], Pm_B, ridge=1e-6)
        _, vmA = _fit_predict(Pm_B, vz[tB], Pm_A, ridge=1e-6)
        umo = np.empty_like(uz); vmo = np.empty_like(vz)
        umo[tA] = umA; umo[tB] = umB
        vmo[tA] = vmA; vmo[tB] = vmB
        # --- baselines and RMS(depth), RMS over time & space (m/s) ---
        ucs = u8 + su * dz; vcs = v8 + sv * dz
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
        rmse['persist_u'][k] = rms(uz - u8);  rmse['persist_v'][k] = rms(vz - v8)
        rmse['cshear_u'][k] = rms(uz - ucs);  rmse['cshear_v'][k] = rms(vz - vcs)
        rmse['reg_loc_u'][k] = rms(uz - uho); rmse['reg_loc_v'][k] = rms(vz - vho)
        rmse['reg_shear_u'][k] = rms(uz - usho); rmse['reg_shear_v'][k] = rms(vz - vsho)
        rmse['reg_multi_u'][k] = rms(uz - umo); rmse['reg_multi_v'][k] = rms(vz - vmo)
        # accumulate per-point vector R² (depth-averaged after the loop)
        r2v_acc['persist'] += _r2_vec(uz, vz, u8, v8)
        r2v_acc['cshear'] += _r2_vec(uz, vz, ucs, vcs)
        r2v_acc['reg_loc'] += _r2_vec(uz, vz, uho, vho)
        r2v_acc['reg_shear'] += _r2_vec(uz, vz, usho, vsho)
        r2v_acc['reg_multi'] += _r2_vec(uz, vz, umo, vmo)
        # accumulate the signed error into the vertical integral (× layer thickness)
        for m, uh, vh in (('persist', u8, v8), ('cshear', ucs, vcs),
                          ('reg_loc', uho, vho),
                          ('reg_shear', usho, vsho), ('reg_multi', umo, vmo)):
            Iu[m] += ((uz - uh) * DZ_FILL).astype(np.float32)
            Iv[m] += ((vz - vh) * DZ_FILL).astype(np.float32)
        if iz == fill_idx[0]:                     # surface level maps
            # full transfer (gain) matrix of the shear-augmented rule at the surface:
            # targets (u,v) x predictors (u8, v8, ∂u/∂z, ∂v/∂z). Shear columns are shown as
            # the RATIO to the geometric extrapolation distance dz (constant shear = 1).
            rs = lambda a: a.reshape(ny, nx)
            surf = {
                'g_uu': rs(bsu[:, 1]), 'g_uv': rs(bsu[:, 2]),
                'g_usu': rs(bsu[:, 3] / dz), 'g_usv': rs(bsu[:, 4] / dz),
                'g_vu': rs(bsv[:, 1]), 'g_vv': rs(bsv[:, 2]),
                'g_vsu': rs(bsv[:, 3] / dz), 'g_vsv': rs(bsv[:, 4] / dz),
                'z': float(z[iz]), 'dz': float(dz),
            }
    r2maps = {m: (r2v_acc[m] / nzf).reshape(ny, nx) for m in methods}
    # time-RMS of the depth-integrated velocity-error VECTOR over 0-8 m (m²/s): the transport
    # error whose horizontal divergence is the w error the layer contributes.
    cumemaps = {m: np.sqrt(np.mean(Iu[m] ** 2 + Iv[m] ** 2, axis=0)).reshape(ny, nx)
                for m in methods}
    return {'zf': zf, 'z_anchor': float(z[ia]), 'prof': prof, 'rmse': rmse,
            'surf': surf, 'dep_z': dep_z, 'r2maps': r2maps, 'cumemaps': cumemaps}


def compute():
    t0 = time.time()
    raw = os.path.join(HERE, 'uv_region.nc')
    if os.path.exists(raw):
        print(f'loading cached U, V region from {raw} ...', flush=True)
        uv = xr.open_dataset(raw).load()
    else:
        print('loading + interpolating U, V (single pass into memory) ...', flush=True)
        uv = _load_centered_uv().compute()      # ~a few GB; box is small
        uv.to_netcdf(raw)
    print(f'  U,V {uv.U.shape} ready in {time.time() - t0:.0f} s', flush=True)
    sh = _shear(uv)                              # in-memory -> all reductions are fast
    # drop auxiliary MITgcm grid coords (drF, rA, ...) so depth-selected fields merge
    keep = {'time', 'Z', 'Zi', 'YC', 'XC'}
    uv = uv.drop_vars([c for c in uv.coords if c not in keep])
    sh = sh.drop_vars([c for c in sh.coords if c not in keep])
    zi = sh.Zi.values

    out = {'zi': zi, 'z_uv': uv.Z.values,
           'box': (LON0, LON1, LAT0, LAT1),
           'sampled_shear_z': SAMPLED_SHEAR_Z,
           'min_depth': MIN_DEPTH}

    # --- band profiles (lat and lon), for SU, SV, Smag, and the raw velocities U, V.
    # Spread (+/- std) is over DAILY MEANS x horizontal points: day-to-day + spatial
    # variability, not sub-daily tidal/inertial noise (matches the "daily-average" ask).
    print('band profiles (daily means) ...', flush=True)
    sh_d = sh.resample(time='1D').mean()
    uv_d = uv.resample(time='1D').mean()
    prof = {}
    for name, fld, zdim in [('SU', sh_d.SU, 'Zi'), ('SV', sh_d.SV, 'Zi'),
                            ('Smag', sh_d.Smag, 'Zi'),
                            ('U', uv_d.U, 'Z'), ('V', uv_d.V, 'Z')]:
        prof['lat_' + name] = _band_profiles(fld, 'YC', LAT_BAND_EDGES, zdim)
        prof['lon_' + name] = _band_profiles(fld, 'XC', LON_BAND_EDGES, zdim)
    out['prof'] = prof

    # --- 2D maps. sh/uv are already in memory, so these are plain numpy reductions.
    print('map reductions ...', flush=True)
    xc = sh.XC.values
    yc = sh.YC.values

    smag = sh.Smag
    su_tm = sh.SU.mean('time')
    sv_tm = sh.SV.mean('time')
    smag_tm = smag.mean('time')                      # time-mean |S|(Zi, YC, XC)
    z020 = slice(0.0, ZMAX)                           # 0-20 m interfaces
    smag8 = smag.sel(Zi=Z_8M, method='nearest')      # |S| at 8 m, all times

    uv_tm = uv.mean('time')
    # native-level geometry (no interpolation): anchor at the shallowest model level
    # at/below 8 m, extrapolate up to the surface-most level using the native top shear.
    zc = uv_tm.Z.values
    z_fill = zc[zc > -MIN_DEPTH]          # levels in 0-8 m: -0.5 .. -7.5
    za = float(zc[zc <= -MIN_DEPTH][0])   # -8.5 m: extrapolation anchor (shallowest sampled level)
    z_surf = float(zc[0])                 # -0.5 m: surface-most model level
    reach = z_surf - za                   # 8 m: anchor -> surface extrapolation distance

    u_anchor = uv_tm.U.sel(Z=za)
    v_anchor = uv_tm.V.sel(Z=za)
    u_surf = uv_tm.U.sel(Z=z_surf)
    v_surf = uv_tm.V.sel(Z=z_surf)
    su_const = su_tm.sel(Zi=SAMPLED_SHEAR_Z, method='nearest')  # native shear at -9 m
    sv_const = sv_tm.sel(Zi=SAMPLED_SHEAR_Z, method='nearest')
    su_deep = su_tm.sel(Zi=slice(za, ZMAX)).mean('Zi')          # native 8.5-20 m mean shear
    sv_deep = sv_tm.sel(Zi=slice(za, ZMAX)).mean('Zi')

    # extrapolation-error (mean state): predicted surface vel = U(-8.5 m) + shear*8, two
    # shear choices - the native top (-9 m) shear vs the native 8.5-20 m mean shear
    uerr_const = (u_anchor + su_const * reach) - u_surf
    verr_const = (v_anchor + sv_const * reach) - v_surf
    uerr_deep = (u_anchor + su_deep * reach) - u_surf
    verr_deep = (v_anchor + sv_deep * reach) - v_surf

    def _v(da):
        return np.asarray(da.values)

    mp = {
        'smag_depthmean': _v(smag_tm.sel(Zi=z020).mean('Zi')),
        'su_depthmean': _v(su_tm.sel(Zi=z020).mean('Zi')),
        'sv_depthmean': _v(sv_tm.sel(Zi=z020).mean('Zi')),
        'smag8_timemean': _v(smag8.mean('time')),
        'smag8_timemax': _v(smag8.max('time')),
        'smag8_timemin': _v(smag8.min('time')),
        'smag_surf': _v(smag_tm.sel(Zi=Z_SURF, method='nearest')),
        'smag_8m': _v(smag_tm.sel(Zi=Z_8M, method='nearest')),
        'smag_20m': _v(smag_tm.sel(Zi=Z_20M, method='nearest')),
        'su_surf': _v(su_tm.sel(Zi=Z_SURF, method='nearest')),
        'su_8m': _v(su_tm.sel(Zi=Z_8M, method='nearest')),
        'sv_surf': _v(sv_tm.sel(Zi=Z_SURF, method='nearest')),
        'sv_8m': _v(sv_tm.sel(Zi=Z_8M, method='nearest')),
        'serr_const': _v(np.sqrt(uerr_const ** 2 + verr_const ** 2)),
        'serr_deep': _v(np.sqrt(uerr_deep ** 2 + verr_deep ** 2)),
    }

    # --- near-surface divergence -> w consequence, WITH vs WITHOUT the constant-shear extrapolation.
    # Every operation (interp, shear, finite-difference divergence, vertical integral)
    # is linear in U, V, so the time-mean error equals the error of the time-mean field;
    # we work from uv_tm. Only the horizontal GRADIENTS of the velocity error feed the
    # divergence, so any spatially uniform part of the U/V bias drops out here.
    print('divergence / w diagnostic ...', flush=True)
    zb = float(zc[zc <= -MIN_DEPTH][1])              # -9.5 m: level below the anchor
    U_true_s = uv_tm.U.sel(Z=z_fill)
    V_true_s = uv_tm.V.sel(Z=z_fill)
    # constant-shear extrapolated field, all on native levels: anchor at -8.5 m, slope =
    # native shear between the two shallowest sampled levels (-8.5, -9.5 m).
    sU = (uv_tm.U.sel(Z=za) - uv_tm.U.sel(Z=zb)) / (za - zb)   # native dU/dz at -9 m
    sV = (uv_tm.V.sel(Z=za) - uv_tm.V.sel(Z=zb)) / (za - zb)
    zc_s = xr.DataArray(z_fill, dims='Z', coords={'Z': z_fill})
    U_ext_s = u_anchor + sU * (zc_s - za)
    V_ext_s = v_anchor + sV * (zc_s - za)

    div_true = _divergence(U_true_s, V_true_s)
    div_ext = _divergence(U_ext_s, V_ext_s)
    dz_top = 1.0                                      # extrapolated levels are 1 m thick
    DAY = 86400.0
    # w at 8 m built from near-surface convergence (w=0 at surface): w(8m)=sum(div*dz)
    w_true8 = div_true.sum('Z') * dz_top             # truth
    w_ext8 = div_ext.sum('Z') * dz_top              # estimator WITH constant-shear extrapolation
    # WITHOUT it the estimator sets w=0 at 8 m, so it omits this layer entirely.
    mp['w_true8'] = _v(w_true8 * DAY)                # m/day
    mp['werr_ext'] = _v((w_ext8 - w_true8) * DAY)   # error WITH extrapolation
    mp['werr_noext'] = _v((-w_true8) * DAY)         # error WITHOUT extrapolation
    mp['uerr_layer'] = _v((U_ext_s - U_true_s).mean('Z'))   # 0-8 m mean U error (m/s)
    mp['verr_layer'] = _v((V_ext_s - V_true_s).mean('Z'))
    mp['div_true_ns'] = _v(div_true.mean('Z'))              # near-surface mean div (1/s)
    mp['div_err_ext'] = _v((div_ext - div_true).mean('Z'))  # near-surface div error (1/s)

    out['maps'] = {k: (v, xc, yc) for k, v in mp.items()}

    # --- PROFILES: truth vs the constant-shear assumption, and the resulting divergence /
    # error profiles, as the box mean AND per 0.5° latitude band. Velocity/shear use the
    # spatial mean (with std); divergence has ~zero mean, so it is summarized by its RMS.
    print('truth-vs-assumed profiles (box + lat bands) ...', flush=True)
    div_true_full = _divergence(uv_tm.U, uv_tm.V)
    zi = su_tm.Zi.values
    zi_fill = zi[(zi < 0) & (zi >= -MIN_DEPTH)]              # fill interfaces: -1..-8
    err_u_s = U_ext_s - U_true_s
    err_v_s = V_ext_s - V_true_s
    div_err = div_ext - div_true

    def _build_prof(ysel):
        """Profile dict for a YC selection (None = whole box)."""
        m = (lambda f: f) if ysel is None else (lambda f: f.sel(YC=ysel))
        box = ['YC', 'XC']
        rms = lambda f: np.sqrt((m(f) ** 2).mean(box))
        return {
            'U_true': _v(m(uv_tm.U).mean(box)), 'V_true': _v(m(uv_tm.V).mean(box)),
            'U_ext': _v(m(U_ext_s).mean(box)), 'V_ext': _v(m(V_ext_s).mean(box)),
            'SU_true': _v(m(su_tm).mean(box)), 'SV_true': _v(m(sv_tm).mean(box)),
            'SU_assumed': float(m(su_const).mean(box)),
            'SV_assumed': float(m(sv_const).mean(box)),
            'SU_true_fill': _v(m(su_tm.sel(Zi=zi_fill)).mean(box)),
            'SV_true_fill': _v(m(sv_tm.sel(Zi=zi_fill)).mean(box)),
            'U_err': _v(m(err_u_s).mean(box)), 'U_err_std': _v(m(err_u_s).std(box)),
            'V_err': _v(m(err_v_s).mean(box)), 'V_err_std': _v(m(err_v_s).std(box)),
            'div_true_rms': _v(rms(div_true_full)), 'div_ext_rms': _v(rms(div_ext)),
            'div_err_rms': _v(rms(div_err)),
        }

    coords = {'za': za, 'z_uv': uv_tm.Z.values, 'z_fill': z_fill,
              'z_shear': zi, 'zi_fill': zi_fill, 'z_div': uv_tm.Z.values}
    out['prof2'] = {**coords, **_build_prof(None)}
    edges = LAT_BAND_EDGES
    out['prof2_bands'] = {
        **coords,
        'centers': np.array([0.5 * (lo + hi) for lo, hi in zip(edges[:-1], edges[1:])]),
        'profs': [_build_prof(slice(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])],
    }

    # --- TIME EVOLUTION: keep time instead of collapsing it (snapshots vs the mean) ---
    print('time-evolution (daily box-mean profiles) ...', flush=True)
    out['tevol'] = _time_evolution_stats(uv)

    # --- REGRESSION rule: extrapolate u(z),v(z) from u(8 m),v(8 m) across time per point ---
    print('regression extrapolation (local, per-point) ...', flush=True)
    out['reg'] = _regression_extrap(uv)

    with open(CACHE, 'wb') as f:
        pickle.dump(out, f)
    print(f'cached -> {CACHE}  ({time.time() - t0:.0f} s)', flush=True)
    return out


# ------------------------------------------------------------------------------- plot
plt.rcParams.update({
    'axes.labelsize': 12.5, 'axes.labelweight': 'bold',
    'axes.titlesize': 12, 'legend.fontsize': 11, 'legend.title_fontsize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
})
MS_PER_S_TO_CM = 100.0
PLOT_ZLIM = -14.0                    # focus depth (m) for the truth-vs-assumption profiles
C_TRUE = 'k'
C_ASSUME = '#d62728'                 # red: the constant-shear assumption / extrapolation


def _band_cmap(n):
    return plt.cm.viridis(np.linspace(0, 1, n))


def _prof_decor(ax, za):
    """Shared decoration for the truth-vs-assumption profile panels."""
    ax.axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.15, lw=0)
    ax.axhline(za, color='0.4', ls=':', lw=1.0)
    ax.axvline(0, color='0.5', lw=0.7, zorder=0)
    ax.set_ylim(PLOT_ZLIM, 0)
    ax.grid(alpha=0.3)


def _fig_profiles_true_vs_assumed(out):
    """Box-mean profiles: true (curved) vs the constant-shear assumption (straight),
    for U, V (top row) and the shear ∂U/∂z, ∂V/∂z (bottom row)."""
    p = out['prof2']
    za = p['za']
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    for ax, tr, ex, name in [
            (axes[0, 0], p['U_true'], p['U_ext'], 'U'),
            (axes[0, 1], p['V_true'], p['V_ext'], 'V')]:
        z = p['z_uv']
        ax.plot(tr * 100, z, color=C_TRUE, lw=2.2, label='true (model)')
        ax.plot(ex * 100, p['z_fill'], color=C_ASSUME, lw=2.2, ls='--',
                label='constant shear')
        ax.set_xlabel(f'{name}  (cm s$^{{-1}}$)')
    for ax, tr, assumed, name in [
            (axes[1, 0], p['SU_true'], p['SU_assumed'], '∂U/∂z'),
            (axes[1, 1], p['SV_true'], p['SV_assumed'], '∂V/∂z')]:
        ax.plot(tr * 100, p['z_shear'], color=C_TRUE, lw=2.2, label='true (model)')
        ax.plot([assumed * 100, assumed * 100], [0, -MIN_DEPTH], color=C_ASSUME,
                lw=2.2, ls='--', label='constant shear')
        ax.set_xlabel(f'{name}  (10$^{{-2}}$ s$^{{-1}}$)')
    for ax in axes.ravel():
        _prof_decor(ax, za)
    axes[0, 0].set_ylabel('depth (m)')
    axes[1, 0].set_ylabel('depth (m)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               frameon=False)
    fig.suptitle('box-mean profiles: truth vs the constant-shear assumption, top 14 m  '
                 '(gray: 0–8 m constant-shear layer · dotted: −8.5 m anchor)',
                 fontsize=12.5, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _save(fig, 'profiles_true_vs_assumed')


def _fig_profiles_errors(out):
    """Profiles of the errors the constant shear produces: shear error, velocity
    error, and the resulting divergence (amplitude + error), all vs depth."""
    p = out['prof2']
    za = p['za']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)

    # (0) shear error = assumed constant − true, over the 0-8 m interfaces
    su_e = (p['SU_assumed'] - p['SU_true_fill']) * 100
    sv_e = (p['SV_assumed'] - p['SV_true_fill']) * 100
    axes[0].plot(su_e, p['zi_fill'], color='C0', lw=2.2, label='∂U/∂z error')
    axes[0].plot(sv_e, p['zi_fill'], color='C1', lw=2.2, label='∂V/∂z error')
    axes[0].set_xlabel('assumed − true shear  (10$^{-2}$ s$^{-1}$)')
    _panel_legend(axes[0], loc='lower left')

    # (1) velocity error = extrapolated − true, over the 0-8 m layer (bias +/- spatial std)
    for err, std, c, name in [(p['U_err'], p['U_err_std'], 'C0', 'U'),
                              (p['V_err'], p['V_err_std'], 'C1', 'V')]:
        axes[1].fill_betweenx(p['z_fill'], (err - std) * 100, (err + std) * 100,
                              color=c, alpha=0.12, lw=0)
        axes[1].plot(err * 100, p['z_fill'], color=c, lw=2.2, label=f'{name} error')
    axes[1].set_xlabel('extrapolated − true velocity  (cm s$^{-1}$)')
    _panel_legend(axes[1], loc='lower left')

    # (2) divergence amplitude (RMS over box): true vs extrapolated, and the error
    S = 1e6
    axes[2].plot(p['div_true_rms'] * S, p['z_div'], color=C_TRUE, lw=2.2,
                 label='true |div| (RMS)')
    axes[2].plot(p['div_ext_rms'] * S, p['z_fill'], color=C_ASSUME, lw=2.2, ls='--',
                 label='constant-shear |div| (RMS)')
    axes[2].plot(p['div_err_rms'] * S, p['z_fill'], color='C2', lw=2.2, ls=':',
                 label='div error (RMS)')
    axes[2].set_xlabel('horizontal divergence  (10$^{-6}$ s$^{-1}$)')
    _panel_legend(axes[2], loc='lower right')
    axes[2].set_xlim(left=0)

    for ax in axes:
        _prof_decor(ax, za)
    axes[0].set_ylabel('depth (m)')
    fig.suptitle('error profiles of the constant shear (0–8 m), top 14 m',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'profiles_errors')


def _band_legend(fig, centers, colors, y=0.955):
    """Top color legend for latitude bands (solid proxies)."""
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors[i], lw=2.4)
               for i in range(len(centers))]
    labels = [f'{c:+.2f}°N' for c in centers]
    fig.legend(handles, labels, title='latitude band', ncol=len(centers),
               loc='upper center', bbox_to_anchor=(0.5, y), frameon=False,
               columnspacing=1.0, handlelength=1.4)


def _fig_profiles_true_vs_assumed_latbands(out):
    """Per-latitude-band truth (solid) vs constant shear (dashed), colored by band."""
    from matplotlib.lines import Line2D
    pb = out['prof2_bands']
    centers = pb['centers']
    colors = _band_cmap(len(centers))
    za, z_uv, z_fill, z_shear = pb['za'], pb['z_uv'], pb['z_fill'], pb['z_shear']
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    for i, band in enumerate(pb['profs']):
        c = colors[i]
        axes[0, 0].plot(band['U_true'] * 100, z_uv, color=c, lw=1.7)
        axes[0, 0].plot(band['U_ext'] * 100, z_fill, color=c, lw=1.7, ls='--')
        axes[0, 1].plot(band['V_true'] * 100, z_uv, color=c, lw=1.7)
        axes[0, 1].plot(band['V_ext'] * 100, z_fill, color=c, lw=1.7, ls='--')
        axes[1, 0].plot(band['SU_true'] * 100, z_shear, color=c, lw=1.7)
        axes[1, 0].plot([band['SU_assumed'] * 100] * 2, [0, -MIN_DEPTH], color=c,
                        lw=1.7, ls='--')
        axes[1, 1].plot(band['SV_true'] * 100, z_shear, color=c, lw=1.7)
        axes[1, 1].plot([band['SV_assumed'] * 100] * 2, [0, -MIN_DEPTH], color=c,
                        lw=1.7, ls='--')
    for ax, xl in [(axes[0, 0], 'U  (cm s$^{-1}$)'), (axes[0, 1], 'V  (cm s$^{-1}$)'),
                   (axes[1, 0], '∂U/∂z  (10$^{-2}$ s$^{-1}$)'),
                   (axes[1, 1], '∂V/∂z  (10$^{-2}$ s$^{-1}$)')]:
        ax.set_xlabel(xl)
        _prof_decor(ax, za)
    axes[0, 0].set_ylabel('depth (m)')
    axes[1, 0].set_ylabel('depth (m)')
    style = [Line2D([0], [0], color='0.3', lw=2, ls='-'),
             Line2D([0], [0], color='0.3', lw=2, ls='--')]
    _panel_legend(axes[0, 0], loc='lower left', fontsize=9.5,
                  handles=style, labels=['true (model)', 'constant shear'])
    _band_legend(fig, centers, colors)
    fig.suptitle('truth (solid) vs constant shear (dashed) by latitude band, top 14 m'
                 '  (gray: 0–8 m constant-shear layer · dotted: −8.5 m anchor)',
                 fontsize=12, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _save(fig, 'profiles_true_vs_assumed_latbands')


def _fig_profiles_errors_latbands(out):
    """Per-latitude-band error profiles: U error, V error, divergence error (RMS)."""
    pb = out['prof2_bands']
    centers = pb['centers']
    colors = _band_cmap(len(centers))
    za, z_fill = pb['za'], pb['z_fill']
    box_true_div = out['prof2']['div_true_rms']            # box reference for scale
    z_div = out['prof2']['z_div']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for i, band in enumerate(pb['profs']):
        c = colors[i]
        axes[0].plot(band['U_err'] * 100, z_fill, color=c, lw=1.8)
        axes[1].plot(band['V_err'] * 100, z_fill, color=c, lw=1.8)
        axes[2].plot(band['div_err_rms'] * 1e6, z_fill, color=c, lw=1.8)
    axes[2].plot(box_true_div * 1e6, z_div, color='0.4', lw=2.0, ls=':',
                 label='box true |div| (RMS)')
    axes[0].set_xlabel('extrapolated − true  U  (cm s$^{-1}$)')
    axes[1].set_xlabel('extrapolated − true  V  (cm s$^{-1}$)')
    axes[2].set_xlabel('divergence error, RMS  (10$^{-6}$ s$^{-1}$)')
    _panel_legend(axes[2], loc='lower right', fontsize=9.5)
    axes[2].set_xlim(left=0)
    for ax in axes:
        _prof_decor(ax, za)
    axes[0].set_ylabel('depth (m)')
    _band_legend(fig, centers, colors, y=0.9)
    fig.suptitle('constant shear error profiles by latitude band, top 14 m',
                 fontsize=13, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    _save(fig, 'profiles_errors_latbands')


def _panel_legend(ax, loc='best', fontsize=9, **kw):
    """In-panel legend with a solid white background so text stays readable over data."""
    return ax.legend(loc=loc, fontsize=fontsize, frameon=True, facecolor='white',
                     framealpha=0.92, edgecolor='0.7', **kw)


def _fit_shear(z, s):
    """Fit the model shear magnitude vs depth to two forms and return params + R².

    power law   |s| = A · d^(−p)        (d = −z, depth below the surface)
    exponential |s| = A · exp(z/L)
    The sign of the mean shear is carried separately so signed components plot correctly.
    """
    from scipy.optimize import curve_fit
    z = np.asarray(z, float)
    s = np.asarray(s, float)
    sgn = float(np.sign(np.nanmean(s)))
    a = np.abs(s)
    d = -z

    def r2(y, yh):
        dd = y - np.mean(y)
        return 1 - np.sum((y - yh) ** 2) / np.sum(dd ** 2)

    (Ap, p), _ = curve_fit(lambda z, A, p: A * (-z) ** (-p), z, a,
                           p0=[a[0], 0.7], maxfev=60000)
    (Ae, L), _ = curve_fit(lambda z, A, L: A * np.exp(z / L), z, a,
                           p0=[a[0], 6.0], maxfev=60000)
    return dict(sgn=sgn, Ap=Ap, p=p, R2p=r2(a, Ap * d ** (-p)),
                Ae=Ae, L=L, R2e=r2(a, Ae * np.exp(z / L)))


def _shear_comps(prof):
    """(name, signed-shear) triples for a profile dict."""
    su, sv = prof['SU_true'], prof['SV_true']
    return [('∂U/∂z', su), ('∂V/∂z', sv), ('|S|', np.sqrt(su ** 2 + sv ** 2))]


def _fig_shear_curvefit(out):
    """Box-mean model shear (points) vs a power-law and an exponential fit, 0–20 m."""
    p = out['prof2']
    z = p['z_shear']
    zg = np.linspace(-0.5, -20.0, 300)
    d = -zg
    iref = int(np.argmin(np.abs(z - SAMPLED_SHEAR_Z)))   # -9 m reference level
    # shear MAGNITUDE is plotted (fits are on |s|) so all three panels share one geometry
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), sharey=True)
    for ax, (name, s) in zip(axes, _shear_comps(p)):
        a = np.abs(s)
        f = _fit_shear(z, s)
        sref = a[iref]
        dn = name if name == '|S|' else f'|{name}|'
        ax.plot(a * 100, z, 'o', color='k', ms=4.5, label='model shear')
        ax.plot(f['Ap'] * d ** (-f['p']) * 100, zg, color=C_ASSUME, lw=2.4,
                label=f"power  A·d$^{{-{f['p']:.2f}}}$  (d=−z; R²={f['R2p']:.2f})")
        ax.plot(f['Ae'] * np.exp(zg / f['L']) * 100, zg, color='C0', lw=2.0, ls='--',
                label=f"exp  L={f['L']:.1f} m  (R²={f['R2e']:.2f})")
        ax.plot([sref * 100] * 2, [0.0, SAMPLED_SHEAR_Z], color='0.45', lw=2.2, ls=':',
                label='constant shear (−9 m)')
        ax.axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.10, lw=0)
        ax.set_xlabel(f'{dn}  (10$^{{-2}}$ s$^{{-1}}$)')
        ax.set_ylim(-20, 0)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3)
        _panel_legend(ax, loc='lower right', fontsize=8.5)
    axes[0].set_ylabel('depth (m)')
    fig.suptitle('curve fits to the model shear magnitude (box mean, 0–20 m): '
                 'power law d$^{-p}$ vs exponential vs constant-shear assumption',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'shear_curvefit')


def _fig_shear_curvefit_latbands(out):
    """Power-law exponent p and fit quality (power vs exp) vs latitude band."""
    pb = out['prof2_bands']
    z = pb['z_shear']
    centers = pb['centers']
    names = ['∂U/∂z', '∂V/∂z', '|S|']
    ccomp = ['C0', 'C1', 'k']
    P = {n: [] for n in names}
    R2p = {n: [] for n in names}
    R2e = {n: [] for n in names}
    for band in pb['profs']:
        for (name, s) in _shear_comps(band):
            f = _fit_shear(z, s)
            P[name].append(f['p'])
            R2p[name].append(f['R2p'])
            R2e[name].append(f['R2e'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    # panel 0: exponent p vs latitude, referenced to the constant-shear limit (p=0)
    axes[0].axhline(0.0, color='0.6', ls=':', lw=1.4)
    axes[0].text(centers[-1], 0.0, 'constant shear (p=0) ', fontsize=8.5, color='0.4',
                 va='bottom', ha='right')
    for name, c in zip(names, ccomp):
        axes[0].plot(centers, P[name], 'o-', color=c, lw=2, label=name)
    axes[0].set_ylim(-0.08, 0.9)
    axes[0].set_ylabel('power-law exponent  p   (|S| ∝ d$^{-p}$)')
    axes[0].set_xlabel('latitude (°N)')
    axes[0].grid(alpha=0.3)
    _panel_legend(axes[0], loc='center left', fontsize=9)
    # panel 1: fit quality per component (∂U/∂z, ∂V/∂z used independently, not combined |S|)
    for name, c in zip(['∂U/∂z', '∂V/∂z'], ['C0', 'C1']):
        axes[1].plot(centers, R2p[name], 'o-', color=c, lw=2.2,
                     label=f'{name} power law d$^{{-p}}$ (d=−z)')
        axes[1].plot(centers, R2e[name], 's--', color=c, lw=2.0, alpha=0.7,
                     label=f'{name} exponential')
    axes[1].set_ylim(0.7, 1.02)
    axes[1].set_ylabel('fit R²')
    axes[1].set_xlabel('latitude (°N)')
    axes[1].grid(alpha=0.3)
    _panel_legend(axes[1], loc='lower left', fontsize=8)
    fig.suptitle('model shear power-law fit vs latitude (0–20 m): exponent & fit quality',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'shear_curvefit_latbands')


def _profile_panel(ax, prof, xlabel, mark_sampled=None, scale=1.0, band_label='lat',
                   shade=True):
    """Overlay per-band mean profiles with +/- std shading; y = depth (0 at top)."""
    centers, z = prof['centers'], prof['z']
    colors = _band_cmap(len(centers))
    for i, c in enumerate(centers):
        m = prof['mean'][i] * scale
        s = prof['std'][i] * scale
        lbl = (f'{c:+.2f}°N' if band_label == 'lat'
               else f'{360 - c:.1f}°W')
        ax.plot(m, z, color=colors[i], lw=2.2, label=lbl)
        if shade:
            ax.fill_betweenx(z, m - s, m + s, color=colors[i], alpha=0.10, lw=0)
    if mark_sampled is not None:
        ax.axhline(mark_sampled, color='k', ls='--', lw=1.2)
        ax.axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.18, lw=0)
    ax.axvline(0, color='0.5', lw=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ZMAX, 0)
    ax.grid(alpha=0.3)


def _fig_shear_profiles(out, band='lat'):
    prof = out['prof']
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    zmark = out['sampled_shear_z']
    _profile_panel(axes[0], prof[f'{band}_Smag'],
                   '|shear| = √((∂U/∂z)²+(∂V/∂z)²)  (10$^{-2}$ s$^{-1}$)',
                   mark_sampled=zmark, scale=100.0, band_label=band)
    _profile_panel(axes[1], prof[f'{band}_SU'],
                   '∂U/∂z  (10$^{-2}$ s$^{-1}$)', mark_sampled=zmark,
                   scale=100.0, band_label=band)
    _profile_panel(axes[2], prof[f'{band}_SV'],
                   '∂V/∂z  (10$^{-2}$ s$^{-1}$)', mark_sampled=zmark,
                   scale=100.0, band_label=band)
    # rescale |shear| panel too (it was plotted with scale in the helper call above)
    axes[0].set_ylabel('depth (m)')
    ttl = ('latitude band' if band == 'lat' else 'longitude band')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=ttl, ncol=len(labels),
               loc='upper center', bbox_to_anchor=(0.5, 0.95),
               frameon=False, columnspacing=1.0, handlelength=1.4)
    fig.suptitle('vertical shear of horizontal currents — top 20 m  '
                 '(dashed: −9 m constant shear · gray: 0–8 m extrapolation layer)',
                 fontsize=12.5, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    _save(fig, f'shear_profiles_{band}bands')


def _fig_velocity_profiles(out, band='lat'):
    prof = out['prof']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6), sharey=True)
    _profile_panel(axes[0], prof[f'{band}_U'], 'U  (cm s$^{-1}$)',
                   mark_sampled=None, scale=MS_PER_S_TO_CM, band_label=band, shade=False)
    _profile_panel(axes[1], prof[f'{band}_V'], 'V  (cm s$^{-1}$)',
                   mark_sampled=None, scale=MS_PER_S_TO_CM, band_label=band, shade=False)
    axes[0].axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.18, lw=0)
    axes[1].axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.18, lw=0)
    axes[0].set_ylabel('depth (m)')
    ttl = ('latitude band' if band == 'lat' else 'longitude band')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=ttl, ncol=len(labels),
               loc='upper center', bbox_to_anchor=(0.5, 0.95),
               frameon=False, columnspacing=1.0, handlelength=1.4)
    fig.suptitle('horizontal velocity — top 20 m  '
                 '(gray: 0–8 m constant-shear extrapolation layer)',
                 fontsize=12.5, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    _save(fig, f'velocity_profiles_{band}bands')


def _map_panel(ax, arr, xc, yc, title, cmap, vmin=None, vmax=None, cbar_label='',
               annot=None, cbar_frac=0.046):
    pc = ax.pcolormesh(xc, yc, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(title)
    ax.set_xlabel('lon (°E)')
    cb = plt.colorbar(pc, ax=ax, fraction=cbar_frac, pad=0.04)
    cb.set_label(cbar_label, fontsize=10)
    ax.axhline(0, color='k', lw=0.6, ls=':')
    if annot:
        ax.text(0.03, 0.03, annot, transform=ax.transAxes, fontsize=9.5, va='bottom',
                ha='left', bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))


def _fig_uv_error(out):
    """Spatial structure of the constant-shear U, V error (0-8 m layer mean).

    Divergence only sees the horizontal GRADIENTS of these fields, so what matters for
    the w estimate is the spatial STD (structure), not the spatial mean (uniform part).
    Both are annotated so the two can be compared directly.
    """
    m = out['maps']
    ue, xc, yc = m['uerr_layer']
    ve = m['verr_layer'][0]
    mag = np.sqrt(ue ** 2 + ve ** 2) * 100
    lim = np.nanpercentile(np.abs(np.concatenate([ue.ravel(), ve.ravel()])) * 100, 99)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, arr, ttl in [(axes[0], ue * 100, 'U error  (U$_{ext}$ − U$_{true}$)'),
                         (axes[1], ve * 100, 'V error  (V$_{ext}$ − V$_{true}$)')]:
        annot = f'mean {np.nanmean(arr):+.2f}\nstd  {np.nanstd(arr):.2f} cm s$^{{-1}}$'
        _map_panel(ax, arr, xc, yc, ttl, cmo.balance, -lim, lim,
                   'error (cm s$^{-1}$)', annot=annot)
    _map_panel(axes[2], mag, xc, yc, '|velocity error|', cmo.amp,
               0, np.nanpercentile(mag, 99), 'error (cm s$^{-1}$)',
               annot=f'mean {np.nanmean(mag):.2f} cm s$^{{-1}}$')
    axes[0].set_ylabel('lat (°N)')
    fig.suptitle('velocity error of the constant shear — 0–8 m layer mean',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'maps_uv_error')


def _fig_div_w_error(out):
    """Near-surface (0-8 m) w consequence: WITH vs WITHOUT constant-shear extrapolation.

    w(8 m) built from the 0-8 m convergence (w=0 at surface). Without the extrapolation the
    estimator omits this layer (w=0 at 8 m), so its error is −w_true; with constant shear its
    error is the divergence error of the extrapolated field. The last panel shows where
    extrapolation actually helps (|err_noext| − |err_ext| > 0).
    """
    m = out['maps']
    wt, xc, yc = m['w_true8']
    ee = m['werr_ext'][0]
    en = m['werr_noext'][0]
    elim = np.nanpercentile(np.abs(np.concatenate([ee.ravel(), en.ravel()])), 98)
    wlim = np.nanpercentile(np.abs(wt), 98)

    def _rms(a):
        return np.sqrt(np.nanmean(a ** 2))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    _map_panel(axes[0, 0], wt, xc, yc, 'true near-surface w(8 m)', cmo.balance,
               -wlim, wlim, 'w (m day$^{-1}$)',
               annot=f'RMS {_rms(wt):.2f} m day$^{{-1}}$')
    _map_panel(axes[0, 1], en, xc, yc, 'w error — WITHOUT extrapolation', cmo.balance,
               -elim, elim, 'error (m day$^{-1}$)',
               annot=f'RMS {_rms(en):.2f}')
    _map_panel(axes[1, 0], ee, xc, yc, 'w error — WITH constant shear', cmo.balance,
               -elim, elim, 'error (m day$^{-1}$)',
               annot=f'RMS {_rms(ee):.2f}')
    improve = np.abs(en) - np.abs(ee)
    ilim = np.nanpercentile(np.abs(improve), 98)
    _map_panel(axes[1, 1], improve, xc, yc,
               '|err$_{no}$| − |err$_{cs}$|  (blue = constant shear worse)', cmo.balance,
               -ilim, ilim, 'Δ|error| (m day$^{-1}$)',
               annot=f'constant shear cuts RMS\n{_rms(en):.2f} → {_rms(ee):.2f}')
    axes[0, 0].set_ylabel('lat (°N)')
    axes[1, 0].set_ylabel('lat (°N)')
    fig.suptitle('vertical velocity at 8 m from the 0–8 m convergence (w = 0 at surface)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'maps_divergence_w_error')


def _fig_depthstats(out):
    m = out['maps']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sm, xc, yc = m['smag_depthmean']
    _map_panel(axes[0], sm * 100, xc, yc, 'depth-mean |shear|, 0-20 m', cmo.speed,
               cbar_label='|shear| (10$^{-2}$ s$^{-1}$)')
    for ax, key, ttl in zip(
            axes[1:], ['su_depthmean', 'sv_depthmean'],
            ['depth-mean ∂U/∂z, 0-20 m', 'depth-mean ∂V/∂z, 0-20 m']):
        arr = m[key][0] * 100
        lim = np.nanpercentile(np.abs(arr), 99)
        _map_panel(ax, arr, xc, yc, ttl, cmo.balance, -lim, lim,
                   cbar_label='shear (10$^{-2}$ s$^{-1}$)')
    axes[0].set_ylabel('lat (°N)')
    fig.suptitle('time-mean shear through the top 20 m of the column',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'maps_depthstats')


def _fig_8m_timestats(out):
    m = out['maps']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, ttl in zip(
            axes,
            ['smag8_timemean', 'smag8_timemax', 'smag8_timemin'],
            ['time-MEAN |shear| @ 8 m', 'time-MAX |shear| @ 8 m',
             'time-MIN |shear| @ 8 m']):
        arr, xc, yc = m[key]
        _map_panel(ax, arr * 100, xc, yc, ttl, cmo.speed,
                   cbar_label='|shear| (10$^{-2}$ s$^{-1}$)')
    axes[0].set_ylabel('lat (°N)')
    fig.suptitle('|shear| at a single depth (8 m) — statistics over time',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'maps_8m_timestats')


def _fig_differences(out):
    m = out['maps']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    # top row: |shear| level differences; bottom: component differences 8m vs surface
    smag_surf, xc, yc = m['smag_surf']
    smag_8m = m['smag_8m'][0]
    smag_20m = m['smag_20m'][0]
    su_surf = m['su_surf'][0]; su_8m = m['su_8m'][0]
    sv_surf = m['sv_surf'][0]; sv_8m = m['sv_8m'][0]

    d1 = (smag_8m - smag_surf) * 100
    d2 = (smag_20m - smag_8m) * 100
    lim1 = np.nanmax(np.abs(np.concatenate([d1.ravel(), d2.ravel()])))
    _map_panel(axes[0, 0], d1, xc, yc, '|shear|(8 m) − |shear|(surf)', cmo.balance,
               -lim1, lim1, '(10$^{-2}$ s$^{-1}$)')
    _map_panel(axes[0, 1], d2, xc, yc, '|shear|(20 m) − |shear|(8 m)', cmo.balance,
               -lim1, lim1, '(10$^{-2}$ s$^{-1}$)')
    # net change surface->20m
    d3 = (smag_20m - smag_surf) * 100
    _map_panel(axes[0, 2], d3, xc, yc, '|shear|(20 m) − |shear|(surf)', cmo.balance,
               -lim1, lim1, '(10$^{-2}$ s$^{-1}$)')

    du = (su_8m - su_surf) * 100
    dv = (sv_8m - sv_surf) * 100
    limc = np.nanmax(np.abs(np.concatenate([du.ravel(), dv.ravel()])))
    _map_panel(axes[1, 0], du, xc, yc, '∂U/∂z(8 m) − ∂U/∂z(surf)', cmo.balance,
               -limc, limc, '(10$^{-2}$ s$^{-1}$)')
    _map_panel(axes[1, 1], dv, xc, yc, '∂V/∂z(8 m) − ∂V/∂z(surf)', cmo.balance,
               -limc, limc, '(10$^{-2}$ s$^{-1}$)')
    axes[1, 2].axis('off')
    axes[0, 0].set_ylabel('lat (°N)')
    axes[1, 0].set_ylabel('lat (°N)')
    fig.suptitle('time-mean shear change between depths (surface = −0.5 m, 8 m, 20 m)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'maps_shear_differences')


def _fig_extrap_error(out):
    m = out['maps']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ec, xc, yc = m['serr_const']
    ed = m['serr_deep'][0]
    both = np.concatenate([ec.ravel(), ed.ravel()]) * 100
    vmin, vmax = np.nanpercentile(both, 2), np.nanpercentile(both, 98)
    _map_panel(axes[0], ec * 100, xc, yc,
               'error: constant −9 m shear', cmo.amp, vmin, vmax,
               'error (cm s$^{-1}$)')
    _map_panel(axes[1], ed * 100, xc, yc,
               'error: 8.5–20 m mean shear', cmo.amp, vmin, vmax,
               'error (cm s$^{-1}$)')
    diff = (ed - ec) * 100
    dl = np.nanmax(np.abs(diff))
    _map_panel(axes[2], diff, xc, yc,
               'deep − constant (blue = deep better)', cmo.balance, -dl, dl,
               'Δerror (cm s$^{-1}$)')
    axes[0].set_ylabel('lat (°N)')
    fig.suptitle('surface (−0.5 m) velocity error from extrapolating over 0–8 m',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'maps_extrap_error')


def _fig_shear_hovmoller(out):
    """Time-depth (Hovmoller) of the box-mean shear: is the surface intensification a
    persistent feature or does it come and go? |S|, ∂U/∂z, ∂V/∂z over the 3-month run."""
    te = out['tevol']
    t, zi = te['t_days'], te['zi']
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    specs = [('Smag', cmo.speed, None, None, '|S|  (10$^{-2}$ s$^{-1}$)'),
             ('SU', cmo.balance, 'sym', None, '∂U/∂z  (10$^{-2}$ s$^{-1}$)'),
             ('SV', cmo.balance, 'sym', None, '∂V/∂z  (10$^{-2}$ s$^{-1}$)')]
    for ax, (key, cmap, mode, _, cl) in zip(axes, specs):
        arr = te['day_' + key] * 100                 # (nz, nday)
        if mode == 'sym':
            lim = np.nanpercentile(np.abs(arr), 99)
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = 0, np.nanpercentile(arr, 99)
        pc = ax.pcolormesh(t, zi, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        cb = plt.colorbar(pc, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label(cl, fontsize=10)
        ax.axhline(-MIN_DEPTH, color='k', ls='--', lw=1.1)
        ax.axhspan(-MIN_DEPTH, 0, color='w', alpha=0.0)
        ax.set_ylim(-20, 0)
        ax.set_ylabel('depth (m)')
    axes[-1].set_xlabel('days since 2012-10-01')
    fig.suptitle('box-mean shear vs time and depth — top 20 m  '
                 '(dashed: 8 m sampling limit; 0–8 m above it is extrapolated)',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, 'shear_hovmoller')


def _fig_profiles_snapshots(out):
    """Do individual daily snapshots look like the time mean? Each thin line is one day's
    box-mean profile; the thick black line is the time mean. If every snapshot shares the
    mean's curved, surface-intensified shape, a single extrapolation rule can work."""
    te = out['tevol']
    z_uv, zi = te['z_uv'], te['zi']
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.6), sharey=True)
    specs = [('U', z_uv, MS_PER_S_TO_CM, 'U  (cm s$^{-1}$)'),
             ('V', z_uv, MS_PER_S_TO_CM, 'V  (cm s$^{-1}$)'),
             ('SU', zi, 100.0, '∂U/∂z  (10$^{-2}$ s$^{-1}$)'),
             ('SV', zi, 100.0, '∂V/∂z  (10$^{-2}$ s$^{-1}$)')]
    for ax, (key, zz, sc, xl) in zip(axes, specs):
        days = te['day_' + key] * sc                 # (nz, nday)
        ax.plot(days, zz, color='0.6', lw=0.5, alpha=0.25)
        lo, hi = np.percentile(days, [10, 90], axis=1)
        ax.fill_betweenx(zz, lo, hi, color='C0', alpha=0.18, lw=0,
                         label='10–90% of days')
        ax.plot(te['tm_' + key] * sc, zz, color=C_TRUE, lw=2.6, label='time mean')
        ax.axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.12, lw=0)
        ax.axvline(0, color='0.5', lw=0.7, zorder=0)
        ax.set_xlabel(xl)
        ax.set_ylim(-14, 0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('depth (m)')
    _panel_legend(axes[0], loc='lower left', fontsize=9)
    fig.suptitle('daily snapshots (thin) vs the time mean (black) — box-mean profiles, top 14 m  '
                 '(gray: 0–8 m extrapolation layer)',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'profiles_snapshots')


def _fig_regression_rule(out):
    """The statistical extrapolation rule: coefficients of u(z),v(z) regressed on
    u(8 m),v(8 m) across time, and how well it predicts (R²), vs depth."""
    r = out['reg']
    p, zf = r['prof'], r['zf']
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=True)
    # (0) self gains: how much the shallower current amplifies the 8 m current
    axes[0].axvline(1.0, color='0.6', ls=':', lw=1.4)
    axes[0].plot(p['b_uu'], zf, 'o-', color='C0', lw=2.2, label='u(z) ← u(8 m)')
    axes[0].plot(p['b_vv'], zf, 's-', color='C1', lw=2.2, label='v(z) ← v(8 m)')
    axes[0].set_xlabel('gain  b')
    # widen to keep the gain=1 reference line and its label clear of the data
    bmax = max(p['b_uu'].max(), p['b_vv'].max())
    axes[0].set_xlim(0.999, bmax + 0.25 * (bmax - 1.0))
    axes[0].text(0.02, 0.95, 'persistence (gain 1)', transform=axes[0].transAxes,
                 fontsize=8.5, color='0.4', va='top', ha='left')
    _panel_legend(axes[0], loc='lower right')
    # (1) cross gains: veering / rotation of the current with depth
    axes[1].axvline(0.0, color='0.6', ls=':', lw=1.4)
    axes[1].plot(p['b_uv'], zf, 'o-', color='C0', lw=2.2, label='u(z) ← v(8 m)')
    axes[1].plot(p['b_vu'], zf, 's-', color='C1', lw=2.2, label='v(z) ← u(8 m)')
    axes[1].set_xlabel('cross gain  b')
    _panel_legend(axes[1], loc='lower left')
    # (2) skill: fraction of variance explained (block-CV held-out)
    axes[2].plot(p['r2u_loc'], zf, 'o-', color='C0', lw=2.2, label='U')
    axes[2].plot(p['r2v_loc'], zf, 's-', color='C1', lw=2.2, label='V')
    axes[2].set_xlabel('held-out R²')
    axes[2].set_xlim(right=1.01)
    _panel_legend(axes[2], loc='lower left', fontsize=8.5)
    for ax in axes:
        ax.axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.10, lw=0)
        ax.set_ylim(-8.5, 0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('depth (m)')
    fig.suptitle('regression rule u(z),v(z) ← u(8 m),v(8 m) across time (box-mean over points): '
                 'gains and skill vs depth',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'regression_rule')


def _fig_regression_skill(out):
    """Does a statistical rule beat constant shear? Held-out RMS reconstruction error vs
    depth for persistence, constant shear, and the regression — first onto u(8 m),v(8 m)
    only, then ALSO onto the measured 8 m shear."""
    r = out['reg']
    rm, zf = r['rmse'], r['zf']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    # every regression is LOCAL (each column predicted from its own 8 m time series) and each
    # predictor set INCLUDES u(8 m),v(8 m) — the "+ shear" rules add the shear on top.
    styles = [('persist', '0.5', '-', 'o', 'persistence (hold 8 m)'),
              ('cshear', C_ASSUME, '-', 's', 'constant shear'),
              ('reg_loc', 'C0', '-', 'D', 'regression:  u,v(8 m)'),
              ('reg_shear', 'C4', '-', 'v', 'regression:  u,v(8 m) + shear(8 m)'),
              ('reg_multi', 'C2', '-', '^', 'regression:  u,v + shear at 4 depths')]
    for ax, comp, name in [(axes[0], 'u', 'U'), (axes[1], 'v', 'V')]:
        for key, c, ls, mk, lbl in styles:
            ax.plot(rm[f'{key}_{comp}'] * 100, zf, marker=mk, color=c, ls=ls, lw=2.2,
                    ms=5, label=lbl)
        ax.set_xlabel(f'{name} reconstruction RMS error  (cm s$^{{-1}}$)')
        ax.axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.10, lw=0)
        ax.set_ylim(-8.5, 0)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('depth (m)')
    _panel_legend(axes[0], loc='lower right', fontsize=8.5)
    fig.suptitle('held-out (block-CV) reconstruction error of the 0–8 m current: '
                 'statistical rule vs constant shear',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, 'regression_skill')


def _fig_regression_matrix_maps(out):
    """One figure for the whole LOCAL shear-augmented rule: the spatial map of every component
    of its 2x4 transfer (gain) matrix — targets (u,v) x predictors (u(8 m), v(8 m), ∂u/∂z,
    ∂v/∂z). All panels share ONE colorscale so the component magnitudes are directly
    comparable: diagonal 'gain' terms ≈ 1 dominate, off-diagonal 'cross gain' terms ≈ 0. The
    two shear columns are shown as the RATIO to the geometric extrapolation distance, so a
    value of 1 there means plain constant shear."""
    s = out['reg']['surf']
    _, xc, yc = out['maps']['w_true8']
    panels = [                                       # 2x4 row-major (row u then row v)
        (s['g_uu'], 'gain  u ← u(8 m)'),
        (s['g_uv'], 'cross gain  u ← v(8 m)'),
        (s['g_usu'], 'gain  u ← ∂u/∂z(8 m)'),
        (s['g_usv'], 'cross gain  u ← ∂v/∂z(8 m)'),
        (s['g_vu'], 'cross gain  v ← u(8 m)'),
        (s['g_vv'], 'gain  v ← v(8 m)'),
        (s['g_vsu'], 'cross gain  v ← ∂u/∂z(8 m)'),
        (s['g_vsv'], 'gain  v ← ∂v/∂z(8 m)'),
    ]
    vmax = np.nanpercentile(np.abs(np.concatenate([a.ravel() for a, _ in panels])), 99)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), layout='constrained')
    for ax, (arr, ttl) in zip(axes.ravel(), panels):
        pc = ax.pcolormesh(xc, yc, arr, cmap=cmo.balance, vmin=-vmax, vmax=vmax,
                           shading='auto')
        ax.set_title(ttl)
        ax.set_xlabel('lon (°E)')
        ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.text(0.03, 0.03, f'mean {np.nanmean(arr):+.2f}\nstd  {np.nanstd(arr):.2f}',
                transform=ax.transAxes, fontsize=9.5, va='bottom', ha='left',
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))
    axes[0, 0].set_ylabel('lat (°N)')
    axes[1, 0].set_ylabel('lat (°N)')
    cb = fig.colorbar(pc, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cb.set_label('gain (×)  — shear columns ÷ constant-shear distance (1 = constant shear)',
                 fontsize=10)
    fig.suptitle('surface (−0.5 m) local shear-augmented rule — every transfer-matrix component '
                 'on one shared colorscale  (predictors: u(8 m), v(8 m), ∂u/∂z, ∂v/∂z)',
                 fontsize=12.5, fontweight='bold')
    _save(fig, 'regression_matrix_maps')


# methods considered, in figure order (all regressions LOCAL: each column predicted from its
# own 8 m time series; every predictor set includes u(8 m),v(8 m)).
_METHOD_ORDER = [
    ('persist', 'persistence (hold 8 m)'),
    ('cshear', 'constant shear'),
    ('reg_loc', 'regression:  u,v(8 m)'),
    ('reg_shear', 'regression:  u,v(8 m) + shear(8 m)'),
    ('reg_multi', 'regression:  u,v + shear at 4 depths'),
]


def _fig_regression_r2_maps(out):
    """Maps of reconstruction skill (vector R² of U,V, averaged over the 0–8 m fill layers)
    for every method considered, all on one shared colorscale. R² is near-saturated for every
    method (u(8 m) explains most of the variance), so it discriminates weakly — the cumulative-
    error map is the more useful comparison for w."""
    rm = out['reg']['r2maps']
    _, xc, yc = out['maps']['w_true8']
    vmin = np.nanpercentile(np.concatenate([rm[k].ravel() for k, _ in _METHOD_ORDER]), 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, ttl) in zip(axes.ravel(), _METHOD_ORDER):
        arr = rm[key]
        _map_panel(ax, arr, xc, yc, ttl, cmo.amp, vmin, 1.0, 'R²',
                   annot=f'mean {np.nanmean(arr):.3f}')
    axes.ravel()[-1].axis('off')                     # 5 methods in a 2x3 grid
    axes[0, 0].set_ylabel('lat (°N)')
    axes[1, 0].set_ylabel('lat (°N)')
    fig.suptitle('reconstruction skill of the 0–8 m current — vector R² (U,V) averaged over '
                 '0–8 m, per method  (block-CV held-out, local rules)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'regression_r2_maps')


def _fig_regression_cumerr_maps(out):
    """Maps of the CUMULATIVE (vertically integrated) velocity error over 0–8 m, per method.
    Unlike the depth-averaged R², errors here integrate: a sign-coherent bias (e.g. constant
    shear's systematic shear underestimate) accumulates, while a zero-mean error cancels — and
    it is this depth-integrated transport error whose horizontal divergence sets the w error.
    Metric: time-RMS of |∫₀⁸ (δu, δv) dz| (m² s⁻¹). Shared colorscale (persistence saturates)."""
    cm = out['reg']['cumemaps']
    _, xc, yc = out['maps']['w_true8']
    # cap the shared scale at the non-persistence methods so the candidates stay resolved
    vmax = np.nanpercentile(np.concatenate(
        [cm[k].ravel() for k, _ in _METHOD_ORDER if k != 'persist']), 98)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, ttl) in zip(axes.ravel(), _METHOD_ORDER):
        arr = cm[key]
        _map_panel(ax, arr, xc, yc, ttl, cmo.amp, 0.0, vmax, 'error (m² s$^{-1}$)',
                   annot=f'area-mean {np.nanmean(arr):.3f}', cbar_frac=0.065)
    axes.ravel()[-1].axis('off')                     # 5 methods in a 2x3 grid
    axes[0, 0].set_ylabel('lat (°N)')
    axes[1, 0].set_ylabel('lat (°N)')
    fig.suptitle('cumulative (0–8 m depth-integrated) velocity error — time-RMS of '
                 '|∫(δu,δv)dz|, per method  (block-CV held-out, local rules)',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'regression_cumerr_maps')


def _save(fig, name):
    path = os.path.join(FIGDIR, name + '.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', path, flush=True)


def plot():
    with open(CACHE, 'rb') as f:
        out = pickle.load(f)
    os.makedirs(FIGDIR, exist_ok=True)
    _fig_shear_profiles(out, 'lat')
    _fig_shear_profiles(out, 'lon')
    _fig_velocity_profiles(out, 'lat')
    _fig_profiles_true_vs_assumed(out)
    _fig_profiles_errors(out)
    _fig_profiles_true_vs_assumed_latbands(out)
    _fig_profiles_errors_latbands(out)
    _fig_shear_curvefit(out)
    _fig_shear_curvefit_latbands(out)
    _fig_depthstats(out)
    _fig_8m_timestats(out)
    _fig_differences(out)
    _fig_extrap_error(out)
    _fig_uv_error(out)
    _fig_div_w_error(out)
    # time evolution + statistical extrapolation rule
    _fig_shear_hovmoller(out)
    _fig_profiles_snapshots(out)
    _fig_regression_rule(out)
    _fig_regression_skill(out)
    _fig_regression_matrix_maps(out)
    _fig_regression_r2_maps(out)
    _fig_regression_cumerr_maps(out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('all', 'compute'):
        compute()
    if mode in ('all', 'plot'):
        plot()
