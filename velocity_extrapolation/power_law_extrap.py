"""Power-law shear and regression-based surface extrapolation, and their effect on w.

This builds on ``run_shear_analysis.py`` (which tests the *constant*-shear surface fill).
Here we ask two things the constant-shear diagnostic cannot:

(1) POWER LAW vs CONSTANT SHEAR for the upwelling.  The mean shear intensifies toward the
    surface roughly as a power law |S| ∝ d^(-p) (d = depth below surface; p ≈ 0.7 from the
    mean-profile fit).  A power-law extrapolation therefore amplifies the measured 8 m shear
    by 1/(1-p) at the surface instead of holding it constant (p = 0 recovers constant shear).
    Does using it — rather than constant shear — improve the sample-based estimate of the
    model's true upwelling w?  We rebuild w at 8 m from the 0-8 m convergence of the
    extrapolated field (as in run_shear_analysis) for each scheme, in the mean AND
    time-varying, and scatter w_error vs w_true.

(2) Does the power law hold for the VARIABILITY as well as the mean?  We regress the model's
    *instantaneous* u(z), v(z) onto its instantaneous u(8 m), v(8 m) — multiple regression,
    so the fit can capture Ekman turning of the current toward the surface (the cross terms).
    The fitted gains b(z) are the empirical "power law for the fluctuations".  We then map
    u_error(z) = u_est(z) - u_true(z) as a 2-D function of (u_true(8 m), v_true(8 m)) at
    several depths, to see where a given extrapolation carries a flow-state-dependent bias —
    i.e. to hone the method.

All schemes are anchored on the OBSERVABLE 8 m quantities (velocity + the 8-10 m shear); the
0-8 m fill is what differs.  Reads the cached box from ``uv_region.nc`` (no model access).

Usage:  python power_law_extrap.py [compute|plot|all]
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
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'uv_region.nc')
CACHE = os.path.join(HERE, 'powerlaw_cache.pkl')
FIGDIR = os.path.join(HERE, 'figs')

MIN_DEPTH = 8.0
DAY = 86400.0
DEG = np.pi / 180 * 6371000.0

plt.rcParams.update({
    'axes.labelsize': 12.5, 'axes.labelweight': 'bold',
    'axes.titlesize': 12, 'legend.fontsize': 10.5, 'legend.title_fontsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
})


# --------------------------------------------------------------------------- helpers
def _div_np(U, V, x_m, y_m):
    """Horizontal divergence dU/dx + dV/dy for arrays with dims (..., ny, nx)."""
    dudx = np.gradient(U, x_m, axis=U.ndim - 1)
    dvdy = np.gradient(V, y_m, axis=U.ndim - 2)
    return dudx + dvdy


def _fit_p(zc, u_mean, v_mean):
    """Power-law exponent p of the MEAN shear magnitude |S| ∝ d^(-p), fit over 0-20 m."""
    zi = 0.5 * (zc[:-1] + zc[1:])
    dz = np.diff(zc)
    su = np.diff(u_mean) / dz
    sv = np.diff(v_mean) / dz
    smag = np.sqrt(su ** 2 + sv ** 2)
    d = -zi
    m = d <= 20.0
    (A, p), _ = curve_fit(lambda d, A, p: A * d ** (-p), d[m], smag[m],
                          p0=[smag[m][0], 0.7], maxfev=60000)
    return float(p), float(A)


def _pooled_fit(cols, targets):
    """Pooled OLS over all (time, point) samples.

    cols: list of predictor arrays, each (N,), the design columns AFTER a leading ones column
          is prepended here.  targets: (N, ncol) stacked targets.  Returns beta (npred, ncol).
    """
    X = np.column_stack([np.ones_like(cols[0])] + list(cols)).astype(np.float64)
    beta, *_ = np.linalg.lstsq(X, targets.astype(np.float64), rcond=None)
    return beta


# --------------------------------------------------------------------------- compute
def compute():
    t0 = time.time()
    print(f'loading {RAW} ...', flush=True)
    uv = xr.open_dataset(RAW).load()
    U = uv.U.values.astype(np.float32)             # (T, Z, ny, nx)
    V = uv.V.values.astype(np.float32)
    z = uv.Z.values
    T, nZ, ny, nx = U.shape
    npts = ny * nx

    ia = int(np.argmin(np.abs(z + 8.5)))           # -8.5 anchor level
    ib = ia + 1                                    # -9.5
    fill = np.where(z > -MIN_DEPTH)[0]             # 0-8 m fill levels (-0.5 .. -7.5)
    z_fill = z[fill]
    d_fill = -z_fill                               # depths 0.5 .. 7.5
    za = float(z[ia]); da = -za                    # anchor depth 8.5
    dz_shear = z[ia] - z[ib]                        # 1 m

    # observable 8 m quantities (fields, per time & point)
    u8 = U[:, ia, :, :]; v8 = V[:, ia, :, :]
    su8 = (U[:, ia, :, :] - U[:, ib, :, :]) / dz_shear      # measured dU/dz at ~9 m
    sv8 = (V[:, ia, :, :] - V[:, ib, :, :]) / dz_shear

    # horizontal metric (small box: cos(lat) at centre)
    lat_c = float(uv.YC.mean())
    x_m = (uv.XC.values - uv.XC.values.mean()) * np.cos(np.radians(lat_c)) * DEG
    y_m = (uv.YC.values - uv.YC.values.mean()) * DEG

    # --- mean power-law exponent ---
    u_mean = U.mean((0, 2, 3)); v_mean = V.mean((0, 2, 3))
    p, A = _fit_p(z, u_mean, v_mean)
    print(f'  mean-shear power law p = {p:.3f}', flush=True)

    # extrapolation displacement factors per fill level (velocity increment = shear * factor)
    const_fac = da - d_fill                                  # linear (constant shear)
    power_fac = da / (1 - p) * (1 - (d_fill / da) ** (1 - p))  # power law (p=0 -> const)

    # --- pooled regressions of instantaneous u(z),v(z) on the 8 m state ---
    # targets: u and v at each fill level, stacked; predictors: [u8, v8] and [u8,v8,su8,sv8].
    print('pooled regressions (instantaneous profiles on 8 m state) ...', flush=True)
    u8f = u8.reshape(-1); v8f = v8.reshape(-1)
    su8f = su8.reshape(-1); sv8f = sv8.reshape(-1)
    Ut = np.column_stack([U[:, iz, :, :].reshape(-1) for iz in fill])   # (N, nfill)
    Vt = np.column_stack([V[:, iz, :, :].reshape(-1) for iz in fill])
    beta_u_reg = _pooled_fit([u8f, v8f], Ut)          # (3, nfill): [1,u8,v8] -> u(z)
    beta_v_reg = _pooled_fit([u8f, v8f], Vt)
    beta_u_rsh = _pooled_fit([u8f, v8f, su8f, sv8f], Ut)   # (5, nfill): +shear
    beta_v_rsh = _pooled_fit([u8f, v8f, su8f, sv8f], Vt)
    del Ut, Vt, u8f, v8f, su8f, sv8f

    # --- schemes: functions returning (u_est_k, v_est_k) fields for fill index k ---
    def est(method, k):
        iz = fill[k]
        if method == 'const':
            return u8 + su8 * const_fac[k], v8 + sv8 * const_fac[k]
        if method == 'power':
            return u8 + su8 * power_fac[k], v8 + sv8 * power_fac[k]
        if method == 'reg':
            bu, bv = beta_u_reg[:, k], beta_v_reg[:, k]
            return (bu[0] + bu[1] * u8 + bu[2] * v8,
                    bv[0] + bv[1] * u8 + bv[2] * v8)
        if method == 'regsh':
            bu, bv = beta_u_rsh[:, k], beta_v_rsh[:, k]
            return (bu[0] + bu[1] * u8 + bu[2] * v8 + bu[3] * su8 + bu[4] * sv8,
                    bv[0] + bv[1] * u8 + bv[2] * v8 + bv[3] * su8 + bv[4] * sv8)
        raise ValueError(method)

    methods = ['const', 'power', 'reg', 'regsh']

    # contour depths (cumulative w evaluated at each of these 0-8 m depths)
    cdepth_k = [int(np.argmin(np.abs(d_fill - dd))) for dd in (0.5, 2.5, 4.5, 6.5)]

    # --- true w(z) from the cumulative 0->z convergence (w = 0 at surface) ---
    # w(z) = -integral_0^z div dz'; the full 0-8 m sum is w at the 8 m sampling limit.
    print('w(z) from cumulative 0-z convergence: truth + each scheme ...', flush=True)
    w_true = np.zeros((T, ny, nx), np.float32)
    w_true_z_all = []                               # cumulative w(z) at every fill depth, m/day
    for k, iz in enumerate(fill):
        w_true += -_div_np(U[:, iz, :, :], V[:, iz, :, :], x_m, y_m) * 1.0
        w_true_z_all.append((w_true * DAY).astype(np.float32).copy())
    w_true *= DAY                                                    # m/day (w at 8 m)
    w_true_z = {round(float(d_fill[k]), 1): w_true_z_all[k] for k in cdepth_k}

    # single-point profiles at 140°W (220°E): equator and ±0.5°N
    yc_v = uv.YC.values; xc_v = uv.XC.values
    prof_pts = []
    for lat0, lon0 in [(0.0, 220.0), (0.5, 220.0), (-0.5, 220.0)]:
        iy = int(np.argmin(np.abs(yc_v - lat0))); ix = int(np.argmin(np.abs(xc_v - lon0)))
        prof_pts.append({
            'lat': float(yc_v[iy]), 'lon': float(xc_v[ix]), 'iy': iy, 'ix': ix,
            'true_rms': np.array([np.sqrt(np.mean(w[:, iy, ix] ** 2)) for w in w_true_z_all]),
            'true_mean': np.array([np.mean(w[:, iy, ix]) for w in w_true_z_all]),
            'err_rms': {}, 'err_mean': {}})
    u8b = np.quantile(u8, [0.01, 0.99]); v8b = np.quantile(v8, [0.01, 0.99])
    ue = np.linspace(u8b[0], u8b[1], 41); ve = np.linspace(v8b[0], v8b[1], 41)
    u8flat = u8.reshape(-1); v8flat = v8.reshape(-1)
    cnt, _, _ = np.histogram2d(u8flat, v8flat, bins=[ue, ve])
    cnt_safe = np.where(cnt < 30, np.nan, cnt)      # hide sparsely sampled bins

    # scatter subsample (same points for every method)
    rng = np.random.default_rng(0)
    idx = rng.choice(T * ny * nx, size=25000, replace=False)
    wt_flat = w_true.reshape(-1)

    out = {'p': p, 'da': da, 'd_fill': d_fill, 'z_fill': z_fill,
           'const_fac': const_fac, 'power_fac': power_fac,
           'ue': ue, 've': ve, 'cnt': cnt_safe, 'cdepth_d': d_fill[cdepth_k],
           'w_true_mean': w_true.mean(0), 'w_true_rms': float(np.sqrt((w_true ** 2).mean())),
           'xc': uv.XC.values, 'yc': uv.YC.values,
           'wt_scatter': wt_flat[idx],
           'w_true_mean_z': {d: wz.mean(0) for d, wz in w_true_z.items()},
           'wprof_d': d_fill,
           'wprof_true_rms': np.array([np.sqrt(np.nanmean(w ** 2)) for w in w_true_z_all]),
           'wprof_true_mean': np.array([np.nanmean(w) for w in w_true_z_all]),
           'wprof_pts': prof_pts,
           'methods': methods, 'contour': {}, 'w_err_mean': {}, 'w_err_rms': {},
           'w_err_scatter': {}, 'contour_w': {}, 'w_err_mean_z': {}, 'w_err_rms_z': {},
           'w_err_rms_prof': {}, 'w_err_mean_prof': {}}

    # regression gain profiles (the "power law for the fluctuations")
    out['reg_gain_uu'] = beta_u_reg[1, :]           # ∂u(z)/∂u8
    out['reg_gain_vv'] = beta_v_reg[2, :]           # ∂v(z)/∂v8
    out['reg_gain_uv'] = beta_u_reg[2, :]           # Ekman turning: u(z) from v8
    out['reg_gain_vu'] = beta_v_reg[1, :]
    # mean surface-ward velocity increment vs the power-law prediction (mean follows power law?)
    du_mean = np.array([(U[:, iz, :, :] - u8).mean() for iz in fill])
    dv_mean = np.array([(V[:, iz, :, :] - v8).mean() for iz in fill])
    out['du_mean'] = du_mean; out['dv_mean'] = dv_mean
    out['su8_mean'] = float(su8.mean()); out['sv8_mean'] = float(sv8.mean())

    for method in methods:
        w_est = np.zeros((T, ny, nx), np.float32)
        werr_rms_prof = np.empty(len(fill)); werr_mean_prof = np.empty(len(fill))
        pt_rms = [np.empty(len(fill)) for _ in prof_pts]
        pt_mean = [np.empty(len(fill)) for _ in prof_pts]
        for k, iz in enumerate(fill):
            ue_k, ve_k = est(method, k)
            w_est += -_div_np(ue_k, ve_k, x_m, y_m) * 1.0
            # cumulative-w error at this fill depth (for the vertical profile)
            werr_z = (w_est * DAY) - w_true_z_all[k]               # m/day
            werr_rms_prof[k] = np.sqrt(np.nanmean(werr_z ** 2))
            werr_mean_prof[k] = np.nanmean(werr_z)
            for pi, pt in enumerate(prof_pts):                    # single-point profiles
                ts = werr_z[:, pt['iy'], pt['ix']]
                pt_rms[pi][k] = np.sqrt(np.mean(ts ** 2)); pt_mean[pi][k] = np.mean(ts)
            # contour bins + maps at the selected depths only
            if k in cdepth_k:
                dep = round(float(d_fill[k]), 1)
                # velocity-error contours (u, v) vs the 8 m flow state
                for comp, err in (('u', (ue_k - U[:, iz, :, :]).reshape(-1)),
                                  ('v', (ve_k - V[:, iz, :, :]).reshape(-1))):
                    s, _, _ = np.histogram2d(u8flat, v8flat, bins=[ue, ve], weights=err)
                    out['contour'][(method, comp, dep)] = s / cnt_safe
                # cumulative-w error at this depth: map and (u8,v8) contour
                out['w_err_mean_z'].setdefault(method, {})[dep] = werr_z.mean(0)
                out['w_err_rms_z'].setdefault(method, {})[dep] = float(werr_rms_prof[k])
                sw, _, _ = np.histogram2d(u8flat, v8flat, bins=[ue, ve],
                                          weights=werr_z.reshape(-1))
                out['contour_w'][(method, dep)] = sw / cnt_safe
        out['w_err_rms_prof'][method] = werr_rms_prof
        out['w_err_mean_prof'][method] = werr_mean_prof
        for pi, pt in enumerate(prof_pts):
            pt['err_rms'][method] = pt_rms[pi]; pt['err_mean'][method] = pt_mean[pi]
        w_est *= DAY
        w_err = w_est - w_true
        out['w_err_mean'][method] = w_err.mean(0)
        out['w_err_rms'][method] = float(np.sqrt((w_err ** 2).mean()))
        out['w_err_scatter'][method] = w_err.reshape(-1)[idx]
        print(f'  {method:6s}  w-error RMS {out["w_err_rms"][method]:.3f} m/day', flush=True)

    with open(CACHE, 'wb') as f:
        pickle.dump(out, f)
    print(f'cached -> {CACHE}  ({time.time() - t0:.0f} s)', flush=True)
    return out


# ------------------------------------------------------------------------------- plot
CONTOUR_LIM = 2.5   # shared symmetric colour limit (cm/s) for all extrap_error_contours figures

METHOD_LABEL = {
    'const': 'constant shear',
    'power': 'power-law shear',
    'reg': 'regression  u,v(8 m)',
    'regsh': 'regression  u,v(8 m) + shear(8 m)',
}


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name + '.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', path, flush=True)


def _fig_extrap_shapes(out):
    """The three extrapolation shapes and the power-law fit: does the mean follow a power law
    (yes, by construction of the fit) and does the fluctuation (regression gain) follow it too?"""
    d = out['d_fill']; p = out['p']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    # (a) mean surface-ward velocity increment vs constant / power-law extrapolation
    su = out['su8_mean']; sv = out['sv8_mean']
    smag = np.hypot(su, sv)
    dmag_mean = np.hypot(out['du_mean'], out['dv_mean'])
    axes[0].plot(dmag_mean * 100, -d, 'ko-', label='model mean |Δu| (z) − (8 m)')
    axes[0].plot(smag * out['const_fac'] * 100, -d, color='#d62728', lw=2.2,
                 label='constant shear')
    axes[0].plot(smag * out['power_fac'] * 100, -d, color='C0', lw=2.2, ls='--',
                 label=f'power law  p={p:.2f}')
    axes[0].set_xlabel('mean |Δvelocity| from 8 m  (cm s$^{-1}$)')
    axes[0].set_ylabel('depth (m)')
    axes[0].set_title('(a) MEAN increment: power law captures the surface intensification')
    axes[0].grid(alpha=0.3); axes[0].legend(loc='lower right', frameon=True)

    # (b) regression gain b(z): how much a fluctuation at 8 m is amplified toward the surface
    axes[1].axvline(1.0, color='0.6', ls=':', lw=1.3)
    axes[1].plot(out['reg_gain_uu'], -d, 'o-', color='C0', lw=2.2, label='∂u(z)/∂u(8 m)')
    axes[1].plot(out['reg_gain_vv'], -d, 's-', color='C1', lw=2.2, label='∂v(z)/∂v(8 m)')
    axes[1].plot(out['reg_gain_vu'], -d, '^--', color='C3', lw=1.6, alpha=0.8,
                 label='∂v(z)/∂u(8 m)  (Ekman turning)')
    amp = 1 / (1 - p)
    axes[1].annotate(f'power law (mean) would\namplify to ×{amp:.1f} at surface →',
                     xy=(1.0, -0.5), xytext=(1.05, -3.0), fontsize=9, color='0.35',
                     va='center', arrowprops=dict(arrowstyle='->', color='0.5'))
    axes[1].set_xlim(0.95, 1.10)
    axes[1].set_xlabel('regression gain  ∂(z)/∂(8 m)')
    axes[1].set_title('(b) FLUCTUATIONS: regression gain ≈ 1 — power law does NOT hold')
    axes[1].grid(alpha=0.3); axes[1].legend(loc='lower left', frameon=True)
    for ax in axes:
        ax.axhspan(-MIN_DEPTH, 0, color='0.6', alpha=0.10, lw=0)
    fig.suptitle('power law describes the MEAN profile, not the variability',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'powerlaw_mean_vs_var')


def _fig_w_scatter(out):
    """w_error vs w_true for each extrapolation scheme (instantaneous, subsampled)."""
    wt = out['wt_scatter']
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    xlim = np.nanpercentile(np.abs(wt), 99.5)
    # y-axis on the ERROR scale (errors are ~20-50x smaller than the w signal) so the
    # scheme-to-scheme differences are visible
    ylim = np.nanpercentile(np.abs(np.concatenate(
        [out['w_err_scatter'][m] for m in out['methods']])), 99.0)
    for ax, method in zip(axes.ravel(), out['methods']):
        we = out['w_err_scatter'][method]
        ax.axhline(0, color='0.6', lw=0.8)
        ax.scatter(wt, we, s=3, alpha=0.12, color='C0', edgecolors='none')
        rms = out['w_err_rms'][method]
        bias = float(np.mean(we))
        sk = 1 - np.nanvar(we) / np.nanvar(wt)
        ax.text(0.03, 0.97, f'{METHOD_LABEL[method]}\nRMS {rms:.3f} m/day\n'
                f'bias {bias:+.3f}\nskill 1−σ²ₑ/σ²ₜ = {sk:.3f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
        ax.set_xlim(-xlim, xlim); ax.set_ylim(-ylim, ylim)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel('true w(8 m)  (m day$^{-1}$)')
    axes[1, 1].set_xlabel('true w(8 m)  (m day$^{-1}$)')
    axes[0, 0].set_ylabel('w error  (m day$^{-1}$)')
    axes[1, 0].set_ylabel('w error  (m day$^{-1}$)')
    fig.suptitle(f'instantaneous upwelling error vs truth  (true w RMS {out["w_true_rms"]:.2f} '
                 'm day$^{-1}$) — extrapolation scheme sets the 0–8 m fill',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, 'powerlaw_w_scatter')


def _fig_w_maps(out):
    """Time-mean true w(z) and each scheme's time-mean w error, by depth (rows)."""
    xc, yc = out['xc'], out['yc']
    methods = out['methods']
    depths = sorted(out['w_true_mean_z'].keys())          # 0.5 .. 6.5
    rows = depths + [8.0]                                  # + full 0-8 m (w at 8 m)

    def tmap(d):  return out['w_true_mean'] if d == 8.0 else out['w_true_mean_z'][d]
    def emap(m, d): return out['w_err_mean'][m] if d == 8.0 else out['w_err_mean_z'][m][d]
    def erms(m, d): return out['w_err_rms'][m] if d == 8.0 else out['w_err_rms_z'][m][d]

    # error columns share one scale across every depth and scheme
    el = max(np.nanpercentile(np.abs(emap(m, d)), 98) for m in methods for d in rows)
    ncol = 1 + len(methods)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.4 * ncol, 3.5 * len(rows)),
                             squeeze=False)
    for r, d in enumerate(rows):
        dlabel = '8 m (full 0–8 m)' if d == 8.0 else f'{d:g} m'
        tt = tmap(d); wl = np.nanpercentile(np.abs(tt), 98)
        _panel(axes[r, 0], tt, xc, yc, 'time-mean true cumulative w(z)' if r == 0 else '',
               cmo.balance, -wl, wl, 'w (m day$^{-1}$)',
               f'mean {np.nanmean(tt):.3f}\nspatial RMS {np.sqrt(np.nanmean(tt**2)):.3f}')
        axes[r, 0].set_ylabel(f'{dlabel}\nlat (°N)')
        for c, m in enumerate(methods, start=1):
            em = emap(m, d)
            _panel(axes[r, c], em, xc, yc,
                   f'error — {METHOD_LABEL[m]}' if r == 0 else '', cmo.balance,
                   -el, el, 'error (m day$^{-1}$)',
                   f'mean {np.nanmean(em):+.4f}\nspatial RMS {erms(m, d):.4f}')
        if r < len(rows) - 1:                              # keep lon label on bottom row only
            for c in range(ncol):
                axes[r, c].set_xlabel('')
    fig.suptitle('time-mean cumulative upwelling w(z) = −∫₀ᶻ∇·u dz′ and extrapolation error, '
                 'by depth (rows); error columns share one scale', fontsize=13,
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, 'w_error_maps')


def _fig_w_error_profiles(out):
    """Vertical profiles of the cumulative-w error (RMS + bias) per scheme, box + points."""
    d = out['wprof_d']
    methods = out['methods']
    colors = dict(zip(methods, ['C3', 'C1', 'C0', 'C2']))
    rows = [{'label': 'box mean', 'true_rms': out['wprof_true_rms'],
             'true_mean': out['wprof_true_mean'], 'err_rms': out['w_err_rms_prof'],
             'err_mean': out['w_err_mean_prof']}]
    for pt in out['wprof_pts']:
        rows.append({'label': f"{pt['lat']:+.2f}°N, {360 - pt['lon']:.0f}°W",
                     'true_rms': pt['true_rms'], 'true_mean': pt['true_mean'],
                     'err_rms': pt['err_rms'], 'err_mean': pt['err_mean']})
    nr = len(rows)
    fig, axes = plt.subplots(nr, 3, figsize=(14, 4.0 * nr), sharey=True, sharex='col',
                             squeeze=False)
    for r, rd in enumerate(rows):
        a0, a1, a2 = axes[r]
        a0.plot(rd['true_rms'], d, 'k-', lw=2, label='RMS in time')
        a0.plot(np.abs(rd['true_mean']), d, 'k--', lw=1.5, label='|time-mean|')
        a0.set_ylabel(f"{rd['label']}\ndepth  (m)")
        for m in methods:
            a1.plot(rd['err_rms'][m], d, color=colors[m], lw=2, label=METHOD_LABEL[m])
        a2.axvline(0, color='0.6', lw=0.8)
        for m in methods:
            a2.plot(rd['err_mean'][m], d, color=colors[m], lw=2)
        for ax in (a0, a1, a2):
            ax.grid(alpha=0.3)
        if r == 0:
            a0.set_title('true cumulative w(z) — the signal', fontsize=11)
            a1.set_title('w-error RMS in time  (√⟨e²⟩$_t$, not the time-mean)', fontsize=11)
            a2.set_title('w-error bias (time mean)', fontsize=11)
            a0.legend(loc='lower right', frameon=True, fontsize=9)
            a1.legend(loc='lower right', frameon=True, fontsize=9)
    for c, xl in enumerate(('w  (m day$^{-1}$)', 'RMS w error  (m day$^{-1}$)',
                            'mean w error  (m day$^{-1}$)')):
        axes[-1, c].set_xlabel(xl)
    axes[0, 0].invert_yaxis()
    fig.suptitle('vertical profiles of the cumulative-w extrapolation error — box mean and '
                 'single points at 140°W (depth 0.5–7.5 m; 7.5 m ≈ w at 8 m; RMS over time, '
                 'box row also over space)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, 'w_error_profiles')


def _fig_w_contours(out):
    """Cumulative-w error vs the 8 m flow state (u8, v8): rows = depth, cols = scheme."""
    ue, ve = out['ue'], out['ve']
    uc = 0.5 * (ue[:-1] + ue[1:]) * 100
    vc = 0.5 * (ve[:-1] + ve[1:]) * 100
    methods = out['methods']
    depths = sorted(out['w_true_mean_z'].keys())
    allv = np.concatenate([out['contour_w'][(m, d)].ravel() for m in methods for d in depths])
    lim = np.nanpercentile(np.abs(allv[np.isfinite(allv)]), 98)
    lev = np.linspace(-lim, lim, 41)
    fig, axes = plt.subplots(len(depths), len(methods),
                             figsize=(4.3 * len(methods), 3.7 * len(depths)),
                             sharex=True, sharey=True, squeeze=False)
    for r, d in enumerate(depths):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            cf = ax.contourf(uc, vc, out['contour_w'][(m, d)].T, levels=lev,
                             cmap=cmo.balance, extend='both')
            ax.contour(uc, vc, out['cnt'].T, levels=[0], colors='0.7', linewidths=0.4)
            ax.axhline(0, color='0.5', lw=0.6); ax.axvline(0, color='0.5', lw=0.6)
            if r == 0:
                ax.set_title(METHOD_LABEL[m], fontsize=11)
            ax.text(0.03, 0.95, f'w error @ {d:g} m', transform=ax.transAxes, va='top',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
            if r == len(depths) - 1:
                ax.set_xlabel('u(8 m)  (cm s$^{-1}$)')
            if c == 0:
                ax.set_ylabel('v(8 m)  (cm s$^{-1}$)')
    cb = fig.colorbar(cf, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label('mean w error  (m day$^{-1}$)', fontsize=10)
    fig.suptitle('cumulative w error vs the 8 m flow state, by depth (rows) and scheme (cols)',
                 fontsize=13, fontweight='bold')
    _save(fig, 'w_error_contours')


def _panel(ax, arr, xc, yc, title, cmap, vmin, vmax, clabel, annot=None):
    pc = ax.pcolormesh(xc, yc, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('lon (°E)')
    cb = plt.colorbar(pc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(clabel, fontsize=9.5)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    if annot:
        ax.text(0.03, 0.03, annot, transform=ax.transAxes, fontsize=9, va='bottom',
                ha='left', bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))


def _contour_grid(out, methods, name, suptitle):
    """Filled contour of mean u/v error vs (u8, v8), rows = component, cols = (method|depth)."""
    ue, ve = out['ue'], out['ve']
    uc = 0.5 * (ue[:-1] + ue[1:]) * 100          # cm/s
    vc = 0.5 * (ve[:-1] + ve[1:]) * 100
    depths = [round(float(d), 1) for d in out['cdepth_d']]
    # collect arrays to share one symmetric scale
    # fixed shared scale (cm/s) so every extrap_error_contours_* figure is comparable
    lim = CONTOUR_LIM
    lev = np.linspace(-lim, lim, 41)
    return uc, vc, lim, lev


def _fig_contours_by_depth(out, method='reg'):
    """Error(u8,v8) for one scheme at several depths — rows u/v, cols depth."""
    uc, vc, lim, lev = _contour_grid(out, [method], None, None)
    depths = [round(float(d), 1) for d in out['cdepth_d']]
    fig, axes = plt.subplots(2, len(depths), figsize=(4.1 * len(depths), 8), sharex=True,
                             sharey=True)
    for j, dep in enumerate(depths):
        for i, comp in enumerate(('u', 'v')):
            ax = axes[i, j]
            arr = out['contour'][(method, comp, dep)].T * 100      # cm/s, (v,u)->(y,x)
            cf = ax.contourf(uc, vc, arr, levels=lev, cmap=cmo.balance, extend='both')
            ax.contour(uc, vc, out['cnt'].T, levels=[0], colors='0.7', linewidths=0.4)
            ax.axhline(0, color='0.5', lw=0.6); ax.axvline(0, color='0.5', lw=0.6)
            ax.set_title(f'{comp}-error @ {dep:g} m', fontsize=11)
            if i == 1:
                ax.set_xlabel('u(8 m)  (cm s$^{-1}$)')
            if j == 0:
                ax.set_ylabel('v(8 m)  (cm s$^{-1}$)')
    cb = fig.colorbar(cf, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label('mean error  u$_{est}$ − u$_{true}$  (cm s$^{-1}$)', fontsize=10)
    fig.suptitle(f'{METHOD_LABEL[method]}: velocity error vs the 8 m flow state, by depth  '
                 '(flat ⇒ a single linear rule suffices)', fontsize=12.5, fontweight='bold')
    _save(fig, f'extrap_error_contours_{method}')


def _fig_contours_by_method(out, dep=None):
    """Error(u8,v8) at one (shallow) depth for constant / power / regression — rows u/v."""
    depths = [round(float(d), 1) for d in out['cdepth_d']]
    if dep is None:
        dep = depths[0]
    methods = ['const', 'power', 'reg', 'regsh']
    uc, vc, lim, lev = _contour_grid(out, methods, None, None)
    fig, axes = plt.subplots(2, len(methods), figsize=(4.3 * len(methods), 8), sharex=True,
                             sharey=True)
    for j, method in enumerate(methods):
        for i, comp in enumerate(('u', 'v')):
            ax = axes[i, j]
            arr = out['contour'][(method, comp, dep)].T * 100
            cf = ax.contourf(uc, vc, arr, levels=lev, cmap=cmo.balance, extend='both')
            ax.contour(uc, vc, out['cnt'].T, levels=[0], colors='0.7', linewidths=0.4)
            ax.axhline(0, color='0.5', lw=0.6); ax.axvline(0, color='0.5', lw=0.6)
            if i == 0:
                ax.set_title(METHOD_LABEL[method], fontsize=11)
            ax.text(0.03, 0.95, f'{comp} error', transform=ax.transAxes, va='top',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
            if i == 1:
                ax.set_xlabel('u(8 m)  (cm s$^{-1}$)')
            if j == 0:
                ax.set_ylabel('v(8 m)  (cm s$^{-1}$)')
    cb = fig.colorbar(cf, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label('mean error  (cm s$^{-1}$)', fontsize=10)
    fig.suptitle(f'velocity error vs the 8 m flow state at {dep:g} m — a state-dependent '
                 'gradient means that scheme mis-scales with the flow', fontsize=12.5,
                 fontweight='bold')
    _save(fig, 'extrap_error_contours_by_method')


def plot():
    with open(CACHE, 'rb') as f:
        out = pickle.load(f)
    _fig_extrap_shapes(out)
    _fig_w_scatter(out)
    _fig_w_maps(out)
    _fig_w_error_profiles(out)
    _fig_w_contours(out)
    for method in out['methods']:
        _fig_contours_by_depth(out, method)
    _fig_contours_by_method(out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('all', 'compute'):
        compute()
    if mode in ('all', 'plot'):
        plot()
