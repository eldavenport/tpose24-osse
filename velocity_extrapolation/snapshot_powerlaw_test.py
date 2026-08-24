"""Does a per-snapshot (time-varying) power-law exponent help, where the fixed
mean-fit exponent did not?

The `power` method in power_law_extrap.py fits |S| ~ d^(-p) to the TIME-MEAN shear
(single p=0.70) and applies it to every snapshot's instantaneous 8 m shear. This
script asks the two things the user raised:

  (1) fit the power law to INDIVIDUAL SNAPSHOTS -> distribution of p(t);
  (2) use a time-varying exponent p(t) to extrapolate and see if w improves.

Also computes, per snapshot, the OPTIMAL surface shear-amplification for the box-mean
increment, to show whether the surface intensification is an instantaneous property
(then a time-varying p could work) or a time-mean-only property (then it cannot).

Reads uv_region.nc; regenerates it from the model if missing.
"""
import os, sys, time
import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'uv_region.nc')
DAY = 86400.0

sys.path.insert(0, HERE)
import run_shear_analysis as rs  # for _load_centered_uv, box constants


def get_box():
    if os.path.exists(RAW):
        print('loading cached', RAW, flush=True)
        return xr.open_dataset(RAW).load()
    print('reading model box (this is the slow part)...', flush=True)
    t0 = time.time()
    uv = rs._load_centered_uv()
    # restrict to the exact box (drop the interpolation halo)
    uv = uv.sel(XC=slice(rs.LON0, rs.LON1), YC=slice(rs.LAT0, rs.LAT1)).load()
    print(f'  loaded in {time.time()-t0:.0f}s, shape U={uv.U.shape}', flush=True)
    uv.to_netcdf(RAW)
    return uv


def main():
    uv = get_box()
    z = uv.Z.values                      # negative, descending
    U = uv.U.values                      # (T, Z, Y, X)
    V = uv.V.values
    T, nz, ny, nx = U.shape
    print(f'T={T} nz={nz} ny={ny} nx={nx}', flush=True)

    # interface shear (between adjacent Z), placed at midpoints
    zi = 0.5 * (z[:-1] + z[1:])
    dz = np.diff(z)                                  # negative
    SU = np.diff(U, axis=1) / dz[None, :, None, None]
    SV = np.diff(V, axis=1) / dz[None, :, None, None]
    Smag = np.sqrt(SU**2 + SV**2)                    # (T, nzi, Y, X)

    # anchor: shallowest sampled level at/below 8 m -> -8.5 m; top shear at -9 m
    ia = int(np.argmin(np.abs(z + 8.5)))             # index of -8.5 m in z
    ib = ia + 1                                       # -9.5 m
    # shear interface index for the -9 m interface (between ia and ib)
    ki = int(np.argmin(np.abs(zi + 9.0)))
    su8 = SU[:, ki, :, :]                             # (T,Y,X) measured 8 m shear
    sv8 = SV[:, ki, :, :]

    # fill levels: z shallower than -8 m (i.e. -0.5 .. -7.5)
    fill = np.where(z > -8.0)[0]
    zf = z[fill]
    da = 8.5
    d_fill = -zf                                      # depth (m), positive: 0.5..7.5

    # deeper interfaces used to fit the power law: 8.5-20 m
    deep = np.where((zi <= -8.0) & (zi >= -20.0))[0]
    dz_deep = -zi[deep]                               # depths (positive)
    print('deep interface depths (m):', np.round(dz_deep, 1), flush=True)

    # ---- (1) per-snapshot BOX-MEAN power-law exponent p(t) --------------------
    # fit log|S_mean(z)| = logA - p*log d over the deep interfaces, per snapshot
    Smag_boxmean = np.nanmean(Smag[:, deep, :, :], axis=(2, 3))   # (T, ndeep)
    logd = np.log(dz_deep)
    X = np.vstack([np.ones_like(logd), -logd]).T                 # [1, -log d]
    # least squares per snapshot
    p_t = np.full(T, np.nan)
    for t in range(T):
        y = np.log(Smag_boxmean[t])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        p_t[t] = coef[1]
    # mean-profile fit for reference
    y_mean = np.log(np.nanmean(Smag_boxmean, axis=0))
    coef_m, *_ = np.linalg.lstsq(X, y_mean, rcond=None)
    p_mean = coef_m[1]
    print('\n=== (1) per-snapshot box-mean power-law exponent p(t) ===', flush=True)
    print(f'  p fit to the time-mean profile : {p_mean:.3f}', flush=True)
    print(f'  p(t): mean={np.nanmean(p_t):.3f}  std={np.nanstd(p_t):.3f}'
          f'  5-95%=[{np.nanpercentile(p_t,5):.2f}, {np.nanpercentile(p_t,95):.2f}]'
          f'  min/max=[{np.nanmin(p_t):.2f}, {np.nanmax(p_t):.2f}]', flush=True)
    amp_t = 1.0 / (1.0 - np.clip(p_t, None, 0.98))
    print(f'  implied surface shear amplification 1/(1-p): '
          f'mean={np.nanmean(amp_t):.2f} std={np.nanstd(amp_t):.2f}', flush=True)

    # ---- divergence helper ---------------------------------------------------
    deg = np.pi / 180 * 6371000.0
    lat_c = float(uv.YC.mean())
    x_m = (uv.XC.values - uv.XC.values.mean()) * np.cos(np.radians(lat_c)) * deg
    y_m = (uv.YC.values - uv.YC.values.mean()) * deg

    def wfill(su_amp_fac, sv_amp_fac):
        """w at 8 m from integrating -div of the extrapolated 0-8 m velocity.
        su_amp_fac, sv_amp_fac: (nfill,) or (T,nfill,1,1) multipliers on su8,sv8
        giving the velocity increment at each fill level. Returns w (T,Y,X) in m/day."""
        w = np.zeros((T, ny, nx))
        for j, k in enumerate(fill):
            fu = su_amp_fac[..., j] if su_amp_fac.ndim > 1 else su_amp_fac[j]
            fv = sv_amp_fac[..., j] if sv_amp_fac.ndim > 1 else sv_amp_fac[j]
            u_lev = U[:, ia, :, :] + su8 * np.asarray(fu).reshape(-1, 1, 1) if np.ndim(fu) else U[:, ia, :, :] + su8 * fu
            v_lev = V[:, ia, :, :] + sv8 * np.asarray(fv).reshape(-1, 1, 1) if np.ndim(fv) else V[:, ia, :, :] + sv8 * fv
            dudx = np.gradient(u_lev, x_m, axis=2)
            dvdy = np.gradient(v_lev, y_m, axis=1)
            w += -(dudx + dvdy) * 1.0     # 1 m layer thickness
        return w * DAY

    # true near-surface w: integrate true convergence over the fill levels
    w_true = np.zeros((T, ny, nx))
    for k in fill:
        dudx = np.gradient(U[:, k, :, :], x_m, axis=2)
        dvdy = np.gradient(V[:, k, :, :], y_m, axis=1)
        w_true += -(dudx + dvdy) * 1.0
    w_true *= DAY

    # amplification factors (velocity increment per unit measured shear) at each fill level
    const_fac = (da - d_fill)                                    # constant shear
    power_fac_fixed = da / (1 - p_mean) * (1 - (d_fill / da) ** (1 - p_mean))

    # per-snapshot power law: same functional form but p = p(t)
    pt = np.clip(p_t, None, 0.95)[:, None]                       # (T,1)
    power_fac_t = da / (1 - pt) * (1 - (d_fill[None, :] / da) ** (1 - pt))  # (T,nfill)

    def wrms(w):
        e = w - w_true
        return np.sqrt(np.nanmean(e**2))

    w_const = wfill(const_fac, const_fac)
    w_powfix = wfill(power_fac_fixed, power_fac_fixed)
    # time-varying: pass (T,nfill) broadcast
    w_powt = np.zeros((T, ny, nx))
    for j in range(len(fill)):
        u_lev = U[:, ia, :, :] + su8 * power_fac_t[:, j][:, None, None]
        v_lev = V[:, ia, :, :] + sv8 * power_fac_t[:, j][:, None, None]
        dudx = np.gradient(u_lev, x_m, axis=2)
        dvdy = np.gradient(v_lev, y_m, axis=1)
        w_powt += -(dudx + dvdy) * 1.0
    w_powt *= DAY

    print('\n=== (2) instantaneous w-error RMS (m/day) ===', flush=True)
    print(f'  true w RMS                         : {np.sqrt(np.nanmean(w_true**2)):.4f}', flush=True)
    print(f'  constant shear                     : {wrms(w_const):.4f}', flush=True)
    print(f'  power law, FIXED p={p_mean:.2f}          : {wrms(w_powfix):.4f}', flush=True)
    print(f'  power law, PER-SNAPSHOT p(t)       : {wrms(w_powt):.4f}', flush=True)

    # ---- (3) is the surface intensification instantaneous or mean-only? ------
    # optimal per-snapshot uniform amplification a(t) on the measured shear that
    # best matches the true 0-8m velocity increment (box-mean), least squares.
    # true increment at each fill level (box mean): mean over space of (u(z)-u(anchor))
    du_true = np.nanmean(U[:, fill, :, :] - U[:, ia:ia+1, :, :], axis=(2, 3))  # (T,nfill)
    dv_true = np.nanmean(V[:, fill, :, :] - V[:, ia:ia+1, :, :], axis=(2, 3))
    su8_bm = np.nanmean(su8, axis=(1, 2))    # (T,)
    sv8_bm = np.nanmean(sv8, axis=(1, 2))
    # model over fill levels: du_true[t,j] ~ a(t) * su8_bm[t] * const_fac[j]
    # solve a(t) = sum_j (du*base) / sum_j base^2  with base = su8_bm*const_fac (+v)
    base_u = su8_bm[:, None] * const_fac[None, :]
    base_v = sv8_bm[:, None] * const_fac[None, :]
    num = np.nansum(du_true * base_u + dv_true * base_v, axis=1)
    den = np.nansum(base_u**2 + base_v**2, axis=1)
    a_opt = num / den
    print('\n=== (3) optimal per-snapshot shear amplification a(t) (box-mean incr) ===', flush=True)
    print('  a=1 -> constant shear is exactly right; a=3.4 -> mean power law right', flush=True)
    print(f'  a(t): mean={np.nanmean(a_opt):.2f}  std={np.nanstd(a_opt):.2f}'
          f'  5-95%=[{np.nanpercentile(a_opt,5):.2f}, {np.nanpercentile(a_opt,95):.2f}]', flush=True)
    # amplification of the time-MEAN increment
    a_meanprofile = (np.nansum(np.nanmean(du_true,0)*np.nanmean(base_u,0)
                               + np.nanmean(dv_true,0)*np.nanmean(base_v,0))
                     / np.nansum(np.nanmean(base_u,0)**2 + np.nanmean(base_v,0)**2))
    print(f'  amplification of the TIME-MEAN increment: {a_meanprofile:.2f}', flush=True)

    np.savez(os.path.join(HERE, 'snapshot_powerlaw_test.npz'),
             p_t=p_t, p_mean=p_mean, a_opt=a_opt, su8_bm=su8_bm, sv8_bm=sv8_bm,
             w_const_rms=wrms(w_const), w_powfix_rms=wrms(w_powfix),
             w_powt_rms=wrms(w_powt), w_true_rms=np.sqrt(np.nanmean(w_true**2)))
    print('\nsaved snapshot_powerlaw_test.npz', flush=True)


if __name__ == '__main__':
    main()
