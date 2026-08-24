"""
Vertical and lateral eddy transport of heat and momentum for the virtual mooring.

Using temporal fluctuations (primes = deviations from the record-mean at each point):
  vertical   w'T' (heat, W m^-2 via rho0*cp),  w'u', w'v' (momentum, m^2 s^-2)
  lateral    u'T', v'T' (heat, W m^-2),         u'v' (Reynolds stress, m^2 s^-2)

Array estimate vs model truth:
  * vertical uses the plane-fit w' with the array-mean tracer'; truth is the hull-point
    mean of w'phi' (the true footprint eddy flux).
  * lateral uses the platform-mean of a'b' over the 6 gliders + mooring; truth is the
    hull-point mean of a'b'.
These are the "fluctuation" transports the user asked for -- a stand-in until model
budget diagnostics are available.

Figures -> transport/
  flux_mean.png    time-mean eddy-flux profiles: rows = vertical / lateral, cols =
                   heat / zonal-momentum / meridional-momentum. Array vs truth per diam.
  flux_skill.png   array-vs-truth correlation r(z) of the instantaneous flux series
                   (same 2x3 layout) -- how well the array TRACKS the true transport.
  wT_hovmoller.png w'T'(z,t) truth vs estimate for the 1.0deg footprint.

Usage:  python run_transport.py
"""

import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo

import common as C
import sd_plot as P

SUB = 'transport'
HFLUX = C.HFLUX

# (truth key in hull.nc, array a, array b, label, unit-scale, unit-string, is_heat)
VERT = [('wT', 'w_est_mid', 'T', "$w'T'$", HFLUX, 'W m$^{-2}$', True),
        ('wU', 'w_est_mid', 'U', "$w'u'$", 1.0, 'm$^2$ s$^{-2}$', False),
        ('wV', 'w_est_mid', 'V', "$w'v'$", 1.0, 'm$^2$ s$^{-2}$', False)]
LAT = [('uT', 'U', 'T', "$u'T'$", HFLUX, 'W m$^{-2}$', True),
       ('vT', 'V', 'T', "$v'T'$", HFLUX, 'W m$^{-2}$', True),
       ('uv', 'U', 'V', "$u'v'$", 1.0, 'm$^2$ s$^{-2}$', False)]


def _anom(da, dim='time'):
    return da - da.mean(dim)


def array_vert_flux(arr, a, b):
    """Instantaneous vertical eddy-flux series from the plane-fit w' and array-mean b'."""
    wp = _anom(arr[a])                     # w_est_mid anomaly (time, obs_depth)
    bp = _anom(arr[b].mean('glider'))      # array-mean tracer anomaly
    return wp * bp


def array_lat_flux(arr, a, b):
    """Instantaneous lateral eddy-flux series: platform mean of a'b'."""
    return (_anom(arr[a]) * _anom(arr[b])).mean('glider')


def flux_series(diam, spec, vertical):
    """(array_series, truth_series) instantaneous flux (time, obs_depth), scaled."""
    key, a, b, _, scale, _, _ = spec
    arr = P.load_array(diam); hull = P.load_hull(diam)
    aser = (array_vert_flux(arr, a, b) if vertical else array_lat_flux(arr, a, b)) * scale
    tser = hull[key] * scale
    return aser, tser


# --------------------------------------------------------------------------- fig 1
def fig_mean(out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharey=True)
    for row, (specs, vertical, rlabel) in enumerate(
            [(VERT, True, 'vertical'), (LAT, False, 'lateral')]):
        for col, spec in enumerate(specs):
            ax = axes[row, col]
            _, _, _, label, _, unit, _ = spec
            for d in C.DIAMETERS:
                aser, tser = flux_series(d, spec, vertical)
                z = aser.obs_depth.values
                c = C.diam_color(d)
                ax.plot(aser.mean('time'), z, color=c, **P.ARRAY_KW)
                ax.plot(tser.mean('time'), z, color=c, **P.TRUTH_KW)
            ax.axvline(0, color='0.6', lw=0.8)
            ax.set_xlabel(f'{label}  ({unit})')
            P.tidy_x(ax, 4)
            if col == 0:
                ax.set_ylabel(f'{rlabel}\ndepth (m)')
    P.top_legend(fig)
    return P.finish(fig, f'{out}/flux_mean.png')


# --------------------------------------------------------------------------- fig 2
def fig_skill(out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharey=True)
    for row, (specs, vertical, rlabel) in enumerate(
            [(VERT, True, 'vertical'), (LAT, False, 'lateral')]):
        for col, spec in enumerate(specs):
            ax = axes[row, col]
            label = spec[3]
            for d in C.DIAMETERS:
                aser, tser = flux_series(d, spec, vertical)
                z = aser.obs_depth.values
                ap = _anom(aser); tp = _anom(tser)
                r = (ap * tp).mean('time') / (aser.std('time') * tser.std('time'))
                ax.plot(r, z, color=C.diam_color(d), **P.ARRAY_KW)
            ax.axvline(1, color='0.6', lw=0.8)
            ax.set_xlabel(f'{label}  array-truth $r$'); ax.set_xlim(0.8, 1.01)
            if col == 0:
                ax.set_ylabel(f'{rlabel}\ndepth (m)')
    P.top_legend(fig, method=False)
    return P.finish(fig, f'{out}/flux_skill.png')


# --------------------------------------------------------------------------- fig 3
def fig_hovmoller(out):
    """w'T'(z,t) truth / estimate / error, one COLUMN per E-W diameter (shared scales)."""
    data = {}
    for d in C.DIAMETERS:
        aser, tser = flux_series(d, VERT[0], True)     # w'T' in W/m^2
        data[d] = (tser, aser, aser - tser)
    z = data[C.DIAMETERS[0]][0].obs_depth.values
    tc = data[C.DIAMETERS[0]][0].time.values
    days = (tc - tc[0]) / np.timedelta64(1, 'D')
    vmax = float(np.nanpercentile(np.abs(np.concatenate(
        [np.abs(data[d][0].values).ravel() for d in C.DIAMETERS])), 99))
    emax = float(np.nanpercentile(np.abs(np.concatenate(
        [np.abs(data[d][2].values).ravel() for d in C.DIAMETERS])), 99))

    rows = [('model truth (disk)', 0, vmax), ('array estimate', 1, vmax),
            ('estimate $-$ truth', 2, emax)]
    nc = len(C.DIAMETERS)
    fig, axes = plt.subplots(3, nc, figsize=(4.2 * nc, 9), sharex=True, sharey=True)
    pcs = {}
    for ci, d in enumerate(C.DIAMETERS):
        axes[0, ci].set_title(f'{d:g}$^\\circ$ footprint')
        for ri, (lbl, idx, vm) in enumerate(rows):
            ax = axes[ri, ci]
            pcs[ri] = ax.pcolormesh(days, z, data[d][idx].T, cmap=cmo.balance,
                                    vmin=-vm, vmax=vm, shading='auto')
            if ci == 0:
                ax.set_ylabel(f'{lbl}\ndepth (m)')
            if ri == 2:
                ax.set_xlabel('days since 2012-10-11')
    fig.colorbar(pcs[0], ax=axes[:2, :].ravel().tolist(), pad=0.01,
                 label="$w'T'$ (W m$^{-2}$)")
    fig.colorbar(pcs[2], ax=axes[2, :].ravel().tolist(), pad=0.01,
                 label="error (W m$^{-2}$)")
    return P.finish(fig, f'{out}/wT_hovmoller.png')


def main():
    out = P.outdir(SUB)
    print(fig_mean(out))
    print(fig_skill(out))
    print(fig_hovmoller(out))


if __name__ == '__main__':
    main()
