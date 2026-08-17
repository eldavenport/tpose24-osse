#!/usr/bin/env python
"""
make_figs.py — w-skill and heat-flux-skill figures for the circling-glider experiment.

Reads only the caches + metrics.csv written by run_experiment_3.py.  Every figure
answers the central question — WHICH circle diameter and HOW MANY gliders best recover
the fixed-disk footprint truth — with the glider count as the colour key and the circle
diameter on the x-axis (or as heatmap columns).

Figures -> experiment_3/figs/
  geometry.png            the orbit geometry: start N-gon, circle, a sampled track, and
                          the disk truth, per (N x diameter), annotated with orbit period.
  w_skill_summary.png     corr / norm-RMS / std-ratio / frac-mean-bias of w vs diameter.
  w_skill_heatmap.png     corr and norm-RMS of w over the N x diameter grid.
  w_profiles.png          time-mean w(z): disk truth vs each N, one panel per diameter.
  heat_skill_summary.png  eddy-flux and advective-heating corr / std-ratio vs diameter.
  heat_skill_heatmap.png  eddy-flux and advective-heating corr over the N x diameter grid.
  heat_profiles.png       time-mean advective heating (degC/day) and eddy flux (W/m2),
                          disk truth vs each N, per diameter.

Run (from the tpose env):  python experiment_3/make_figs.py
"""
import os
import sys
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circle_common as C          # noqa: E402
import circle_plot as P            # noqa: E402
sys.path.insert(0, C.REPO)
import osse_tools as ot            # noqa: E402

DAY = C.SEC_PER_DAY                # s/day; heating degC/s -> degC/day, w m/s -> m/day
REP_DIAM = 1.0                     # representative diameter for single-figure panels


# --------------------------------------------------------------------------- geometry
def fig_geometry(out):
    nrows, ncols = len(C.N_GLIDERS), len(C.DIAMETERS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows),
                             sharex=False, sharey=False)
    # a sampled orbit over ~1.2 periods, dense enough to read the sense of rotation
    for ri, n in enumerate(C.N_GLIDERS):
        for ci, d in enumerate(C.DIAMETERS):
            ax = axes[ri, ci]
            period_s = 2 * np.pi / C.orbit_omega(d)
            t = np.datetime64(C.SPINUP_END) + (np.linspace(0, 1.15 * period_s, 240)
                                               * 1e9).astype('timedelta64[ns]')
            lat, lon = C.glider_positions(n, d, t)
            # disk (truth footprint)
            th = np.linspace(0, 2 * np.pi, 200)
            r = d / 2.0
            ax.fill(C.CENTER[1] + r * np.cos(th), C.CENTER[0] + r * np.sin(th),
                    color='0.85', zorder=0)
            ax.plot(C.CENTER[1] + r * np.cos(th), C.CENTER[0] + r * np.sin(th),
                    color='0.5', lw=1.0)
            # one glider's track (fades in time) + all start positions
            ax.plot(lon.isel(glider=0), lat.isel(glider=0), color=P.n_color(n),
                    lw=1.0, alpha=0.5)
            ax.scatter(lon.isel(time=0), lat.isel(time=0), color=P.n_color(n),
                       s=45, zorder=5, edgecolor='k', linewidth=0.5)
            ax.plot(C.CENTER[1], C.CENTER[0], marker='*', color='k', ms=10, zorder=6)
            ax.set_aspect('equal')
            if ri == 0:
                ax.set_title(f'{d:g}$^\\circ$ circle')
            if ci == 0:
                ax.set_ylabel(f'{n} gliders\nlatitude ($^\\circ$N)')
            if ri == nrows - 1:
                ax.set_xlabel('longitude ($^\\circ$E)')
            ax.text(0.5, 0.02, f'{period_s / 86400:.1f} d/orbit', transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(fc='white', ec='0.8', alpha=0.8))
    fig.suptitle('')
    fig.tight_layout()
    return P.finish(fig, f'{out}/geometry.png')


# --------------------------------------------------------------------------- tracks over mean-w
def fig_tracks_meanw(out):
    """DEMONSTRATION figure: the glider orbits drawn over the model's time-mean vertical
    velocity field (depth-averaged, m/day).  One panel per (N gliders x diameter): the
    fixed disk (truth footprint), each glider's sampled track over ~1.5 orbits, the start
    positions, a clockwise arrow, and the TAO mooring at the centre."""
    Wbg = P.load_meanw().mean('obs_depth')                      # depth-avg mean w (m/day)
    lonf, latf = Wbg.XC.values, Wbg.YC.values
    vmax = float(np.nanpercentile(np.abs(Wbg.values), 99))
    nrows, ncols = len(C.N_GLIDERS), len(C.DIAMETERS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows),
                             sharex=True, sharey=True)
    pcm = None
    for ri, n in enumerate(C.N_GLIDERS):
        for ci, d in enumerate(C.DIAMETERS):
            ax = axes[ri, ci]
            pcm = ax.pcolormesh(lonf, latf, Wbg.values, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, shading='auto', zorder=0)
            period_s = 2 * np.pi / C.orbit_omega(d)
            t = np.datetime64(C.SPINUP_END) + (np.linspace(0, 1.5 * period_s, 320)
                                               * 1e9).astype('timedelta64[ns]')
            lat, lon = C.glider_positions(n, d, t)
            th = np.linspace(0, 2 * np.pi, 200)
            r = d / 2.0
            ax.plot(C.CENTER[1] + r * np.cos(th), C.CENTER[0] + r * np.sin(th),
                    color='0.25', lw=1.0, zorder=3)
            for g in range(n):
                ax.plot(lon.isel(glider=g), lat.isel(glider=g), color='0.15',
                        lw=0.8, alpha=0.5, zorder=4)
            ax.scatter(lon.isel(time=0), lat.isel(time=0), color=P.n_color(n),
                       s=42, zorder=6, edgecolor='k', linewidth=0.5)
            # clockwise arrow on the first glider's early motion
            p0 = (float(lon.isel(time=0, glider=0)), float(lat.isel(time=0, glider=0)))
            p1 = (float(lon.isel(time=6, glider=0)), float(lat.isel(time=6, glider=0)))
            ax.annotate('', xy=p1, xytext=p0, zorder=7,
                        arrowprops=dict(arrowstyle='-|>', color='k', lw=1.4))
            ax.plot(C.CENTER[1], C.CENTER[0], marker='*', color='k', ms=11, zorder=8)
            ax.set_aspect('equal')
            ax.set_xlim(lonf.min(), lonf.max())
            ax.set_ylim(latf.min(), latf.max())
            if ri == 0:
                ax.set_title(f'{d:g}$^\\circ$ circle')
            if ci == 0:
                ax.set_ylabel(f'{n} gliders\nlatitude ($^\\circ$N)')
            if ri == nrows - 1:
                ax.set_xlabel('longitude ($^\\circ$E)')
    cb = fig.colorbar(pcm, ax=axes, fraction=0.020, pad=0.01)
    cb.set_label('time-mean $w$  (m day$^{-1}$)')
    fig.suptitle('')
    return P.finish(fig, f'{out}/tracks_meanw.png')


# --------------------------------------------------------------------------- skill vs diameter
def _line_vs_diam(ax, m, col, ylabel, hline=None):
    for n in C.N_GLIDERS:
        sub = m[m.n_gliders == n].sort_values('diameter')
        ax.plot(sub.diameter, sub[col], '-o', color=P.n_color(n), lw=2, ms=6)
    if hline is not None:
        ax.axhline(hline, color='0.5', lw=0.9, ls='--')
    ax.set_xlabel('circle diameter ($^\\circ$)')
    ax.set_ylabel(ylabel)
    P.tidy_x(ax, 4)


def fig_w_skill_summary(out):
    m = P.load_metrics()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _line_vs_diam(axes[0, 0], m, 'w_corr', 'w correlation')
    _line_vs_diam(axes[0, 1], m, 'w_norm_rms', 'w norm-RMS  (RMS / $\\sigma_w$)')
    _line_vs_diam(axes[1, 0], m, 'w_std_ratio', 'w std ratio  (est / truth)', hline=1.0)
    _line_vs_diam(axes[1, 1], m, 'w_frac_mean_bias', 'fractional mean-w bias', hline=0.0)
    P.n_legend(fig, method=False, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return P.finish(fig, f'{out}/w_skill_summary.png')


def fig_heat_skill_summary(out, zk=50):
    m = P.load_metrics()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _line_vs_diam(axes[0, 0], m, f'flux_corr_{zk}', f"eddy flux $w'T'$ corr ({zk} m)")
    _line_vs_diam(axes[0, 1], m, f'flux_std_ratio_{zk}',
                  f"$w'T'$ std ratio ({zk} m)", hline=1.0)
    _line_vs_diam(axes[1, 0], m, f'adv_corr_{zk}',
                  f'advective heating $w\\,dT/dz$ corr ({zk} m)')
    _line_vs_diam(axes[1, 1], m, f'adv_std_ratio_{zk}',
                  f'$w\\,dT/dz$ std ratio ({zk} m)', hline=1.0)
    P.n_legend(fig, method=False, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return P.finish(fig, f'{out}/heat_skill_summary.png')


# --------------------------------------------------------------------------- heatmaps
def _heatmap(ax, m, col, title, cmap, vmin=None, vmax=None, fmt='{:.2f}'):
    grid = np.full((len(C.N_GLIDERS), len(C.DIAMETERS)), np.nan)
    for i, n in enumerate(C.N_GLIDERS):
        for j, d in enumerate(C.DIAMETERS):
            r = m[(m.n_gliders == n) & (np.isclose(m.diameter, d))]
            if len(r):
                grid[i, j] = float(r[col].iloc[0])
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(C.DIAMETERS)))
    ax.set_xticklabels([f'{d:g}' for d in C.DIAMETERS])
    ax.set_yticks(range(len(C.N_GLIDERS)))
    ax.set_yticklabels(C.N_GLIDERS)
    ax.set_xlabel('circle diameter ($^\\circ$)')
    ax.set_ylabel('number of gliders')
    ax.set_title(title)
    ax.grid(False)
    for i in range(len(C.N_GLIDERS)):
        for j in range(len(C.DIAMETERS)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, fmt.format(grid[i, j]), ha='center', va='center',
                        fontsize=10, color='k')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def fig_w_skill_heatmap(out):
    m = P.load_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _heatmap(axes[0], m, 'w_corr', 'w correlation', 'viridis')
    _heatmap(axes[1], m, 'w_norm_rms', 'w norm-RMS (lower = better)', 'viridis_r')
    fig.tight_layout()
    return P.finish(fig, f'{out}/w_skill_heatmap.png')


def fig_heat_skill_heatmap(out, zk=50):
    m = P.load_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _heatmap(axes[0], m, f'flux_corr_{zk}', f"eddy flux $w'T'$ corr ({zk} m)", 'viridis')
    _heatmap(axes[1], m, f'adv_corr_{zk}', f'adv. heating corr ({zk} m)', 'viridis')
    fig.tight_layout()
    return P.finish(fig, f'{out}/heat_skill_heatmap.png')


# --------------------------------------------------------------------------- profiles
def fig_w_profiles(out):
    fig, axes = plt.subplots(1, len(C.DIAMETERS), figsize=(3.4 * len(C.DIAMETERS), 5.2),
                             sharey=True)
    for ax, d in zip(axes, C.DIAMETERS):
        disk = P.load_disk(d)
        wt = disk.w_true.mean('time') * P.SEC_PER_DAY
        ax.plot(wt, wt.depth, color='0.2', **P.TRUTH_KW)
        for n in C.N_GLIDERS:
            arr = P.load_array(n, d)
            we = arr.w_est.mean('time') * P.SEC_PER_DAY
            ax.plot(we, we.depth, color=P.n_color(n), **P.ARRAY_KW)
            arr.close()
        disk.close()
        ax.axvline(0, color='0.6', lw=0.8)
        ax.set_title(f'{d:g}$^\\circ$ circle')
        ax.set_xlabel('mean w (m day$^{-1}$)')
    axes[0].set_ylabel('depth (m)')
    P.n_legend(fig, method=True, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return P.finish(fig, f'{out}/w_profiles.png')


def fig_heat_profiles(out):
    # two row-groups (advective heating, eddy heat flux); each panel carries its own
    # x-axis label so the group labels never collide with tick labels between the rows.
    adv_lbl = 'mean adv. heating $w\\,\\partial_z T$\n(°C day$^{-1}$)'
    flux_lbl = "mean eddy heat flux $w'T'$\n(W m$^{-2}$)"
    fig, axes = plt.subplots(2, len(C.DIAMETERS),
                             figsize=(3.4 * len(C.DIAMETERS), 9.4), sharey=True)
    for ci, d in enumerate(C.DIAMETERS):
        disk = P.load_disk(d)
        adv_t = disk.adv_total.mean('time') * DAY                     # degC/day
        flux_t = disk.eddy_flux.mean('time') * C.HFLUX                # W/m2
        z = disk.obs_depth
        axes[0, ci].plot(adv_t, z, color='0.2', **P.TRUTH_KW)
        axes[1, ci].plot(flux_t, z, color='0.2', **P.TRUTH_KW)
        for n in C.N_GLIDERS:
            arr = P.load_array(n, d)
            a_est = ot.advective_heating(arr.w_est_mid, arr.Tbar_glider)['total']
            a_est = a_est.mean('time') * DAY
            f_est = ot.flux_total_and_eddy(arr.w_est_mid, arr.Tbar_glider)['eddy']
            f_est = f_est.mean('time') * C.HFLUX
            axes[0, ci].plot(a_est, arr.obs_depth, color=P.n_color(n), **P.ARRAY_KW)
            axes[1, ci].plot(f_est, arr.obs_depth, color=P.n_color(n), **P.ARRAY_KW)
            arr.close()
        disk.close()
        for r in (0, 1):
            axes[r, ci].axvline(0, color='0.6', lw=0.8)
        axes[0, ci].set_title(f'{d:g}$^\\circ$ circle')
        axes[0, ci].set_xlabel(adv_lbl)
        axes[1, ci].set_xlabel(flux_lbl)
    axes[0, 0].set_ylabel('depth (m)')
    axes[1, 0].set_ylabel('depth (m)')
    P.n_legend(fig, method=True, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=3.0)
    return P.finish(fig, f'{out}/heat_profiles.png')


# --------------------------------------------------------------------------- heat decomposition
def _heating_to_flux(H):
    """Cumulative top-down depth integral of a mean-heating profile (°C/day on obs_depth)
    into a heat flux ρ₀cp∫ w∂_zT dz (W m⁻²).  Integration is linear, so it commutes with
    the true−estimate differences taken on the heating profiles."""
    z = np.asarray(H['obs_depth'].values, float)       # negative, top(-9) -> bottom(-79)
    Hs = np.asarray(H.values, float) / DAY             # °C/day -> °C/s
    F = ot.HFLUX * cumulative_trapezoid(Hs, x=-z, initial=0.0)
    return xr.DataArray(F, dims=H.dims, coords=H.coords)


def _disk_components(disk, arr):
    """AREA structural decomposition (°C/day profiles) for one (diameter, N):
      true_total = ⟨[w ∂_zT]⟩            disk mean of the pointwise product   (N-independent)
      true_mean  = ⟨[w]_true [∂_zT]_true⟩ product of TRUE disk means           (N-independent)
      est_mean   = ⟨[w]_est  [∂_zT]_est⟩  product of ESTIMATED array means      (per N)
      est_err    = true_mean − est_mean   observable estimate error (reducible)
      subgrid    = true_total − true_mean unobservable within-footprint covariance
    Identity: true_total = est_mean + est_err + subgrid."""
    dTh = disk.Tbar.differentiate('obs_depth')         # [∂_zT]_true
    dTg = arr.Tbar_glider.differentiate('obs_depth')   # [∂_zT]_est
    true_total = disk.adv_total.mean('time') * DAY
    true_mean = (disk.wbar * dTh).mean('time') * DAY
    est_mean = (arr.w_est_mid * dTg).mean('time') * DAY
    return dict(z=disk.obs_depth, true_total=true_total, true_mean=true_mean,
                est_mean=est_mean, est_err=true_mean - est_mean,
                subgrid=true_total - true_mean)


def _prof(ax, xlim=None):
    ax.axvline(0, color='0.7', lw=0.8, zorder=0)
    ax.set_ylim(-80, 0)
    ax.grid(alpha=0.3)
    P.tidy_x(ax, 4)                     # few, uncrowded x ticks (shared ×10ⁿ offset)
    if xlim is not None:
        ax.set_xlim(*xlim)


def _span(*arrs, pad=0.05):
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrs])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), float(v.max())
    d = pad * (hi - lo or 1.0)
    return (lo - d, hi + d)


def fig_heat_area_components(out, flux=False):
    """Structural heat-flux decomposition — columns = circle diameter, three rows:
      1. true area mean ⟨[w][∂_zT]⟩ (black) vs each array's estimated mean (coloured by N)
      2. estimate error (true mean − est mean), coloured by N        (reducible)
      3. sub-array covariance (true total − true mean), N-independent (unobservable).
    `flux=True` renders the units-only heat-flux (W m⁻²) duplicate."""
    conv = _heating_to_flux if flux else (lambda x: x)
    unit = 'W m$^{-2}$' if flux else '°C day$^{-1}$'
    labels = [f'advective heat flux  ({unit})' if flux else f'advective heating  ({unit})',
              f'estimate error  ({unit})', f'sub-array  ({unit})']
    ncol = len(C.DIAMETERS)
    fig, axes = plt.subplots(3, ncol, figsize=(4.3 * ncol, 11.4), sharey=True)
    # gather per-diameter decompositions, then share x-limits per row across columns
    store, m1, m2, m3 = {}, [], [], []
    for d in C.DIAMETERS:
        disk = P.load_disk(d)
        per_n = {}
        for n in C.N_GLIDERS:
            arr = P.load_array(n, d)
            R = _disk_components(disk, arr)
            per_n[n] = {k: (conv(v) if k != 'z' else v) for k, v in R.items()}
            arr.close()
        disk.close()
        R0 = per_n[C.N_GLIDERS[0]]
        store[d] = per_n
        m1 += [R0['true_mean'].values] + [per_n[n]['est_mean'].values for n in C.N_GLIDERS]
        m2 += [per_n[n]['est_err'].values for n in C.N_GLIDERS]
        m3 += [R0['subgrid'].values]
    x1, x2, x3 = _span(*m1), _span(*m2), _span(*m3)
    for ci, d in enumerate(C.DIAMETERS):
        per_n = store[d]
        R0 = per_n[C.N_GLIDERS[0]]
        z = R0['z']
        a1, a2, a3 = axes[0, ci], axes[1, ci], axes[2, ci]
        a1.plot(R0['true_mean'], z, color='0.15', lw=2.4, zorder=5)          # true (N-indep)
        a3.plot(R0['subgrid'], z, color='0.15', lw=2.4, zorder=5)            # subgrid (N-indep)
        for n in C.N_GLIDERS:
            a1.plot(per_n[n]['est_mean'], z, color=P.n_color(n), ls='--', lw=1.9)
            a2.plot(per_n[n]['est_err'], z, color=P.n_color(n), lw=2.0)
        _prof(a1, x1); _prof(a2, x2); _prof(a3, x3)
        a1.set_title(f'{d:g}$^\\circ$ circle')
        for ax, lbl in zip((a1, a2, a3), labels):
            ax.set_xlabel(lbl, fontsize=10)
    for r in range(3):
        axes[r, 0].set_ylabel('depth (m)')
    axes[0, 0].legend(handles=[Line2D([], [], color='0.15', lw=2.4, label='true area mean'),
                               Line2D([], [], color='0.4', ls='--', lw=1.9,
                                      label='estimated mean (per N)')],
                      fontsize=9, loc='lower left')
    P.n_legend(fig, method=False, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    name = 'heat_area_components_flux' if flux else 'heat_area_components'
    return P.finish(fig, f'{out}/{name}.png')


_REDUCIBLE_ROWS = [   # (key, quantity label [left col], error label [right col])
    ('w', 'w  (m day$^{-1}$)', 'w estimate error  (m day$^{-1}$)'),
    ('dTdz', '∂T/∂z  (°C m$^{-1}$)', '∂T/∂z estimate error  (°C m$^{-1}$)'),
    ('flux', 'advective heating  (°C day$^{-1}$)', 'heat-flux estimate error  (°C day$^{-1}$)'),
    ('fluxint', 'advective heat flux  (W m$^{-2}$)', 'heat-flux estimate error  (W m$^{-2}$)'),
]


def _reducible_profiles(disk, arr):
    """Per-row (true, est) time-mean profiles for the reducible figure at one (diam, N)."""
    dTh = disk.Tbar.differentiate('obs_depth')
    dTg = arr.Tbar_glider.differentiate('obs_depth')
    flux_true = (disk.wbar * dTh).mean('time') * DAY
    flux_est = (arr.w_est_mid * dTg).mean('time') * DAY
    return {
        'w': (disk.wbar.mean('time') * DAY, arr.w_est_mid.mean('time') * DAY),
        'dTdz': (dTh.mean('time'), dTg.mean('time')),
        'flux': (flux_true, flux_est),
        'fluxint': (_heating_to_flux(flux_true), _heating_to_flux(flux_est)),
    }


def fig_heat_area_reducible(out, diam=REP_DIAM):
    """Reducible plane-fit / sampling errors at one representative diameter — four rows
    (w, ∂T/∂z, advective heating, and that heating integrated into a heat flux) × two
    columns (left: true black vs each array estimate dashed-coloured; right: estimate
    error true−est per N).  The heating-row error equals the components estimate-error row."""
    disk = P.load_disk(diam)
    profs = {}
    for n in C.N_GLIDERS:
        arr = P.load_array(n, diam)
        profs[n] = _reducible_profiles(disk, arr)
        arr.close()
    z = disk.obs_depth
    nrow = len(_REDUCIBLE_ROWS)
    fig, axes = plt.subplots(nrow, 2, figsize=(10, 3.9 * nrow), sharey=True)
    for r, (key, qlbl, elbl) in enumerate(_REDUCIBLE_ROWS):
        aL, aR = axes[r, 0], axes[r, 1]
        true0 = profs[C.N_GLIDERS[0]][key][0]                # N-independent truth
        aL.plot(true0, z, color='0.15', lw=2.4, zorder=5)
        tv, dv = [true0.values], []
        for n in C.N_GLIDERS:
            t, e = profs[n][key]
            aL.plot(e, z, color=P.n_color(n), ls='--', lw=1.9)
            aR.plot(t - e, z, color=P.n_color(n), lw=2.0)
            tv.append(e.values); dv.append((t - e).values)
        _prof(aL, _span(*tv)); _prof(aR, _span(*dv))
        aL.set_xlabel(qlbl, fontsize=10)
        aR.set_xlabel(elbl, fontsize=10)
    disk.close()
    axes[0, 0].set_title(f'true vs estimate  ({diam:g}$^\\circ$ circle)')
    axes[0, 1].set_title(f'estimate error (true − estimate)  ({diam:g}$^\\circ$)')
    for r in range(nrow):
        axes[r, 0].set_ylabel('depth (m)')
    axes[0, 0].legend(handles=[Line2D([], [], color='0.15', lw=2.4, label='true (disk)'),
                               Line2D([], [], color='0.4', ls='--', lw=1.9,
                                      label='estimate (per N)')],
                      fontsize=9, loc='lower left')
    P.n_legend(fig, method=False, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return P.finish(fig, f'{out}/heat_area_reducible_d{diam:g}.png')


# --------------------------------------------------------------------------- fig7-style w CI
F7_EST_C = '#1f77b4'      # array estimate
F7_MOD_C = '#d62728'      # model / disk truth
F7_Z = 1.96               # 95% CI = mean +/- 1.96 * SE (normal approximation)


def fig_w_fig7(out, diam=REP_DIAM):
    """Time-mean ⟨w⟩(z) with autocorrelation-aware 95% confidence intervals — one column
    per glider count (+ a total-over-depth forest panel), at one circle diameter.  The SE
    (osse_tools.mean_se_autocorr) treats the record as one autocorrelated sample, deflating
    N to N_eff; a band straddling zero means the record does not resolve a nonzero mean, and
    overlapping est/true bands mean the estimate is statistically consistent with truth."""
    m = P.load_metrics()
    rows = [(n, f'{n} gliders') for n in C.N_GLIDERS]
    ncfg = len(rows)
    fig, axes = plt.subplots(1, ncfg + 1, figsize=(3.3 * (ncfg + 1), 6.3))
    disk = P.load_disk(diam)
    profs, lo, hi = [], 0.0, 0.0
    for n, lbl in rows:
        arr = P.load_array(n, diam)
        we, wt = xr.align(arr.w_est, disk.w_true, join='inner')   # common depth axis
        B = ot.w_skill_by_depth(we, wt)
        arr.close()
        profs.append((B, lbl))
        for mean, se in ((B.w_model_mean, B.w_model_mean_se),
                         (B.w_est_mean, B.w_est_mean_se)):
            lo = min(lo, float(((mean - F7_Z * se) * DAY).min()))
            hi = max(hi, float(((mean + F7_Z * se) * DAY).max()))
    disk.close()
    pad = 0.08 * (hi - lo) or 0.1
    for ax, (B, lbl) in zip(axes[:ncfg], profs):
        z = B.depth.values
        for mean, se, c, ls in ((B.w_model_mean, B.w_model_mean_se, F7_MOD_C, ':'),
                                (B.w_est_mean, B.w_est_mean_se, F7_EST_C, '-')):
            mu = mean.values * DAY; hw = F7_Z * se.values * DAY
            ax.fill_betweenx(z, mu - hw, mu + hw, color=c, alpha=0.18, lw=0)
            ax.plot(mu, z, color=c, ls=ls, lw=1.9)
        ax.axvline(0, color='0.5', lw=0.8, zorder=0)
        ax.set_title(lbl); ax.set_xlim(lo - pad, hi + pad); ax.grid(alpha=0.3)
        ax.set_xlabel(r'$\langle w\rangle$  (m day$^{-1}$)')
    axes[0].set_ylabel('depth (m)')
    for ax in axes[1:ncfg]:
        ax.set_yticklabels([])
    # total-over-depth forest panel from the metrics scalars
    axT = axes[ncfg]
    yv = np.arange(ncfg)[::-1]
    for y, (n, lbl) in zip(yv, rows):
        r = m[(m.n_gliders == n) & np.isclose(m.diameter, diam)].iloc[0]
        for off, c, mk, mkey, skey in (
                (+0.15, F7_MOD_C, 'o', 'w_model_mean', 'w_model_mean_se'),
                (-0.15, F7_EST_C, 's', 'w_est_mean', 'w_est_mean_se')):
            axT.errorbar(r[mkey] * DAY, y + off, xerr=F7_Z * r[skey] * DAY,
                         fmt=mk, color=c, capsize=3, ms=6, lw=1.6)
    axT.axvline(0, color='0.5', lw=0.8)
    axT.set_yticks(yv); axT.set_yticklabels([lbl for _, lbl in rows])
    axT.set_ylim(-0.6, ncfg - 0.4)
    axT.set_xlabel(r'total $\langle w\rangle$  (m day$^{-1}$)')
    axT.set_title('total over depth'); axT.grid(alpha=0.3, axis='x')
    handles = [Line2D([0], [0], color=F7_EST_C, ls='-', lw=1.9, marker='s',
                      label=r'estimated ($\pm$95% CI)'),
               Line2D([0], [0], color=F7_MOD_C, ls=':', lw=1.9, marker='o',
                      label=r'disk truth ($\pm$95% CI)')]
    fig.legend(handles=handles, title=f'{diam:g}$^\\circ$ circle — glider count',
               loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return P.finish(fig, f'{out}/w_fig7_d{diam:g}.png')


# --------------------------------------------------------------------------- scatter grids
def _scatter_grid(pts, lim, zmax, cblabel, xlabel, ylabel, title, path):
    """Shared renderer: rows = N gliders, columns = diameter; each panel scatters pooled
    est-vs-true samples coloured by depth, with the 1:1 line and the correlation."""
    nrow, ncol = len(C.N_GLIDERS), len(C.DIAMETERS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    sc = None
    for i, n in enumerate(C.N_GLIDERS):
        for j, d in enumerate(C.DIAMETERS):
            ax = axes[i, j]
            x, y, z = pts[(n, d)]
            ax.plot([-lim, lim], [-lim, lim], color='0.4', lw=1.0, zorder=1)
            sc = ax.scatter(x, y, c=z, cmap='viridis', vmin=0, vmax=zmax, s=4, alpha=0.35,
                            linewidths=0, zorder=2, rasterized=True)
            r = np.corrcoef(x, y)[0, 1] if x.size > 2 else np.nan
            ax.text(0.04, 0.93, f'r = {r:.3f}', transform=ax.transAxes, fontsize=9,
                    va='top', bbox=dict(fc='w', ec='none', alpha=0.7))
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(f'{d:g}$^\\circ$ circle', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'{n} gliders\n{ylabel}', fontsize=10)
            if i == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=10)
    cb = fig.colorbar(sc, ax=axes, fraction=0.022, pad=0.01)
    cb.set_label('depth (m)')
    fig.suptitle(title, y=0.995, fontsize=14)
    return P.finish(fig, path)


def _subsample(x, y, z, rng, cap=8000):
    g = np.isfinite(x) & np.isfinite(y)
    x, y, z = x[g], y[g], z[g]
    if x.size > cap:
        sel = rng.choice(x.size, cap, replace=False)
        x, y, z = x[sel], y[sel], z[sel]
    return x, y, z


def fig_w_scatter(out):
    """Estimated vs true w scatter (m/day), pooling every (time, depth) sample of the
    plane-fit w against the disk-mean truth.  Rows = N gliders, columns = diameter."""
    rng = np.random.default_rng(0)
    pts, lim, zmax = {}, 0.0, 0.0
    for d in C.DIAMETERS:
        disk = P.load_disk(d)
        for n in C.N_GLIDERS:
            arr = P.load_array(n, d)
            we, wt = xr.align(arr.w_est, disk.w_true, join='inner')
            z = np.broadcast_to(np.abs(we.depth.values)[None, :], we.shape)
            x = (wt.values * DAY).ravel(); y = (we.values * DAY).ravel()
            x, y, z = _subsample(x, y, z.ravel(), rng)
            pts[(n, d)] = (x, y, z)
            lim = max(lim, np.nanpercentile(np.abs(np.concatenate([x, y])), 99.5))
            zmax = max(zmax, float(z.max()) if z.size else 0.0)
            arr.close()
        disk.close()
    return _scatter_grid(pts, lim, zmax, 'depth (m)',
                         r'true w  (m day$^{-1}$)', 'est w  (m day$^{-1}$)',
                         'estimated vs true $w$  (all depths & times)',
                         f'{out}/w_scatter.png')


def fig_heat_scatter(out):
    """Estimated vs true eddy heat flux w'T' scatter (W/m²), pooling every (time, depth)
    sample of the array's flux against the disk truth.  Rows = N gliders, cols = diameter."""
    rng = np.random.default_rng(0)
    pts, lim, zmax = {}, 0.0, 0.0
    for d in C.DIAMETERS:
        disk = P.load_disk(d)
        ft = disk.eddy_flux * C.HFLUX
        for n in C.N_GLIDERS:
            arr = P.load_array(n, d)
            fe = ot.flux_total_and_eddy(arr.w_est_mid, arr.Tbar_glider)['eddy'] * C.HFLUX
            fe, ftn = xr.align(fe, ft, join='inner')
            z = np.broadcast_to(np.abs(fe.obs_depth.values)[None, :], fe.shape)
            x = ftn.values.ravel(); y = fe.values.ravel()
            x, y, z = _subsample(x, y, z.ravel(), rng)
            pts[(n, d)] = (x, y, z)
            lim = max(lim, np.nanpercentile(np.abs(np.concatenate([x, y])), 99.5))
            zmax = max(zmax, float(z.max()) if z.size else 0.0)
            arr.close()
        disk.close()
    return _scatter_grid(pts, lim, zmax, 'depth (m)',
                         r"true $w'T'$  (W m$^{-2}$)", r"est $w'T'$  (W m$^{-2}$)",
                         "estimated vs true eddy heat flux $w'T'$  (all depths & times)",
                         f'{out}/heat_scatter.png')


def main():
    out = P.outdir()
    print(fig_geometry(out))
    print(fig_tracks_meanw(out))
    print(fig_w_skill_summary(out))
    print(fig_w_skill_heatmap(out))
    print(fig_w_profiles(out))
    print(fig_w_fig7(out))
    print(fig_w_scatter(out))
    print(fig_heat_skill_summary(out))
    print(fig_heat_skill_heatmap(out))
    print(fig_heat_profiles(out))
    print(fig_heat_area_components(out))
    print(fig_heat_area_components(out, flux=True))
    print(fig_heat_area_reducible(out))
    print(fig_heat_scatter(out))


if __name__ == '__main__':
    main()
