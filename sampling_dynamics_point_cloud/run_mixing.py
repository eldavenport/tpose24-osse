"""
KPP mixing diagnostics for the virtual mooring.

The array sees vertical mixing only where its gliders + mooring sit; the truth is the
mixing averaged over every grid point in the footprint.  We compare:
  * kappa_T  KPP vertical diffusivity for heat   (m^2 s^-1, log scale)
  * nu       KPP vertical eddy viscosity          (m^2 s^-1, log scale)
  * N^2      stratification from DRHODR            (s^-2)
  * hbl      KPP boundary-layer depth              (m) -- the mixed layer the array sits in
Array = platform mean (6 gliders + mooring); truth = hull mean.  Diffusivity/viscosity
are strongly intermittent (diurnal convection, wind + shear bursts), so max/median as
well as the mean are reported.

Figures -> mixing/
  mix_profiles.png    time-mean kappa_T, nu, N^2 profiles, array vs truth per diameter.
  hbl_timeseries.png  boundary-layer depth vs time (array vs truth) + its distribution.
  mix_hovmoller.png   kappa_T(z,t) truth vs array for 1.0deg, hbl overlaid.
  mix_stats.png       max/median/mean of kappa_T and hbl per diameter, array vs truth.

Usage:  python run_mixing.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmocean.cm as cmo

import common as C
import sd_plot as P

SUB = 'mixing'
HBL_COLOR = '#ff1493'           # deep pink -- stands out against cmo.tempo / balance


def _pair(diam, var):
    """Platform-mean (array) and hull-mean (truth) profile series for a var."""
    arr = P.load_array(diam)
    hull = P.load_hull(diam)
    return P.array_profile(arr, var), hull[var]


# --------------------------------------------------------------------------- fig 1
def fig_profiles(out):
    specs = [('kappaT', r'$\kappa_T$ (m$^2$ s$^{-1}$)', True),
             ('nu', r'$\nu$ (m$^2$ s$^{-1}$)', True),
             ('N2', '$N^2$ (s$^{-2}$)', False)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), sharey=True)
    for ax, (var, xlabel, logx) in zip(axes, specs):
        for d in C.DIAMETERS:
            a, t = _pair(d, var)
            z = a.obs_depth.values
            c = C.diam_color(d)
            ax.plot(a.mean('time'), z, color=c, **P.ARRAY_KW)
            ax.plot(t.mean('time'), z, color=c, **P.TRUTH_KW)
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale('log')
        else:
            P.tidy_x(ax, 4)
    axes[0].set_ylabel('depth (m)')
    P.top_legend(fig)
    return P.finish(fig, f'{out}/mix_profiles.png')


# --------------------------------------------------------------------------- fig 2
def fig_hbl(out, tlim=None):
    """hbl time series (array vs truth) + distribution, ONE ROW PER DIAMETER.
    tlim=(day0,day1) zooms in time."""
    nr = len(C.DIAMETERS)
    fig, axes = plt.subplots(nr, 2, figsize=(13, 2.6 * nr), sharex='col',
                             gridspec_kw={'width_ratios': [3, 1]})
    ylim = None
    for ri, d in enumerate(C.DIAMETERS):
        ax, axd = axes[ri]
        arr = P.load_array(diam=d); hull = P.load_hull(d)
        t = (arr.time.values - arr.time.values[0]) / np.timedelta64(1, 'D')
        m = np.ones_like(t, bool) if tlim is None else (t >= tlim[0]) & (t <= tlim[1])
        ha = arr['hbl'].mean('glider').values; ht = hull['hbl'].values
        c = C.diam_color(d)
        ax.plot(t[m], ha[m], color=c, **P.ARRAY_KW, alpha=0.9)
        ax.plot(t[m], ht[m], color=c, **P.TRUTH_KW)
        axd.hist(ht[m], bins=30, orientation='horizontal', color=c, alpha=0.3,
                 density=True)
        axd.hist(ha[m], bins=30, orientation='horizontal', histtype='step',
                 color=c, lw=1.8, density=True)
        if tlim is not None:
            ax.set_xlim(*tlim)
        ax.text(0.01, 0.9, f'{d:g}$^\\circ$', transform=ax.transAxes, fontweight='bold',
                color=c, va='top', bbox=dict(fc='white', ec='none', alpha=0.7))
        ax.set_ylabel('BL depth (m)')
        if ylim is None:
            ylim = (0, max(ht[m].max(), ha[m].max()) * 1.05)
        ax.set_ylim(*ylim); ax.invert_yaxis()
        axd.set_ylim(*ylim); axd.invert_yaxis(); axd.set_yticklabels([])
    axes[-1, 0].set_xlabel('days since 2012-10-11')
    axes[-1, 1].set_xlabel('density')
    P.top_legend(fig, diam=False)
    suffix = '' if tlim is None else f'_day{tlim[0]:g}-{tlim[1]:g}'
    return P.finish(fig, f'{out}/hbl_timeseries{suffix}.png')


# --------------------------------------------------------------------------- fig 3
def fig_hovmoller(out, tlim=None):
    """kappa_T(z,t) truth / estimate / error, one COLUMN per diameter (KPP BL depth
    overlaid on the truth and estimate rows; shared scales across footprints).
    tlim=(day0,day1) crops the time axis to reveal more structure; colour scales stay
    the FULL-record scales so the zoom is directly comparable to the full figure."""
    from matplotlib.colors import LogNorm
    data = {}
    for d in C.DIAMETERS:
        a, t = _pair(d, 'kappaT')
        arr = P.load_array(d); hull = P.load_hull(d)
        data[d] = dict(truth=t, est=a, err=a - t,
                       hbl_t=hull['hbl'], hbl_a=arr['hbl'].mean('glider'))
    z = data[C.DIAMETERS[0]]['truth'].obs_depth.values
    tc = data[C.DIAMETERS[0]]['truth'].time.values
    days_full = (tc - tc[0]) / np.timedelta64(1, 'D')
    m = (np.ones_like(days_full, bool) if tlim is None
         else (days_full >= tlim[0]) & (days_full <= tlim[1]))
    days = days_full[m]
    vmax = float(np.nanpercentile(np.concatenate(
        [data[d]['truth'].values.ravel() for d in C.DIAMETERS]), 99.5))
    norm = LogNorm(vmin=max(vmax * 1e-4, 1e-6), vmax=vmax)
    emax = float(np.nanpercentile(np.abs(np.concatenate(
        [np.abs(data[d]['err'].values).ravel() for d in C.DIAMETERS])), 99))

    rows = [('model truth (disk)', 'truth', 'hbl_t'),
            ('array estimate', 'est', 'hbl_a'),
            ('estimate $-$ truth', 'err', None)]
    nc = len(C.DIAMETERS)
    fig, axes = plt.subplots(3, nc, figsize=(4.2 * nc, 9), sharex=True, sharey=True)
    pc_k = pc_e = None
    for ci, d in enumerate(C.DIAMETERS):
        axes[0, ci].set_title(f'{d:g}$^\\circ$ footprint')
        for ri, (lbl, key, hkey) in enumerate(rows):
            ax = axes[ri, ci]
            fld = data[d][key].values[m].T          # (obs_depth, time-window)
            if key == 'err':
                pc_e = ax.pcolormesh(days, z, fld, cmap=cmo.balance,
                                     vmin=-emax, vmax=emax, shading='auto')
            else:
                pc_k = ax.pcolormesh(days, z, np.clip(fld, norm.vmin, None),
                                     cmap=cmo.tempo, norm=norm, shading='auto')
                ax.plot(days, -data[d][hkey].values[m], color=HBL_COLOR, lw=1.4)
            if ci == 0:
                ax.set_ylabel(f'{lbl}\ndepth (m)')
            if ri == 2:
                ax.set_xlabel('days since 2012-10-11')
    fig.colorbar(pc_k, ax=axes[:2, :].ravel().tolist(), pad=0.01,
                 label=r'$\kappa_T$ (m$^2$ s$^{-1}$)')
    fig.colorbar(pc_e, ax=axes[2, :].ravel().tolist(), pad=0.01,
                 label=r'error (m$^2$ s$^{-1}$)')
    fig.legend(handles=[Line2D([0], [0], color=HBL_COLOR, lw=1.6, label='KPP BL depth')],
               loc='upper center', ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.02))
    suffix = '' if tlim is None else f'_day{tlim[0]:g}-{tlim[1]:g}'
    return P.finish(fig, f'{out}/mix_hovmoller{suffix}.png')


# --------------------------------------------------------------------------- fig 4
def fig_stats(out):
    specs = [('kappaT', r'$\kappa_T$ (m$^2$ s$^{-1}$)', True),
             ('hbl', 'KPP BL depth (m)', False)]
    stat_names = ['median', 'mean', 'max']
    x = np.arange(len(C.DIAMETERS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (var, ylabel, logy) in zip(axes, specs):
        for si, sname in enumerate(stat_names):
            ea, ta = [], []
            for d in C.DIAMETERS:
                if var == 'hbl':
                    arr = P.load_array(d); hull = P.load_hull(d)
                    a = arr['hbl'].mean('glider'); tt = hull['hbl']
                else:
                    a, tt = _pair(d, var)
                    a = a.mean('obs_depth'); tt = tt.mean('obs_depth')
                ea.append(float(P.time_stats(a)[sname]))
                ta.append(float(P.time_stats(tt)[sname]))
            col = plt.cm.plasma(si / (len(stat_names) - 1))
            ax.plot(x, ta, 'o', color=col, ms=9, mfc='none', mew=2)
            ax.plot(x, ea, 's', color=col, ms=8)
        ax.set_xticks(x); ax.set_xticklabels([f'{d:g}$^\\circ$' for d in C.DIAMETERS])
        ax.set(xlabel='E-W diameter', ylabel=ylabel)
        if logy:
            ax.set_yscale('log')
    stat_h = [Line2D([0], [0], marker='o', ls='', color=plt.cm.plasma(i/(len(stat_names)-1)),
                     label=s, ms=8) for i, s in enumerate(stat_names)]
    mk_h = [Line2D([0], [0], marker='s', ls='', color='0.3', label='array estimate', ms=8),
            Line2D([0], [0], marker='o', ls='', color='0.3', mfc='none', mew=2,
                   label='model truth', ms=9)]
    fig.legend(handles=stat_h + mk_h, loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    return P.finish(fig, f'{out}/mix_stats.png')


ZOOM_HBL = (30, 60)             # day window for the zoomed hbl time series
ZOOM_HOV = (40, 50)             # day window for the zoomed kappa_T Hovmoller


def main():
    out = P.outdir(SUB)
    print(fig_profiles(out))
    print(fig_hbl(out))
    print(fig_hbl(out, ZOOM_HBL))
    print(fig_hovmoller(out))
    print(fig_hovmoller(out, ZOOM_HOV))
    print(fig_stats(out))


if __name__ == '__main__':
    main()
