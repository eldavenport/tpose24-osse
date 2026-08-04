"""
Vertical-velocity diagnostics for the virtual mooring.

How well does the hexagon + centre mooring recover the area-mean vertical velocity w
that the model produces at 0degN,140degW, and how does that depend on the E-W diameter?
The array estimate is the plane-fit w (compute_w_planefit defaults, 8-80 m); the truth
is the hull-mean model WVEL.  All figures compare array (solid) vs truth (dotted),
coloured light->dark with diameter.  w is shown in m day^-1.

Figures -> vertical_velocity/
  w_profiles.png     time-mean w(z), std(z) (variability), and array-vs-truth r(z).
  w_stats.png        max/min/mean/median of the w time series (depth-avg + 30 m).
  w_timemean_ci.png  time-mean w(z) with autocorrelation-aware 95% CI, est vs true.
  w_hovmoller.png    w(z,t) truth, estimate, and error for the 1.0deg footprint.

Usage:  python run_w.py
"""

import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo

import common as C
import osse_tools as ot
import sd_plot as P

SUB = 'vertical_velocity'
DAY = C.SEC_PER_DAY


def _w_pair(diam):
    """(w_est_mid, w_truth) both (time, obs_depth) in m/day, on the same obs axis."""
    arr = P.load_array(diam)
    hull = P.load_hull(diam)
    return arr['w_est_mid'] * DAY, hull['W'] * DAY


# --------------------------------------------------------------------------- fig 1
def fig_profiles(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True)
    for d in C.DIAMETERS:
        we, wt = _w_pair(d)
        z = we.obs_depth.values
        c = C.diam_color(d)
        axes[0].plot(we.mean('time'), z, color=c, **P.ARRAY_KW)
        axes[0].plot(wt.mean('time'), z, color=c, **P.TRUTH_KW)
        axes[1].plot(we.std('time'), z, color=c, **P.ARRAY_KW)
        axes[1].plot(wt.std('time'), z, color=c, **P.TRUTH_KW)
        # array-vs-truth correlation per depth
        r = xr_corr(we, wt, 'time')
        axes[2].plot(r, z, color=c, **P.ARRAY_KW)
    axes[0].set(xlabel='mean $w$ (m day$^{-1}$)', ylabel='depth (m)')
    axes[1].set(xlabel='$w$ variability, std (m day$^{-1}$)')
    axes[2].set(xlabel='array-truth correlation $r$', xlim=(0.9, 1.005))
    axes[2].axvline(1, color='0.6', lw=0.8)
    P.top_legend(fig)
    return P.finish(fig, f'{out}/w_profiles.png')


def xr_corr(a, b, dim):
    ap = a - a.mean(dim); bp = b - b.mean(dim)
    return (ap * bp).mean(dim) / (a.std(dim) * b.std(dim))


# --------------------------------------------------------------------------- fig 2
def fig_stats(out):
    """max/min/mean/median of the w series, array vs truth, per diameter."""
    depths = [('0-80 m avg', None), ('30 m', -30.0)]
    stat_names = ['min', 'median', 'mean', 'max']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    x = np.arange(len(C.DIAMETERS))
    for ax, (dlabel, zsel) in zip(axes, depths):
        for si, sname in enumerate(stat_names):
            ea, ta = [], []
            for d in C.DIAMETERS:
                we, wt = _w_pair(d)
                if zsel is None:
                    we = we.mean('obs_depth'); wt = wt.mean('obs_depth')
                else:
                    we = we.interp(obs_depth=zsel); wt = wt.interp(obs_depth=zsel)
                s_e = P.time_stats(we)[sname]; s_t = P.time_stats(wt)[sname]
                ea.append(float(s_e)); ta.append(float(s_t))
            col = plt.cm.viridis(si / (len(stat_names) - 1))
            ax.plot(x, ta, 'o', color=col, ms=9, mfc='none', mew=2)
            ax.plot(x, ea, 's', color=col, ms=8)
        ax.set_xticks(x); ax.set_xticklabels([f'{d:g}$^\\circ$' for d in C.DIAMETERS])
        ax.set(xlabel='E-W diameter', title=dlabel)
        ax.axhline(0, color='0.6', lw=0.8)
    axes[0].set_ylabel('$w$ (m day$^{-1}$)')
    # legends: statistic color + marker meaning (self-legending)
    from matplotlib.lines import Line2D
    stat_h = [Line2D([0], [0], marker='o', ls='', color=plt.cm.viridis(i/(len(stat_names)-1)),
                     label=s, ms=8) for i, s in enumerate(stat_names)]
    mk_h = [Line2D([0], [0], marker='s', ls='', color='0.3', label='array estimate', ms=8),
            Line2D([0], [0], marker='o', ls='', color='0.3', mfc='none', mew=2,
                   label='model truth', ms=9)]
    fig.legend(handles=stat_h + mk_h, loc='upper center', ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    return P.finish(fig, f'{out}/w_stats.png')


# --------------------------------------------------------------------------- fig 3
def fig_timemean_ci(out):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for d in C.DIAMETERS:
        we, wt = _w_pair(d)
        z = we.obs_depth.values
        c = C.diam_color(d)
        me = np.array([ot.mean_se_autocorr(we.isel(obs_depth=k).values) for k in range(z.size)])
        mt = np.array([ot.mean_se_autocorr(wt.isel(obs_depth=k).values) for k in range(z.size)])
        ax.plot(me[:, 0], z, color=c, **P.ARRAY_KW)
        ax.fill_betweenx(z, me[:, 0] - 1.96 * me[:, 1], me[:, 0] + 1.96 * me[:, 1],
                         color=c, alpha=0.18)
        ax.plot(mt[:, 0], z, color=c, **P.TRUTH_KW)
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set(xlabel='time-mean $w$ (m day$^{-1}$)', ylabel='depth (m)')
    P.top_legend(fig)
    return P.finish(fig, f'{out}/w_timemean_ci.png')


# --------------------------------------------------------------------------- fig 4
def fig_hovmoller(out):
    """w(z,t) truth / estimate / error, one COLUMN per E-W diameter (shared scales)."""
    data = {}
    for d in C.DIAMETERS:
        we, wt = _w_pair(d)
        data[d] = (wt, we, we - wt)
    z = data[C.DIAMETERS[0]][0].obs_depth.values
    tcoord = data[C.DIAMETERS[0]][0].time.values
    t = (tcoord - tcoord[0]) / np.timedelta64(1, 'D')
    # shared color scales: w (truth+estimate) and the error, over ALL diameters
    vmax = float(np.nanpercentile(np.abs(np.concatenate(
        [np.abs(data[d][0].values).ravel() for d in C.DIAMETERS])), 99))
    emax = float(np.nanpercentile(np.abs(np.concatenate(
        [np.abs(data[d][2].values).ravel() for d in C.DIAMETERS])), 99))

    rows = [('model truth', 0, vmax), ('array estimate', 1, vmax),
            ('estimate $-$ truth', 2, emax)]
    nc = len(C.DIAMETERS)
    fig, axes = plt.subplots(3, nc, figsize=(4.2 * nc, 9), sharex=True, sharey=True)
    pcs = {}
    for ci, d in enumerate(C.DIAMETERS):
        axes[0, ci].set_title(f'{d:g}$^\\circ$ footprint')
        for ri, (lbl, idx, vm) in enumerate(rows):
            ax = axes[ri, ci]
            pcs[ri] = ax.pcolormesh(t, z, data[d][idx].T, cmap=cmo.balance,
                                    vmin=-vm, vmax=vm, shading='auto')
            if ci == 0:
                ax.set_ylabel(f'{lbl}\ndepth (m)')
            if ri == 2:
                ax.set_xlabel('days since 2012-10-11')
    # one shared colorbar per row (w rows share the w scale, error its own)
    fig.colorbar(pcs[0], ax=axes[:2, :].ravel().tolist(), pad=0.01,
                 label='$w$ (m day$^{-1}$)')
    fig.colorbar(pcs[2], ax=axes[2, :].ravel().tolist(), pad=0.01,
                 label='error (m day$^{-1}$)')
    return P.finish(fig, f'{out}/w_hovmoller.png')


def main():
    out = P.outdir(SUB)
    print(fig_profiles(out))
    print(fig_stats(out))
    print(fig_timemean_ci(out))
    print(fig_hovmoller(out))


if __name__ == '__main__':
    main()
