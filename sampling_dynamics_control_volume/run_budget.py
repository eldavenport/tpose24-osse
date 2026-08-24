"""
Budget profiles: area-averaged w (from the OCV divergence theorem) and the vertical advective
HEATING w*dT/dz, truth vs obs. Per shape, plus a cross-shape comparison at each diameter.

The heat term is the advective heating RATE w*dT/dz [degC/day], NOT the flux w*T: it depends
on the temperature GRADIENT, so it is reference-independent (matches osse_tools convention).

  budget/<shape>/w_profiles.png        time-mean area-averaged w(z), truth (dotted) vs obs
                                       (solid) per diameter. (Truth-integrated w matches the
                                       model area-mean w to r=1.000, verified in run_ocv.)
  budget/<shape>/heating_profiles.png  time-mean advective heating w*dT/dz(z) [degC/day].
  budget/compare/w_profiles.png        w(z) with the three shapes overlaid, one panel per D.
  budget/compare/heating_profiles.png  same for the advective heating.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common as C
import cv_plot as P

# advective-heating components (reference-free, degC/day)
HEAT_COMPS = [('horiz', 'horizontal  $u\\cdot\\nabla_h T$'),
              ('vert', 'vertical  $w\\,\\partial_z T$'),
              ('total', 'total  $u\\cdot\\nabla T$')]
HEAT_UNIT = '$^\\circ$C day$^{-1}$'


def _profile(ds, var, which):
    """Time-mean profile (value, z). 'heat' = reference-free advective heating w*dT/dz."""
    if var == 'heat':
        da, zc = P.adv_heating_da(ds, which), 'obs_depth'
    else:
        da, zc = ds[f'{var}_{which}'], 'depth'
    m = da.mean('time')
    return m.values, m[zc].values


def fig_profiles(out, shape, var, scale, xlabel, fname):
    """Per-shape: time-mean budget profile, truth vs obs, all diameters (coloured by D)."""
    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    for d in C.DIAMETERS:
        ds = P.load_ocv(d, shape)
        col = C.diam_color(d)
        vt, z = _profile(ds, var, 'true')
        vo, _ = _profile(ds, var, 'obs')
        ax.plot(vt * scale, z, color=col, **P.TRUTH_KW)
        ax.plot(vo * scale, z, color=col, **P.OBS_KW)
    ax.axvline(0, color='0.7', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('depth (m)')
    P.tidy_x(ax)
    P.top_legend(fig, diam=True, method=True, y=1.06)
    return P.finish(fig, os.path.join(out, fname))


def fig_profiles_compare(out, var, scale, xlabel, fname):
    """Cross-shape: one panel per diameter, the three shapes overlaid (obs solid / truth
    dotted, coloured by shape)."""
    fig, axes = plt.subplots(1, len(C.DIAMETERS), figsize=(15, 5.2), sharey=True)
    for ax, d in zip(np.ravel(axes), C.DIAMETERS):
        for shape in C.SHAPES:
            ds = P.load_ocv(d, shape)
            col = C.SHAPE_COLOR[shape]
            vt, z = _profile(ds, var, 'true')
            vo, _ = _profile(ds, var, 'obs')
            ax.plot(vt * scale, z, color=col, **P.TRUTH_KW)
            ax.plot(vo * scale, z, color=col, **P.OBS_KW)
        ax.axvline(0, color='0.7', lw=1)
        ax.set_title(f'D = {d:g}$^\\circ$')
        ax.set_xlabel(xlabel, fontsize=10)
        P.tidy_x(ax, 3)
        ax.tick_params(axis='x', labelsize=9)
    axes[0].set_ylabel('depth (m)')
    P.shape_top_legend(fig, method=True, y=1.06)
    return P.finish(fig, os.path.join(out, fname))


def _heat_prof(ds, which, comp):
    m = P.adv_heating_da(ds, which, comp).mean('time')
    return m.values, m['obs_depth'].values


def fig_heating_profiles(out, shape):
    """Per-shape: time-mean advective-heating profiles, one panel per component
    (horizontal / vertical / total), all diameters overlaid (truth dotted, obs solid)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2), sharey=True)
    for ax, (comp, title) in zip(axes, HEAT_COMPS):
        for d in C.DIAMETERS:
            ds = P.load_ocv(d, shape)
            vt, z = _heat_prof(ds, 'true', comp)
            vo, _ = _heat_prof(ds, 'obs', comp)
            ax.plot(vt, z, color=C.diam_color(d), **P.TRUTH_KW)
            ax.plot(vo, z, color=C.diam_color(d), **P.OBS_KW)
        ax.axvline(0, color='0.7', lw=1)
        ax.set_title(title)
        ax.set_xlabel(HEAT_UNIT)
        P.tidy_x(ax, 4)
    axes[0].set_ylabel('depth (m)')
    P.top_legend(fig, diam=True, method=True, y=1.05)
    return P.finish(fig, os.path.join(out, 'heating_profiles.png'))


def fig_heating_profiles_compare(out):
    """Cross-shape: advective-heating profiles, rows = component (horizontal / vertical /
    total), cols = diameter; the three shapes overlaid (truth dotted, obs solid)."""
    short = {'horiz': 'horizontal', 'vert': 'vertical', 'total': 'total'}
    nr, nc = len(HEAT_COMPS), len(C.DIAMETERS)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 4.0 * nr), sharey=True)
    for r, (comp, _t) in enumerate(HEAT_COMPS):
        for cidx, d in enumerate(C.DIAMETERS):
            ax = axes[r][cidx]
            for shape in C.SHAPES:
                ds = P.load_ocv(d, shape)
                vt, z = _heat_prof(ds, 'true', comp)
                vo, _ = _heat_prof(ds, 'obs', comp)
                ax.plot(vt, z, color=C.SHAPE_COLOR[shape], **P.TRUTH_KW)
                ax.plot(vo, z, color=C.SHAPE_COLOR[shape], **P.OBS_KW)
            ax.axvline(0, color='0.7', lw=1)
            P.tidy_x(ax, 3)
            if r == 0:
                ax.set_title(f'D = {d:g}$^\\circ$')
            if r == nr - 1:
                ax.set_xlabel(HEAT_UNIT)
        axes[r][0].set_ylabel(f'{short[comp]}\ndepth (m)')
    P.shape_top_legend(fig, method=True, y=1.02)
    return P.finish(fig, os.path.join(out, 'heating_profiles.png'))


def fig_budget_closure(out, shape):
    """Truth advective heat budget: does -(u.grad T) explain the OCV warming/cooling?
    solid = advective tendency -u.grad T; dotted = actual d<T>/dt; the gap is the
    non-advective residual (KPP mixing + surface forcing + sub-OCV covariance)."""
    fig, ax = plt.subplots(figsize=(6.6, 6.8))
    for d in C.DIAMETERS:
        ds = P.load_ocv(d, shape)
        adv = (-P.adv_heating_da(ds, 'true', 'total')).mean('time')
        st = P.dTdt_da(ds, 'true').mean('time')
        ax.plot(adv.values, adv['obs_depth'].values, color=C.diam_color(d), ls='-', lw=2)
        ax.plot(st.values, st['obs_depth'].values, color=C.diam_color(d), ls=':', lw=2.4)
    ax.axvline(0, color='0.7', lw=1)
    ax.set_xlabel(HEAT_UNIT)
    ax.set_ylabel('depth (m)')
    P.tidy_x(ax, 5)
    extra = [Line2D([0], [0], color='0.3', ls='-', lw=2, label='$-u\\cdot\\nabla T$ (advective)'),
             Line2D([0], [0], color='0.3', ls=':', lw=2.4, label='$\\partial_t\\langle T\\rangle$ (actual)')]
    fig.legend(handles=[Line2D([0], [0], color=C.diam_color(d), lw=2.6, label=f'{d:g}$^\\circ$')
                        for d in C.DIAMETERS] + extra, loc='upper center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.09))
    return P.finish(fig, os.path.join(out, 'budget_closure.png'))


def _budget_nmetric(shape, var, metric='rmse'):
    """Depth-averaged normalized metric (obs vs truth) of the vertically-integrated budget,
    per diameter, over the OBSERVED 8-80 m range (0-8 m is shear-extrapolated, not observed).
    The per-depth std normalization stays well-behaved near the w=0 surface because the error
    and std shrink together. metric='rmse' or 'bias' (signed mean_time(obs-truth)/std_time)."""
    out = []
    for d in C.DIAMETERS:
        ds = P.load_ocv(d, shape)
        if var == 'heat':
            t, o, zc = P.adv_heating_da(ds, 'true'), P.adv_heating_da(ds, 'obs'), 'obs_depth'
        else:
            t, o, zc = ds[f'{var}_true'], ds[f'{var}_obs'], 'depth'
        z = t[zc].values
        m = (z <= -C.MIN_DEPTH) & (z >= -C.MAX_DEPTH)
        tt, oo = t.isel({zc: m}), o.isel({zc: m})
        if metric == 'bias':
            val = ((oo - tt).mean('time') / tt.std('time')).mean(zc)
        else:
            val = (np.sqrt(((oo - tt) ** 2).mean('time')) / tt.std('time')).mean(zc)
        out.append(float(val))
    return out


def fig_budget_error_compare(out):
    """Cross-shape budget error vs D. Row 1: normalized RMSE; row 2: signed normalized bias,
    of the integrated area-averaged w and vertical heat flux (obs vs truth), one line/shape."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), sharex=True)
    for c, (var, title) in enumerate([('w', 'area-averaged w'),
                                      ('heat', 'total advective heating')]):
        for r, metric in enumerate(('rmse', 'bias')):
            ax = axes[r][c]
            for shape in C.SHAPES:
                ax.plot(C.DIAMETERS, _budget_nmetric(shape, var, metric), 'o-',
                        color=C.SHAPE_COLOR[shape], lw=2.2, ms=7, label=C.SHAPE_LABEL[shape])
            if metric == 'bias':
                ax.axhline(0, color='0.6', lw=1)
            else:
                ax.set_ylim(bottom=0)
                ax.set_title(title)
        axes[1][c].set_xlabel('diameter D ($^\\circ$)')
    axes[0][0].set_ylabel('normalized RMSE (8-80 m)')
    axes[1][0].set_ylabel('normalized bias (8-80 m)')
    axes[0][0].legend(frameon=False, title='shape')
    return P.finish(fig, os.path.join(out, 'error_vs_D.png'))


def main():
    for shape in C.SHAPES:
        out = P.outdir('budget', shape)
        fig_profiles(out, shape, 'w', C.SEC_PER_DAY, 'area-averaged w (m day$^{-1}$)',
                     'w_profiles.png')
        fig_heating_profiles(out, shape)
        fig_budget_closure(out, shape)
    cmp = P.outdir('budget', 'compare')
    fig_profiles_compare(cmp, 'w', C.SEC_PER_DAY, 'area-averaged w (m day$^{-1}$)',
                         'w_profiles.png')
    fig_heating_profiles_compare(cmp)
    fig_budget_error_compare(cmp)
    print('wrote budget figures -> budget/<shape>/ and budget/compare/')


if __name__ == '__main__':
    main()
