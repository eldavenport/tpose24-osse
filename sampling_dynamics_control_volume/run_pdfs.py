"""
Temporal PDFs of the space-averaged OCV quantities: truth vs obs, per shape, and their
convergence with diameter. Spatial averaging is already done, so every PDF is a distribution
over TIME. Every distribution is drawn as a SOLID histogram line with a DASHED skew-normal
fit; an upper-left box lists the fit moments (mean/std/skew) for each series (truth included).

Per shape (distributions/<shape>/):
  w_pdfs.png / heating_pdfs.png   area-avg w / total advective heating u.gradT at OCV base, panel per D.
  convergence.png                 JS(obs,truth) vs D for w and heat flux, at base + mid depth.
  volmean_pdfs_30m.png / _70m.png OCV volume-mean T,S,U,V (rows) x diameter (cols), truth vs obs.
  face_pdfs_30m.png / _70m.png    per-face outward current (rows=face) x diameter (cols).

Cross-shape (distributions/compare/):
  convergence.png                 JS(obs,truth) vs D, one line per SHAPE, for w and heat flux.
  w_pdfs.png / heating_pdfs.png   truth + one fit per shape, one panel per D.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common as C
import cv_plot as P

BASE = -70.0        # base of the OCV column (m)
MID = -30.0
DEPTHS = [(MID, '30m'), (BASE, '70m')]
TRUTH_COL = '0.45'

# style key: solid = histogram, dashed = parametric (skew-normal) fit
_STYLE = [Line2D([0], [0], color='0.25', lw=2.0, ls='-', label='histogram'),
          Line2D([0], [0], color='0.25', lw=1.8, ls='--', label='skew-normal fit')]


def _getter(kind):
    return P.w_series if kind == 'w' else P.heat_series


def _vol(ds, v, which, z):
    return np.asarray(ds[f'{v}bar_{which}'].interp(obs_depth=z).values).ravel()


def _face(ds, which, f, z):
    return np.asarray(ds[f'un_{which}'].isel(face=f).interp(obs_depth=z).values).ravel()


# ---------------------------------------------------------------- per-shape: w / hflux PDFs
def fig_series_pdfs(out, shape, kind, z, unit, fname, scale=1.0):
    getter = _getter(kind)
    fig, axes = plt.subplots(1, len(C.DIAMETERS), figsize=(15, 4), sharey=True)
    for ax, d in zip(np.ravel(axes), C.DIAMETERS):
        ds = P.load_ocv(d, shape)
        series = [('truth', getter(ds, 'true', z) * scale, TRUTH_COL),
                  ('array', getter(ds, 'obs', z) * scale, C.diam_color(d))]
        P.series_pdf_panel(ax, series, unit)
        ax.set_title(f'D = {d:g}$^\\circ$', color=C.diam_color(d))
    axes[0].set_ylabel('probability density')
    fig.legend(handles=_STYLE, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.10))
    return P.finish(fig, os.path.join(out, fname))


# ---------------------------------------------------------------- per-shape: convergence
def fig_convergence(out, shape):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for kind, title, ax in [('w', 'area-averaged w', axes[0]),
                            ('heat', 'advective heating', axes[1])]:
        getter = _getter(kind)
        for z, mk, lab in [(BASE, 'o-', f'{-BASE:.0f} m'), (MID, 's--', f'{-MID:.0f} m')]:
            js = [P.js_dist(getter(P.load_ocv(d, shape), 'obs', z),
                            getter(P.load_ocv(d, shape), 'true', z)) for d in C.DIAMETERS]
            ax.plot(C.DIAMETERS, js, mk, color='#08306b' if z == BASE else '#6baed6',
                    lw=2, ms=7, label=lab)
        ax.set_title(title)
        ax.set_xlabel('diameter D ($^\\circ$)')
        ax.set_ylabel('JS distance (obs vs truth)')
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, title='depth')
    return P.finish(fig, os.path.join(out, 'convergence.png'))


# ---------------------------------------------------------------- per-shape: volume means
def fig_volmean(out, shape, z, tag):
    """Rows = OCV volume-mean quantity, cols = diameter; each cell truth vs obs + fit box."""
    specs = [('T', '$^\\circ$C'), ('S', 'g kg$^{-1}$'),
             ('U', 'm s$^{-1}$'), ('V', 'm s$^{-1}$')]
    nrow, ncol = len(specs), len(C.DIAMETERS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
    for i, (v, unit) in enumerate(specs):
        for j, d in enumerate(C.DIAMETERS):
            ax = axes[i][j]
            ds = P.load_ocv(d, shape)
            series = [('truth', _vol(ds, v, 'true', z), TRUTH_COL),
                      ('array', _vol(ds, v, 'obs', z), C.diam_color(d))]
            P.series_pdf_panel(ax, series, unit if i == nrow - 1 else '', box=True)
            if i == 0:
                ax.set_title(f'D = {d:g}$^\\circ$', color=C.diam_color(d))
        axes[i][0].set_ylabel(f'{v}  ({unit})')
    fig.legend(handles=_STYLE, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return P.finish(fig, os.path.join(out, f'volmean_pdfs_{tag}.png'))


# ---------------------------------------------------------------- per-shape: per-face current
def fig_face_pdfs(out, shape, z, tag):
    """Rows = face, cols = diameter; each cell outward-current truth vs obs + fit box."""
    nf = P.load_ocv(C.DIAMETERS[0], shape).sizes['face']
    ncol = len(C.DIAMETERS)
    unit = 'outward current (m s$^{-1}$)'
    fig, axes = plt.subplots(nf, ncol, figsize=(3.4 * ncol, 2.5 * nf), squeeze=False)
    for f in range(nf):
        for j, d in enumerate(C.DIAMETERS):
            ax = axes[f][j]
            ds = P.load_ocv(d, shape)
            series = [('truth', _face(ds, 'true', f, z), TRUTH_COL),
                      ('array', _face(ds, 'obs', f, z), C.diam_color(d))]
            P.series_pdf_panel(ax, series, unit if f == nf - 1 else '', box=True)
            if f == 0:
                ax.set_title(f'D = {d:g}$^\\circ$', color=C.diam_color(d))
        b = np.degrees(np.arctan2(float(P.load_ocv(C.DIAMETERS[0], shape).face_lat.isel(face=f)),
                                  float(P.load_ocv(C.DIAMETERS[0], shape).face_lon.isel(face=f)) - 220))
        axes[f][0].set_ylabel(f'face {f}  ({b:+.0f}$^\\circ$)')
    fig.legend(handles=_STYLE, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return P.finish(fig, os.path.join(out, f'face_pdfs_{tag}.png'))


# ---------------------------------------------------------------- cross-shape: convergence
def fig_convergence_compare(out, z=BASE):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for kind, title, ax in [('w', 'area-averaged w', axes[0]),
                            ('heat', 'advective heating', axes[1])]:
        getter = _getter(kind)
        for shape in C.SHAPES:
            js = [P.js_dist(getter(P.load_ocv(d, shape), 'obs', z),
                            getter(P.load_ocv(d, shape), 'true', z)) for d in C.DIAMETERS]
            ax.plot(C.DIAMETERS, js, 'o-', color=C.SHAPE_COLOR[shape], lw=2.2, ms=7,
                    label=C.SHAPE_LABEL[shape])
        ax.set_title(f'{title}  ({-z:.0f} m)')
        ax.set_xlabel('diameter D ($^\\circ$)')
        ax.set_ylabel('JS distance (obs vs truth)')
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, title='shape')
    return P.finish(fig, os.path.join(out, 'convergence.png'))


# ---------------------------------------------------------------- cross-shape: PDFs per D
def fig_series_pdfs_compare(out, kind, z, unit, fname, scale=1.0):
    getter = _getter(kind)
    fig, axes = plt.subplots(1, len(C.DIAMETERS), figsize=(15, 4), sharey=True)
    for ax, d in zip(np.ravel(axes), C.DIAMETERS):
        # truths across shapes nearly coincide -> pool into one reference distribution
        pooled = np.concatenate([getter(P.load_ocv(d, s), 'true', z) for s in C.SHAPES]) * scale
        series = [('truth', pooled, TRUTH_COL)]
        series += [(C.SHAPE_LABEL[s], getter(P.load_ocv(d, s), 'obs', z) * scale, C.SHAPE_COLOR[s])
                   for s in C.SHAPES]
        P.series_pdf_panel(ax, series, unit)
        ax.set_title(f'D = {d:g}$^\\circ$')
    axes[0].set_ylabel('probability density')
    fig.legend(handles=_STYLE, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.10))
    return P.finish(fig, os.path.join(out, fname))


def main():
    for shape in C.SHAPES:
        out = P.outdir('distributions', shape)
        fig_series_pdfs(out, shape, 'w', BASE, 'area-averaged w (m day$^{-1}$)', 'w_pdfs.png')
        fig_series_pdfs(out, shape, 'heat', BASE,
                        '$u\\cdot\\nabla T$ ($^\\circ$C day$^{-1}$)', 'heating_pdfs.png')
        fig_convergence(out, shape)
        for z, tag in DEPTHS:
            fig_volmean(out, shape, z, tag)
            fig_face_pdfs(out, shape, z, tag)
    cmp = P.outdir('distributions', 'compare')
    fig_convergence_compare(cmp)
    fig_series_pdfs_compare(cmp, 'w', BASE, 'area-averaged w (m day$^{-1}$)', 'w_pdfs.png')
    fig_series_pdfs_compare(cmp, 'heat', BASE,
                            '$u\\cdot\\nabla T$ ($^\\circ$C day$^{-1}$)', 'heating_pdfs.png')
    print('wrote distribution figures -> distributions/<shape>/ and distributions/compare/')


if __name__ == '__main__':
    main()
