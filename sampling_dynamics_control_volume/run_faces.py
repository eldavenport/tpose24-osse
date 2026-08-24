"""
Per-face diagnostics of the OCV, for every shape, plus a cross-shape comparison.

  faces/<shape>/layout.png            OCV geometry: inscribed circle (diameter D), polygon
                                      (vertices outside), face-centre gliders on the circle,
                                      outward normals, numbered faces -- one panel per D.
  faces/<shape>/current_profiles.png  time-mean outward-normal current profile per face,
                                      truth (dotted) vs obs (solid), all diameters overlaid.
  faces/<shape>/anomflux_profiles.png same for the reference-free outward heat-flux
                                      anomaly u_n(T-<T>) (sums over faces -> horiz. heating).
  faces/<shape>/midpoint_error.png    per-face normalized RMSE (face-centre glider vs full
                                      edge average) vs D -- the midpoint-rule sampling error.
  faces/compare/midpoint_error.png    face-mean midpoint error vs D, one line per SHAPE.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common as C
import cv_plot as P


def fig_layout(out, shape):
    fig, axes = plt.subplots(1, len(C.DIAMETERS), figsize=(16, 4.2))
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, d in zip(np.ravel(axes), C.DIAMETERS):
        fg = C.face_geometry(d, shape)
        r = d / 2.0
        ax.plot(220 + r * np.cos(th), r * np.sin(th), color='0.6', lw=1.4)  # inscribed circle
        vx = [v[1] for v in fg.vertices] + [fg.vertices[0][1]]
        vy = [v[0] for v in fg.vertices] + [fg.vertices[0][0]]
        ax.plot(vx, vy, '-', color=C.diam_color(d), lw=2)                    # polygon
        fc = np.array(fg.face_centers)
        ax.plot(fc[:, 1], fc[:, 0], 'o', color=C.diam_color(d), ms=8)        # gliders
        ax.plot(220, 0, 'kx', ms=8, mew=2)                                   # mooring
        for f, (c, n) in enumerate(zip(fg.face_centers, fg.normals)):
            ax.annotate('', xy=(c[1] + 0.18 * r * n[0], c[0] + 0.18 * r * n[1]),
                        xytext=(c[1], c[0]),
                        arrowprops=dict(arrowstyle='->', color='0.35', lw=1.3))
            ax.text(c[1] + 0.30 * r * n[0], c[0] + 0.30 * r * n[1], str(f),
                    color='0.15', fontsize=11, fontweight='bold', ha='center', va='center')
        ax.set_title(f'D = {d:g}$^\\circ$', color=C.diam_color(d))
        ax.set_aspect('equal')
        ax.margins(0.22)
        ax.set_xlabel('lon ($^\\circ$E)')
    axes[0].set_ylabel('lat ($^\\circ$N)')
    fig.text(0.5, 1.02, f'{C.SHAPE_LABEL[shape]}: gliders (o) on the inscribed circle, '
             'vertices outside, mooring (x) at centre; numbers = face indices',
             ha='center', fontsize=11)
    return P.finish(fig, os.path.join(out, 'layout.png'))


def _bearing(ds, f):
    """D-independent face bearing (deg): 0=E, +-180=W, +90=N."""
    return np.degrees(np.arctan2(float(ds.face_lat.isel(face=f)),
                                 float(ds.face_lon.isel(face=f)) - 220))


def _face_da(ds, var, which):
    """Per-face series (time, face, obs_depth). 'anomflux' = reference-free outward
    heat-flux anomaly u_n*(T-<T>); otherwise the raw cached var (e.g. 'un')."""
    if var == 'anomflux':
        return P.face_anom_flux_da(ds, which)
    return ds[f'{var}_{which}']


def fig_face_profiles(out, shape, var, scale, xlabel, fname):
    """Time-mean outward profile per face, truth (dotted) vs obs (solid), all diameters."""
    dss = {d: P.load_ocv(d, shape) for d in C.DIAMETERS}
    nf = dss[C.DIAMETERS[0]].sizes['face']
    ncol = min(3, nf)
    nrow = int(np.ceil(nf / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False,
                             sharey=True)
    for f in range(nf):
        ax = axes[f // ncol][f % ncol]
        for d in C.DIAMETERS:
            ds = dss[d]; col = C.diam_color(d)
            t = _face_da(ds, var, 'true').isel(face=f).mean('time')
            o = _face_da(ds, var, 'obs').isel(face=f).mean('time')
            ax.plot(t.values * scale, t['obs_depth'].values, color=col, **P.TRUTH_KW)
            ax.plot(o.values * scale, o['obs_depth'].values, color=col, **P.OBS_KW)
        ax.axvline(0, color='0.7', lw=1)
        ax.set_title(f'face {f}  ({_bearing(dss[C.DIAMETERS[0]], f):+.0f}$^\\circ$)')
        ax.set_xlabel(xlabel)
        P.tidy_x(ax, 4)
    for k in range(nf, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    axes[0][0].set_ylabel('depth (m)')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    P.top_legend(fig, diam=True, method=True, y=1.02)
    return P.finish(fig, os.path.join(out, fname))


def _nmetric_by_face(shape, var, metric='rmse'):
    """Depth-averaged normalized error metric per face, per diameter. metric='rmse' ->
    normalized RMSE (rmse_time/std_time); 'bias' -> signed mean_time(obs-truth)/std_time."""
    nf = P.load_ocv(C.DIAMETERS[0], shape).sizes['face']
    err = {f: [] for f in range(nf)}
    for d in C.DIAMETERS:
        ds = P.load_ocv(d, shape)
        t, o = _face_da(ds, var, 'true'), _face_da(ds, var, 'obs')
        if metric == 'bias':
            m = ((o - t).mean('time') / t.std('time')).mean('obs_depth')
        else:
            m = (np.sqrt(((o - t) ** 2).mean('time')) / t.std('time')).mean('obs_depth')
        for f in range(nf):
            err[f].append(float(m.isel(face=f).values))
    return err, nf


def fig_midpoint_error(out, shape):
    """Per-face midpoint-rule error vs D. Rows: normalized RMSE (scatter+bias) and signed
    normalized bias (systematic over/under-estimate); cols: current / heat flux."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), sharex=True)
    ds0 = P.load_ocv(C.DIAMETERS[0], shape)
    for c, (var, title) in enumerate([('un', 'outward current'), ('anomflux', 'outward heat-flux anomaly')]):
        for r, metric in enumerate(('rmse', 'bias')):
            ax = axes[r][c]
            err, nf = _nmetric_by_face(shape, var, metric)
            cols = plt.cm.viridis(np.linspace(0, 0.88, nf))
            for f in range(nf):
                ax.plot(C.DIAMETERS, err[f], 'o-', color=cols[f], lw=2, ms=6,
                        label=f'face {f}  ({_bearing(ds0, f):+.0f}$^\\circ$)')
            if metric == 'bias':
                ax.axhline(0, color='0.6', lw=1)
            else:
                ax.set_ylim(bottom=0)
                ax.set_title(title)
        axes[1][c].set_xlabel('diameter D ($^\\circ$)')
    axes[0][0].set_ylabel('per-face normalized RMSE')
    axes[1][0].set_ylabel('per-face normalized bias')
    axes[0][0].legend(frameon=False, fontsize=8.5,
                      title='face (bearing: 0$^\\circ$=E, 180$^\\circ$=W)')
    return P.finish(fig, os.path.join(out, 'midpoint_error.png'))


def fig_midpoint_error_compare(out):
    """Cross-shape midpoint error/bias vs D. Rows 1-2: normalized RMSE (face-mean, then every
    face by opacity). Rows 3-4: signed normalized bias (same layout). Cols: current / heat
    flux. Shapes are green/red/blue; individual faces vary by line opacity."""
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 17), sharex=True)
    for c, (var, title) in enumerate([('un', 'outward current'), ('anomflux', 'outward heat-flux anomaly')]):
        for mi, metric in enumerate(('rmse', 'bias')):
            ax_mean, ax_face = axes[2 * mi][c], axes[2 * mi + 1][c]
            for shape in C.SHAPES:
                err, nf = _nmetric_by_face(shape, var, metric)
                mean_e = [np.mean([err[f][i] for f in range(nf)]) for i in range(len(C.DIAMETERS))]
                ax_mean.plot(C.DIAMETERS, mean_e, 'o-', color=C.SHAPE_COLOR[shape], lw=2.2,
                             ms=7, label=C.SHAPE_LABEL[shape])
                alphas = np.linspace(0.3, 1.0, nf)
                for f in range(nf):
                    ax_face.plot(C.DIAMETERS, err[f], '-', color=C.SHAPE_COLOR[shape],
                                 lw=2.0, alpha=alphas[f])
            for ax in (ax_mean, ax_face):
                if metric == 'bias':
                    ax.axhline(0, color='0.6', lw=1)
                else:
                    ax.set_ylim(bottom=0)
        axes[0][c].set_title(title)
        axes[3][c].set_xlabel('diameter D ($^\\circ$)')
    axes[0][0].set_ylabel('face-mean normalized RMSE')
    axes[1][0].set_ylabel('per-face normalized RMSE')
    axes[2][0].set_ylabel('face-mean normalized bias')
    axes[3][0].set_ylabel('per-face normalized bias')
    axes[0][0].legend(frameon=False, title='shape')
    axes[1][0].legend(handles=[Line2D([0], [0], color=C.SHAPE_COLOR[s], lw=2,
                                      label=C.SHAPE_LABEL[s]) for s in C.SHAPES],
                      frameon=False, title='shape (opacity = face)')
    return P.finish(fig, os.path.join(out, 'midpoint_error.png'))


def main():
    for shape in C.SHAPES:
        out = P.outdir('faces', shape)
        fig_layout(out, shape)
        fig_face_profiles(out, shape, 'un', 1.0, 'outward current (m s$^{-1}$)',
                          'current_profiles.png')
        fig_face_profiles(out, shape, 'anomflux', 1.0,
                          "outward $u_n(T-\\langle T\\rangle)$ ($^\\circ$C m s$^{-1}$)",
                          'anomflux_profiles.png')
        fig_midpoint_error(out, shape)
    fig_midpoint_error_compare(P.outdir('faces', 'compare'))
    print('wrote face figures -> faces/<shape>/ and faces/compare/')


if __name__ == '__main__':
    main()
