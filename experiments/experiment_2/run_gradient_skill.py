#!/usr/bin/env python
"""
run_gradient_skill.py — plane-fit skill for the INDIVIDUAL horizontal gradients
du/dx and dv/dy (not just their sum, the divergence), for the symmetric regular-shape
sweep (symhex / symdia / symsq) in experiment_2.

Motivation: the hexagon almost always has the lowest error in the divergence δ =
du/dx + dv/dy (hence in w). This asks whether that win survives in the individual
gradients, or whether a shape can get a good δ from du/dx and dv/dy errors that
compensate (one over-, the other under-estimated by a similar amount).

Estimate: the single plane-fit slopes compute_w_planefit reports (constant over the
footprint) — now returned as `du_dx`, `dv_dy` alongside `div`.
Truth: the footprint area-mean of the model's true du/dx and dv/dy, on the same
C-grid flux form as model_divergence (osse_tools.model_gradient_components), using
the symmetric-shape circular-disk footprint (sym_disk).

Same physics/sampling as run_experiment_2.py (extrapolate to surface, 8–80 m).

Usage (tpose env):
  python run_gradient_skill.py [compute|plot|all]   (default all)

Outputs (experiment_2/):
  data_grad_skill/<config>__cell_<center>.nc  dudx/dvdy/div est+true (time, depth)
  data_grad_skill/metrics.csv                 long table, one row per (cell, quantity)
  summary_figs/gradient_sweep/                key comparison figures
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXP1 = os.path.join(REPO, "experiments", "experiment_1")
sys.path.insert(0, REPO)
import osse_tools as ot          # noqa: E402
import summary_fig_tools as sft  # noqa: E402  (sym constants, colors, styles)

RUN_DIR    = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS      = list(range(36, 26173, 36))
SPINUP_END = '2012-10-11'
MIN_DEPTH  = 8
MAX_DEPTH  = 80
DZ_OBS     = 2
MAX_WORKERS = 8

DATA_DIR = os.path.join(HERE, 'data_grad_skill')
FIG_DIR  = os.path.join(HERE, 'summary_figs', 'gradient_sweep')
CFG_GLOB = os.path.join(EXP1, 'configs', 'sym*', '*.json')  # symhex/symdia/symsq

GRAD2DAY = 86400.0                                   # 1/s -> 1/day
QTYS = [('du_dx', r'$\partial u/\partial x$'),
        ('dv_dy', r'$\partial v/\partial y$'),
        ('div',   r'divergence $\delta$')]


# --------------------------------------------------------------------------- compute
def _footprint_key(positions):
    return tuple(sorted((round(p[0], 6), round(p[1], 6)) for p in positions))


def _as_depth(da):
    """Rename the obs-midpoint axis to 'depth' so w_skill_metrics can align est/truth."""
    return da.rename({'obs_depth': 'depth'})


def _metrics_rows(cfg, center_lat, nc_path):
    """One skill row per quantity (du/dx, dv/dy, div) for a cell."""
    rows = []
    with xr.open_dataset(nc_path) as ds:
        for q, _ in QTYS:
            m = ot.w_skill_metrics(ds[f'{q}_est'], ds[f'{q}_true'])
            rows.append(dict(
                config=cfg['name'], family=cfg['family'], quantity=q,
                width=cfg['width'], center_lat=center_lat, **m))
    return rows


def compute():
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = sorted(glob.glob(CFG_GLOB))
    print(f'Found {len(paths)} sym configs')

    cached, pending = [], []     # (cfg, center_lat, [pos], nc_path)
    for path in paths:
        with open(path) as f:
            cfg = json.load(f)
        for center_lat, pos in ot.load_cells(path):
            nc = os.path.join(DATA_DIR, f'{cfg["name"]}__cell_{center_lat:+.2f}.nc')
            (cached if os.path.exists(nc) else pending).append((cfg, center_lat, pos, nc))
    print(f'{len(cached)} cached, {len(pending)} to compute')

    rows = [r for cfg, cl, _, nc in cached for r in _metrics_rows(cfg, cl, nc)]

    if pending:
        all_pos = sorted({(round(p[0], 6), round(p[1], 6))
                          for _, _, pos, _ in pending for p in pos})
        pidx = {p: i for i, p in enumerate(all_pos)}
        ds = ot.load_model(RUN_DIR, ITERS).sel(time=slice(SPINUP_END, None))
        print(f'Model loaded: {ds.sizes["time"]} timesteps')
        uv_all = ot.sample_fields(ds, all_pos, vars=('UVEL', 'VVEL'),
                                  max_depth=MAX_DEPTH, dz_obs=DZ_OBS,
                                  min_depth=MIN_DEPTH).compute()

        # truth gradients are one computation per unique footprint (shared across shapes)
        unique = {}
        for cfg, _, pos, _ in pending:
            unique.setdefault(_footprint_key(pos), (pos, ot.sym_disk(cfg)))

        def _truth(pos, disk):
            return ot.model_gradient_components(ds, pos, max_depth=MAX_DEPTH,
                                                dz_obs=DZ_OBS, min_depth=MIN_DEPTH, disk=disk)
        keys = list(unique)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            truth = dict(zip(keys, ex.map(lambda k: _truth(*unique[k]), keys)))
        print('Truth gradients computed')

        for cfg, center_lat, pos, nc in pending:
            dudx_true, dvdy_true = truth[_footprint_key(pos)]
            uv = uv_all.isel(glider=[pidx[(round(p[0], 6), round(p[1], 6))] for p in pos])
            est = ot.compute_w_planefit(uv)
            out = xr.Dataset(dict(
                du_dx_est=_as_depth(est.du_dx), dv_dy_est=_as_depth(est.dv_dy),
                div_est=_as_depth(est.div),
                du_dx_true=_as_depth(dudx_true), dv_dy_true=_as_depth(dvdy_true),
                div_true=_as_depth(dudx_true + dvdy_true)))
            out.to_netcdf(nc)
            rows += _metrics_rows(cfg, center_lat, nc)

    m = pd.DataFrame(rows).sort_values(
        ['family', 'quantity', 'width', 'center_lat']).reset_index(drop=True)
    m.to_csv(os.path.join(DATA_DIR, 'metrics.csv'), index=False)
    print(f'Wrote {len(m)} rows -> {os.path.join(DATA_DIR, "metrics.csv")}')


# --------------------------------------------------------------------------- plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

# four rows: mean (est vs true), signed bias (compensation), RMS/sigma, correlation
def _draw_grid(groups, legend_title, outfile):
    """groups: list of (df_subset, color, marker, label). Each df has one row per
    (quantity, diameter). Columns = du/dx, dv/dy, div; rows = the four metrics."""
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), sharex=True)
    for j, (q, qlab) in enumerate(QTYS):
        for df, color, marker, label in groups:
            d = df[df.quantity == q].copy()
            if d.empty:
                continue
            d['diameter'] = 2.0 * d['width']
            d = d.sort_values('diameter')
            x = d['diameter']
            mk = dict(color=color, marker=marker, ms=8, mec='k', mew=0.4, lw=2.0)
            # row 0: mean gradient, est (solid) vs true (dotted)
            axes[0, j].plot(x, d.w_model_mean * GRAD2DAY,
                            ls=':', lw=1.3, alpha=0.6, color=color)
            axes[0, j].plot(x, d.w_est_mean * GRAD2DAY, label=label, **mk)
            # row 1: signed mean bias = est - true (the compensation diagnostic)
            axes[1, j].plot(x, d.mean_bias * GRAD2DAY, label=label, **mk)
            # row 2: relative error RMS / sigma_true
            axes[2, j].plot(x, d.norm_rms, label=label, **mk)
            # row 3: correlation
            axes[3, j].plot(x, d['corr'], label=label, **mk)
        axes[0, j].set_title(qlab, fontsize=14, fontweight='bold')

    rowlab = [r'mean  (day$^{-1}$)', r'mean bias est$-$true  (day$^{-1}$)',
              r'relative error  RMS/$\sigma$', 'correlation  r']
    for i, lab in enumerate(rowlab):
        axes[i, 0].set_ylabel(lab, fontsize=12, fontweight='bold')
        for j in range(3):
            axes[i, j].grid(alpha=0.3)
            if i in (0, 1):
                axes[i, j].axhline(0, color='0.5', lw=0.8)
    for j in range(3):
        axes[-1, j].set_xlabel('array diameter (deg)', fontsize=12, fontweight='bold')
        axes[-1, j].set_xticks(sft.SYM_DIAMS)
    axes[0, 0].plot([], [], ls=':', color='0.4', lw=1.3, label='true (model)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, title=legend_title, loc='upper center',
               bbox_to_anchor=(0.5, 1.0), ncol=len(l), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', outfile)


def _scatter_deepest():
    """Time-mean est-vs-true at the deepest observed level, one point per config —
    rows = du/dx, dv/dy, div, w; cols = hexagon/diamond/square; hue = center latitude,
    shading = diameter. Rows 1-3 are the gradients (day^-1, data_grad_skill/); row 4 is
    the plane-fit w (m day^-1, from run_experiment_2's data/) — the same quantity as
    make_sym_w_scatter_deepest, stacked here for a direct comparison."""
    # (ylabel, data_dir, est_var, true_var, unit) per row
    rows = [(qlab, DATA_DIR, f'{q}_est', f'{q}_true', 'day$^{-1}$') for q, qlab in QTYS]
    rows.append(('$w$', os.path.join(HERE, 'data'), 'w_est', 'w_model', 'm day$^{-1}$'))
    centers = sorted(sft.SYMHEX_CENTERS)
    fig, axes = plt.subplots(len(rows), len(sft.SYM_FAMILIES),
                             figsize=(5.0 * len(sft.SYM_FAMILIES), 5.0 * len(rows)))
    zdeep = None
    for i, (ylab, ddir, evar, tvar, unit) in enumerate(rows):
        pts, vmin, vmax = {}, np.inf, -np.inf
        for fam in sft.SYM_FAMILIES:
            for c in centers:
                for d in sft.SYM_DIAMS:
                    cfg = f'{fam}_d{d}_c{c:+.1f}'
                    with xr.open_dataset(os.path.join(ddir, f'{cfg}__cell_{c:+.2f}.nc')) as ds:
                        kd = int(np.argmax(np.abs(ds.depth.values)))
                        zdeep = float(np.abs(ds.depth.values)[kd])
                        x = float(ds[tvar].isel(depth=kd).mean('time')) * GRAD2DAY
                        y = float(ds[evar].isel(depth=kd).mean('time')) * GRAD2DAY
                    pts[(fam, c, d)] = (x, y)
                    vmin, vmax = min(vmin, x, y), max(vmax, x, y)
        pad = 0.08 * (vmax - vmin)
        lo, hi = vmin - pad, vmax + pad                 # shared square limits per row
        for j, fam in enumerate(sft.SYM_FAMILIES):
            ax = axes[i, j]
            ax.plot([lo, hi], [lo, hi], color='0.4', lw=1.0, zorder=1)
            for c in centers:
                for d in sft.SYM_DIAMS:
                    x, y = pts[(fam, c, d)]
                    ax.scatter(x, y, s=95, marker='o',
                               color=sft._shade(sft.LAT_COLORS[c], sft.SYM_DIAM_FRAC[d]),
                               edgecolor='k', linewidths=0.6, zorder=3)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(sft.SYM_SHAPE[fam], fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{ylab}\nest ({unit})', fontsize=12, fontweight='bold')
            if i == len(rows) - 1:
                ax.set_xlabel(f'true ({unit})', fontsize=12, fontweight='bold')
    c_handles = [Line2D([], [], marker='o', ls='', color=sft.LAT_COLORS[c], ms=9,
                        label=f'center {c:+.1f}°') for c in centers]
    d_handles = [Line2D([], [], marker='o', ls='', color=sft._shade('#444444', sft.SYM_DIAM_FRAC[d]),
                        markeredgecolor='k', markeredgewidth=0.5, ms=9,
                        label=f'diameter {d:g}°') for d in sft.SYM_DIAMS]
    leg1 = fig.legend(handles=c_handles, loc='upper left', bbox_to_anchor=(0.10, 1.0),
                      ncol=len(centers), frameon=False, fontsize=10)
    fig.add_artist(leg1)
    fig.legend(handles=d_handles, loc='upper right', bbox_to_anchor=(0.92, 1.0),
               ncol=len(sft.SYM_DIAMS), frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, 'fig_sym_gradient_scatter_deepest.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out, f'(deepest ~{zdeep:.0f} m)')


def plot():
    os.makedirs(FIG_DIR, exist_ok=True)
    sft.apply_style()
    m = pd.read_csv(os.path.join(DATA_DIR, 'metrics.csv'))
    _scatter_deepest()

    # 1) shape comparison at each center latitude (the headline: is the hexagon best
    #    in du/dx AND dv/dy, or does it only win on the compensated divergence?)
    for c in sft.SYMHEX_CENTERS:
        groups = []
        for fam in sft.SYM_FAMILIES:
            d = m[(m.family == fam) & np.isclose(m.center_lat, c)]
            if not d.empty:
                groups.append((d, sft.EQS_SHAPE_COLOR[sft.SYM_SHAPE[fam]],
                               sft.SYM_MARKER[fam], sft.SYM_SHAPE[fam]))
        if groups:
            _draw_grid(groups,
                       f'regular shape (center {c:+.1f}°)',
                       os.path.join(FIG_DIR, f'fig_sym_shape_gradient_skill_c{c:+.1f}.png'))

    # 2) per-family vs diameter, one line per center latitude
    for fam in sft.SYM_FAMILIES:
        groups = []
        for c in sft.SYMHEX_CENTERS:
            d = m[(m.family == fam) & np.isclose(m.center_lat, c)]
            if not d.empty:
                groups.append((d, sft.LAT_COLORS[c], sft.SYM_MARKER[fam], f'{c:+.1f}°'))
        if groups:
            _draw_grid(groups,
                       f'{sft.SYM_SHAPE[fam]}: cell center latitude',
                       os.path.join(FIG_DIR, f'fig_{fam}_gradient_skill.png'))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('compute', 'all'):
        compute()
    if mode in ('plot', 'all'):
        plot()
