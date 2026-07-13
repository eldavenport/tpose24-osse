"""
plotting_tools.py — general OSSE diagnostic plotting (field / velocity / w
comparisons).
Also holds the per-config experiment-figure loops shared by the
make_experiment_figs notebooks. 
"""

import os
import re
import json
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import cmocean.cm as cmo

import osse_tools as ot
from osse_tools import dist_stats, _js_distance


def plot_flux_compare(array_flux, model_total):
    """
    Vertical eddy flux profiles vs depth: true total (model) and glider estimate.

    Each panel annotates the fraction of the depth-integrated flux the gliders recover,
    i.e. how much of the true vertical transport survives this sampling.
    """
    panels = [('wT', "w'T' (m s⁻¹ °C)"), ('wS', "w'S' (m s⁻¹ g/kg)"),
              ('wU', "w'u' (m² s⁻²)"),   ('wV', "w'v' (m² s⁻²)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, (k, lab) in zip(axes, panels):
        z = array_flux[k].obs_depth.values
        tot, est = model_total[k].values, array_flux[k].values
        itot, iest = np.trapz(tot, z), np.trapz(est, z)
        frac = iest / itot if itot != 0 else np.nan
        ax.plot(tot, z, color='0.4', lw=2.5, label='model total')
        ax.plot(est, z, 'C3-', lw=1.5, label='glider est')
        ax.axvline(0, color='k', lw=0.5, ls=':')
        ax.set_xlabel(lab); ax.set_title(f'recovered {frac:.0%}', fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('depth (m)'); axes[0].legend(fontsize=8)
    return fig


def plot_pdf_compare(obs, true, label='', units='', bins=60, ax=None):
    """Overlay normalised histograms of observed (gliders) and true (model) values."""
    o = np.asarray(obs).ravel(); o = o[np.isfinite(o)]
    t = np.asarray(true).ravel(); t = t[np.isfinite(t)]
    lo, hi = np.percentile(np.concatenate([o, t]), [0.5, 99.5])
    edges = np.linspace(lo, hi, bins + 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(t, edges, density=True, color='0.6', alpha=0.6, label='model')
    ax.hist(o, edges, density=True, histtype='step', color='C3', lw=1.8, label='gliders')
    s = dist_stats(o, t)
    ax.text(0.97, 0.97, f"JS={s['js']:.2f}\nKS={s['ks']:.2f}\nW={s['wasserstein']:.1e}",
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', fc='w', ec='0.7', alpha=0.85))
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(f'{label} ({units})' if units else label)
    ax.set_ylabel('pdf'); ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    return ax.figure


def plot_joint_compare(obs_x, obs_y, true_x, true_y, labels=('x', 'y'),
                       units=('', ''), bins=60, max_pts=30000):
    """
    Side-by-side scatter of model vs glider samples for a pair of quantities,
    each point coloured by depth.

    The annotated covariance is the eddy flux / Reynolds stress when the inputs are
    anomalies (e.g. (V', T') = meridional eddy heat flux, (U', V') = stress u'v').
    A Jensen-Shannon distance (0 identical, 1 disjoint) summarises how close the two
    joint distributions are. Inputs must carry an obs_depth coordinate.
    """
    rng = np.random.default_rng(0)

    def prep(a, b):
        depth = (-a.obs_depth).broadcast_like(a)
        a = np.asarray(a).ravel(); b = np.asarray(b).ravel(); d = np.asarray(depth).ravel()
        m = np.isfinite(a) & np.isfinite(b)
        return a[m], b[m], d[m]
    ox, oy, od = prep(obs_x, obs_y)
    tx, ty, td = prep(true_x, true_y)

    xlo, xhi = np.percentile(np.concatenate([ox, tx]), [0.5, 99.5])
    ylo, yhi = np.percentile(np.concatenate([oy, ty]), [0.5, 99.5])
    xe = np.linspace(xlo, xhi, bins + 1)
    ye = np.linspace(ylo, yhi, bins + 1)
    js = _js_distance(np.column_stack([tx, ty]), np.column_stack([ox, oy]), [xe, ye])
    vmax = max(od.max(), td.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, (X, Y, D, name) in zip(axes, [(tx, ty, td, 'model'), (ox, oy, od, 'gliders')]):
        cov, corr = np.cov(X, Y)[0, 1], np.corrcoef(X, Y)[0, 1]
        if X.size > max_pts:                 # thin dense populations for legibility
            i = rng.choice(X.size, max_pts, replace=False)
            X, Y, D = X[i], Y[i], D[i]
        sc = ax.scatter(X, Y, c=D, s=5, cmap=cmo.deep, vmin=0, vmax=vmax, alpha=0.5, lw=0)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)  # before 0-lines so they don't rescale
        ax.axhline(0, color='0.5', lw=0.5, ls=':'); ax.axvline(0, color='0.5', lw=0.5, ls=':')
        ax.set_title(f"{name}   cov={cov:.2e}  r={corr:.2f}", fontsize=9)
        ax.set_xlabel(f'{labels[0]} ({units[0]})' if units[0] else labels[0])
        plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, label='depth (m)')
    axes[1].text(0.97, 0.97, f'JS={js:.2f}', transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', fc='w', ec='0.7', alpha=0.85))
    axes[0].set_ylabel(f'{labels[1]} ({units[1]})' if units[1] else labels[1])
    return fig


def plot_field_pdfs(obs, true, vars=('T', 'S', 'U', 'V', 'sigma0'),
                    units=('°C', 'g/kg', 'm/s', 'm/s', 'kg/m³')):
    """Grid of 1-D PDF comparisons (model vs gliders) for each field."""
    ncol = 3
    nrow = int(np.ceil(len(vars) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, v, u in zip(axes, vars, units):
        plot_pdf_compare(obs[v], true[v], label=v, units=u, ax=ax)
    for ax in axes[len(vars):]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_w_comparison(w_est, w_model, depth_range=None, time_range=None, point_depth=-50):
    """
    Six-panel comparison of estimated and model w.

    Row 0: w_est Hovmöller | w_model Hovmöller | bias Hovmöller | depth profiles
    Row 1: depth-mean time series with ±σ shading (spans first three columns)
    Row 2: w and bias at point_depth vs time (spans first three columns)

    Parameters
    ----------
    w_est : xr.DataArray, dims (time, depth)
    w_model : xr.DataArray, dims (time, depth)
    depth_range : (z_shallow, z_deep) in model convention (e.g. (0, -50)), optional
    time_range : (t_start, t_end) as strings or datetimes, optional
    point_depth : float
        Depth for the bottom time series panel. Default -50.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if time_range is not None:
        w_est   = w_est.sel(time=slice(*time_range))
        w_model = w_model.sel(time=slice(*time_range))
    if depth_range is not None:
        w_est   = w_est.sel(depth=slice(*depth_range))
        w_model = w_model.sel(depth=slice(*depth_range))

    bias = w_est - w_model

    w_est_tmean   = w_est.mean('time')
    w_est_tstd    = w_est.std('time')
    w_model_tmean = w_model.mean('time')
    w_model_tstd  = w_model.std('time')
    bias_tmean    = bias.mean('time')
    bias_tstd     = bias.std('time')

    w_est_dm   = w_est.mean('depth')
    w_model_dm = w_model.mean('depth')
    bias_dm    = bias.mean('depth')

    actual_depth = float(w_est.depth.sel(depth=point_depth, method='nearest'))
    w_est_pt   = w_est.sel(depth=actual_depth, method='nearest')
    w_model_pt = w_model.sel(depth=actual_depth, method='nearest')
    bias_pt    = w_est_pt - w_model_pt

    T = w_est.time.values
    Z = w_est.depth.values

    vmax = float(np.nanpercentile(
        np.abs(np.concatenate([w_est.values.ravel(), w_model.values.ravel()])), 98
    ))
    vmax_bias = float(np.nanpercentile(np.abs(bias.values.ravel()), 98))

    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(
        3, 4,
        width_ratios=[3, 3, 3, 2],
        height_ratios=[3, 2, 2],
        hspace=0.45, wspace=0.32,
    )
    ax_h1    = fig.add_subplot(gs[0, 0])
    ax_h2    = fig.add_subplot(gs[0, 1], sharey=ax_h1)
    ax_h3    = fig.add_subplot(gs[0, 2], sharey=ax_h1)
    ax_prof  = fig.add_subplot(gs[0, 3], sharey=ax_h1)
    ax_ts    = fig.add_subplot(gs[1, :3])
    ax_ts2   = fig.add_subplot(gs[2, :3], sharex=ax_ts)
    for ax in (fig.add_subplot(gs[1, 3]), fig.add_subplot(gs[2, 3])):
        ax.axis('off')

    def _hovm(ax, data, cmap, vmax, title):
        im = ax.pcolormesh(T, Z, data.values.T, cmap=cmap,
                           vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_title(title)
        ax.set_ylabel('Depth (m)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label='m s⁻¹')

    _hovm(ax_h1, w_est,   cmo.balance, vmax,      'w estimated')
    _hovm(ax_h2, w_model, cmo.balance, vmax,      'w model')
    _hovm(ax_h3, bias,    cmo.balance, vmax_bias, 'bias (est − model)')
    plt.setp(ax_h2.get_yticklabels(), visible=False)
    plt.setp(ax_h3.get_yticklabels(), visible=False)

    for data_m, data_s, color, label in [
        (w_est_tmean,   w_est_tstd,   'C0', 'est'),
        (w_model_tmean, w_model_tstd, 'C1', 'model'),
        (bias_tmean,    bias_tstd,    'C2', 'bias'),
    ]:
        ax_prof.plot(data_m.values, Z, color=color, lw=1.5, label=label)
        ax_prof.fill_betweenx(Z, (data_m - data_s).values, (data_m + data_s).values,
                              color=color, alpha=0.2)
    ax_prof.axvline(0, color='k', lw=0.7, ls=':')
    ax_prof.set_xlabel('w (m s⁻¹)')
    ax_prof.set_title('Time mean ± σ\nvs depth')
    ax_prof.legend(fontsize=8)
    ax_prof.grid(alpha=0.3)

    for data_m, data_s, color, label in [
        (w_est_dm,   w_est.std('depth'),   'C0', 'est'),
        (w_model_dm, w_model.std('depth'), 'C1', 'model'),
        (bias_dm,    bias.std('depth'),    'C2', 'bias'),
    ]:
        ax_ts.plot(T, data_m.values, color=color, lw=1, label=label)
        ax_ts.fill_between(T, (data_m - data_s).values, (data_m + data_s).values,
                           color=color, alpha=0.15)
    ax_ts.axhline(0, color='k', lw=0.5, ls=':')
    ax_ts.set_ylabel('Depth-mean w (m s⁻¹)')
    ax_ts.set_title('Depth-mean w and bias vs time  (shading = ±σ over depth)')
    ax_ts.legend(fontsize=9)
    ax_ts.grid(alpha=0.3)
    plt.setp(ax_ts.get_xticklabels(), visible=False)

    for data, color, label in [
        (w_est_pt,   'C0', 'est'),
        (w_model_pt, 'C1', 'model'),
        (bias_pt,    'C2', 'bias'),
    ]:
        ax_ts2.plot(T, data.values, color=color, lw=1, label=label)
    ax_ts2.axhline(0, color='k', lw=0.5, ls=':')
    ax_ts2.set_ylabel(f'w at {abs(actual_depth):.0f} m (m s⁻¹)')
    ax_ts2.set_title(f'w and bias at {abs(actual_depth):.0f} m depth vs time')
    ax_ts2.legend(fontsize=9)
    ax_ts2.grid(alpha=0.3)
    ax_ts2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_ts2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_ts2.xaxis.get_majorticklabels(), rotation=30, ha='right')

    return fig


def compute_velocity_means(ds, max_depth=70, time_range=None, level_depth=None):
    """Compute the three U, V, W mean fields for a velocity map.

    These depend only on the model and the averaging choice — NOT on the array
    configuration (which only sets the marker overlay and zoom) — so compute this
    once and reuse it across every config's `plot_velocity_map` call.

    If `level_depth` is None, each field is the depth- AND time-mean over surface
    to `max_depth`. Otherwise each field is the TIME mean at the single model level
    nearest to `level_depth` (e.g. 70 for the bottom of the sampling range).

    Returns a list of 3 panel tuples ``(data, xx, yy, title)`` (U, V, W).
    """
    ds_t = ds if time_range is None else ds.sel(time=slice(*time_range))

    if level_depth is None:
        # Depth/time mean — mask below max_depth, then average lazily before computing
        u_mean = ds_t.UVEL.where(ds_t.Z  >= -max_depth).mean(['time', 'Z']).compute()
        v_mean = ds_t.VVEL.where(ds_t.Z  >= -max_depth).mean(['time', 'Z']).compute()
        w_mean = ds_t.WVEL.where(ds_t.Zl >= -max_depth).mean(['time', 'Zl']).compute()
        u_ttl, v_ttl, w_ttl = (f'Depth/time mean {c}  (m s⁻¹)' for c in 'UVW')
    else:
        # Time mean at the single model level nearest to level_depth (U/V on Z, W on Zl)
        u_sel = ds_t.UVEL.sel(Z=-level_depth, method='nearest')
        v_sel = ds_t.VVEL.sel(Z=-level_depth, method='nearest')
        w_sel = ds_t.WVEL.sel(Zl=-level_depth, method='nearest')
        u_mean = u_sel.mean('time').compute()
        v_mean = v_sel.mean('time').compute()
        w_mean = w_sel.mean('time').compute()
        zc, zl = -float(u_sel.Z.values), -float(w_sel.Zl.values)
        u_ttl = f'Time mean U at {zc:.0f} m  (m s⁻¹)'
        v_ttl = f'Time mean V at {zc:.0f} m  (m s⁻¹)'
        w_ttl = f'Time mean W at {zl:.0f} m  (m s⁻¹)'

    return [
        (u_mean, ds_t.XG.values, ds_t.YC.values, u_ttl),
        (v_mean, ds_t.XC.values, ds_t.YG.values, v_ttl),
        (w_mean, ds_t.XC.values, ds_t.YC.values, w_ttl),
    ]


def plot_velocity_map(ds, positions, max_depth=70, time_range=None, cells=None,
                      level_depth=None, means=None):
    """
    Three-panel map of U, V, W with glider positions overlaid.

    Parameters
    ----------
    ds : xr.Dataset
        From load_model. Only used to compute the mean fields; may be None if
        `means` is supplied.
    positions : list of (lat, lon)
        Used only to set the map's zoom extent; ignored for markers if cells is given.
    max_depth : float
        Depth range to average over (surface to max_depth). Default 70.
    time_range : (t_start, t_end) as strings or datetimes, optional
    cells : list of (label, positions, color), optional
        Overlay each cell in its own color: the cell outline (convex hull of its
        positions) is drawn as a colored line and its centre (centroid of its
        positions) as a colored dot. Gliders (positions offset from the mooring
        meridian) are black stars and moorings (positions on the meridian) black
        dots, drawn once regardless of how many cells share them.
    level_depth : float, optional
        If None (default), each panel is the depth- AND time-mean over surface to
        max_depth. If set, each panel is the TIME mean at the single model level
        nearest to `level_depth` (e.g. 70 for the bottom of the sampling range).
    means : list of (data, xx, yy, title), optional
        Precomputed mean fields from `compute_velocity_means`. When looping over
        many configs, compute these once and pass them in — the fields are the same
        for every config, so this skips the expensive re-read/re-reduce each time.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if means is None:
        means = compute_velocity_means(ds, max_depth=max_depth,
                                       time_range=time_range, level_depth=level_depth)

    glider_lats = [p[0] for p in positions]
    glider_lons = [p[1] for p in positions]
    buf = 0.25
    lon_lim = (min(glider_lons) - buf, max(glider_lons) + buf)
    lat_lim = (min(glider_lats) - buf, max(glider_lats) + buf)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panels = [(data, xx, yy, cmo.balance, title) for data, xx, yy, title in means]

    if cells is not None:
        from scipy.spatial import ConvexHull
        # Moorings sit on the array's central meridian (the mean longitude, about
        # which each array is symmetric); gliders are offset in longitude from it.
        # Dedup across cells that share points.
        all_pos = sorted({p for _, pos, _ in cells for p in pos})
        mooring_lon = float(np.mean([p[1] for p in all_pos]))
        moorings = [p for p in all_pos if abs(p[1] - mooring_lon) < 1e-6]
        gliders_ = [p for p in all_pos if abs(p[1] - mooring_lon) >= 1e-6]

    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize
    for ax, (data, xx, yy, cmap, title) in zip(axes, panels):
        vmax = float(np.nanpercentile(np.abs(data.values), 98))
        levels = np.linspace(-vmax, vmax, 100)   # 100 filled levels, symmetric about 0
        ax.contourf(xx, yy, data.values, levels=levels, cmap=cmap, extend='both')
        # Build the colorbar from a continuous Normalize (not the contourf's discrete
        # boundaries) so evenly-spaced tick values plot at evenly-spaced positions —
        # a contourf colorbar snaps ticks to level edges and looks irregular.
        sm = mcm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap=cmap)
        plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, label='m s⁻¹', extend='both',
                     ticks=mticker.MaxNLocator(nbins=5, symmetric=True))
        # clean, evenly spaced lon/lat ticks (contourf leaves the raw grid otherwise).
        # Few x ticks + 1-decimal labels so narrow-extent configs don't overlap.
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        if cells is None:
            ax.scatter(glider_lons, glider_lats, c='k', s=40, zorder=5, marker='o')
        else:
            for label, pos, color in cells:
                pts = np.array([(p[1], p[0]) for p in pos])  # (lon, lat)
                if len(pts) >= 3:  # outline = convex hull boundary
                    verts = ConvexHull(pts).vertices
                    ring = np.append(verts, verts[0])
                    ax.plot(pts[ring, 0], pts[ring, 1], color=color, lw=1.5, zorder=4)
                # cell centre = centroid of its positions
                ax.scatter(pts[:, 0].mean(), pts[:, 1].mean(), c=[color], s=70,
                           marker='o', zorder=6, edgecolor='k', linewidth=0.5)
            # gliders as black stars, moorings as black dots (once, all cells)
            if gliders_:
                ax.scatter([p[1] for p in gliders_], [p[0] for p in gliders_],
                           c='k', s=90, marker='*', zorder=5)
            if moorings:
                ax.scatter([p[1] for p in moorings], [p[0] for p in moorings],
                           c='k', s=30, marker='o', zorder=5)
        ax.set_xlim(*lon_lim)
        ax.set_ylim(*lat_lim)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_title(title)
        ax.axhline(0, color='k', lw=0.5, ls=':')

    if cells is not None:
        handles = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k',
                       markeredgecolor='k', markersize=12, label='glider'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
                       markeredgecolor='k', markersize=7, label='mooring'),
        ]
        handles += [plt.Line2D([0], [0], marker='o', color=c, markerfacecolor=c,
                              markeredgecolor='k', markersize=9, label=f'cell {l}')
                    for l, _, c in cells]
        axes[0].legend(handles=handles, fontsize=8, loc='upper right')

    return fig


# --- experiment-figure loops (shared by make_experiment_figs notebooks) ------

def render_w_comparisons(m, here, efig, point_depth=-50):
    """Per-cell w comparison (Hovmoller + time series) from the saved arrays; no
    model load. Skips any figure already on disk. Returns the number written."""
    n_new = 0
    for _, r in m.iterrows():
        outdir = os.path.join(efig, r.config)
        outpath = os.path.join(outdir, f"w_comparison_cell_{r.center_lat:+.2f}.png")
        if os.path.exists(outpath):
            continue
        ds = xr.open_dataset(os.path.join(here, r.nc_path))
        fig = plot_w_comparison(ds.w_est, ds.w_model, point_depth=point_depth)
        fig.suptitle(f"{r.config}  |  cell {r.center_lat:+.1f}N  |  "
                     f"RMS/sigma={r.norm_rms:.2f}  r={r['corr']:.2f}", y=1.01, fontsize=12)
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(outpath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        n_new += 1
    return n_new


def render_velocity_maps(here, efig, run_dir, iters, spinup_end="2012-10-11", max_depth=70,
                         delta_t=300):
    """Per-config velocity/vorticity context maps. Writes two figures per config:
    `velocity_map.png` (depth/time mean over surface->max_depth) and
    `velocity_map_{max_depth}m.png` (time mean at the bottom sampling level).
    Opens the model only if at least one config still lacks a figure. Returns the
    number of configs (re)rendered. `delta_t` is the model timestep (s) used to map
    iters -> time (60 for the dt60 runs, 300 for the standard runs)."""
    bottom_name = f"velocity_map_{max_depth:g}m.png"
    cfg_paths = sorted(glob.glob(os.path.join(here, "configs", "**", "*.json"), recursive=True))
    def missing(p):
        d = os.path.join(efig, json.load(open(p))["name"])
        return not (os.path.exists(os.path.join(d, "velocity_map.png"))
                    and os.path.exists(os.path.join(d, bottom_name)))
    pending = [p for p in cfg_paths if missing(p)]
    print(f"{len(pending)} configs need velocity maps "
          f"({len(cfg_paths) - len(pending)} already complete)")
    if not pending:
        return 0
    ds_model = ot.load_model(run_dir, iters, delta_t=delta_t).sel(time=slice(spinup_end, None))
    # The mean fields are identical for every config (config only sets the overlay
    # and zoom), so compute each set ONCE up front and reuse it across all configs.
    print("computing depth/time-mean and bottom-level mean fields (once)...")
    depth_means = compute_velocity_means(ds_model, max_depth=max_depth)
    bottom_means = compute_velocity_means(ds_model, max_depth=max_depth, level_depth=max_depth)
    for path in pending:
        cfg = json.load(open(path))
        cells = ot.load_cells(path)
        positions = sorted({p for _, pos in cells for p in pos})
        cells_plot = [(f"{cl:+.1f}", pos, f"C{i}") for i, (cl, pos) in enumerate(cells)]
        outdir = os.path.join(efig, cfg["name"]); os.makedirs(outdir, exist_ok=True)
        # depth/time-mean map
        fig = plot_velocity_map(None, positions, cells=cells_plot, means=depth_means)
        fig.suptitle(f"{cfg['name']}  ({cfg['description']})", fontsize=11, y=1.02)
        fig.savefig(os.path.join(outdir, "velocity_map.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        # time-mean at the bottom sampling level
        fig = plot_velocity_map(None, positions, cells=cells_plot, means=bottom_means)
        fig.suptitle(f"{cfg['name']}  ({cfg['description']})", fontsize=11, y=1.02)
        fig.savefig(os.path.join(outdir, bottom_name), dpi=130, bbox_inches="tight")
        plt.close(fig)
    return len(pending)



N_CONTOUR_LEVELS = 100


def _group_limits(panels, groups, diverging, pct):
    """
    Per-panel (vmin, vmax) with panels sharing a `groups` label sharing color limits.

    Limits come from the pooled finite values of every panel in the group: symmetric
    ±percentile for diverging fields, else the (low, high) percentiles. This lets e.g.
    the U and V gradient panels share one scale so they're directly comparable.
    """
    lims = {}
    for g in dict.fromkeys(groups):
        pooled = np.concatenate([np.asarray(v).ravel()
                                 for (v, *_), gg in zip(panels, groups) if gg == g])
        pooled = pooled[np.isfinite(pooled)]
        if diverging:
            m = float(np.nanpercentile(np.abs(pooled), pct[1]))
            lims[g] = (-m, m)
        else:
            lims[g] = (float(np.nanpercentile(pooled, pct[0])),
                       float(np.nanpercentile(pooled, pct[1])))
    return [lims[g] for g in groups]


def plot_domain_grid(panels, cbar_label, cmap, suptitle, fname,
                     diverging=False, ncols=3, pct=(2, 98), vlim=None, groups=None):
    """
    Grid of filled-contour domain maps, styled like the demo_domain mean-velocity figs.

    Rendered with contourf (N_CONTOUR_LEVELS filled levels) and clean, auto-located
    colorbar ticks.

    Parameters
    ----------
    panels : list of (values2d, x, y, title)
        Each panel's 2-D field with its lon (x) and lat (y) coordinates.
    cbar_label, cmap, suptitle, fname : str / colormap
    diverging : bool
        If True, symmetric limits +/- the `pct[1]`-th percentile of |values| with a
        zero-centred cmap; otherwise sequential limits at the `pct` percentiles.
    ncols : int
        Panels per row (extra axes are hidden).
    vlim : (vmin, vmax) or None
        One fixed color limit for every panel (e.g. (-1, 1) for a correlation map);
        overrides percentile scaling and `groups`.
    groups : list of hashable or None
        Same length as `panels`; panels sharing a label share color limits (pooled
        percentiles). Overrides per-panel scaling. Ignored when `vlim` is given.
    """
    import matplotlib.ticker as mticker
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.7 * ncols, 4.8 * nrows),
                             squeeze=False)
    axf = axes.ravel()

    if vlim is not None:
        panel_lims = [vlim] * n
    elif groups is not None:
        panel_lims = _group_limits(panels, groups, diverging, pct)
    else:
        panel_lims = None  # per-panel, computed below

    cbars = []
    for i, (ax, (vals, xx, yy, title)) in enumerate(zip(axf, panels)):
        vals = np.asarray(vals)
        if panel_lims is not None:
            vmin, vmax = panel_lims[i]
        elif diverging:
            vmax = float(np.nanpercentile(np.abs(vals), pct[1])); vmin = -vmax
        else:
            vmin = float(np.nanpercentile(vals, pct[0]))
            vmax = float(np.nanpercentile(vals, pct[1]))
        levels = np.linspace(vmin, vmax, N_CONTOUR_LEVELS)
        ax.contourf(xx, yy, vals, levels=levels, cmap=cmap, extend='both')
        # Build the colorbar from a continuous Normalize (not the contourf's discrete
        # boundaries) so evenly-spaced tick values plot at evenly-spaced positions —
        # a contourf colorbar snaps ticks to level edges and looks irregular.
        sm = mcm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.03, label=cbar_label,
                            extend='both',
                            ticks=mticker.MaxNLocator(nbins=6, symmetric=diverging))
        cbar.ax.tick_params(labelsize=9)
        cbars.append(cbar)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_title(title)
    for ax in axf[n:]:
        ax.axis('off')
    # Fold each colorbar's scientific-notation scale (e.g. "1e-6") into its label so
    # the floating offset text at the top of the bar can't overlap the map above it.
    fig.canvas.draw()
    for cbar in cbars:
        off = cbar.ax.yaxis.get_major_formatter().get_offset()
        if off:
            cbar.ax.yaxis.get_offset_text().set_visible(False)
            off = off.replace('−', '-')          # matplotlib uses a unicode minus
            m = re.fullmatch(r'1e([+-]?\d+)', off.strip())
            scale = rf'$\times 10^{{{int(m.group(1))}}}$' if m else f'×{off}'
            cbar.set_label(f'{cbar_label}  ({scale})')
    # reserve a constant absolute headroom for the suptitle regardless of nrows
    h = 4.8 * nrows
    if suptitle:
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.45 / h])
        fig.suptitle(suptitle, fontsize=13, y=1 - 0.10 / h)
    else:
        fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def plot_footprint_profiles(profiles, coord, rows, widths, shapes, colors,
                            xlabel, legend_title, fname, ref_x=None):
    """
    Line-profile summary of footprint w-error, faceted by (statistic, height) rows ×
    width cols, one colored line per shape -- so the lowest line at any point on the
    axis is the best shape there. The reducing axis is baked into `profiles` (the
    median/mean |w-error| over the other axis); this just plots vs `coord`.

    profiles : dict (stat, shape, height, width) -> 1-D array aligned to `coord`.
    coord    : 1-D array for the x-axis (latitude or longitude, deg).
    rows     : list of (stat, height) -- one plot row each (e.g. median h1, median h2,
               mean h1, mean h2).
    shapes   : ordered shape labels (lines).  colors : dict label -> color.
    ref_x    : optional x for a vertical guide line.

    Follows the repo summary-figure style: figure-level top legend, no suptitle,
    bold labels, shared y-scale.
    """
    nrows, ncols = len(rows), len(widths)
    ymax = np.nanpercentile(
        np.concatenate([np.abs(p).ravel() for p in profiles.values()]), 99)
    with plt.rc_context({'axes.labelsize': 12.5, 'axes.labelweight': 'bold',
                         'legend.fontsize': 12, 'legend.title_fontsize': 14}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.1 * nrows),
                                 squeeze=False, sharex=True, sharey=True)
        for r, (stat, h) in enumerate(rows):
            for c, w in enumerate(widths):
                ax = axes[r][c]
                for s in shapes:
                    ax.plot(coord, profiles[(stat, s, h, w)], color=colors[s],
                            lw=1.8, label=s)
                if ref_x is not None:
                    ax.axvline(ref_x, color='k', lw=0.6, ls=':')
                ax.set_ylim(0, ymax)
                ax.grid(True, lw=0.3, alpha=0.5)
                ax.set_title(f'{stat} · {w:g}° × {h:g}°  (W×H)')
                if r == nrows - 1:
                    ax.set_xlabel(xlabel)
                if c == 0:
                    ax.set_ylabel(f'{stat} |w error| (m day$^{{-1}}$)')
        handles = [plt.Line2D([], [], color=colors[s], lw=2.4) for s in shapes]
        fig.legend(handles, list(shapes), title=legend_title, ncol=len(shapes),
                   loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.35 / nrows])
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fname


def plot_point_autocorr(per_var, suptitle, fname, xmax_deg=None):
    """
    Spatial autocorrelation function anchored at a single grid point, one panel per
    variable: zonal (solid) and meridional (dashed) correlation vs separation.

    r=0 is marked so any negative lobe is readable; no 1/e reference/crossing markers --
    the curve shape is the point.

    per_var : list of (var_title, dict) from osse_tools.mean_field_point_autocorr:
              sep_x_deg, r_x, sep_y_deg, r_y.
    """
    from matplotlib.lines import Line2D
    n = len(per_var)
    # shared, data-driven lower limit so a deep negative lobe is never clipped
    ymin = min(float(np.nanmin(np.concatenate([p['r_x'], p['r_y']])))
               for _, p in per_var)
    ylo = max(-1.02, min(-0.3, ymin - 0.08))
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False, sharey=True)
    for ax, (title, pac) in zip(axes.ravel(), per_var):
        ax.plot(pac['sep_x_deg'], pac['r_x'], '-', color='#1b6ca8', lw=1.9)
        ax.plot(pac['sep_y_deg'], pac['r_y'], '--', color='#c0392b', lw=1.9)
        ax.axhline(0, color='k', lw=0.7)
        xm = xmax_deg or max(pac['sep_x_deg'][-1], pac['sep_y_deg'][-1])
        ax.set_xlim(0, xm)
        ax.set_ylim(ylo, 1.02)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_xlabel('separation (°)')
        ax.set_ylabel('autocorrelation')
        ax.set_title(title)
    handles = [Line2D([0], [0], color='#1b6ca8', lw=1.9, ls='-', label='zonal'),
               Line2D([0], [0], color='#c0392b', lw=1.9, ls='--', label='meridional')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(suptitle, fontsize=13, y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def plot_anchor_corr_map(panels, anchor_lonlat, suptitle, fname, ncols=3,
                         ref_levels=(1.0 / np.e, 0.0)):
    """
    Grid of anchored 2-D spatial-correlation maps, one panel per variable.

    Each panel is a filled correlation map on fixed diverging limits (-1..1, zero-centred
    cmo.balance) so panels are directly comparable; the anchor is marked with a star and
    `ref_levels` are drawn as labelled contours (r=0 dotted, positive levels solid) so the
    coherent footprint -- and whether an array fits inside it -- is readable. Used for both
    the temporal one-point map (osse_tools.point_corr_map) and the mean-field 2-D pattern
    (mean_field_point_corr_map).

    panels : list of (r2d, lon, lat, title) with lon/lat the map coords of the pattern.
    anchor_lonlat : (lon0, lat0) marked on every panel.
    ref_levels : one sequence of contour levels shared by all panels, OR a list of such
        sequences (one per panel) for per-panel control (e.g. drop the 0 contour on W).
    """
    from matplotlib.colors import Normalize
    import matplotlib.cm as mcm
    lon0, lat0 = anchor_lonlat
    n = len(panels)
    # broadcast a shared level sequence, or take one sequence per panel
    if not ref_levels:
        per_panel = [()] * n
    elif np.isscalar(ref_levels[0]):
        per_panel = [tuple(ref_levels)] * n
    else:
        per_panel = [tuple(lv) for lv in ref_levels]
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.7 * ncols, 4.8 * nrows),
                             squeeze=False)
    axf = axes.ravel()
    levels = np.linspace(-1, 1, N_CONTOUR_LEVELS)
    for ax, (r, xx, yy, title), lv in zip(axf, panels, per_panel):
        ax.contourf(xx, yy, r, levels=levels, cmap=cmo.balance, extend='neither')
        if lv:
            slv = sorted(lv)
            ax.contour(xx, yy, r, levels=slv, colors='k', linewidths=0.8,
                       linestyles=['-' if v != 0 else ':' for v in slv])
        ax.plot(lon0, lat0, marker='*', ms=15, mfc='yellow', mec='k', mew=0.8)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        sm = mcm.ScalarMappable(norm=Normalize(-1, 1), cmap=cmo.balance)
        plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.03, label='correlation',
                     ticks=mticker.MaxNLocator(nbins=6, symmetric=True))
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_title(title)
    for ax in axf[n:]:
        ax.axis('off')
    h = 4.8 * nrows
    if suptitle:
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.45 / h])
        fig.suptitle(suptitle, fontsize=13, y=1 - 0.10 / h)
    else:
        fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname
