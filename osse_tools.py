"""
osse_tools.py — Wave glider array OSSE analysis.

Typical workflow:
    ds        = load_model(run_dir, iters)
    uv        = sample_uv(ds, positions, max_depth=70, dz_obs=2)
    w_result  = compute_w_planefit(uv)
    w_model   = sample_model_w(ds, positions, max_depth=70, dz_obs=2)
    fig       = plot_w_comparison(w_result['w_est'], w_model)
    fig2      = plot_velocity_map(ds, positions, max_depth=70)
"""

import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import cmocean.cm as cmo
from xmitgcm import open_mdsdataset


def load_positions(path):
    """
    Load glider positions from a JSON config file.

    Returns
    -------
    list of (lat, lon) tuples
    """
    with open(path) as f:
        cfg = json.load(f)
    return [tuple(p) for p in cfg['positions']]


def load_model(run_dir, iters, ref_date='2012-10-01', delta_t=300):
    """Open MITgcm diag_state diagnostics lazily, masking fill values."""
    ds = open_mdsdataset(
        data_dir=run_dir, grid_dir=run_dir,
        iters=iters, prefix=['diag_state'],
        ref_date=ref_date, delta_t=delta_t,
    )
    # xmitgcm returns big-endian coordinates; cast to native float so
    # pandas/scipy indexing works on little-endian systems
    for c in ('XC', 'YC', 'XG', 'YG', 'Z', 'Zl'):
        if c in ds.coords:
            ds[c] = ds[c].astype(float)
    return ds.where(ds != -999.0)


def sample_uv(ds, positions, max_depth=70, dz_obs=2):
    """
    Lazily interpolate UVEL and VVEL to each glider position at uniform obs depths.

    Parameters
    ----------
    ds : xr.Dataset
        From load_model.
    positions : list of (lat, lon)
        Glider positions in degrees.
    max_depth : float
        Maximum sampling depth in metres (positive). Default 70.
    dz_obs : float
        Depth interval between obs levels in metres. Default 2.

    Returns
    -------
    xr.Dataset with dims (time, glider, depth).
        Glider lat/lon stored as coordinates. depth coordinate holds layer midpoints
        (e.g. -1, -3, ..., -69 for max_depth=70, dz_obs=2).
    """
    n = int(max_depth / dz_obs)
    obs_z = -(np.arange(n) * dz_obs + dz_obs / 2)  # layer midpoints

    lats = [p[0] for p in positions]
    lons = [p[1] for p in positions]
    g = np.arange(len(positions))

    lat_da = xr.DataArray(lats, dims='glider', coords={'glider': g})
    lon_da = xr.DataArray(lons, dims='glider', coords={'glider': g})
    obs_z_da = xr.DataArray(obs_z, dims='obs_depth', coords={'obs_depth': obs_z})

    # UVEL on (YC, XG); VVEL on (YG, XC) — interpolate in staggered coords + depth
    U = ds.UVEL.interp(XG=lon_da, YC=lat_da, Z=obs_z_da).transpose('time', 'glider', 'obs_depth')
    V = ds.VVEL.interp(XC=lon_da, YG=lat_da, Z=obs_z_da).transpose('time', 'glider', 'obs_depth')

    return xr.Dataset({'U': U, 'V': V}).assign_coords(lat=lat_da, lon=lon_da)


def compute_w_planefit(uv_samples, remove_barotropic=False):
    """
    Estimate w via plane fit to U and V across the array, then integrate divergence.

    At each (time, depth): fits u = a + b*x + c*y and v = a + b*x + c*y over the
    glider positions to extract du/dx and dv/dy. Integrates div = du/dx + dv/dy
    downward from w=0 at the surface using the continuity equation:
        w(z_bottom) = w(z_top) + div * dz
    which follows from dw/dz = -div with z negative downward.

    Parameters
    ----------
    uv_samples : xr.Dataset
        From sample_uv, dims (time, glider, obs_depth).
    remove_barotropic : bool
        If True, subtract the depth-mean from U and V at each glider and timestep
        before the plane fit, so the result estimates the baroclinic w only.

    Returns
    -------
    xr.Dataset with:
        w_est : (time, depth)  estimated w at layer interfaces [m/s]
                               depth coordinate = [0, -dz, -2*dz, ..., -max_depth]
        div   : (time, obs_depth)  horizontal divergence at obs midpoints [1/s]
    """
    lats = uv_samples.lat.values
    lons = uv_samples.lon.values
    lat_c, lon_c = lats.mean(), lons.mean()

    deg_to_m = np.pi / 180 * 6371000.0
    x_m = (lons - lon_c) * np.cos(np.radians(lat_c)) * deg_to_m
    y_m = (lats - lat_c) * deg_to_m

    # Pseudoinverse of design matrix — computed once, applied to all (time, depth)
    A = np.column_stack([np.ones(len(lats)), x_m, y_m])  # (N, 3)
    Ainv = np.linalg.pinv(A)                              # (3, N)

    uv = uv_samples.compute()
    U = uv['U'].values  # (ntime, nglider, n_obs)
    V = uv['V'].values
    ntime, nglider, n_obs = U.shape

    if remove_barotropic:
        # Subtract depth-mean at each glider and timestep before fitting
        U = U - U.mean(axis=2, keepdims=True)
        V = V - V.mean(axis=2, keepdims=True)

    # Transpose to (nglider, ntime, n_obs) then flatten for a single matrix multiply
    cu = Ainv @ U.transpose(1, 0, 2).reshape(nglider, ntime * n_obs)  # (3, ntime*n_obs)
    cv = Ainv @ V.transpose(1, 0, 2).reshape(nglider, ntime * n_obs)

    du_dx = cu[1].reshape(ntime, n_obs)
    dv_dy = cv[2].reshape(ntime, n_obs)
    div_vals = du_dx + dv_dy

    obs_z = uv_samples.obs_depth.values      # midpoints, e.g. [-1, -3, ..., -69]
    dz_obs = float(abs(obs_z[1] - obs_z[0])) if n_obs > 1 else float(abs(obs_z[0]) * 2)
    w_z = -np.arange(n_obs + 1) * dz_obs    # interfaces: [0, -dz, ..., -max_depth]

    # Integrate downward: w(interface k+1) = w(interface k) + div(layer k) * dz
    w_vals = np.concatenate([
        np.zeros((ntime, 1)),
        np.cumsum(div_vals * dz_obs, axis=1)
    ], axis=1)  # (ntime, n_obs+1)

    time_coord = uv['U'].time
    # obs_depth (midpoints) and depth (interfaces) are kept as separate dims to
    # avoid xarray aligning them into a NaN-filled union when returned together
    div_da   = xr.DataArray(div_vals, dims=('time', 'obs_depth'),
                            coords={'time': time_coord, 'obs_depth': obs_z})
    w_est_da = xr.DataArray(w_vals,   dims=('time', 'depth'),
                            coords={'time': time_coord, 'depth': w_z})

    return xr.Dataset({'w_est': w_est_da, 'div': div_da})


def sample_model_w(ds, positions, max_depth=70, dz_obs=2, remove_barotropic=False):
    """
    Sample WVEL at the centroid of the glider array, interpolated to the same
    interface depths as compute_w_planefit.

    Parameters
    ----------
    ds : xr.Dataset
    positions : list of (lat, lon)
    max_depth : float
        Must match the value used in sample_uv. Default 70.
    dz_obs : float
        Must match the value used in sample_uv. Default 2.
    remove_barotropic : bool
        If True, subtract the depth-mean from the returned WVEL at each timestep,
        consistent with compute_w_planefit(remove_barotropic=True).

    Returns
    -------
    xr.DataArray, dims (time, depth)
        depth coordinate = [0, -dz_obs, ..., -max_depth]
    """
    n = int(max_depth / dz_obs)
    w_z = -np.arange(n + 1) * dz_obs  # interface depths matching compute_w_planefit

    lat_c = np.mean([p[0] for p in positions])
    lon_c = np.mean([p[1] for p in positions])
    w_z_da = xr.DataArray(w_z, dims='depth', coords={'depth': w_z})

    w = ds.WVEL.interp(XC=lon_c, YC=lat_c, Zl=w_z_da).compute()
    if remove_barotropic:
        # Remove the linear barotropic trend: the component that takes w from 0
        # at the surface to w(-H) at the bottom of the sampled layer.
        # This is consistent with compute_w_planefit(remove_barotropic=True),
        # which integrates div - <div>_z = d/dz[w - w(-H)/H * |z|].
        w_bottom = w.isel(depth=-1)          # WVEL at z = -max_depth, shape (time,)
        w = w + (w_bottom / max_depth) * w.depth   # w.depth is negative, so this subtracts
    return w


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


def plot_velocity_map(ds, positions, max_depth=70, time_range=None):
    """
    Three-panel map of depth- and time-averaged U, V, W with glider positions overlaid.

    Parameters
    ----------
    ds : xr.Dataset
        From load_model.
    positions : list of (lat, lon)
    max_depth : float
        Depth range to average over (surface to max_depth). Default 70.
    time_range : (t_start, t_end) as strings or datetimes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    ds_t = ds if time_range is None else ds.sel(time=slice(*time_range))

    # Depth/time mean — mask below max_depth, then average lazily before computing
    u_mean = ds_t.UVEL.where(ds_t.Z  >= -max_depth).mean(['time', 'Z']).compute()
    v_mean = ds_t.VVEL.where(ds_t.Z  >= -max_depth).mean(['time', 'Z']).compute()
    w_mean = ds_t.WVEL.where(ds_t.Zl >= -max_depth).mean(['time', 'Zl']).compute()

    glider_lats = [p[0] for p in positions]
    glider_lons = [p[1] for p in positions]
    buf = 0.25
    lon_lim = (min(glider_lons) - buf, max(glider_lons) + buf)
    lat_lim = (min(glider_lats) - buf, max(glider_lats) + buf)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        (u_mean, ds_t.XG.values, ds_t.YC.values, cmo.balance, 'Depth/time mean U  (m s⁻¹)'),
        (v_mean, ds_t.XC.values, ds_t.YG.values, cmo.balance, 'Depth/time mean V  (m s⁻¹)'),
        (w_mean, ds_t.XC.values, ds_t.YC.values, cmo.balance, 'Depth/time mean W  (m s⁻¹)'),
    ]

    for ax, (data, xx, yy, cmap, title) in zip(axes, panels):
        vmax = float(np.nanpercentile(np.abs(data.values), 98))
        im = ax.pcolormesh(xx, yy, data.values, cmap=cmap,
                           vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label='m s⁻¹')
        ax.scatter(glider_lons, glider_lats, c='k', s=40, zorder=5, marker='o')
        ax.set_xlim(*lon_lim)
        ax.set_ylim(*lat_lim)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_title(title)
        ax.axhline(0, color='k', lw=0.5, ls=':')

    return fig
