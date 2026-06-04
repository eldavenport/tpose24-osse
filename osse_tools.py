"""
osse_tools.py — AUV array OSSE analysis.

workflow:
    ds        = load_model(run_dir, iters)
    uv        = sample_uv(ds, positions)
    w_result  = compute_w_planefit(uv, ds)
    w_model   = sample_model_w(ds, positions)
    fig       = plot_w_comparison(w_result['w_est'], w_model)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import cmocean.cm as cmo
from xmitgcm import open_mdsdataset


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


def sample_uv(ds, positions):
    """
    Interpolate UVEL and VVEL to each glider position.

    Parameters
    ----------
    ds : xr.Dataset
        From load_model.
    positions : list of (lat, lon)
        Glider positions in degrees.

    Returns
    -------
    xr.Dataset with dims (time, glider, Z).
        Glider lat and lon stored as coordinates on the glider dimension.
    """
    lats = [p[0] for p in positions]
    lons = [p[1] for p in positions]
    g = np.arange(len(positions))

    lat_da = xr.DataArray(lats, dims='glider', coords={'glider': g})
    lon_da = xr.DataArray(lons, dims='glider', coords={'glider': g})

    # UVEL lives on (YC, XG); VVEL on (YG, XC) — interpolate in native staggered coords
    U = ds.UVEL.interp(XG=lon_da, YC=lat_da).transpose('time', 'glider', 'Z')
    V = ds.VVEL.interp(XC=lon_da, YG=lat_da).transpose('time', 'glider', 'Z')

    return xr.Dataset({'U': U, 'V': V}).assign_coords(lat=lat_da, lon=lon_da)


def compute_w_planefit(uv_samples, ds):
    """
    Estimate W via plane fit to U and V across the array, then integrate divergence.

    At each (time, depth): fits u = a + b*x + c*y and v = a + b*x + c*y over the
    glider positions to extract du/dx and dv/dy. Integrates div = du/dx + dv/dy
    downward from w=0 at the surface.

    Parameters
    ----------
    uv_samples : xr.Dataset
        From sample_uv, dims (time, glider, Z).
    ds : xr.Dataset
        Model dataset, provides drF (layer thicknesses) and Zl coordinates.

    Returns
    -------
    xr.Dataset with:
        w_est : (time, Zl)  estimated vertical velocity [m/s]
        div   : (time, Z)   horizontal divergence [1/s]
    """
    lats = uv_samples.lat.values
    lons = uv_samples.lon.values
    lat_c = lats.mean()
    lon_c = lons.mean()

    deg_to_m = np.pi / 180 * 6371000.0
    x_m = (lons - lon_c) * np.cos(np.radians(lat_c)) * deg_to_m
    y_m = (lats - lat_c) * deg_to_m

    # Pseudoinverse of design matrix — computed once, applied to all (time, Z)
    A = np.column_stack([np.ones(len(lats)), x_m, y_m])  # (N, 3)
    Ainv = np.linalg.pinv(A)                              # (3, N)

    uv = uv_samples.compute()
    U = uv['U'].values  # (ntime, nglider, nz)
    V = uv['V'].values
    ntime, nglider, nz = U.shape

    # Transpose to (nglider, ntime, nz) so the glider axis aligns with Ainv columns,
    # then flatten the last two axes for a single matrix multiply
    cu = Ainv @ U.transpose(1, 0, 2).reshape(nglider, ntime * nz)  # (3, ntime*nz)
    cv = Ainv @ V.transpose(1, 0, 2).reshape(nglider, ntime * nz)

    du_dx = cu[1].reshape(ntime, nz)  # coefficient of x in U fit
    dv_dy = cv[2].reshape(ntime, nz)  # coefficient of y in V fit
    div_vals = du_dx + dv_dy

    dz = ds.drF.values  # (nz,)

    # w at Zl[k] = -sum_{j<k} div[j]*drF[j], w at surface = 0
    w_vals = np.concatenate([
        np.zeros((ntime, 1)),
        -np.cumsum(div_vals * dz[np.newaxis, :], axis=1)
    ], axis=1)[:, :nz]   # Zl has nz levels (top faces); drop the extra bottom face

    time_coord = uv['U'].time
    Z_coord    = uv['U'].Z
    Zl_coord   = ds.Zl

    div_da  = xr.DataArray(div_vals, dims=('time', 'Z'),
                           coords={'time': time_coord, 'Z': Z_coord})
    w_est_da = xr.DataArray(w_vals, dims=('time', 'Zl'),
                            coords={'time': time_coord, 'Zl': Zl_coord})

    return xr.Dataset({'w_est': w_est_da, 'div': div_da})


def sample_model_w(ds, positions):
    """
    Sample WVEL at the centroid of the array.

    Parameters
    ----------
    ds : xr.Dataset
    positions : list of (lat, lon)

    Returns
    -------
    xr.DataArray, dims (time, Zl)
    """
    lat_c = np.mean([p[0] for p in positions])
    lon_c = np.mean([p[1] for p in positions])
    return ds.WVEL.interp(XC=lon_c, YC=lat_c).compute()


def plot_w_comparison(w_est, w_model, depth_range=None, time_range=None):
    """
    Five-panel comparison of estimated and model w.

    Top row: w_est Hovmöller | w_model Hovmöller | bias Hovmöller | depth profiles
    Bottom row: depth-mean time series (spans first three columns)

    Parameters
    ----------
    w_est : xr.DataArray, dims (time, Zl)
    w_model : xr.DataArray, dims (time, Zl)
    depth_range : (z_shallow, z_deep) in model convention (e.g. (0, -200)), optional
    time_range : (t_start, t_end) as strings or datetimes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if time_range is not None:
        w_est   = w_est.sel(time=slice(*time_range))
        w_model = w_model.sel(time=slice(*time_range))
    if depth_range is not None:
        w_est   = w_est.sel(Zl=slice(*depth_range))
        w_model = w_model.sel(Zl=slice(*depth_range))

    bias = w_est - w_model

    # Time-mean statistics for profile panel
    w_est_tmean  = w_est.mean('time')
    w_est_tstd   = w_est.std('time')
    w_model_tmean = w_model.mean('time')
    w_model_tstd  = w_model.std('time')
    bias_tmean   = bias.mean('time')
    bias_tstd    = bias.std('time')

    # Depth-mean time series
    w_est_dm   = w_est.mean('Zl')
    w_model_dm = w_model.mean('Zl')
    bias_dm    = bias.mean('Zl')
    bias_dm_std = bias.std('Zl')   # spread across depth at each timestep

    T = w_est.time.values
    Z = w_est.Zl.values

    vmax = float(np.nanpercentile(
        np.abs(np.concatenate([w_est.values.ravel(), w_model.values.ravel()])), 98
    ))
    vmax_bias = float(np.nanpercentile(np.abs(bias.values.ravel()), 98))

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(
        2, 4,
        width_ratios=[3, 3, 3, 2],
        height_ratios=[3, 2],
        hspace=0.38, wspace=0.32,
    )
    ax_h1   = fig.add_subplot(gs[0, 0])
    ax_h2   = fig.add_subplot(gs[0, 1], sharey=ax_h1)
    ax_h3   = fig.add_subplot(gs[0, 2], sharey=ax_h1)
    ax_prof = fig.add_subplot(gs[0, 3], sharey=ax_h1)
    ax_ts   = fig.add_subplot(gs[1, :3])
    ax_stats = fig.add_subplot(gs[1, 3])
    ax_stats.axis('off')

    def _hovm(ax, data, cmap, vmax, title):
        im = ax.pcolormesh(T, Z, data.values.T, cmap=cmap,
                           vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_title(title)
        ax.set_ylabel('Depth (m)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label='m s⁻¹')

    _hovm(ax_h1, w_est,  cmo.balance, vmax,      'w estimated')
    _hovm(ax_h2, w_model, cmo.balance, vmax,     'w model')
    _hovm(ax_h3, bias,   cmo.balance, vmax_bias, 'bias (est − model)')
    plt.setp(ax_h2.get_yticklabels(), visible=False)
    plt.setp(ax_h3.get_yticklabels(), visible=False)

    # Depth profile panel — time-mean ± std
    for data_m, data_s, color, label in [
        (w_est_tmean,   w_est_tstd,   'C0', 'est'),
        (w_model_tmean, w_model_tstd, 'C1', 'model'),
        (bias_tmean,    bias_tstd,    'C2', 'bias'),
    ]:
        ax_prof.plot(data_m.values, Z, color=color, lw=1.5, label=label)
        ax_prof.fill_betweenx(Z,
                              (data_m - data_s).values,
                              (data_m + data_s).values,
                              color=color, alpha=0.2)
    ax_prof.axvline(0, color='k', lw=0.7, ls=':')
    ax_prof.set_xlabel('w (m s⁻¹)')
    ax_prof.set_title('Time mean ± σ\nvs depth')
    ax_prof.legend(fontsize=8)
    ax_prof.grid(alpha=0.3)

    # Time series — depth-mean ± std over depth
    for data_m, data_s, color, label in [
        (w_est_dm,   w_est.std('Zl'),   'C0', 'est'),
        (w_model_dm, w_model.std('Zl'), 'C1', 'model'),
        (bias_dm,    bias_dm_std,       'C2', 'bias'),
    ]:
        ax_ts.plot(T, data_m.values, color=color, lw=1, label=label)
        ax_ts.fill_between(T,
                           (data_m - data_s).values,
                           (data_m + data_s).values,
                           color=color, alpha=0.15)
    ax_ts.axhline(0, color='k', lw=0.5, ls=':')
    ax_ts.set_ylabel('Depth-mean w (m s⁻¹)')
    ax_ts.set_title('Depth-mean w and bias vs time (shading = ±σ over depth)')
    ax_ts.legend(fontsize=9)
    ax_ts.grid(alpha=0.3)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_ts.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha='right')

    return fig
