"""
osse_tools.py — Wave glider array OSSE analysis.

Vertical-velocity workflow:
    ds       = load_model(run_dir, iters)
    uv       = sample_fields(ds, positions, vars=('UVEL', 'VVEL'))
    w_est    = compute_w_planefit(uv)['w_est']
    w_model  = sample_model_w(ds, positions)
    fig      = plot_w_comparison(w_est, w_model)

Distribution workflow (observed = glider points, true = model field in the hull):
    obs  = eddy_anomalies(add_density(sample_fields(ds, positions)))
    true = eddy_anomalies(add_density(model_region(ds, positions)))
    plot_field_pdfs(obs, true)
    plot_joint_compare(obs.Vp, obs.Tp, true.Vp, true.Tp, ('v\\'', 'T\\''))
"""

import json
import warnings
import numpy as np
import xarray as xr
import gsw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import cmocean.cm as cmo
from xmitgcm import open_mdsdataset

# MITgcm C-grid stagger of each diagnostic, its vertical coord, and a short alias
_GRID   = {'UVEL': ('XG', 'YC'), 'VVEL': ('XC', 'YG'),
           'THETA': ('XC', 'YC'), 'SALT': ('XC', 'YC'), 'WVEL': ('XC', 'YC')}
_ZCOORD = {'WVEL': 'Zl'}  # default 'Z' (cell centers); WVEL on Zl (interfaces)
_RENAME = {'UVEL': 'U', 'VVEL': 'V', 'THETA': 'T', 'SALT': 'S', 'WVEL': 'W'}


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


def load_cells(path):
    """
    Load a multi-estimate array config (e.g. configs/with_TAO/north_shift/*.json).

    Each config holds one or more independent 'cells' — small position sets
    (e.g. a 4-point TAO+glider diamond) that each support their own plane fit.
    The number of cells returned is the number of independent w estimates the
    config is designed to produce; iterate and pass each cell's positions to
    sample_fields / compute_w_planefit / sample_model_w in turn.

    Returns
    -------
    list of (center_lat, positions) tuples, positions a list of (lat, lon) tuples
    """
    with open(path) as f:
        cfg = json.load(f)
    return [(c['center_lat'], [tuple(p) for p in c['positions']]) for c in cfg['cells']]


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
    # mask fill values in the diagnostics only, never the grid coordinates
    return ds.where(ds[list(ds.data_vars)] != -999.0)


def _latlon_to_m(lats, lons):
    """projection of lat/lon (deg) to meters about their centroid."""
    lats, lons = np.asarray(lats), np.asarray(lons)
    lat_c, lon_c = lats.mean(), lons.mean()
    deg_to_m = np.pi / 180 * 6371000.0
    x_m = (lons - lon_c) * np.cos(np.radians(lat_c)) * deg_to_m
    y_m = (lats - lat_c) * deg_to_m
    return x_m, y_m


def _hull_bbox(positions, buf=3 / 24):
    """Bounding box (lon_min, lon_max, lat_min, lat_max) around positions, with buffer."""
    lats = [p[0] for p in positions]
    lons = [p[1] for p in positions]
    return (min(lons) - buf, max(lons) + buf, min(lats) - buf, max(lats) + buf)


def _obs_z(max_depth, dz_obs, min_depth=0):
    """Layer-midpoint depths for sampling: -(min_depth+dz/2), ..., down to max_depth."""
    n = int((max_depth - min_depth) / dz_obs)
    z = -(min_depth + np.arange(n) * dz_obs + dz_obs / 2)
    return xr.DataArray(z, dims='obs_depth', coords={'obs_depth': z})


def sample_fields(ds, positions, vars=('UVEL', 'VVEL', 'THETA', 'SALT'),
                  max_depth=70, dz_obs=2, min_depth=0):
    """
    Lazily interpolate model fields to each glider position at uniform obs depths.

    Parameters
    ----------
    ds : xr.Dataset
        From load_model.
    positions : list of (lat, lon)
        Glider positions in degrees.
    vars : tuple of str
        MITgcm diagnostics to sample. Default UVEL, VVEL, THETA, SALT.
    max_depth, dz_obs : float
        Sampling depth range and interval in meters. Defaults 70 and 2.
    min_depth : float
        Shallowest depth sampled, e.g. 8 if the array can't see above 8 m.
        compute_w_planefit assumes w=0 at this depth rather than at the surface.

    Returns
    -------
    xr.Dataset, dims (time, glider, obs_depth)
        Variables renamed U, V, T, S. Glider lat/lon stored as coordinates;
        obs_depth holds layer midpoints (-1, -3, ..., -69 for 70 m / 2 m / 0 m min).
    """
    obs_z_da = _obs_z(max_depth, dz_obs, min_depth)
    g = np.arange(len(positions))
    lat_da = xr.DataArray([p[0] for p in positions], dims='glider', coords={'glider': g})
    lon_da = xr.DataArray([p[1] for p in positions], dims='glider', coords={'glider': g})

    out = {}
    for v in vars:
        gx, gy = _GRID[v]
        out[_RENAME[v]] = ds[v].interp({gx: lon_da, gy: lat_da, _ZCOORD.get(v, 'Z'): obs_z_da}) \
                               .transpose('time', 'glider', 'obs_depth')
    return xr.Dataset(out).assign_coords(lat=lat_da, lon=lon_da)


def model_region(ds, positions, vars=('UVEL', 'VVEL', 'THETA', 'SALT'),
                 max_depth=70, dz_obs=2, min_depth=0):
    """
    The 'true' estimate: model fields at every grid point inside the array hull.

    Each field is interpolated to the tracer cell centers (co-locating U, V, T, S)
    and to the obs depths, then masked to the convex hull of positions and stacked
    over the horizontal points. Compare its distribution against sample_fields output.

    For a large hull or long record, subsample time before calling for memory usage purposes.

    Returns
    -------
    xr.Dataset, dims (time, point, obs_depth)
        Variables renamed U, V, T, S; lat/lon stored as coordinates on point.
    """
    obs_z_da = _obs_z(max_depth, dz_obs, min_depth)
    lon0, lon1, lat0, lat1 = _hull_bbox(positions)
    xc = ds.XC.sel(XC=slice(lon0, lon1)).values
    yc = ds.YC.sel(YC=slice(lat0, lat1)).values
    xt = xr.DataArray(xc, dims='XC', coords={'XC': xc})
    yt = xr.DataArray(yc, dims='YC', coords={'YC': yc})
    mask = xr.DataArray(_convex_hull_mask(xc, yc, positions),
                        dims=('YC', 'XC'), coords={'YC': yc, 'XC': xc})

    keep = {'time', 'YC', 'XC', 'obs_depth'}
    out = {}
    for v in vars:
        gx, gy = _GRID[v]
        da = ds[v].interp({gx: xt, gy: yt, _ZCOORD.get(v, 'Z'): obs_z_da})
        # drop MITgcm grid coords (hFac, dxG, ...) that carry the now-unused stagger dims
        da = da.drop_vars([c for c in da.coords if c not in keep])
        out[_RENAME[v]] = da.where(mask)
    reg = xr.Dataset(out).stack(point=('YC', 'XC'))
    reg = reg.assign_coords(lat=reg.YC, lon=reg.XC)
    return reg.transpose('time', 'point', 'obs_depth')


def add_density(samp):
    """Add potential density anomaly sigma0 (kg/m^3) from T, S via TEOS-10."""
    p = xr.apply_ufunc(gsw.p_from_z, samp.obs_depth, samp.lat,
                       dask='parallelized', output_dtypes=[float])
    SA = xr.apply_ufunc(gsw.SA_from_SP, samp.S, p, samp.lon, samp.lat,
                        dask='parallelized', output_dtypes=[float])
    CT = xr.apply_ufunc(gsw.CT_from_pt, SA, samp.T,
                        dask='parallelized', output_dtypes=[float])
    samp['sigma0'] = xr.apply_ufunc(gsw.sigma0, SA, CT,
                                    dask='parallelized', output_dtypes=[float])
    return samp


def eddy_anomalies(samp, mean_dim='time'):
    """Add anomalies U', V', T', S' as deviations from the mean over mean_dim."""
    for v in ('U', 'V', 'T', 'S'):
        if v in samp:
            samp[v + 'p'] = samp[v] - samp[v].mean(mean_dim)
    return samp


def extrapolate_currents_to_surface(uv_samples):
    """
    Extend the sampled profile upward to the surface using the vertical shear at
    the top of the profile, so a plane-fit w can integrate from w=0 at the surface
    rather than from the shallowest sampled depth.

    The vertical shear dU/dz (and dV/dz, etc.) between the two shallowest obs_depth
    levels is assumed constant above the shallowest level. New layer midpoints are
    added at the same spacing from the shallowest level up toward the surface (the
    topmost midpoint sits at -dz/2 so its upper interface is 0 m). Every field with
    an obs_depth dimension is extrapolated by its own top shear, i.e., U and V use dU/dz and dV/dz.

    Parameters
    ----------
    uv_samples : xr.Dataset
        dims (time, glider, obs_depth), with U, V and lat/lon coords.
        obs_depth midpoints must be uniformly spaced, shallowest (least negative) first.

    Returns
    -------
    xr.Dataset
        Same variables as the input with extra shallower obs_depth levels prepended.
        Returned unchanged if the shallowest level is already within dz/2 of 0 m,
        or if there are fewer than two obs_depth levels to define a shear.
    """
    obs_z = uv_samples.obs_depth.values
    if obs_z.size < 2:
        return uv_samples
    dz = obs_z[1] - obs_z[0]                  # signed spacing (negative: deeper index)
    z_top = float(obs_z[0])                   # shallowest midpoint, e.g. -9

    # New midpoints from just above z_top up toward the surface, staying below 0 m.
    # e.g. z_top=-9, dz=-2 -> [-7, -5, -3, -1]
    new_mids = np.arange(z_top - dz, 0.0, -dz)
    if new_mids.size == 0:
        return uv_samples                     # already reaches the surface
    new_mids = new_mids[::-1]                 # shallowest first: [-1, -3, -5, -7]

    out = {}
    for v in uv_samples.data_vars:
        arr = uv_samples[v]
        if 'obs_depth' not in arr.dims:
            out[v] = arr
            continue
        top0 = arr.isel(obs_depth=0)
        shear = (arr.isel(obs_depth=1) - top0) / dz        # d/dz at the top of the profile
        extra = xr.concat([top0 + shear * (zm - z_top) for zm in new_mids],
                          dim='obs_depth').assign_coords(obs_depth=new_mids)
        # concat puts obs_depth first; restore the caller's dim order (compute_w_planefit
        # relies on U being (time, glider, obs_depth))
        out[v] = xr.concat([extra, arr], dim='obs_depth').transpose(*arr.dims)
    return xr.Dataset(out)


def compute_w_planefit(uv_samples, remove_barotropic=False, extrapolate_to_surface=True):
    """
    Estimate w via plane fit to U and V across the array, then integrate divergence.
    At each (time, depth): fits u = a + b*x + c*y and v = a + b*x + c*y over the
    glider positions to extract du/dx and dv/dy. Integrates div = du/dx + dv/dy
    downward from w=0 at the shallowest sampled depth using the continuity equation:
        w(z_bottom) = w(z_top) + div * dz
    which follows from dw/dz = -div with z negative downward.

    Parameters
    ----------
    uv_samples : xr.Dataset
        From sample_uv, dims (time, glider, obs_depth).
    remove_barotropic : bool
        If True, subtract the depth-mean from U and V at each glider and timestep
        before the plane fit, so the result estimates the baroclinic w only. (Not using this)
    extrapolate_to_surface : bool
        If True (default), first extend U and V up to the surface using the vertical
        shear at the top of the profile (see extrapolate_currents_to_surface), so w
        is integrated from w=0 at the surface. If False, w=0 is assumed at the
        shallowest sampled depth. For an 8-to-75 m depth range this fills in 0-8 m before
        the plane fit.

    Returns
    -------
    xr.Dataset with:
        w_est : (time, depth)  estimated w at layer interfaces [m/s]
                               depth coordinate = [z_top, z_top-dz, ..., z_top-n*dz]
        div   : (time, obs_depth)  horizontal divergence at obs midpoints [1/s]

    Notes
    -----
    Positions are projected onto a flat plane via _latlon_to_m.
    w=0 is assumed at at the surface by default, otherwise 0 at z_top if no extrapolation happens.
    """
    if extrapolate_to_surface:
        uv_samples = extrapolate_currents_to_surface(uv_samples)

    lats = uv_samples.lat.values
    lons = uv_samples.lon.values
    x_m, y_m = _latlon_to_m(lats, lons)

    # Pseudoinverse of design matrix — computed once, applied to all (time, depth)
    A = np.column_stack([np.ones(len(lats)), x_m, y_m])  # (N, 3)
    Ainv = np.linalg.pinv(A)                              # (3, N): solves overdetermined plane fit in one multiply

    uv = uv_samples.compute()
    U = uv['U'].values  # (ntime, nglider, n_obs) — must stay in this order; see reshape below
    V = uv['V'].values
    ntime, nglider, n_obs = U.shape

    if remove_barotropic:
        # Subtract depth-mean at each glider and timestep before fitting
        U = U - U.mean(axis=2, keepdims=True)
        V = V - V.mean(axis=2, keepdims=True)

    # Transpose to (nglider, ntime, n_obs) so the glider axis aligns with Ainv's (3, N),
    # then collapse (ntime, n_obs) → one axis to fit all times and depths in a single multiply.
    # WARNING: assumes U.shape == (ntime, nglider, n_obs); assert this if dim ordering is ever uncertain.
    cu = Ainv @ U.transpose(1, 0, 2).reshape(nglider, ntime * n_obs)  # (3, ntime*n_obs)
    cv = Ainv @ V.transpose(1, 0, 2).reshape(nglider, ntime * n_obs)  # row 0=intercept, 1=d/dx, 2=d/dy
    du_dx = cu[1].reshape(ntime, n_obs)
    dv_dy = cv[2].reshape(ntime, n_obs)
    div_vals = du_dx + dv_dy

    obs_z = uv_samples.obs_depth.values      # midpoints, e.g. [-9, -11, ..., -69] for an 8 m min_depth
    # WARNING: only the first interval is used — assumes uniform depth spacing throughout
    dz_obs = float(abs(obs_z[1] - obs_z[0])) if n_obs > 1 else float(abs(obs_z[0]) * 2)
    z_top = obs_z[0] + dz_obs / 2            # shallowest interface; w=0 assumed here, not at 0 m
    w_z = z_top - np.arange(n_obs + 1) * dz_obs   # interfaces: [z_top, z_top-dz, ..., z_top-n_obs*dz]

    # Integrate downward from w=0 at surface: w(k+1) = w(k) + div(k) * dz
    # Sign: integrating dw/dz = -div with dz < 0 gives Δw = div * |dz|, so no explicit minus needed
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


def _convex_hull_mask(xc_vals, yc_vals, positions):
    """
    Boolean mask (nYC, nXC) — True for model grid points inside the convex hull
    of positions. Uses scipy ConvexHull + matplotlib Path

    Args:
        xc_vals: 1D array of grid X (lon) coordinates, length nXC
        yc_vals: 1D array of grid Y (lat) coordinates, length nYC
        positions: iterable of (lat, lon) pairs — NOTE: lat-first ordering assumed

    Returns:
        (nYC, nXC) numpy bool array; True where grid point is inside the hull
    """
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path

    # positions is (lat, lon) which is MITgcm default order; flip to (lon, lat) = (x, y) for spatial ops
    # WARNING: if caller passes (lon, lat) already, the hull will be mirrored
    pts = np.array([[p[1], p[0]] for p in positions])  # shape (N, 2)

    hull = ConvexHull(pts)          # raises QhullError if < 3 non-collinear pts
    path = Path(pts[hull.vertices]) # boundary vertices only; relies on contains_points to close

    # meshgrid with default indexing='xy': rows vary with Y, cols vary with X → (nYC, nXC)
    XC, YC = np.meshgrid(xc_vals, yc_vals)

    inside = path.contains_points(
        np.column_stack([XC.ravel(), YC.ravel()])  # (nYC*nXC, 2) in (x, y) order
    ).reshape(XC.shape)                             # back to (nYC, nXC)

    return inside  # (nYC, nXC) numpy bool


def _hull_mean(field, positions):
    """Average a (..., YC, XC) field over model grid points inside the array hull."""
    lon0, lon1, lat0, lat1 = _hull_bbox(positions)
    sub = field.sel(XC=slice(lon0, lon1), YC=slice(lat0, lat1))
    mask = xr.DataArray(
        _convex_hull_mask(sub.XC.values, sub.YC.values, positions),
        dims=('YC', 'XC'), coords={'YC': sub.YC.values, 'XC': sub.XC.values})
    return sub.where(mask).mean(['XC', 'YC'])


def sample_model_w(ds, positions, max_depth=70, dz_obs=2,
                   remove_barotropic=False, spatial_mean=True, min_depth=0):
    """
    Sample WVEL interpolated to the interface depths of compute_w_planefit.

    Parameters
    ----------
    ds : xr.Dataset
    positions : list of (lat, lon)
    max_depth, dz_obs, min_depth : float
        Must match the values used in sample_fields. Defaults 70, 2, 0.
    remove_barotropic : bool
        If True, subtract the linear barotropic trend from the returned w.
    spatial_mean : bool
        If True (default), return WVEL averaged over all model grid points inside
        the convex hull — the area-mean w that the plane fit estimates. If False,
        return WVEL at the array centroid.

    Returns
    -------
    xr.DataArray, dims (time, depth)
        depth coordinate = [z_top, z_top-dz_obs, ..., -max_depth], z_top = -min_depth.
        Note this is the model's true w (generally nonzero at z_top if min_depth > 0),
        unlike compute_w_planefit's w_est which assumes w=0 there.
    """
    z_top = -min_depth
    n = int((max_depth - min_depth) / dz_obs)
    w_z = z_top - np.arange(n + 1) * dz_obs
    w_z_da = xr.DataArray(w_z, dims='depth', coords={'depth': w_z})

    if spatial_mean:
        w = _hull_mean(ds.WVEL.interp(Zl=w_z_da), positions).compute()
    else:
        lat_c = np.mean([p[0] for p in positions])
        lon_c = np.mean([p[1] for p in positions])
        w = ds.WVEL.interp(XC=lon_c, YC=lat_c, Zl=w_z_da).compute()

    if remove_barotropic:
        # Remove the linear barotropic trend: the component that takes w from its
        # value at z_top to w(-max_depth) at the bottom of the sampled layer.
        # This is consistent with compute_w_planefit(remove_barotropic=True),
        # which integrates div - <div>_z = d/dz[w - w(-H)/H * |z|].
        w_bottom = w.isel(depth=-1)
        span = max_depth - min_depth
        w = w + (w_bottom / span) * (w.depth - z_top)
    return w


def model_divergence(ds, positions, max_depth=70, dz_obs=2, min_depth=0):
    """
    True horizontal divergence, area-averaged over the array hull.

    Computed on the native C-grid from the flux form du/dx + dv/dy using the
    cell-edge lengths and areas, so it is the truth that compute_w_planefit's
    'div' field estimates from the sparse array.

    Returns
    -------
    xr.DataArray, dims (time, obs_depth)
    """
    from xgcm import Grid
    grid = Grid(ds, periodic=False)
    div = (grid.diff(ds.UVEL * ds.dyG, 'X', boundary='fill') +
           grid.diff(ds.VVEL * ds.dxG, 'Y', boundary='fill')) / ds.rA
    div = div.interp(Z=_obs_z(max_depth, dz_obs, min_depth))
    return _hull_mean(div, positions).compute()


# ---------------------------------------------------------------------------
# Vertical-velocity skill metrics
# ---------------------------------------------------------------------------

def _integrated_autocorr_time(x):
    """Integrated autocorrelation time tau = 1 + 2*sum_k rho_k of a 1-D series,
    in units of the sampling interval (samples).

    The sum is truncated by: accumulate lags until the sample ACF first 
    goes non-positive. tau is the factor by which serial correlation inflates the
    variance of the sample mean, so the effective sample size is N/tau.
    """
    x = np.asarray(x, float)
    n = x.size
    xa = x - x.mean()
    denom = np.dot(xa, xa)                        # == n * sample variance
    if n < 2 or denom <= 0:
        return 1.0
    maxlag = max(1, n // 4)
    s = 0.0
    for k in range(1, maxlag + 1):
        rho_k = np.dot(xa[:-k], xa[k:]) / denom   # the 1/n on top and bottom cancels
        if rho_k <= 0:
            break
        s += rho_k
    return max(1.0 + 2.0 * s, 1.0)


def mean_se_autocorr(series):
    """Time-mean of a 1-D series and the standard error of that mean, with the SE
    inflated for serial correlation.

    SE = sd/sqrt(N_eff) with N_eff = N/tau, so a 95% CI is mean +/- 1.96*SE (assuming 
    normal distribution). 

    Returns
    -------
    (mean, se, n_eff, tau)
        mean   sample mean of the series
        se     standard error of that mean (NaN if N<2)
        n_eff  effective sample size N/tau
        tau    integrated autocorrelation time (samples); NaN if N<2
    """
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, 0.0, np.nan
    mean = float(x.mean())
    if n < 2:
        return mean, np.nan, float(n), np.nan
    tau   = _integrated_autocorr_time(x)
    n_eff = n / tau
    se    = float(x.std(ddof=1) / np.sqrt(n_eff))
    return mean, se, float(n_eff), float(tau)


def w_skill_metrics(w_est, w_model, depth_range=None):
    """
    Scalar skill of an estimated w against model-truth w, pooled over all
    (time, depth) samples.

    Parameters
    ----------
    w_est, w_model : xr.DataArray, dims (time, depth)
        Both from compute_w_planefit / sample_model_w. They need not share the
        same interface depth grid: when compute_w_planefit extrapolated to the
        surface, w_est carries extra shallow interfaces above -min_depth. The two
        are aligned to their shared (time, depth) grid before pooling.
    depth_range : (z_shallow, z_deep) in model convention (e.g. (0, -50)), optional
        Restrict the statistics to this depth slice before pooling.

    Returns
    -------
    dict
        rms         root-mean-square of (w_est - w_model) [m/s]
        mean_bias   mean of (w_est - w_model) [m/s]
        corr        Pearson correlation of w_est and w_model over (time, depth)
        w_est_std   std of w_est [m/s]
        w_model_std std of w_model, i.e. the signal being estimated [m/s]
        w_est_mean  mean of w_est, i.e. the estimated mean vertical velocity [m/s]
        w_model_mean mean of w_model, i.e. the mean vertical velocity [m/s]
        norm_rms    rms / w_model_std (error relative to the signal); NaN if signal std is 0
        n           number of finite sample pairs used
        w_est_mean_se, w_model_mean_se, mean_bias_se
                    standard errors of the depth-averaged
                    time means (est, true, and their paired difference) [m/s].
                    95% CI = mean +/- 1.96*se. See mean_se_autocorr for the
                    estimand (expected/long-run mean, not the exact window mean).
        n_eff, n_eff_model, n_eff_est
                    effective sample sizes (N/tau) of the difference, true, and
                    estimated column-mean series
        tau         integrated autocorrelation time of the difference (samples)
    """
    if depth_range is not None:
        w_est   = w_est.sel(depth=slice(*depth_range))
        w_model = w_model.sel(depth=slice(*depth_range))
    # Compare only where both are defined: an extrapolated-to-surface w_est spans
    # shallower interfaces than w_model, so raveling raw .values would mismatch.
    w_est, w_model = xr.align(w_est, w_model, join='inner')
    # Autocorrelation-aware SE of the depth-averaged ("total over depth") time
    # means. Each is the time series of the column-mean w; its mean is the total
    # <w> and its SE accounts for w's ~1-day decorrelation (see mean_se_autocorr).
    est_col = w_est.mean('depth', skipna=True).values
    mod_col = w_model.mean('depth', skipna=True).values
    _, est_mean_se, n_eff_est, _        = mean_se_autocorr(est_col)
    _, mod_mean_se, n_eff_model, _      = mean_se_autocorr(mod_col)
    _, bias_mean_se, n_eff, tau         = mean_se_autocorr(est_col - mod_col)
    est = np.asarray(w_est.values, float).ravel()
    mod = np.asarray(w_model.values, float).ravel()
    good = np.isfinite(est) & np.isfinite(mod)
    est, mod = est[good], mod[good]
    bias    = est - mod
    est_std = float(est.std()) if est.size else np.nan
    mod_std = float(mod.std()) if mod.size else np.nan
    rms     = float(np.sqrt((bias ** 2).mean())) if bias.size else np.nan
    corr    = (float(np.corrcoef(est, mod)[0, 1])
               if est.size > 1 and est_std > 0 and mod_std > 0 else np.nan)
    return dict(
        rms=rms,
        mean_bias=float(bias.mean()) if bias.size else np.nan,
        corr=corr,
        w_est_std=est_std,
        w_model_std=mod_std,
        w_est_mean=float(est.mean()) if est.size else np.nan,
        w_model_mean=float(mod.mean()) if mod.size else np.nan,
        norm_rms=rms / mod_std if mod_std and mod_std > 0 else np.nan,
        n=int(est.size),
        # autocorrelation-aware SEs of the depth-averaged time means (m/s)
        w_est_mean_se=est_mean_se,
        w_model_mean_se=mod_mean_se,
        mean_bias_se=bias_mean_se,
        n_eff=n_eff,               # effective sample size of the difference series
        n_eff_model=n_eff_model,   # effective sample size of the true-w series
        n_eff_est=n_eff_est,
        tau=tau,                   # integrated autocorr time of the difference (samples)
    )


def frac_mean_bias(mean_bias, w_model_mean, alpha=0.05):
    """
    Fractional error in the time-mean w: (<w_est> - <w_model>) / <w_model>, i.e.
    mean_bias / w_model_mean. Perfect estimate = 0.

    Guarded against the mean's zero-crossings: the ratio is undefined where the
    denominator passes through zero, so any point whose |w_model_mean| is below
    alpha times the peak |w_model_mean| of the set is returned as NaN (the line
    simply breaks there rather than spiking to +/-inf). The threshold is relative
    to the mean's OWN scale, not the signal std, because |mean w| is typically
    tens of times smaller than sigma here.

    Accepts scalars, numpy arrays, or xarray/pandas objects (compared point-wise
    across a depth profile or a config sweep); returns a numpy array (or scalar).
    """
    mean_bias    = np.asarray(mean_bias, float)
    w_model_mean = np.asarray(w_model_mean, float)
    peak = np.nanmax(np.abs(w_model_mean))
    tau  = alpha * peak if np.isfinite(peak) else 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        out = mean_bias / w_model_mean
    return np.where(np.abs(w_model_mean) >= tau, out, np.nan)


def w_skill_by_depth(w_est, w_model):
    """
    Depth-resolved skill: the same descriptive statistics as w_skill_metrics,
    computed at each depth by pooling over time. Use to show how error grows
    with integration depth within a single sampled range.

    Returns
    -------
    xr.Dataset, dim (depth)
        rms, mean_bias, corr, w_est_std, w_model_std, norm_rms,
        w_est_mean, w_model_mean, and the standard errors
        of the three time means at each depth: w_est_mean_se, w_model_mean_se,
        mean_bias_se (95% CI = mean +/- 1.96*se), plus n_eff (of the difference)
        and tau (integrated autocorr time, samples). See mean_se_autocorr.
    """
    bias      = w_est - w_model
    rms       = np.sqrt((bias ** 2).mean('time'))
    est_std   = w_est.std('time')
    mod_std   = w_model.std('time')
    # correlation over time at each depth, from the time anomalies
    ea = w_est   - w_est.mean('time')
    ma = w_model - w_model.mean('time')
    corr = (ea * ma).mean('time') / (est_std * mod_std)

    # Autocorrelation-aware SE of each time mean, computed per depth (each depth's
    # own decorrelation time). Looping over ~32 depths is cheap.
    depth = w_est['depth']
    nd = depth.size
    est_se = np.full(nd, np.nan); mod_se = np.full(nd, np.nan)
    bias_se = np.full(nd, np.nan); neff = np.full(nd, np.nan); tau = np.full(nd, np.nan)
    for k in range(nd):
        _, est_se[k], _, _        = mean_se_autocorr(w_est.isel(depth=k).values)
        _, mod_se[k], _, _        = mean_se_autocorr(w_model.isel(depth=k).values)
        _, bias_se[k], neff[k], tau[k] = mean_se_autocorr(bias.isel(depth=k).values)
    _da = lambda a: xr.DataArray(a, dims='depth', coords={'depth': depth})
    return xr.Dataset(dict(
        rms=rms,
        mean_bias=bias.mean('time'),
        corr=corr,
        w_est_std=est_std,
        w_model_std=mod_std,
        norm_rms=rms / mod_std,
        w_est_mean=w_est.mean('time'),
        w_model_mean=w_model.mean('time'),
        w_est_mean_se=_da(est_se),
        w_model_mean_se=_da(mod_se),
        mean_bias_se=_da(bias_se),
        n_eff=_da(neff),
        tau=_da(tau),
    ))


def vertical_eddy_flux(w, tracers, mean_dim='time'):
    """
    Vertical eddy flux <w' phi'> over mean_dim for each tracer present (U,V,T,S).

    w and tracers must share dims and depth grid; primes are deviations from the
    mean over mean_dim. Returns a Dataset with wU, wV, wT, wS as available.
    """
    wp = w - w.mean(mean_dim)
    out = {}
    for v in ('U', 'V', 'T', 'S'):
        if v in tracers:
            out['w' + v] = (wp * (tracers[v] - tracers[v].mean(mean_dim))).mean(mean_dim)
    return xr.Dataset(out)


def array_vertical_flux(w_est, fields, mean_dim='time'):
    """
    Array-estimated vertical eddy flux: plane-fit w_est paired with the array-mean
    tracers. w_est (on interfaces) is interpolated to the tracer obs depths.

    Returns Dataset of flux profiles (obs_depth).
    """
    z = fields.obs_depth.values
    w = w_est.interp(depth=xr.DataArray(z, dims='obs_depth', coords={'obs_depth': z}))
    return vertical_eddy_flux(w, fields.mean('glider'), mean_dim)


def model_vertical_flux(region, mean_dim='time'):
    """
    True total vertical eddy flux profiles <w' phi'> over the hull, from a
    model_region that includes WVEL: full eddy flux averaged over hull points and time.
    """
    return vertical_eddy_flux(region.W, region, mean_dim).mean('point')


def _js_distance(sa, sb, edges):
    """Jensen-Shannon distance (0 identical, 1 disjoint) between two histograms.

    sa, sb : (N, D) sample arrays; edges : list of D bin-edge arrays. Works for
    1-D PDFs (D=1) and joint PDFs (D=2) on the shared grid.
    """
    from scipy.spatial.distance import jensenshannon
    Ha, _ = np.histogramdd(sa, bins=edges)
    Hb, _ = np.histogramdd(sb, bins=edges)
    return float(jensenshannon(Ha.ravel(), Hb.ravel(), base=2))


def dist_stats(obs, true):
    """Summary stats and observed-vs-true distance metrics (NaNs dropped)."""
    from scipy.stats import skew, ks_2samp, wasserstein_distance
    o = np.asarray(obs).ravel(); o = o[np.isfinite(o)]
    t = np.asarray(true).ravel(); t = t[np.isfinite(t)]
    edges = np.linspace(*np.percentile(np.concatenate([o, t]), [0.5, 99.5]), 61)
    return {
        'obs_mean': o.mean(),  'true_mean': t.mean(),
        'obs_std':  o.std(),   'true_std':  t.std(),
        'obs_skew': skew(o),   'true_skew': skew(t),
        'ks': ks_2samp(o, t).statistic,
        'wasserstein': wasserstein_distance(o, t),
        'js': _js_distance(o.reshape(-1, 1), t.reshape(-1, 1), [edges]),
    }




# ---------------------------------------------------------------------------
# Domain-map diagnostics: depth-mean time series, spatial decorrelation, and
# horizontal gradients of the mean velocity field. Used by run_domain_maps.py
# to build the domain/ maps (mirrors the mean-velocity figures in demo_domain).
# ---------------------------------------------------------------------------

_KM_PER_DEG = 111.195  # mean km per degree of latitude


def depth_mean_series(ds, var, max_depths):
    """
    Depth-average one MITgcm velocity diagnostic over 0..max_depth, keeping time.

    Reads the 0-250 m water column once and returns the depth-mean *time series*
    (time, y, x) for each requested cutoff, so several depth means share a single
    read of the data. WVEL uses the Zl interface coordinate, U/V the Z centers.

    Parameters
    ----------
    ds : xr.Dataset  (from load_model)
    var : str        one of 'UVEL', 'VVEL', 'WVEL'
    max_depths : iterable of float   depth cutoffs in meters, e.g. (70, 120, 250)

    Returns
    -------
    dict {max_depth: xr.DataArray(time, y, x), float32} on the variable's own grid
    """
    zc = _ZCOORD.get(var, 'Z')
    da = ds[var]
    zmax = max(max_depths)
    # trim to the shallowest levels covering zmax so we never read the deep ocean
    kkeep = int((da[zc].values >= -zmax - 1e-6).sum())
    da = da.isel({zc: slice(0, kkeep)})
    means = {d: da.where(da[zc] >= -d).mean(zc) for d in max_depths}
    import dask
    computed = dask.compute(*means.values())  # one pass over the trimmed column
    return {d: c.astype('float32') for d, c in zip(means.keys(), computed)}


def _grid_spacing_deg(coord):
    """Mean absolute spacing (deg) of a 1-D coordinate."""
    return float(np.mean(np.abs(np.diff(np.asarray(coord, float)))))


def gradient_components(field, lon, lat):
    """
    Horizontal derivatives d/dx, d/dy (per meter) of a field on a lon/lat grid.

    Accepts a 2-D (y, x) field or a 3-D (time, y, x) stack; the last two axes are
    latitude then longitude. Uses second-order centered differences, converting the
    degree spacing to meters with the local cos(lat) factor for the zonal term.

    Returns
    -------
    (fx, fy) : same shape as `field`
    """
    field = np.asarray(field, float)
    yax, xax = field.ndim - 2, field.ndim - 1
    dlon = _grid_spacing_deg(lon)
    dlat = _grid_spacing_deg(lat)
    m_per_deg = _KM_PER_DEG * 1000.0
    cos = np.cos(np.radians(np.asarray(lat, float)))
    cos = cos.reshape([1] * yax + [-1, 1])  # broadcast over any leading axes and x
    fy = np.gradient(field, axis=yax) / (dlat * m_per_deg)
    fx = np.gradient(field, axis=xax) / (dlon * m_per_deg * cos)
    return fx, fy


def gradient_magnitude(field, lon, lat):
    """Magnitude sqrt((d/dx)^2 + (d/dy)^2) (per meter) of a field; see gradient_components."""
    fx, fy = gradient_components(field, lon, lat)
    return np.hypot(fx, fy)


def _lag_corr_curve(an, nlag, axis):
    """
    Lag-correlation curve of a normalised anomaly field along one spatial axis.

    `an` is (time, y, x) with zero time-mean and unit time-std at every point, so
    the time average of an_i * an_j is the Pearson correlation between points i and
    j. For each separation d = 0..nlag the forward (i, i+d) and backward (i, i-d)
    correlations are averaged, giving c[d] on the full grid (NaN where a point has
    no in-domain neighbour at that lag). `axis` is 1 (meridional) or 2 (zonal).
    """
    T = an.shape[0]
    shp = an.shape[1:]
    Np = an.shape[axis]
    c = np.full((nlag + 1,) + shp, np.nan, np.float32)
    c[0] = 1.0
    for d in range(1, nlag + 1):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[axis] = slice(0, Np - d); s2[axis] = slice(d, Np)
        r = (an[tuple(s1)] * an[tuple(s2)]).sum(0) / T          # (grid minus d along axis)
        fwd = np.full(shp, np.nan, np.float32)
        bwd = np.full(shp, np.nan, np.float32)
        fs = [slice(None)] * 2; bs = [slice(None)] * 2
        fs[axis - 1] = slice(0, Np - d); bs[axis - 1] = slice(d, Np)
        fwd[tuple(fs)] = r; bwd[tuple(bs)] = r
        with warnings.catch_warnings():           # edge points have no neighbour
            warnings.simplefilter('ignore', RuntimeWarning)
            c[d] = np.nanmean(np.stack([fwd, bwd]), 0)
    return c


def _efold_lag(c, thresh):
    """
    Fractional lag at which a lag-correlation curve c[0..nlag] first drops to thresh.

    Linear interpolation between the last lag >= thresh and the first below it. If
    the curve never crosses within the window it is capped at the maximum lag; NaN
    where the curve is undefined (edges) at the crossing.
    """
    nlagp1 = c.shape[0]
    below = c < thresh
    below[0] = False
    ever = below.any(0)
    first = np.argmax(below, 0)                    # first True lag; 0 if never
    k = np.clip(np.where(ever, first, nlagp1 - 1), 1, nlagp1 - 1)
    c1 = np.take_along_axis(c, k[None] - 1, 0)[0]  # >= thresh (or capped)
    c2 = np.take_along_axis(c, k[None], 0)[0]      # <  thresh (or capped)
    denom = c1 - c2
    with np.errstate(divide='ignore', invalid='ignore'):
        step = np.where(denom > 0, (c1 - thresh) / denom, 0.0)
    frac = (k - 1) + step
    frac = np.where(ever, frac, nlagp1 - 1).astype(np.float32)
    frac[~(np.isfinite(c1) & np.isfinite(c2))] = np.nan
    return frac


def autocorr_curves(field, lon, lat, max_lag_km=600.0):
    """
    Zonal and meridional lag-correlation curves of a (time, y, x) field.

    At every grid point the temporal anomaly is normalised to unit variance, so the
    time average of the product of two points IS their Pearson correlation. For each
    separation the forward/backward correlations are averaged (see _lag_corr_curve),
    giving the dimensionless autocorrelation vs separation along x and along y.

    Returns (bundled in a dict so callers can pick what they need)
    -------
    cx, cy : (nlag+1, y, x) float32   correlation at lag 0..nlag (zonal, meridional)
    sep_x_deg, sep_y_deg : (nlag+1,)  separation of each lag in degrees
    dx_km : (y,)  zonal grid spacing in km (varies with lat);  dy_km : float
    valid : (y, x) bool  points with non-degenerate time series
    """
    f = np.asarray(field, np.float32)
    mean = f.mean(0)
    std = np.where(f.std(0) == 0, np.nan, f.std(0))
    an = (f - mean) / std
    valid = np.isfinite(an[0])
    an = np.nan_to_num(an, copy=False)

    dlon = _grid_spacing_deg(lon)
    dlat = _grid_spacing_deg(lat)
    dx_km = dlon * _KM_PER_DEG * np.cos(np.radians(np.asarray(lat, float)))  # (Ny,)
    dy_km = dlat * _KM_PER_DEG
    nlag_x = int(np.ceil(max_lag_km / float(np.nanmin(dx_km))))
    nlag_y = int(np.ceil(max_lag_km / dy_km))

    cx = _lag_corr_curve(an, nlag_x, axis=2)
    cy = _lag_corr_curve(an, nlag_y, axis=1)
    return dict(cx=cx, cy=cy,
                sep_x_deg=np.arange(nlag_x + 1) * dlon,
                sep_y_deg=np.arange(nlag_y + 1) * dlat,
                dx_km=dx_km, dy_km=dy_km, valid=valid)


def decorr_scale_from_curves(cur, thresh=1.0 / np.e):
    """
    Isotropic decorrelation length (km) from precomputed autocorr_curves output.

    The zonal (Lx) and meridional (Ly) separations at which each point's curve first
    falls to `thresh` (default 1/e) are found by interpolation; the isotropic scale
    is the geometric mean sqrt(Lx * Ly). Returns (L, Lx, Ly) as (y, x) km arrays.
    """
    frac_x = _efold_lag(cur['cx'], thresh)
    frac_y = _efold_lag(cur['cy'], thresh)
    Lx = frac_x * cur['dx_km'][:, None]
    Ly = frac_y * cur['dy_km']
    L = np.sqrt(Lx * Ly)
    for a in (Lx, Ly, L):
        a[~cur['valid']] = np.nan
    return L, Lx, Ly


def fixed_lag_corr(cur, sep_deg):
    """
    Isotropic autocorrelation at a fixed separation (degrees), from autocorr_curves.

    Averages the zonal and meridional correlation at the lag nearest `sep_deg`,
    giving a dimensionless (y, x) map in [-1, 1] (NaN where undefined). The actual
    separations used are returned so the caller can label the figure exactly.
    """
    dlon = cur['sep_x_deg'][1]
    dlat = cur['sep_y_deg'][1]
    kx = int(round(sep_deg / dlon))
    ky = int(round(sep_deg / dlat))
    r = 0.5 * (cur['cx'][kx] + cur['cy'][ky])
    r = np.where(cur['valid'], r, np.nan).astype(np.float32)
    return r, cur['sep_x_deg'][kx], cur['sep_y_deg'][ky]


def band_autocorr(cur, lat, bands):
    """
    Latitude-band-averaged zonal & meridional autocorrelation curves.

    For each (label, lo, hi) band the correlation curves are averaged over all points
    with lo <= |lat| < hi. Returns {label: dict(sep_z, r_z, sep_m, r_m)} with
    separations in degrees and dimensionless correlations.
    """
    alat = np.abs(np.asarray(lat, float))
    out = {}
    for label, lo, hi in bands:
        rows = (alat >= lo) & (alat < hi)
        if not rows.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r_z = np.nanmean(cur['cx'][:, rows, :], axis=(1, 2))
            r_m = np.nanmean(cur['cy'][:, rows, :], axis=(1, 2))
        out[label] = dict(sep_z=cur['sep_x_deg'], r_z=r_z,
                          sep_m=cur['sep_y_deg'], r_m=r_m)
    return out


def decorrelation_scale(field, lon, lat, max_lag_km=600.0, thresh=1.0 / np.e):
    """
    Isotropic spatial decorrelation length (km) at each grid point of a field.

    Thin wrapper over autocorr_curves + decorr_scale_from_curves; see those for the
    method. Returns (L, Lx, Ly) as (y, x) float32 km arrays (NaN where undefined).
    """
    cur = autocorr_curves(field, lon, lat, max_lag_km=max_lag_km)
    return decorr_scale_from_curves(cur, thresh=thresh)


def point_autocorr(field, lon, lat, anchor_lon, anchor_lat, max_lag_km=600.0):
    """
    Zonal and meridional spatial autocorrelation function anchored at ONE point.

    Same temporal-anomaly Pearson correlation as autocorr_curves (anomalies
    normalized to unit time-variance, forward/backward lags averaged), but evaluated
    only for the nearest grid point to (anchor_lat, anchor_lon) -- the correlation of
    that point's time series with its neighbours along the row (zonal) and column
    (meridional). No decorrelation cutoff is applied; this is the raw curve.

    field : (time, y, x).  Returns dict(sep_x_deg, r_x, sep_y_deg, r_y, lon0, lat0).
    """
    f = np.asarray(field, np.float32)
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    ix = int(np.argmin(np.abs(lon - anchor_lon)))
    iy = int(np.argmin(np.abs(lat - anchor_lat)))

    std = np.where(f.std(0) == 0, np.nan, f.std(0))
    an = np.nan_to_num((f - f.mean(0)) / std)      # unit-variance anomalies
    T = f.shape[0]
    a0 = an[:, iy, ix]                              # anchor series (unit variance)

    dlon = _grid_spacing_deg(lon); dlat = _grid_spacing_deg(lat)
    dx_km = dlon * _KM_PER_DEG * np.cos(np.radians(lat[iy]))
    dy_km = dlat * _KM_PER_DEG
    nlag_x = int(np.ceil(max_lag_km / dx_km))
    nlag_y = int(np.ceil(max_lag_km / dy_km))

    def _curve(strip, i0, nlag):                   # strip: (time, n) along the axis
        n = strip.shape[1]
        r = np.full(nlag + 1, np.nan, np.float32)
        for k in range(nlag + 1):
            vals = []
            if i0 + k < n:
                vals.append(float((a0 * strip[:, i0 + k]).sum() / T))
            if i0 - k >= 0:
                vals.append(float((a0 * strip[:, i0 - k]).sum() / T))
            if vals:
                r[k] = np.mean(vals)
        return r

    r_x = _curve(an[:, iy, :], ix, nlag_x)         # along the anchor's latitude row
    r_y = _curve(an[:, :, ix], iy, nlag_y)         # along the anchor's longitude column
    return dict(sep_x_deg=np.arange(nlag_x + 1) * dlon, r_x=r_x,
                sep_y_deg=np.arange(nlag_y + 1) * dlat, r_y=r_y,
                lon0=float(lon[ix]), lat0=float(lat[iy]))


def point_corr_map(field, lon, lat, anchor_lon, anchor_lat):
    """
    Two-dimensional spatial autocorrelation anchored at ONE point: the Pearson
    correlation of that point's temporal anomaly with every other grid point.

    The 2-D generalization of point_autocorr -- instead of slicing along the anchor's
    row and column it maps r(dlon, dlat), so the anisotropy AND the orientation/tilt
    of the coherent structure are visible. r = 1 at the anchor by construction.

    field : (time, y, x).  Returns (r2d (y, x) float32, lon0, lat0).
    """
    f = np.asarray(field, np.float32)
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    ix = int(np.argmin(np.abs(lon - anchor_lon)))
    iy = int(np.argmin(np.abs(lat - anchor_lat)))

    sd = f.std(0)
    valid = np.isfinite(sd) & (sd != 0)
    an = np.nan_to_num((f - f.mean(0)) / np.where(valid, sd, np.nan))
    T = f.shape[0]
    # both series carry unit time-variance, so the time-mean product IS Pearson r
    r = np.tensordot(an[:, iy, ix], an, axes=(0, 0)) / T
    r = np.where(valid, r, np.nan).astype(np.float32)
    return r, float(lon[ix]), float(lat[iy])


# ---------------------------------------------------------------------------
# Footprint plane-fit error maps  (array-design diagnostic)
# ---------------------------------------------------------------------------
# For a candidate glider footprint of a given shape and size, map the error the
# plane-fit w estimator makes across the whole domain. The estimator (sample ->
# plane fit -> divergence -> depth integrate) and the truth (area-mean divergence)
# are both LINEAR in the velocity field, and time-averaging commutes through them,
# so the error in the 3-month / depth-mean w equals the error evaluated on the
# time-and-depth-mean field. These maps therefore operate on the mean U, V
# (means['U'], means['V'] from run_domain_maps' cache) -- no per-snapshot loop.
#
# w at the base of the sampled layer = H * depth-mean divergence (depth-averaging
# commutes with the horizontal divergence), so we map
#     w_err = H * (planefit_divergence - true_area_mean_divergence)
# Tier 2 uses the discrete glider stencil; Tier 1 fills the footprint (best case,
# dense sampling -> the intrinsic aliasing floor of a footprint of that size).

FOOTPRINT_SHAPES = ('hexagon', 'square', 'square4', 'diamond')


def footprint_offsets(shape, width_deg, height_deg):
    """
    Glider (lat, lon) offsets (deg, relative to the array centre) for a footprint
    inscribed in a width_deg (zonal) x height_deg (meridional) box.

    hexagon : 6 gliders -- N/S vertices on the centre meridian, 4 side gliders at
              +/-width/2 lon and +/-height/4 lat (pointy N-S). Matches
              experiment_1/generate_configs.py:_equator_hex_cell, where the N/S
              vertices land on the mooring line. Regular iff width = 0.866*height.
    square  : 6 gliders -- 4 box corners + the 2 E/W edge midpoints (matches the
              existing rectangle configs).
    square4 : 4 gliders -- the 4 box corners only (fair 4-glider peer of diamond).
    diamond : 4 gliders -- the N/S/E/W tips (edge midpoints of the box).
    """
    w, h = width_deg / 2.0, height_deg / 2.0
    if shape == 'hexagon':
        return [(h, 0.0), (h / 2, w), (-h / 2, w),
                (-h, 0.0), (-h / 2, -w), (h / 2, -w)]
    if shape == 'square':
        return [(h, w), (h, -w), (-h, -w), (-h, w), (0.0, w), (0.0, -w)]
    if shape == 'square4':
        return [(h, w), (h, -w), (-h, -w), (-h, w)]
    if shape == 'diamond':
        return [(h, 0.0), (0.0, w), (-h, 0.0), (0.0, -w)]
    raise ValueError(f'unknown shape {shape!r}; choose from {FOOTPRINT_SHAPES}')


def footprint_outline(shape, width_deg, height_deg):
    """Closed polygon vertices (lat, lon) tracing a footprint's boundary, for the
    convex-hull area (truth averaging) and the filled-fit region (Tier 1)."""
    w, h = width_deg / 2.0, height_deg / 2.0
    if shape == 'hexagon':
        return [(h, 0.0), (h / 2, w), (-h / 2, w),
                (-h, 0.0), (-h / 2, -w), (h / 2, -w)]
    if shape in ('square', 'square4'):
        return [(h, w), (h, -w), (-h, -w), (-h, w)]
    if shape == 'diamond':
        return [(h, 0.0), (0.0, w), (-h, 0.0), (0.0, -w)]
    raise ValueError(f'unknown shape {shape!r}; choose from {FOOTPRINT_SHAPES}')


def colocate_uv(meanU, meanV):
    """
    Interpolate the staggered mean U (YC, XG) and V (YG, XC) onto the common tracer
    grid (YC, XC). Returns (U, V, lon, lat) with U, V as (nlat, nlon) float arrays.
    """
    lon = np.asarray(meanV['XC'].values, float)   # V already on XC
    lat = np.asarray(meanU['YC'].values, float)   # U already on YC
    U = meanU.interp(XG=xr.DataArray(lon, dims='XC', coords={'XC': lon})).values
    V = meanV.interp(YG=xr.DataArray(lat, dims='YC', coords={'YC': lat})).values
    return np.asarray(U, float), np.asarray(V, float), lon, lat


def _planefit_slope_weights(offsets, lat_deg):
    """(wx, wy) weight vectors s.t. du/dx = wx . u_samples, dv/dy = wy . v_samples
    for a plane fit u = a + b*x + c*y over `offsets`, at latitude `lat_deg`.
    Longitude offsets are scaled to metres with cos(lat); latitude with a constant."""
    offs = np.asarray(offsets, float)                 # (N, 2) = (dlat, dlon)
    deg_to_m = _KM_PER_DEG * 1000.0
    x = offs[:, 1] * np.cos(np.radians(lat_deg)) * deg_to_m
    y = offs[:, 0] * deg_to_m
    A = np.column_stack([np.ones(len(offs)), x, y])   # (N, 3)
    Ainv = np.linalg.pinv(A)                          # (3, N)
    return Ainv[1], Ainv[2]


def planefit_divergence_stencil(U, V, lon, lat, offsets):
    """
    Plane-fit horizontal divergence estimated by a discrete glider stencil centred
    at every grid point (Tier 2). U, V are (nlat, nlon) on (lat, lon).

    The stencil is fixed in offset space, so sampling glider k over all centres is
    one interpolation of the whole field onto the grid shifted by that offset.
    """
    from scipy.interpolate import RegularGridInterpolator
    rgiU = RegularGridInterpolator((lat, lon), U, bounds_error=False, fill_value=np.nan)
    rgiV = RegularGridInterpolator((lat, lon), V, bounds_error=False, fill_value=np.nan)
    LON, LAT = np.meshgrid(lon, lat)
    ny, nx = LAT.shape
    N = len(offsets)
    Us = np.empty((N, ny, nx)); Vs = np.empty((N, ny, nx))
    for k, (dlat, dlon) in enumerate(offsets):
        pts = np.stack([(LAT + dlat).ravel(), (LON + dlon).ravel()], axis=-1)
        Us[k] = rgiU(pts).reshape(ny, nx)
        Vs[k] = rgiV(pts).reshape(ny, nx)
    div = np.full((ny, nx), np.nan)
    for i, la in enumerate(lat):                      # weights depend on lat via cos
        wx, wy = _planefit_slope_weights(offsets, la)
        div[i] = wx @ Us[:, i, :] + wy @ Vs[:, i, :]
    return div


def _shape_cell_mask(shape, width_deg, height_deg, dlon, dlat):
    """Boolean footprint mask on a local (2*ry+1, 2*rx+1) cell window, plus the
    local x/y cell-index coordinate arrays (integers, centred at 0)."""
    from matplotlib.path import Path
    rx = max(int(round((width_deg / 2.0) / dlon)), 1)
    ry = max(int(round((height_deg / 2.0) / dlat)), 1)
    jx = np.arange(-rx, rx + 1)
    iy = np.arange(-ry, ry + 1)
    CX, CY = np.meshgrid(jx * dlon, iy * dlat)        # local degrees
    poly = np.array([[p[1], p[0]] for p in footprint_outline(shape, width_deg, height_deg)])
    mask = Path(poly).contains_points(
        np.column_stack([CX.ravel(), CY.ravel()])).reshape(CX.shape)
    IX, IY = np.meshgrid(jx.astype(float), iy.astype(float))
    return mask, IX, IY


def true_areamean_divergence(U, V, lon, lat, shape, width_deg, height_deg):
    """True horizontal divergence of the mean field, area-averaged over the
    footprint hull centred at every grid point (the like-for-like truth)."""
    from scipy.ndimage import correlate
    fx, _ = gradient_components(U, lon, lat)
    _, fy = gradient_components(V, lon, lat)
    div_true = fx + fy
    dlon = _grid_spacing_deg(lon); dlat = _grid_spacing_deg(lat)
    mask, _, _ = _shape_cell_mask(shape, width_deg, height_deg, dlon, dlat)
    kern = mask.astype(float) / mask.sum()
    return correlate(div_true, kern, mode='nearest')


def filled_planefit_divergence(U, V, lon, lat, shape, width_deg, height_deg):
    """
    Plane-fit divergence using EVERY grid point inside the footprint (Tier 1) at
    every centre -- the best-case, dense-sampling fit. For a footprint symmetric in
    x and y the least-squares slopes decouple, so du/dx = <x*u>/<x^2> etc., which is
    a correlation of the field with the (x*mask) / (y*mask) kernels.
    """
    from scipy.ndimage import correlate
    dlon = _grid_spacing_deg(lon); dlat = _grid_spacing_deg(lat)
    deg_to_m = _KM_PER_DEG * 1000.0
    mask, IX, IY = _shape_cell_mask(shape, width_deg, height_deg, dlon, dlat)
    m = mask.astype(float)
    Kx = IX * m; Ky = IY * m
    Sxx = float((IX ** 2 * m).sum()); Syy = float((IY ** 2 * m).sum())
    # du/d(col index) and dv/d(row index); convert index -> metres (cos(lat) per row)
    du_di = correlate(U, Kx, mode='nearest') / Sxx
    dv_dj = correlate(V, Ky, mode='nearest') / Syy
    cos = np.cos(np.radians(lat))[:, None]
    du_dx = du_di / (dlon * cos * deg_to_m)
    dv_dy = dv_dj / (dlat * deg_to_m)
    return du_dx + dv_dy


def footprint_w_error(means, shape, width_deg, height_deg, depth_m, tier=2):
    """
    Map the error (m/day) a footprint's plane fit makes in the depth-/time-mean w at
    the base of the 0..depth_m layer: w_err = depth_m * (div_est - div_true) * 86400.

    means : dict with 'U' (YC,XG) and 'V' (YG,XC) time-and-depth-mean DataArrays.
    tier  : 2 = discrete glider stencil; 1 = filled footprint (dense-sampling floor).
    Returns an (nlat, nlon) DataArray on the tracer grid (coords lon=XC, lat=YC).
    """
    U, V, lon, lat = colocate_uv(means['U'], means['V'])
    div_true = true_areamean_divergence(U, V, lon, lat, shape, width_deg, height_deg)
    if tier == 2:
        div_est = planefit_divergence_stencil(U, V, lon, lat,
                                              footprint_offsets(shape, width_deg, height_deg))
    elif tier == 1:
        div_est = filled_planefit_divergence(U, V, lon, lat, shape, width_deg, height_deg)
    else:
        raise ValueError('tier must be 1 or 2')
    w_err = depth_m * (div_est - div_true) * 86400.0
    return xr.DataArray(w_err, dims=('YC', 'XC'), coords={'YC': lat, 'XC': lon})
