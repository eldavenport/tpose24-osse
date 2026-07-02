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
_ZCOORD = {'WVEL': 'Zl'}  # default 'Z' (cell centres); WVEL on Zl (interfaces)
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
    """Equirectangular projection of lat/lon (deg) to metres about their centroid."""
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
        Sampling depth range and interval in metres. Defaults 70 and 2.
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
    The 'true' population: model fields at every grid point inside the array hull.

    Each field is interpolated to the tracer cell centres (co-locating U, V, T, S)
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
    an obs_depth dimension is extrapolated by its own top shear, keeping the dataset
    rectangular; for the w workflow this is what extends U and V.

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
        shallowest sampled depth. For an 8-to-75 m array this fills in 0-8 m before
        the plane fit.

    Returns
    -------
    xr.Dataset with:
        w_est : (time, depth)  estimated w at layer interfaces [m/s]
                               depth coordinate = [z_top, z_top-dz, ..., z_top-n*dz]
        div   : (time, obs_depth)  horizontal divergence at obs midpoints [1/s]

    Notes
    -----
    Positions are projected onto a flat plane via _latlon_to_m (flat-Earth approximation).
    w=0 is assumed at z_top, the shallowest depth in uv_samples.obs_depth after any
    surface extrapolation (the surface when extrapolate_to_surface is True).
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

def w_skill_metrics(w_est, w_model, depth_range=None):
    """
    Scalar skill of an estimated w against model-truth w, pooled over all
    (time, depth) samples. Every metric is a plain descriptive statistic — no
    thresholds or pass/fail judgements are applied.

    Parameters
    ----------
    w_est, w_model : xr.DataArray, dims (time, depth)
        Must share the same interface depth grid (as returned by
        compute_w_planefit and sample_model_w).
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
        norm_rms    rms / w_model_std (error relative to the signal); NaN if signal std is 0
        n           number of finite sample pairs used
    """
    if depth_range is not None:
        w_est   = w_est.sel(depth=slice(*depth_range))
        w_model = w_model.sel(depth=slice(*depth_range))
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
        norm_rms=rms / mod_std if mod_std and mod_std > 0 else np.nan,
        n=int(est.size),
    )


def w_skill_by_depth(w_est, w_model):
    """
    Depth-resolved skill: the same descriptive statistics as w_skill_metrics,
    computed at each depth by pooling over time. Use to show how error grows
    with integration depth within a single sampled range.

    Returns
    -------
    xr.Dataset, dim (depth)
        rms, mean_bias, corr, w_est_std, w_model_std, norm_rms
    """
    bias      = w_est - w_model
    rms       = np.sqrt((bias ** 2).mean('time'))
    est_std   = w_est.std('time')
    mod_std   = w_model.std('time')
    # correlation over time at each depth, from the time anomalies
    ea = w_est   - w_est.mean('time')
    ma = w_model - w_model.mean('time')
    corr = (ea * ma).mean('time') / (est_std * mod_std)
    return xr.Dataset(dict(
        rms=rms,
        mean_bias=bias.mean('time'),
        corr=corr,
        w_est_std=est_std,
        w_model_std=mod_std,
        norm_rms=rms / mod_std,
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


def plot_velocity_map(ds, positions, max_depth=70, time_range=None, cells=None):
    """
    Three-panel map of depth- and time-averaged U, V, W with glider positions overlaid.

    Parameters
    ----------
    ds : xr.Dataset
        From load_model.
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

    if cells is not None:
        from scipy.spatial import ConvexHull
        # Moorings sit on the array's central meridian (the mean longitude, about
        # which each array is symmetric); gliders are offset in longitude from it.
        # Dedup across cells that share points.
        all_pos = sorted({p for _, pos, _ in cells for p in pos})
        mooring_lon = float(np.mean([p[1] for p in all_pos]))
        moorings = [p for p in all_pos if abs(p[1] - mooring_lon) < 1e-6]
        gliders_ = [p for p in all_pos if abs(p[1] - mooring_lon) >= 1e-6]

    for ax, (data, xx, yy, cmap, title) in zip(axes, panels):
        vmax = float(np.nanpercentile(np.abs(data.values), 98))
        im = ax.pcolormesh(xx, yy, data.values, cmap=cmap,
                           vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label='m s⁻¹')
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
