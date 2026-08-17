"""
Shared core for experiment_3 — a CIRCLING glider array.

Concept
-------
Unlike experiment_2 (and the sampling_dynamics virtual mooring), where the gliders
sit at FIXED positions and behave like a small mooring array, here the TAO mooring
stays put at 0degN, 140degW and the wave gliders ORBIT it: N gliders spread evenly
around a circle of a given diameter, all travelling CLOCKWISE at 2 knots through the
water.  Because the through-water speed is fixed, a smaller circle is orbited faster
(period = circumference / speed), so each diameter samples its footprint at a
different angular cadence.

The question this experiment answers is: for a circling array, WHAT CIRCLE DIAMETER
and HOW MANY GLIDERS best recover the footprint-mean vertical velocity, the vertical
advective heat flux, and the field distributions?

Design choices (documented; see README)
--------------------------------------
* Circle diameter == the labelled diameter (0.3/0.5/0.75/1.0 deg); radius = diameter/2.
  The gliders travel ON this circle, so the disk they enclose is the footprint.
* N in {3, 4, 5, 6}.  At t=0 the gliders form a regular N-gon with one vertex due
  north (matching the symhex "pointy-top" convention); they all rotate together.
* TRUTH is the FIXED circular DISK of that diameter centred at 0degN,140degW — the
  area-mean over every model grid point inside the disk.  It is the same target for
  every N at a given diameter, so skill differences isolate N and the orbit, not the
  truth region.  (Centre is on the equator, where 1deg lon == 1deg lat in metres, so a
  circle in degrees is a circle in metres.)
* Model run + w method match experiment_2 exactly (transp_cons, 3-hourly, plane-fit w
  with shear extrapolated to the surface, 8-80 m, w=0 at the surface).

Everything is computed from ONE in-memory read of the array bounding box: the moving
glider samples come from xarray's diagonal advanced interpolation (a (time, glider)
position array shares the model 'time' dim, so each timestep is interpolated at that
timestep's glider positions), and the disk truth / distribution cloud come from a
tracer-centred region over the same bbox.
"""

import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import osse_tools as ot  # noqa: E402

# WVEL is not in osse_tools' horizontal-grid table (sample_fields is normally called
# with U,V,T only); register it (tracer-centred, on Zl) so the region builder and the
# moving sampler can place it like any other field.
ot._GRID.setdefault('WVEL', ('XC', 'YC'))

# --------------------------------------------------------------------------- paths
CACHE_DIR = os.path.join(HERE, 'cache')
DATA_DIR = os.path.join(HERE, 'data')

# --------------------------------------------------------------------------- model run
# Same run + iteration axis + spin-up as experiment_2 (run_experiment_2.py / run_heat_flux.py).
RUN_DIR = '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons'
ITERS = list(range(36, 26173, 36))          # 3-hourly diag_state steps
SPINUP_END = '2012-10-11'                    # drop model spin-up before this date

# --------------------------------------------------------------------------- geometry
CENTER = (0.0, 220.0)                        # (lat, lon) = 0degN, 140degW (TAO mooring)
DIAMETERS = [0.3, 0.5, 0.75, 1.0]            # circle diameters (deg), matching symhex
N_GLIDERS = [3, 4, 5, 6]                     # gliders evenly spread on the circle
GLIDER_SPEED_KTS = 2.0                       # through-water speed (knots), clockwise
DEG_TO_M = np.pi / 180.0 * 6371000.0         # metres per degree (matches ot._latlon_to_m)

# --------------------------------------------------------------------------- depth / obs axis
MIN_DEPTH = 8                                # shallowest sampled depth (m); w=0 at surface after extrapolation
MAX_DEPTH = 80
DZ_OBS = 2
KEY_DEPTHS = [25, 50, 70]                    # m; summary depths (as in experiment_2 heat flux)

# --------------------------------------------------------------------------- physical
SEC_PER_DAY = 86400.0
HFLUX = ot.HFLUX                             # rho0*cp, degC m/s -> W/m^2

SAMPLE_VARS = ('UVEL', 'VVEL', 'THETA', 'WVEL')   # everything the three analyses need


# ============================================================ configuration helpers
def config_name(n, diam):
    return f'circle_n{n}_d{diam}'


def all_configs():
    """[(n, diam, name), ...] for every (N gliders x diameter) circling array."""
    return [(n, d, config_name(n, d)) for d in DIAMETERS for n in N_GLIDERS]


def glider_speed_ms():
    return GLIDER_SPEED_KTS * 1852.0 / 3600.0    # knots -> m/s


def start_angles(n):
    """Start angles (radians) of the N gliders: a regular N-gon, one vertex due north."""
    return np.pi / 2.0 + 2.0 * np.pi * np.arange(n) / n


def orbit_omega(diam):
    """Angular speed (rad/s) of the orbit for a circle diameter (deg). Positive magnitude;
    clockwise sense is applied as a negative increment in glider_positions."""
    r_m = (diam / 2.0) * DEG_TO_M
    return glider_speed_ms() / r_m


def glider_positions(n, diam, times):
    """Time-varying glider positions for a circling array.

    Parameters
    ----------
    n : int          number of gliders (evenly spread on the circle)
    diam : float     circle diameter (deg); radius = diam/2
    times : array of np.datetime64   the model output times to place the gliders at

    Returns
    -------
    lat_da, lon_da : xr.DataArray, dims (time, glider)
        Latitude/longitude (deg) of each glider at each time, sharing the model 'time'
        coordinate so they can be used as diagonal interpolation indexers.  The array
        rotates CLOCKWISE (angle decreases with time) at 2 kt; at t=times[0] it is a
        regular N-gon with a vertex due north.
    """
    times = np.asarray(times)
    t_sec = (times - times[0]) / np.timedelta64(1, 's')     # seconds since first sample
    r_deg = diam / 2.0
    omega = orbit_omega(diam)                               # rad/s
    theta0 = start_angles(n)                                # (glider,)
    # clockwise: angle decreases with time.  theta[time, glider]
    theta = theta0[None, :] - omega * t_sec[:, None]
    lat = CENTER[0] + r_deg * np.sin(theta)                 # equator: 1deg lat == 1deg lon in metres
    lon = CENTER[1] + r_deg * np.cos(theta)
    g = np.arange(n)
    coords = {'time': times, 'glider': g}
    lat_da = xr.DataArray(lat, dims=('time', 'glider'), coords=coords)
    lon_da = xr.DataArray(lon, dims=('time', 'glider'), coords=coords)
    return lat_da, lon_da


# ============================================================ model loader (bbox in memory)
def _max_half_deg(buf=3 / 24):
    return max(DIAMETERS) / 2.0 + buf


def load_bbox_memory(iters=None, buf=3 / 24, nz_pad=6):
    """Read the array bounding box for the whole record into memory ONCE.

    The bbox spans CENTER +- (max radius + buf) horizontally (on both the tracer XC/YC
    and velocity XG/YG grids) and a little past MAX_DEPTH vertically.  Returns an
    in-memory xr.Dataset with UVEL/VVEL/WVEL/THETA, fill values masked.  All downstream
    interpolation (moving glider points + tracer-centred region) is then in memory.
    """
    if iters is None:
        iters = ITERS
    ds = ot.load_model(RUN_DIR, iters).sel(time=slice(SPINUP_END, None))
    half = _max_half_deg(buf)
    lon0, lon1 = CENTER[1] - half, CENTER[1] + half
    lat0, lat1 = CENTER[0] - half, CENTER[0] + half
    kz = int(np.argmin(np.abs(ds.Z.values + (MAX_DEPTH + 4)))) + nz_pad
    keep = [v for v in SAMPLE_VARS if v in ds]
    sub = ds[keep]
    hsel = dict(XC=slice(lon0, lon1), YC=slice(lat0, lat1),
                XG=slice(lon0, lon1), YG=slice(lat0, lat1))
    sub = sub.sel({k: v for k, v in hsel.items() if k in sub.dims})
    isel = {k: slice(0, kz) for k in ('Z', 'Zl') if k in sub.dims}
    sub = sub.isel(isel)
    return sub.load()


# ============================================================ moving-glider sampling
def sample_moving(ds, lat_da, lon_da, vars=('UVEL', 'VVEL', 'THETA'),
                  max_depth=MAX_DEPTH, dz_obs=DZ_OBS, min_depth=MIN_DEPTH):
    """Interpolate model fields to MOVING glider positions on the obs-depth axis.

    lat_da / lon_da are (time, glider) and share ds' 'time' coordinate, so xarray's
    advanced interpolation is DIAGONAL in time: each timestep is interpolated at that
    timestep's glider positions.  Mirrors osse_tools.sample_fields but with
    time-varying positions.  Returns dims (time, glider, obs_depth), vars renamed
    U/V/T/W, with lat/lon (time, glider) coords.
    """
    obs_z = ot._obs_z(max_depth, dz_obs, min_depth)
    out = {}
    for v in vars:
        gx, gy = ot._GRID[v]
        da = ds[v].interp({gx: lon_da, gy: lat_da, ot._ZCOORD.get(v, 'Z'): obs_z})
        out[ot._RENAME[v]] = da.transpose('time', 'glider', 'obs_depth')
    return xr.Dataset(out).assign_coords(lat=lat_da, lon=lon_da)


def compute_w_planefit_moving(uv_samples, extrapolate_to_surface=True):
    """Plane-fit w for a MOVING array (positions vary in time).

    Same estimator as osse_tools.compute_w_planefit (fit u = a + b x + c y and
    v = a + b x + c y across the gliders to get du/dx + dv/dy, integrate continuity
    downward from w=0 at the surface), but the design matrix is rebuilt at EACH
    timestep from that timestep's glider positions instead of once for a fixed array.

    uv_samples : Dataset dims (time, glider, obs_depth) with U, V and lat/lon
                 (time, glider) coords (from sample_moving).
    Returns Dataset with w_est (time, depth interfaces) and div (time, obs_depth).
    """
    if extrapolate_to_surface:
        uv_samples = ot.extrapolate_currents_to_surface(uv_samples)

    uv = uv_samples.compute()
    lat = uv.lat.values                      # (time, glider)
    lon = uv.lon.values
    U = uv['U'].values                       # (time, glider, obs_depth)
    V = uv['V'].values
    ntime, nglider, n_obs = U.shape

    # per-timestep plane fit: project that timestep's positions to a local metre plane
    # about their centroid and solve the least-squares slopes across depth in one go.
    du_dx = np.empty((ntime, n_obs))
    dv_dy = np.empty((ntime, n_obs))
    for it in range(ntime):
        x_m, y_m = ot._latlon_to_m(lat[it], lon[it])
        A = np.column_stack([np.ones(nglider), x_m, y_m])    # (N, 3)
        Ainv = np.linalg.pinv(A)                             # (3, N)
        du_dx[it] = (Ainv @ U[it])[1]                        # row 1 = d/dx of u
        dv_dy[it] = (Ainv @ V[it])[2]                        # row 2 = d/dy of v
    div_vals = du_dx + dv_dy

    obs_z = uv.obs_depth.values
    dz_obs = float(abs(obs_z[1] - obs_z[0]))
    z_top = obs_z[0] + dz_obs / 2.0                          # shallowest interface (0 m after extrapolation)
    w_z = z_top - np.arange(n_obs + 1) * dz_obs
    w_vals = np.concatenate([np.zeros((ntime, 1)),
                             np.cumsum(div_vals * dz_obs, axis=1)], axis=1)

    time_coord = uv['U'].time
    div_da = xr.DataArray(div_vals, dims=('time', 'obs_depth'),
                          coords={'time': time_coord, 'obs_depth': obs_z})
    w_est_da = xr.DataArray(w_vals, dims=('time', 'depth'),
                            coords={'time': time_coord, 'depth': w_z})
    return xr.Dataset({'w_est': w_est_da, 'div': div_da})


# ============================================================ tracer-centred region + disk truth
def region_bbox(ds, half_deg=None, buf=3 / 24, max_depth=MAX_DEPTH, dz_obs=DZ_OBS,
                min_depth=MIN_DEPTH):
    """Co-locate every field to tracer centres over a bbox -> (time, point, obs_depth).

    Like osse_tools.model_region / sampling_dynamics.region_bbox but WITHOUT any hull
    mask.  half_deg (default: largest circle radius) sets the bbox half-width; pass a
    single diameter's radius to keep the co-located cloud small (peak memory) and just
    big enough to enclose that disk.  Carries U, V, T, W with lat/lon on the point axis.
    """
    obs_z = ot._obs_z(max_depth, dz_obs, min_depth)
    half = (_max_half_deg(buf) if half_deg is None else half_deg + buf)
    lon0, lon1 = CENTER[1] - half, CENTER[1] + half
    lat0, lat1 = CENTER[0] - half, CENTER[0] + half
    xc = ds.XC.sel(XC=slice(lon0, lon1)).values
    yc = ds.YC.sel(YC=slice(lat0, lat1)).values
    xt = xr.DataArray(xc, dims='XC', coords={'XC': xc})
    yt = xr.DataArray(yc, dims='YC', coords={'YC': yc})
    keep = {'time', 'YC', 'XC', 'obs_depth'}
    out = {}
    for v in SAMPLE_VARS:
        gx, gy = ot._GRID[v]
        da = ds[v].interp({gx: xt, gy: yt, ot._ZCOORD.get(v, 'Z'): obs_z})
        da = da.drop_vars([c for c in da.coords if c not in keep])
        out[ot._RENAME[v]] = da
    reg = xr.Dataset(out).stack(point=('YC', 'XC'))
    reg = reg.assign_coords(lat=reg.YC, lon=reg.XC)
    return reg.transpose('time', 'point', 'obs_depth')


def disk_select(region, diam):
    """Subset a region point cloud to grid points inside the fixed disk of diameter `diam`.

    Distance is measured in metres about CENTER (equator, so lon/lat degrees are equal
    in metres); a point is kept if it lies within radius = diam/2 of the centre.
    """
    r_deg = diam / 2.0
    dlat = (region.lat.values - CENTER[0])
    dlon = (region.lon.values - CENTER[1]) * np.cos(np.radians(CENTER[0]))
    inside = (dlat ** 2 + dlon ** 2) <= r_deg ** 2
    return region.isel(point=np.where(inside)[0])


def disk_mean_w(ds, diam, max_depth=MAX_DEPTH, dz_obs=DZ_OBS, min_depth=MIN_DEPTH):
    """Disk-area-mean true w on the plane-fit interface depths (the target of w_est).

    Mirrors osse_tools.sample_model_w but averages over the fixed circular disk instead
    of a convex hull.  WVEL is interpolated to the interface depths z_top..-max_depth and
    averaged over every grid point within radius diam/2 of the centre.
    """
    z_top = -min_depth
    n = int((max_depth - min_depth) / dz_obs)
    w_z = z_top - np.arange(n + 1) * dz_obs
    w_z_da = xr.DataArray(w_z, dims='depth', coords={'depth': w_z})
    W = ds.WVEL.interp(Zl=w_z_da)                         # (time, depth, YC, XC)
    r_deg = diam / 2.0
    dlat = W.YC - CENTER[0]
    dlon = (W.XC - CENTER[1]) * np.cos(np.radians(CENTER[0]))
    mask = (dlat ** 2 + dlon ** 2) <= r_deg ** 2
    return W.where(mask).mean(['XC', 'YC']).transpose('time', 'depth').compute()
