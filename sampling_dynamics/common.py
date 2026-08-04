"""
Shared configuration and data loaders for the sampling_dynamics "virtual mooring"
study.

Concept
-------
Sit in ONE spot at 0degN, 140degW (=220degE) and let a long temporal sample do the
work as tidal, diurnal, wind, TIW, Kelvin-wave and MJO regimes advect past. The
"virtual mooring" is a symmetric hexagon of 6 wave gliders PLUS a TAO mooring at the
centre. We compare what this array can SAMPLE against the model truth (every grid
point inside the array hull), for vertical velocity w, KPP mixing, and the vertical
and lateral eddy transports of heat and momentum, across the symhex E-W diameters.

Model run
---------
`oct2012_3mo_dt60_AB3` (deltaT=60 s, 3-hourly output). This is the ONLY tpose24 run
that carries the KPP mixing diagnostics, so the whole study uses it for a fully
self-consistent set of fields:
  * diag_state : THETA, SALT, UVEL, VVEL, WVEL, PHIHYD, DRHODR
  * diag_surf  : ETAN, KPPhbl        (boundary-layer depth)
  * diag_kpp   : KPPviscA, KPPdiffT  (vertical viscosity, T diffusivity)
diag_state is opened lazily with osse_tools.load_model. The KPP streams have no
available_diagnostics.log in the run dir, so they are read straight from the raw MDS
records (memory-mapped, sliced to the array bounding box) and placed on the model
grid here. All three mixing quantities (nu, kappa_T, and N^2 from DRHODR) live on the
vertical interfaces Zl, co-located with WVEL.

Configurations
--------------
The symhex family from experiments/experiment_1/configs/symhex, centre +0.0degN only
(the ones that actually "sit" at 0degN,140degW), E-W diameter 0.3/0.5/0.75/1.0deg,
6 gliders each. We ADD the TAO mooring at the centre as a 7th interior sample point
(the field array would have a mooring there); the convex hull is unchanged (the
mooring is interior), so the truth footprint is the hexagon.
"""

import os
import sys
import json

import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset

# the computational core (osse_tools.py) lives in the repo root, one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import osse_tools as ot

# --------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(REPO, 'experiments', 'experiment_1', 'configs', 'symhex')
CACHE_DIR = os.path.join(HERE, 'cache')

RUN_DIR = '/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3'
DELTA_T = 60.0
ITER_STEP = 180                     # 180 * 60 s = 3 h
ITER_MAX = 129240
REF_DATE = '2012-10-01'
SPINUP_END = '2012-10-11'           # drop model spin-up before this date

# --------------------------------------------------------------------------- geometry
CENTER = (0.0, 220.0)               # (lat, lon) = 0degN, 140degW
MOORING = (0.0, 220.0)             # TAO mooring at the centre (interior sample point)
DIAMETERS = [0.3, 0.5, 0.75, 1.0]  # symhex E-W diameters (deg), centre +0.0 only

# --------------------------------------------------------------------------- depth / obs axis
MIN_DEPTH = 8                       # shallowest sampled depth (m)
MAX_DEPTH = 80                      # deepest sampled depth (m)
DZ_OBS = 2                          # obs layer thickness (m)

# depth treatments used by the distribution / summary figures
KEY_DEPTHS = [15, 30, 45, 60, 75]   # m (interior of the 8-80 m column)

# --------------------------------------------------------------------------- physical
G = 9.81
RHO0 = ot.RHO0                       # 1027 kg/m^3 (matches the model rhonil)
HFLUX = ot.HFLUX                     # rho0*cp, degC m/s -> W/m^2
SEC_PER_DAY = 86400.0

# fields sampled at every array point / hull point (obs-depth profiles)
PROFILE_VARS = ['U', 'V', 'T', 'S', 'W', 'kappaT', 'nu', 'N2']

# MITgcm diagnostics we sample, and how osse_tools should place them.  U,V,T,S,W are
# already registered in osse_tools; we ADD the KPP mixing coefficients and DRHODR
# (all on the vertical interfaces Zl, tracer-centred horizontally) so sample_fields /
# model_region can co-locate them onto the obs axis exactly like WVEL.
SAMPLE_VARS = ('UVEL', 'VVEL', 'THETA', 'SALT', 'WVEL', 'KPPviscA', 'KPPdiffT', 'DRHODR')
_STREAMS = ['diag_state', 'diag_surf', 'diag_kpp']


def _register_kpp_vars():
    """Teach osse_tools' sample_fields/model_region about the KPP + DRHODR fields."""
    ot._GRID.update({'KPPviscA': ('XC', 'YC'), 'KPPdiffT': ('XC', 'YC'),
                     'DRHODR': ('XC', 'YC')})
    ot._ZCOORD.update({'KPPviscA': 'Zl', 'KPPdiffT': 'Zl', 'DRHODR': 'Zl'})
    ot._RENAME.update({'KPPviscA': 'nu', 'KPPdiffT': 'kappaT', 'DRHODR': 'DRHODR'})


_register_kpp_vars()

# ============================================================ configuration helpers
def config_path(diam):
    """Path to the symhex centre-0 config JSON for an E-W diameter (deg)."""
    return os.path.join(CONFIG_DIR, f'symhex_d{diam}_c+0.0.json')


def config_name(diam):
    return f'symhex_d{diam}_c+0.0'


def glider_positions(diam):
    """The 6 hexagon glider positions [(lat, lon), ...] for a diameter."""
    return ot.load_positions(config_path(diam))


def array_positions(diam):
    """Glider hexagon + the centre TAO mooring (7 interior sample points)."""
    return glider_positions(diam) + [MOORING]


def mooring_index(diam):
    """Index of the mooring within array_positions (it is appended last)."""
    return len(glider_positions(diam))


def all_configs():
    """[(diam, name, glider_pos, array_pos), ...] for every studied diameter."""
    out = []
    for d in DIAMETERS:
        out.append((d, config_name(d), glider_positions(d), array_positions(d)))
    return out


# ============================================================ time / iteration axis
def _iter_times():
    """(iters, times) for the full 3-hourly record."""
    iters = np.arange(ITER_STEP, ITER_MAX + ITER_STEP, ITER_STEP)
    times = np.datetime64(REF_DATE) + (iters * DELTA_T * 1e9).astype('timedelta64[ns]')
    return iters, times


def iters_after_spinup():
    """Iteration numbers whose model time is at/after SPINUP_END (aligned to load_model)."""
    iters, times = _iter_times()
    keep = times >= np.datetime64(SPINUP_END)
    return iters[keep].tolist()


# ============================================================ model loaders
def load_state(iters=None):
    """Open the dt60 run (diag_state + diag_surf + diag_kpp), lazy, after spin-up.

    Now that available_diagnostics.log is present in the run dir, xmitgcm places
    KPPviscA / KPPdiffT / DRHODR on Zl (interfaces) and KPPhbl on the tracer grid
    automatically -- no manual coordinate attachment needed.
    """
    if iters is None:
        iters = iters_after_spinup()
    ds = open_mdsdataset(
        data_dir=RUN_DIR, grid_dir=RUN_DIR, iters=iters, prefix=_STREAMS,
        ref_date=REF_DATE, delta_t=DELTA_T,
    )
    for c in ('XC', 'YC', 'XG', 'YG', 'Z', 'Zl'):
        if c in ds.coords:
            ds[c] = ds[c].astype(float)
    return ds.where(ds != -999.0)


def load_bbox_memory(half_deg, iters=None, buf=3 / 24, max_depth=MAX_DEPTH, nz_pad=6):
    """Load the array bounding box into memory ONCE (fields + grid subset).

    Reading the small bbox columns for the whole record up front (~minutes) makes all
    downstream point/region interpolation in-memory and instant, instead of the lazy
    per-point interp re-reading full-domain slabs.  Returns an xr.Dataset restricted to
    lon/lat = CENTER +- half_deg (+buf) and depth to a little past max_depth, on all
    the staggered grids (XC/XG, YC/YG, Z/Zl), fill values already masked.
    """
    ds = load_state(iters)
    lon0, lon1 = CENTER[1] - half_deg - buf, CENTER[1] + half_deg + buf
    lat0, lat1 = CENTER[0] - half_deg - buf, CENTER[0] + half_deg + buf
    kz = int(np.argmin(np.abs(ds.Zl.values + (max_depth + 4)))) + nz_pad
    keep = ['UVEL', 'VVEL', 'THETA', 'SALT', 'WVEL', 'KPPviscA', 'KPPdiffT', 'DRHODR',
            'KPPhbl']
    sub = ds[[v for v in keep if v in ds]]
    # horizontal: label slices (increasing lon/lat); vertical: POSITIONAL (Z labels are
    # negative and decreasing, so a label slice would empty the dimension).
    hsel = dict(XC=slice(lon0, lon1), YC=slice(lat0, lat1),
                XG=slice(lon0, lon1), YG=slice(lat0, lat1))
    sub = sub.sel({k: v for k, v in hsel.items() if k in sub.dims})
    isel = {k: slice(0, kz) for k in ('Z', 'Zl') if k in sub.dims}
    sub = sub.isel(isel)
    return sub.load()


def n2_from_drhodr(drhodr):
    """Brunt-Vaisala frequency squared N^2 (s^-2) from DRHODR = d(rho)/dz (kg/m^4).

    N^2 = -g/rho0 * d(rho)/dz.  DRHODR lives on Zl (interfaces), co-located with WVEL.
    """
    return (-G / RHO0) * drhodr


def _add_n2(samp):
    """Convert a sampled DRHODR variable into N2 (s^-2); drop DRHODR."""
    if 'DRHODR' in samp:
        samp['N2'] = n2_from_drhodr(samp['DRHODR'])
        samp = samp.drop_vars('DRHODR')
    return samp


# ============================================================ array-point sampling
def sample_array(ds, positions, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, dz=DZ_OBS):
    """Sample every field at the array points -> Dataset (time, glider, obs_depth).

    Includes U, V, T, S, W, kappaT, nu, N2 (from DRHODR) on the obs axis, plus the
    surface KPP boundary-layer depth hbl (time, glider).  Density sigma0 is added.
    """
    samp = ot.sample_fields(ds, positions, vars=SAMPLE_VARS,
                            max_depth=max_depth, dz_obs=dz, min_depth=min_depth)
    samp = _add_n2(ot.add_density(samp))
    g = np.arange(len(positions))
    lat_da = xr.DataArray([p[0] for p in positions], dims='glider', coords={'glider': g})
    lon_da = xr.DataArray([p[1] for p in positions], dims='glider', coords={'glider': g})
    samp['hbl'] = ds.KPPhbl.interp(XC=lon_da, YC=lat_da).transpose('time', 'glider')
    return samp


# ============================================================ co-located hull region
def region_bbox(ds, half_deg, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, dz=DZ_OBS,
                buf=3 / 24):
    """Co-locate every field to tracer centres over the array bbox -> point cloud.

    Like osse_tools.model_region but WITHOUT the hull mask, so the (expensive)
    horizontal interpolation is done ONCE for the biggest footprint and reused for
    every diameter via select_hull().  Returns dims (time, point, obs_depth) with
    U, V, T, S, W, kappaT, nu, N2, sigma0 + a 2-D hbl (time, point); lat/lon on point.
    """
    obs_z = ot._obs_z(max_depth, dz, min_depth)
    lon0, lon1 = CENTER[1] - half_deg - buf, CENTER[1] + half_deg + buf
    lat0, lat1 = CENTER[0] - half_deg - buf, CENTER[0] + half_deg + buf
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
    hbl = ds.KPPhbl.interp(XC=xt, YC=yt)
    out['hbl'] = hbl.drop_vars([c for c in hbl.coords if c not in keep])
    reg = xr.Dataset(out).stack(point=('YC', 'XC'))
    reg = reg.assign_coords(lat=reg.YC, lon=reg.XC)
    reg = _add_n2(ot.add_density(reg))          # needs lat/lon, so after stacking
    return reg.transpose('time', 'point', 'obs_depth')


def select_hull(region, positions):
    """Subset a region_bbox point cloud to the model points inside the array hull."""
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path
    pts = np.array([[p[1], p[0]] for p in positions])       # (lon, lat)
    hull = ConvexHull(pts)
    path = Path(pts[hull.vertices])
    lon = region.lon.values
    lat = region.lat.values
    inside = path.contains_points(np.column_stack([lon, lat]))
    return region.isel(point=np.where(inside)[0])


# ============================================================ obs-depth axis helper
def obs_depths(min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, dz=DZ_OBS):
    """Obs-depth midpoints (negative, shallow first): -(min+dz/2) ... down to max."""
    return ot._obs_z(max_depth, dz, min_depth)


# ============================================================ style / labels
# quantity -> (short label, units, matplotlib color) ; American spelling, plain units
VAR_LABEL = {
    'U': ('U', 'm s$^{-1}$'),
    'V': ('V', 'm s$^{-1}$'),
    'W': ('w', 'm day$^{-1}$'),
    'T': ('T', '$^\\circ$C'),
    'S': ('S', 'g kg$^{-1}$'),
    'kappaT': (r'$\kappa_T$', 'm$^2$ s$^{-1}$'),
    'nu': (r'$\nu$', 'm$^2$ s$^{-1}$'),
    'N2': ('$N^2$', 's$^{-2}$'),
    'hbl': ('KPP BL depth', 'm'),
}

# diameter -> color (light->dark with size), shared across every figure
DIAM_COLOR = {0.3: '#a6cee3', 0.5: '#4292c6', 0.75: '#2171b5', 1.0: '#08306b'}


def diam_color(d):
    return DIAM_COLOR.get(d, '#333333')


os.makedirs(CACHE_DIR, exist_ok=True)
