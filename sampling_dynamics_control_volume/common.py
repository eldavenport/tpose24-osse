"""
Shared configuration, data loaders and OCV geometry for the
sampling_dynamics_control_volume study.

Concept
-------
Instead of pooling every model point x every time into one histogram (that is the
sibling sampling_dynamics_point_cloud study), we define an **observation control
volume (OCV)**: a polygonal prism whose FACE CENTRES (the gliders) lie on the circle of
diameter D -- i.e. the circle is INSCRIBED in the polygon, tangent to each face at its
midpoint, so the vertices sit outside at circumradius (D/2)/cos(pi/n) --
centred at 0degN,140degW, extruded over the 8-80 m column in dz=2 m layers. We reduce
the high-res truth to a SMALL set of space-averaged quantities first --

  * the outward-normal current and advective heat flux averaged over each lateral FACE,
  * the OCV volume-mean of each field,

-- and then, by the divergence theorem applied to continuity, vertically integrate the
net lateral flux to get the area-averaged w at every interface (and, with a heat-storage
term, the vertical heat flux). Only AFTER this spatial reduction do we look at temporal
PDFs, so spatial and temporal variability are never mixed.

The array estimate parks one glider at the CENTER of each face (the edge midpoints,
derived here from the on-circle vertices) plus a TAO mooring at the centre. Each glider's
current is the midpoint-rule estimate of its face-average outward-normal flux; the sum
around the polygon is the OCV horizontal divergence, integrated to give the array's w.

Model run
---------
`oct2012_3mo_dt60_AB3` (deltaT=60 s, 3-hourly output) -- the only tpose24 run carrying the
KPP diagnostics, so the loaders match the point-cloud study exactly.

Scope
-----
symhex only for now (E-W diameter 0.3/0.5/0.75/1.0deg, centre +0.0degN). The geometry
below is written for a general regular n-gon, so symsq/symdia (and other n-gons) drop in
by adding them to SHAPES; nothing here assumes six faces.
"""

import os
import sys
import collections

import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset

# the computational core (osse_tools.py) lives in the repo root, one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import osse_tools as ot

# --------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG_BASE = os.path.join(REPO, 'experiments', 'experiment_1', 'configs')
CACHE_ROOT = '/data/SO3/edavenport/tpose24-osse/cache'
CACHE_DIR = os.path.join(CACHE_ROOT, 'sampling_dynamics_control_volume')

RUN_DIR = '/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3'
DELTA_T = 60.0
ITER_STEP = 180                     # 180 * 60 s = 3 h
ITER_MAX = 129240
REF_DATE = '2012-10-01'
SPINUP_END = '2012-10-11'

# --------------------------------------------------------------------------- geometry
CENTER = (0.0, 220.0)               # (lat, lon) = 0degN, 140degW
MOORING = (0.0, 220.0)             # TAO mooring at the centre (interior sample point)
DIAMETERS = [0.3, 0.5, 0.75, 1.0]  # symhex E-W diameters (deg), centre +0.0 only
SHAPES = ['symhex', 'symsq', 'symdia']   # regular n-gons sampled (hexagon, square, diamond)

# --------------------------------------------------------------------------- depth / obs axis
MIN_DEPTH = 8                       # shallowest sampled depth (m)
MAX_DEPTH = 80                      # deepest sampled depth (m)
DZ_OBS = 2                          # obs layer thickness (m)

# --------------------------------------------------------------------------- physical
G = 9.81
RHO0 = ot.RHO0                       # 1027 kg/m^3 (matches the model rhonil)
HFLUX = ot.HFLUX                     # rho0*cp, degC m/s -> W/m^2
SEC_PER_DAY = 86400.0

# fields sampled on the obs axis
SAMPLE_VARS = ('UVEL', 'VVEL', 'THETA', 'SALT', 'WVEL', 'KPPviscA', 'KPPdiffT', 'DRHODR')
_STREAMS = ['diag_state', 'diag_surf', 'diag_kpp']

SHAPE_LABEL = {'symhex': 'hexagon', 'symsq': 'square', 'symdia': 'diamond'}
SHAPE_COLOR = {'symdia': '#1f77b4', 'symsq': '#d62728', 'symhex': '#2ca02c'}


def _register_kpp_vars():
    """Teach osse_tools' sample_fields/model_region about the KPP + DRHODR fields."""
    ot._GRID.update({'KPPviscA': ('XC', 'YC'), 'KPPdiffT': ('XC', 'YC'),
                     'DRHODR': ('XC', 'YC')})
    ot._ZCOORD.update({'KPPviscA': 'Zl', 'KPPdiffT': 'Zl', 'DRHODR': 'Zl'})
    ot._RENAME.update({'KPPviscA': 'nu', 'KPPdiffT': 'kappaT', 'DRHODR': 'DRHODR'})


_register_kpp_vars()

# ============================================================ configuration helpers
def config_path(diam, shape='symhex'):
    return os.path.join(CONFIG_BASE, shape, f'{shape}_d{diam}_c+0.0.json')


def config_name(diam, shape='symhex'):
    return f'{shape}_d{diam}_c+0.0'


# ============================================================ OCV geometry
# The OCV is the regular n-gon whose FACE CENTRES (gliders) sit on the circle of diameter
# D -- the circle is inscribed in the polygon (apothem = D/2), so the vertices are outside
# at circumradius (D/2)/cos(pi/n). Config JSONs place the vertices ON the circle
# (circumradius D/2) and NOT in polygon order (symhex order is N,S,NW,NE,SW,SE), so we sort
# them by azimuth about the centre and scale them outward by 1/cos(pi/n) to move the face
# centres onto the circle. All metric geometry uses osse_tools._latlon_to_m (local tangent
# plane about the centroid): x is the zonal (U) direction, y the meridional (V), in metres.

FaceGeom = collections.namedtuple(
    'FaceGeom', ['vertices', 'face_centers', 'normals', 'lengths', 'area', 'n_faces'])


def ocv_vertices(diam, shape='symhex'):
    """Polygon vertices [(lat, lon), ...] sorted CCW, scaled so the FACE CENTRES lie on
    the circle of diameter D (circle inscribed in the polygon).

    The config vertices sit on the circle (circumradius D/2); scaling them outward about
    the centre by 1/cos(pi/n) makes the polygon's apothem = D/2, so each face midpoint --
    where a glider sits -- lands on the circle, and the vertices move outside."""
    pos = ot.load_positions(config_path(diam, shape))
    lat0, lon0 = CENTER
    ang = [np.arctan2(p[0] - lat0, p[1] - lon0) for p in pos]  # atan2(dlat, dlon)
    order = np.argsort(ang)
    verts = [pos[i] for i in order]
    s = 1.0 / np.cos(np.pi / len(verts))                        # inscribe the circle
    return [(lat0 + s * (p[0] - lat0), lon0 + s * (p[1] - lon0)) for p in verts]


def face_geometry(diam, shape='symhex'):
    """OCV face geometry for a diameter: per-face centre / outward normal / length,
    plus the polygon cross-sectional area A (m^2).

    Returns a FaceGeom with
      vertices     : [(lat, lon)] CCW,
      face_centers : [(lat, lon)] edge midpoints (where the gliders sit),
      normals      : (n_faces, 2) outward UNIT normals (nx=zonal, ny=meridional),
      lengths      : (n_faces,) edge lengths in metres,
      area         : polygon area in m^2 (shoelace on the metre projection),
      n_faces      : number of faces.
    """
    verts = ocv_vertices(diam, shape)
    lats = np.array([v[0] for v in verts])
    lons = np.array([v[1] for v in verts])
    x, y = ot._latlon_to_m(lats, lons)             # metre projection about the centroid
    n = len(verts)

    face_centers, normals, lengths = [], [], []
    for i in range(n):
        j = (i + 1) % n
        # edge i->j in metres and in lat/lon
        ex, ey = x[j] - x[i], y[j] - y[i]
        L = float(np.hypot(ex, ey))
        # outward normal candidate: rotate edge by -90 deg -> (ey, -ex); flip to point away
        # from the centroid (~origin of the projection)
        nx, ny = ey, -ex
        mx, my = 0.5 * (x[i] + x[j]), 0.5 * (y[i] + y[j])   # face midpoint in metres
        if nx * mx + ny * my < 0:
            nx, ny = -nx, -ny
        norm = np.hypot(nx, ny)
        normals.append((nx / norm, ny / norm))
        lengths.append(L)
        face_centers.append((0.5 * (lats[i] + lats[j]), 0.5 * (lons[i] + lons[j])))

    # shoelace area on the metre projection
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return FaceGeom(verts, face_centers, np.array(normals), np.array(lengths),
                    float(area), n)


def glider_positions(diam, shape='symhex'):
    """The obs platforms: one glider at each FACE CENTRE (edge midpoint)."""
    return list(face_geometry(diam, shape).face_centers)


def array_positions(diam, shape='symhex'):
    """Face-centre gliders + the centre TAO mooring (interior sample point)."""
    return glider_positions(diam, shape) + [MOORING]


def mooring_index(diam, shape='symhex'):
    return len(glider_positions(diam, shape))


def edge_sample_positions(diam, shape='symhex', n_per_face=None, grid_deg=None):
    """Dense (lat, lon) sample points along every face, for the TRUTH face-average.

    Returns (positions, n_faces, n_per_face): positions is a flat list ordered
    face-major (all points of face 0, then face 1, ...), so a single sample_fields call
    reshapes to (n_faces, n_per_face). n_per_face is chosen so the along-edge spacing is
    <= grid_deg/2 (half the native grid spacing) unless given explicitly.
    """
    verts = ocv_vertices(diam, shape)
    n = len(verts)
    if n_per_face is None:
        # longest edge in degrees sets a common n_per_face for a rectangular reshape
        seg = [np.hypot(verts[(i + 1) % n][0] - verts[i][0],
                        verts[(i + 1) % n][1] - verts[i][1]) for i in range(n)]
        step = (grid_deg or 1 / 24) / 2.0
        n_per_face = int(max(3, np.ceil(max(seg) / step)))
    pos = []
    for i in range(n):
        j = (i + 1) % n
        la = np.linspace(verts[i][0], verts[j][0], n_per_face)
        lo = np.linspace(verts[i][1], verts[j][1], n_per_face)
        pos.extend(list(zip(la, lo)))
    return pos, n, n_per_face


# ============================================================ time / iteration axis
def _iter_times():
    iters = np.arange(ITER_STEP, ITER_MAX + ITER_STEP, ITER_STEP)
    times = np.datetime64(REF_DATE) + (iters * DELTA_T * 1e9).astype('timedelta64[ns]')
    return iters, times


def iters_after_spinup():
    iters, times = _iter_times()
    keep = times >= np.datetime64(SPINUP_END)
    return iters[keep].tolist()


# ============================================================ model loaders
def load_state(iters=None):
    """Open the dt60 run (diag_state + diag_surf + diag_kpp), lazy, after spin-up."""
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
    """Load the array bounding box into memory ONCE (fields + grid subset)."""
    ds = load_state(iters)
    lon0, lon1 = CENTER[1] - half_deg - buf, CENTER[1] + half_deg + buf
    lat0, lat1 = CENTER[0] - half_deg - buf, CENTER[0] + half_deg + buf
    kz = int(np.argmin(np.abs(ds.Zl.values + (max_depth + 4)))) + nz_pad
    keep = ['UVEL', 'VVEL', 'THETA', 'SALT', 'WVEL', 'KPPviscA', 'KPPdiffT', 'DRHODR',
            'KPPhbl']
    sub = ds[[v for v in keep if v in ds]]
    hsel = dict(XC=slice(lon0, lon1), YC=slice(lat0, lat1),
                XG=slice(lon0, lon1), YG=slice(lat0, lat1))
    sub = sub.sel({k: v for k, v in hsel.items() if k in sub.dims})
    isel = {k: slice(0, kz) for k in ('Z', 'Zl') if k in sub.dims}
    sub = sub.isel(isel)
    return sub.load()


def n2_from_drhodr(drhodr):
    """N^2 (s^-2) = -g/rho0 * d(rho)/dz, DRHODR on Zl (interfaces)."""
    return (-G / RHO0) * drhodr


def _add_n2(samp):
    if 'DRHODR' in samp:
        samp['N2'] = n2_from_drhodr(samp['DRHODR'])
        samp = samp.drop_vars('DRHODR')
    return samp


def native_grid_deg(ds):
    """Native tracer grid spacing (deg) near the centre, for edge-sampling density."""
    dx = float(np.abs(np.diff(ds.XC.values)).mean())
    dy = float(np.abs(np.diff(ds.YC.values)).mean())
    return min(dx, dy)


# ============================================================ obs-point sampling
def sample_points(ds, positions, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, dz=DZ_OBS):
    """Sample U, V, T, S, W, kappaT, nu, N2 at (lat, lon) points -> (time, glider, obs_depth)."""
    samp = ot.sample_fields(ds, positions, vars=SAMPLE_VARS,
                            max_depth=max_depth, dz_obs=dz, min_depth=min_depth)
    return _add_n2(samp)


# ============================================================ co-located interior region
def region_bbox(ds, half_deg, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, dz=DZ_OBS,
                buf=3 / 24):
    """Co-locate every field to tracer centres over the bbox -> point cloud
    (time, point, obs_depth), for the OCV volume mean. Like the point-cloud study's
    region_bbox but without density (not needed here)."""
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
    reg = xr.Dataset(out).stack(point=('YC', 'XC'))
    reg = reg.assign_coords(lat=reg.YC, lon=reg.XC)
    reg = _add_n2(reg)
    return reg.transpose('time', 'point', 'obs_depth')


def select_polygon(region, verts):
    """Subset a region point cloud to model points inside the OCV polygon (convex hull
    of the vertices) -- the truth interior for the OCV volume mean."""
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path
    pts = np.array([[v[1], v[0]] for v in verts])           # (lon, lat)
    hull = ConvexHull(pts)
    path = Path(pts[hull.vertices])
    inside = path.contains_points(np.column_stack([region.lon.values, region.lat.values]))
    return region.isel(point=np.where(inside)[0])


# ============================================================ style / labels
DIAM_COLOR = {0.3: '#a6cee3', 0.5: '#4292c6', 0.75: '#2171b5', 1.0: '#08306b'}


def diam_color(d):
    return DIAM_COLOR.get(d, '#333333')


os.makedirs(CACHE_DIR, exist_ok=True)
