#!/usr/bin/env python
"""
run_subarray_flux_maps.py — 2-D maps of the SUB-ARRAY-SCALE vertical advective
heat flux over the equatorial box, for a hexagonal array centered at each grid point.

The heat-flux analysis (run_heat_flux / heat_flux_summary) defines the sub-array
("unobservable") flux as the TRUE TOTAL minus the flux from MEAN ADVECTION:

    sub-array  =  <[w·∂ᵤT]>  −  <[w][∂ᵤT]>
                  true total     product of the footprint means

where [·] is a spatial average over the array footprint and <·> a time average — the
within-footprint spatial covariance no single footprint-average value can see. That
number is one scalar per array. This script maps it: it slides a **hexagonal footprint**
(the equator_hex1deg shape, 1° diameter) to be centered at every grid point of the
equatorial box and evaluates the sub-array flux there, as a depth-integrated heat flux
ρ₀cp∫·dz (W m⁻²) over 0–75 m — read straight from the model truth (no plane fit). The
integral runs from the surface (where w=0, so the integrand vanishes) to 75 m.

Two panels on one figure:
  1. AT EACH POINT      sub_point(x) = <w·∂ᵤT>(x) − <[w][∂ᵤT]>(x)
       the LOCAL pointwise flux minus the footprint mean-advection of the array
       centered at x (panel 2 is the footprint average of this field).
  2. OVER THE ARRAY     sub_area(x)  = <[w·∂ᵤT]>(x) − <[w][∂ᵤT]>(x)
       the footprint-averaged product minus the product of footprint means.

Footprint means are computed by convolving with a fixed normalized hexagon kernel on
the model grid (the box is loaded with a 0.5° margin so every displayed center has full
kernel coverage). ρ₀, cp match the TPOSE24 MITgcm run (osse_tools.RHO0/CP).

Split into compute (reads the model once, caches the two 2-D maps) and plot, like
run_domain_maps.py.

Run (tpose env):  python experiments/experiment_2/run_subarray_flux_maps.py [all|compute|plot]
"""
import os
import sys
import time
import pickle

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as mcm
from matplotlib.colors import Normalize
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXP1 = os.path.join(REPO, 'experiments', 'experiment_1')
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import osse_tools as ot            # noqa: E402
import run_heat_flux as rhf        # noqa: E402  (RUN_DIR, ITERS, SPINUP_END)

CACHE = '/data/SO3/edavenport/tpose24/cache/subarray_flux_maps.pkl'
OUTDIR = os.path.join(HERE, 'heat_flux_figs')

# display box (matches the context maps) and the margin needed for full footprint coverage
BOX_LON = slice(217, 223)
BOX_LAT = slice(-3, 3)
MARGIN = 0.55                                     # deg; > footprint half-width (0.5°)
LOAD_LON = slice(BOX_LON.start - MARGIN, BOX_LON.stop + MARGIN)
LOAD_LAT = slice(BOX_LAT.start - MARGIN, BOX_LAT.stop + MARGIN)

# depth sampling: obs midpoints 9..75 m (MAX_DEPTH=76 -> deepest midpoint 75 m)
MIN_DEPTH, MAX_DEPTH, DZ_OBS = 8, 76, 2
DZ = 2.0                                          # m, half-stencil for ∂T/∂z

# the three 1° array footprints (all centered at 0°N, 220°E) whose sub-array flux is mapped
SHAPES = ['hexagon', 'square', 'diamond']
SHAPE_CONFIG = {'hexagon': 'equator_hex1deg_w0.5',
                'square': 'equator_sq1deg_w0.5',
                'diamond': 'equator_1deg_w0.5'}


def _shape_kernel(config, xc, yc):
    """Normalized footprint-averaging kernel for one array config on the model grid.

    Builds the footprint (1° diameter) as a boolean mask on a local offset grid at the
    model's Δlon/Δlat, then normalizes to sum 1 so convolving a field with it returns the
    footprint spatial mean at each center point. Returns (kernel, (lat,lon) vertices)."""
    path = os.path.join(EXP1, 'configs', 'equator', f'{config}.json')
    (_, pos), = ot.load_cells(path)                   # (lat, lon) vertices, center 0°,220°E
    rel = [(p[0] - 0.0, p[1] - 220.0) for p in pos]   # positions relative to the cell center
    dx = float(np.mean(np.diff(xc)))
    dy = float(np.mean(np.diff(yc)))
    nx = int(np.ceil(0.5 / dx)) + 1
    ny = int(np.ceil(0.5 / dy)) + 1
    lon_off = np.arange(-nx, nx + 1) * dx
    lat_off = np.arange(-ny, ny + 1) * dy
    mask = ot._convex_hull_mask(lon_off, lat_off, rel)   # (nlat, nlon) inside the footprint
    return mask.astype(float) / mask.sum(), np.array(pos)


def _foot_mean(field3d, K):
    """Footprint spatial mean of a (time, YC, XC) array via convolution with kernel K."""
    return fftconvolve(field3d, K[None, :, :], mode='same', axes=(1, 2))


def compute():
    """Read the model once; return the depth-integrated sub-array-flux maps (W/m²)."""
    ds = ot.load_model(rhf.RUN_DIR, rhf.ITERS).sel(time=slice(rhf.SPINUP_END, None))
    box = ds.sel(XC=LOAD_LON, YC=LOAD_LAT)
    xc, yc = box.XC.values, box.YC.values
    print(f'box {xc.size}×{yc.size}, {box.sizes["time"]} times')
    kernels, verts = {}, {}
    for s in SHAPES:
        kernels[s], verts[s] = _shape_kernel(SHAPE_CONFIG[s], xc, yc)
        print(f'  {s} kernel {kernels[s].shape}, {int((kernels[s] > 0).sum())} points')

    obs_z = ot._obs_z(MAX_DEPTH, DZ_OBS, MIN_DEPTH).values      # negative, 9..75 m
    nz, ny, nx = obs_z.size, yc.size, xc.size
    Pbar = np.empty((nz, ny, nx))        # <w·∂ᵤT>            pointwise time-mean product
    Wbar = np.empty((nz, ny, nx))        # <w>                pointwise time-mean w
    Gbar = np.empty((nz, ny, nx))        # <∂ᵤT>              pointwise time-mean gradient
    # per-shape footprint quantities <[w·∂ᵤT]> and <[w][∂ᵤT]>
    Parea = {s: np.empty((nz, ny, nx)) for s in SHAPES}
    Pmean = {s: np.empty((nz, ny, nx)) for s in SHAPES}
    t0 = time.time()
    for iz, z in enumerate(obs_z):
        W = box.WVEL.interp(Zl=float(z)).transpose('time', 'YC', 'XC').values
        Tsh = box.THETA.interp(Z=float(z) + DZ)      # shallower (less negative)
        Tdp = box.THETA.interp(Z=float(z) - DZ)      # deeper
        g = ((Tsh - Tdp) / (2 * DZ)).transpose('time', 'YC', 'XC').values   # ∂T/∂z, °C/m
        Pbar[iz] = (W * g).mean(0)
        Wbar[iz], Gbar[iz] = W.mean(0), g.mean(0)
        for s in SHAPES:
            K = kernels[s]
            Parea[s][iz] = fftconvolve(Pbar[iz], K, mode='same')          # footprint avg (linear)
            Pmean[s][iz] = (_foot_mean(W, K) * _foot_mean(g, K)).mean(0)  # <[w][∂ᵤT]>
        print(f'  z={float(z):+.0f} m  ({time.time() - t0:.0f}s)')

    # depth-integrate 0..75 m into a heat flux ρ₀cp∫·dz (W/m²). The model fields here are
    # truth only (no plane fit), so integrate from the surface: prepend a z=0 plane where
    # w=0 (the surface condition), so every integrand (w·∂ᵤT, ⟨w⟩⟨∂ᵤT⟩, footprint terms)
    # vanishes there and the 0..9 m slab is a clean trapezoid down to the shallowest obs.
    zp = -obs_z                                       # positive depth, 9..75 m increasing
    zp_full = np.concatenate(([0.0], zp))             # add the surface
    scale = ot.RHO0 * ot.CP
    def itg(A):
        A0 = np.concatenate((np.zeros((1,) + A.shape[1:]), A), axis=0)
        return scale * np.trapz(A0, x=zp_full, axis=0)
    # sub-array flux per shape: <[w·∂ᵤT]> - <[w][∂ᵤT]> (footprint avg minus product of means)
    sub_area = {s: itg(Parea[s] - Pmean[s]) for s in SHAPES}
    # "at each point": local pointwise flux minus the hexagon footprint mean-advection
    sub_point = itg(Pbar - Pmean['hexagon'])
    # pointwise TEMPORAL (Reynolds-in-time) decomposition — no array stencil:
    #   total <w·∂ᵤT>, mean-advection <w><∂ᵤT>, eddy covariance <w'·∂ᵤT'> = total − mean-adv
    total = itg(Pbar)
    mean_adv = itg(Wbar * Gbar)
    eddy = total - mean_adv

    # crop to the display box
    fields = dict(sub_point=sub_point, total=total, mean_adv=mean_adv, eddy=eddy,
                  **{f'sub_area_{s}': sub_area[s] for s in SHAPES})
    names = list(fields)
    da = xr.DataArray(np.stack([fields[k] for k in names]), dims=('panel', 'YC', 'XC'),
                      coords={'YC': yc, 'XC': xc}).sel(XC=BOX_LON, YC=BOX_LAT)
    out = dict(lon=da.XC.values, lat=da.YC.values,
               verts=verts, depth_top=0.0, depth_bot=float(zp.max()))
    out.update({k: da.values[i] for i, k in enumerate(names)})
    with open(CACHE, 'wb') as f:
        pickle.dump(out, f)
    print(f'wrote {CACHE}  ({time.time() - t0:.0f}s)')
    return out


def _overlay_shape(ax, verts, lon0, lat0):
    """Draw the array footprint (its convex hull) centered on the box middle."""
    h = ConvexHull(verts[:, ::-1])                         # hull in (lon, lat)
    vv = np.append(h.vertices, h.vertices[0])
    ax.plot(lon0 + verts[:, 1][vv] - 220.0, lat0 + verts[:, 0][vv], 'k-', lw=1.5)


def plot(cache=None):
    if cache is None:
        with open(CACHE, 'rb') as f:
            cache = pickle.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    lon, lat = cache['lon'], cache['lat']
    lon0 = 0.5 * (lon.min() + lon.max())                   # box center for the footprint overlay
    lat0 = 0.5 * (lat.min() + lat.max())
    cmap = __import__('cmocean').cm.balance
    sub = r'$\langle[w\,\partial_z T]\rangle-\langle[w][\partial_z T]\rangle$'
    # 4 panels: the pointwise field (no shape overlay), then the sub-array flux over each
    # 1° footprint (hexagon/square/diamond) with its shape overlaid.
    panels = [('sub_point', 'at each point\n'
               r'$\langle w\,\partial_z T\rangle-\langle[w][\partial_z T]\rangle$', None)]
    panels += [(f'sub_area_{s}', f'over {s} array\n{sub}', s) for s in SHAPES]

    # the three array panels share one symmetric scale (so shapes are comparable); the
    # pointwise panel is ~2× larger, so it keeps its own scale.
    amax = max(float(np.nanpercentile(np.abs(cache[f'sub_area_{s}']), 99)) for s in SHAPES)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0), squeeze=False)
    for ax, (key, title, shape) in zip(axes.ravel(), panels):
        vals = cache[key]
        vmax = float(np.nanpercentile(np.abs(vals), 99)) if shape is None else amax
        vmin = -vmax
        print(f'  {title.splitlines()[0]:>20s}: ±{vmax:.1f} W/m²')
        ax.contourf(lon, lat, vals, levels=np.linspace(vmin, vmax, 60), cmap=cmap, extend='both')
        sm = mcm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
        cb = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.03,
                          label='sub-array heat flux  (W m$^{-2}$)',
                          ticks=mticker.MaxNLocator(6, symmetric=True))
        cb.ax.tick_params(labelsize=9)
        if shape is not None:
            _overlay_shape(ax, cache['verts'][shape], lon0, lat0)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
        ax.set_title(title, fontsize=11)
    fig.suptitle('Sub-array-scale vertical advective heat flux, 0–75 m  '
                 '(1° footprints)', fontsize=12.5, y=1.005)
    fig.tight_layout()
    fname = os.path.join(OUTDIR, 'subarray_flux_maps.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {fname}')


def plot_model(cache=None):
    """Companion figure — the pointwise TEMPORAL (Reynolds-in-time) decomposition of the
    advective heat flux, straight from the model fields (NO array stencil): total
    ⟨w·∂ᵤT⟩, mean advection ⟨w⟩⟨∂ᵤT⟩, and the eddy covariance ⟨w·∂ᵤT⟩−⟨w⟩⟨∂ᵤT⟩=⟨w′·∂ᵤT′⟩
    (the time analog of the spatial sub-array term). Total & mean-advection share one
    color scale; the eddy panel gets its own."""
    from plotting_tools import plot_domain_grid
    if cache is None:
        with open(CACHE, 'rb') as f:
            cache = pickle.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    lon, lat = cache['lon'], cache['lat']
    panels = [
        (cache['total'], lon, lat, r'total  $\langle w\,\partial_z T\rangle$'),
        (cache['mean_adv'], lon, lat, r'mean advection  $\langle w\rangle\langle\partial_z T\rangle$'),
        (cache['eddy'], lon, lat,
         r'eddy  $\langle w\,\partial_z T\rangle-\langle w\rangle\langle\partial_z T\rangle$'),
    ]
    plot_domain_grid(
        panels, cbar_label='vertical advective heat flux  (W m$^{-2}$)',
        cmap=__import__('cmocean').cm.balance,
        suptitle='Vertical advective heat flux, 0–75 m — pointwise temporal decomposition (no array)',
        fname=os.path.join(OUTDIR, 'model_flux_decomp_maps.png'),
        diverging=True, ncols=3, groups=['tot', 'tot', 'eddy'])
    print('wrote', os.path.join(OUTDIR, 'model_flux_decomp_maps.png'))


def main(mode='all'):
    cache = compute() if mode in ('all', 'compute') else None
    if mode in ('all', 'plot'):
        plot_model(cache)
        plot(cache)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
