"""
Tests + demonstration for the footprint w-estimate: how well an array's plane fit
reproduces the TRUE area-mean vertical velocity, and how that splits into a
plane-fit-to-shape error vs a sparse-sampling error.

WHAT THE PLANE FIT DOES, AND WHAT ERROR WE MEASURE
--------------------------------------------------
An array's plane fit reports ONE slope per component -- it fits u = a + b x + c y over
its samples and reads off du/dx = b, dv/dy = c, so its divergence estimate is a single
constant, div_est = b + c. The truth it is graded against is the AREA-MEAN of the model's
real (curved) divergence over the footprint, <div_true>. The w error at depth H is

    w_err = H * (div_est - <div_true>) * 86400          [m/day].

TWO SAMPLINGS (the tier-1 / tier-2 idea from the repo experiments, renamed)
--------------------------------------------------------------------------
  DENSE-SAMPLING : a plane fit to the footprint filled at the MODEL RESOLUTION (1/24 deg)
        -- "the best a single plane fit can do given the full field under the footprint."
        Its error is the plane-fit-TO-SHAPE error: a flat plane cannot equal the area-mean
        of a curved field, even with complete information.
  EDGE-SAMPLING  : a plane fit to the actual array vertices (square4 = the 4 corners; an
        optional TAO center adds a (0,0) point that does not change the slope). This is what
        a real array measures, so its error is DENSE + the SPARSE-SAMPLING penalty.

So EDGE - DENSE is meant to isolate the cost of sampling sparsely. That decomposition is
meaningful, with one caveat that the tests make explicit: the area-mean divergence is a
BOUNDARY quantity (divergence theorem, <div> = net edge flux / area), so for a SYMMETRIC /
SEPARABLE field there is no along-edge structure for the sparse points to miss, and edge-
sampling can MATCH or even BEAT dense-sampling (a negative apparent penalty). Once the field
varies ALONG the edges (a non-separable / cross-term field) the sparse corners miss that
structure and DENSE-SAMPLING wins -- the real, positive sparsity penalty.

SIX SYNTHETIC CASES (velocity U(dlon,dlat), V by symmetry; centered at 220E/eq)
-------------------------------------------------------------------------------
  1. linear   U = S x                 -> constant divergence  -> both exact (err 0)
  2. even     U = C x^2               -> odd divergence       -> both cancel (err 0)
  3. odd      U = C x^3  (separable)  -> even divergence      -> EDGE ~ exact, DENSE errs
                                          (symmetric special case: edge beats dense)
  4. mixed    U = C(x y^2 + 0.5 x^3)  -> NON-separable        -> DENSE beats EDGE
                                          (varies along the edges: real sparsity penalty)
  5. fine     U = C sin(K x + B y)    -> NON-separable, SUB-ARRAY scale -> both err a LOT
                                          (structure smaller than the array: plane fit fails)
  6. combo    U = wave1 + wave2       -> two tilted waves at different scales -> irregular,
                                          non-uniform ("beating") field, not one constant ripple

The square truth is computed in CLOSED FORM (analytic area-mean divergence over the box);
for the hexagon / diamond arrays the truth is a fine masked average over the shape polygon.
The plane fits use osse_tools._planefit_slope_weights -- the same projection the production
pipeline (compute_w_planefit / planefit_divergence_stencil) uses -- and test_pipeline_estimator
checks that the production estimator reproduces these numbers.

FIGURES (test_figs/)
--------------------
  dense_vs_edge_error.png   : the headline -- |w err| of dense vs edge sampling for all cases.
        Symmetric cases: edge <= dense; the non-separable cases: dense < edge.
  <shape>/mechanism_<case>.png : one figure PER array shape (square / hexagon / diamond) AND
        case. Two columns -- velocity and divergence -- each with three rows: (1) the 2-D field
        with the array footprint polygon + gliders, (2) the 3-D surface (height = the value),
        (3) the dense (purple) / edge (orange) plane fits over the true field; the divergence
        column also draws the true mean (green) whose gap from the flat plane-fit IS the w error.
        The shape sets the gliders (edge sampling), the dense fill, and the truth.
  <shape>/mechanism_odd_rotation.png : the rotation-alignment demo -- SIX columns = the odd field
        separable ∥ lon/lat vs the SAME field rotated 45 deg (separable ∥ diagonal), each shown as
        a U-velocity | V-velocity | divergence block. The U/V component maps are frame-dependent
        (they look different between the two experiments even though the flow is one rigidly
        rotated field); the divergence is isotropic, so the truth is unchanged and only the plane
        fit moves. Shows that edge exactness is about the field's
        alignment with the footprint edges, not the shape: the square is exact standard / errs
        rotated, the diamond (a rotated square) is the mirror image, and the isotropic hexagon
        errs the same both ways. The diamond is drawn at diameter 1.5*sqrt2 so it is a TRUE 45
        deg rotation of the 1.5 deg square (same size) -- then its two errors are exactly the
        square's flipped, unlike the smaller box-inscribed diamond used elsewhere in the study.
  error_vs_width.png        : dense/edge |w err| vs footprint width (log-log, slope-2 guide).
  real_field_shrink.png     : on the real TPOSE24 mean field, shrinking the footprint -- error
        falls to a grid-scale floor then RISES as the plane fit degenerates (NOT -> 0).

A DOCUMENTED CAVEAT (does not affect the study): when a footprint half-width is an exact
integer number of grid cells, matplotlib's Path.contains_points drops one boundary row of
the footprint mask -> an O(dx) bias in the GRID-based truth. The synthetic tests above use a
CLOSED-FORM truth, so they are immune; report_mask_alignment_caveat() shows the artifact and
confirms it is absent at the real 1/24 deg grid.

Run with:  python test_footprint_pipeline.py
"""

import os
import sys
import pickle

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import osse_tools as ot

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'test_figs')
CACHE = '/data/SO3/edavenport/tpose24/cache/domain_maps_70m_3mo.pkl'
M_PER_DEG = ot._KM_PER_DEG * 1000.0

LON0, LAT0 = 220.0, 0.0
DEPTH_M = 70.0
SHAPE = 'square4'         # 4-corner square = the array geometry for the tests / headline figs
WIDTH = HEIGHT = 1.5      # deg (array footprint; the gliders sit at its vertices)
MODEL_DX = 1.0 / 24.0     # dense-sampling spacing = TPOSE24 model resolution

# per-array mechanism figures are drawn for these shapes (osse id -> output subfolder name)
SHAPE_DIR = {'square4': 'square', 'hexagon': 'hexagon', 'diamond': 'diamond'}
FIG_SHAPES = list(SHAPE_DIR)

# amplitudes chosen so each field peaks at ~0.25 m/s about 2 deg from center -- a fair
# like-for-like comparison of the curvatures.
S_LIN = 0.125            # linear:  U = S x
C_EVEN = 0.0625          # even:    U = C x^2
C_ODD = 0.03125          # odd:     U = C x^3
C_MIX = 0.05             # mixed:   U = C (x y^2 + 0.5 x^3)   (x=dlon, y=dlat, deg)
C_FINE = 0.04            # fine:  U = C sin(K x + B y)  -- sub-array scale, non-separable
K_FINE = 2.8 * np.pi / WIDTH   # x-wavenumber: ~1.4 half-waves across the half-width (< array)
B_FINE = 1.1 * np.pi / WIDTH   # y-phase tilt: couples x,y (non-separable) and breaks symmetry

# combo: sum of two tilted waves at different scales/orientations -> beating, irregular field
# (not a single constant-wavelength ripple). (amplitude, x-wavenumber, y-tilt) per wave:
C6A, K6A, B6A = 0.030, 2.3 * np.pi / WIDTH, 0.9 * np.pi / WIDTH
C6B, K6B, B6B = 0.025, 3.7 * np.pi / WIDTH, -1.5 * np.pi / WIDTH

CASE_ORDER = ['linear', 'even', 'odd', 'mixed', 'fine', 'combo']
CASE_LABEL = {
    'linear': '(1) linear:  U = S x\nconstant divergence',
    'even':   '(2) even:  U = C x²\nsymmetric, separable',
    'odd':    '(3) odd:  U = C x³\nsymmetric, separable',
    'mixed':  '(4) mixed:  U = C(x y² + ½ x³)\nnon-separable',
    'fine':   '(5) fine:  U = C sin(kx + by)\nsub-array scale, non-separable',
    'combo':  '(6) combo:  U = wave₁ + wave₂\ntwo scales, irregular',
}


# --------------------------------------------------- analytic fields (deg in; SI out)
def _UV(name, dlon, dlat):
    """Synthetic velocity (U, V) [m/s] as a function of longitude/latitude offset (deg)
    from the array center. dlon, dlat may be arrays."""
    x, y = np.asarray(dlon, float), np.asarray(dlat, float)
    if name == 'linear':
        return S_LIN * x, S_LIN * y
    if name == 'even':
        return C_EVEN * x**2, C_EVEN * y**2
    if name == 'odd':
        return C_ODD * x**3, C_ODD * y**3
    if name == 'mixed':
        return C_MIX * (x * y**2 + 0.5 * x**3), C_MIX * (y * x**2 + 0.5 * y**3)
    if name == 'fine':
        return (C_FINE * np.sin(K_FINE * x + B_FINE * y),
                C_FINE * np.sin(K_FINE * y + B_FINE * x))
    if name == 'combo':
        return (C6A * np.sin(K6A * x + B6A * y) + C6B * np.sin(K6B * x + B6B * y),
                C6A * np.sin(K6A * y + B6A * x) + C6B * np.sin(K6B * y + B6B * x))
    raise ValueError(f'unknown case {name!r}')


def _div_true(name, dlon, dlat):
    """True horizontal divergence du/dx + dv/dy [1/s] at (dlon, dlat) (deg). cos=1 at eq,
    so a degree derivative converts to per-meter by / M_PER_DEG."""
    x, y = np.asarray(dlon, float), np.asarray(dlat, float)
    if name == 'linear':
        return (S_LIN + S_LIN) / M_PER_DEG * np.ones_like(x + y)
    if name == 'even':
        return (2 * C_EVEN * x + 2 * C_EVEN * y) / M_PER_DEG
    if name == 'odd':
        return (3 * C_ODD * x**2 + 3 * C_ODD * y**2) / M_PER_DEG
    if name == 'mixed':
        return (C_MIX * (y**2 + 1.5 * x**2) + C_MIX * (x**2 + 1.5 * y**2)) / M_PER_DEG
    if name == 'fine':
        return (C_FINE * K_FINE * np.cos(K_FINE * x + B_FINE * y) +
                C_FINE * K_FINE * np.cos(K_FINE * y + B_FINE * x)) / M_PER_DEG
    if name == 'combo':
        return (C6A * K6A * np.cos(K6A * x + B6A * y) + C6B * K6B * np.cos(K6B * x + B6B * y) +
                C6A * K6A * np.cos(K6A * y + B6A * x) + C6B * K6B * np.cos(K6B * y + B6B * x)
                ) / M_PER_DEG
    raise ValueError(f'unknown case {name!r}')


def _areamean_div(name, hx=WIDTH / 2, hy=HEIGHT / 2):
    """CLOSED-FORM area-mean of _div_true over the square footprint [-hx,hx] x [-hy,hy]
    [1/s]. Uses <x^2> = hx^2/3 etc. This is the exact 'truth' -- no grid, no artifact."""
    mx, my = hx**2 / 3.0, hy**2 / 3.0            # <x^2>, <y^2>
    if name == 'linear':
        return 2 * S_LIN / M_PER_DEG
    if name == 'even':
        return 0.0                                # <x> = <y> = 0 over the symmetric box
    if name == 'odd':
        return 3 * C_ODD * (mx + my) / M_PER_DEG
    if name == 'mixed':
        return C_MIX * (2.5 * mx + 2.5 * my) / M_PER_DEG
    if name == 'fine':
        # closed-form <du/dx + dv/dy> over the box for U = C sin(Kx+By), V = C sin(Ky+Bx)
        return (C_FINE * (np.sin(K_FINE * hx) * np.sin(B_FINE * hy) +
                          np.sin(K_FINE * hy) * np.sin(B_FINE * hx))
                / (B_FINE * hx * hy)) / M_PER_DEG
    if name == 'combo':
        def _wave(C, k, b):   # box area-mean divergence of one tilted wave (linear -> summable)
            return C * (np.sin(k * hx) * np.sin(b * hy) + np.sin(k * hy) * np.sin(b * hx)) / (b * hx * hy)
        return (_wave(C6A, K6A, B6A) + _wave(C6B, K6B, B6B)) / M_PER_DEG
    raise ValueError(f'unknown case {name!r}')


def _edge_offsets(width=WIDTH, height=HEIGHT, center=False):
    """(dlat, dlon) offsets of the array vertices (square4 = 4 corners). An optional TAO
    center point (0,0) can be included; it does not change the plane-fit slope."""
    offs = list(ot.footprint_offsets('square4', width, height))
    return offs + [(0.0, 0.0)] if center else offs


def _dense_offsets(width=WIDTH, height=HEIGHT, dx=MODEL_DX):
    """(dlat, dlon) offsets filling the footprint at the model resolution -- the densest a
    plane fit could ever be given the field under the square footprint."""
    # floor (not round) so the sample grid stays INSIDE the footprint at any width
    nx = max(int(np.floor((width / 2) / dx + 1e-9)), 1)
    ny = max(int(np.floor((height / 2) / dx + 1e-9)), 1)
    xs = np.arange(-nx, nx + 1) * dx
    ys = np.arange(-ny, ny + 1) * dx
    return [(float(y), float(x)) for y in ys for x in xs]


def _shape_dense_offsets(shape, width=WIDTH, height=HEIGHT, dx=MODEL_DX):
    """(dlat, dlon) offsets filling a SHAPE's interior at the model resolution -- the box grid
    (for square/square4) or the box grid clipped to the shape polygon (hexagon/diamond)."""
    base = _dense_offsets(width, height, dx)
    if shape in ('square', 'square4'):
        return base
    from matplotlib.path import Path
    poly = np.array([[p[1], p[0]] for p in ot.footprint_outline(shape, width, height)])
    inside = Path(poly).contains_points(np.array([[o[1], o[0]] for o in base]))
    return [o for o, ins in zip(base, inside) if ins]


def _shape_areamean_div(name, shape, width=WIDTH, height=HEIGHT, nfine=241):
    """True area-mean divergence [1/s] over a SHAPE. Closed form for the square (exact,
    matches _areamean_div); a fine masked average over the polygon for hexagon/diamond."""
    if shape in ('square', 'square4'):
        return _areamean_div(name, width / 2, height / 2)
    from matplotlib.path import Path
    xs = np.linspace(-width / 2, width / 2, nfine)
    ys = np.linspace(-height / 2, height / 2, nfine)
    GX, GY = np.meshgrid(xs, ys)
    poly = np.array([[p[1], p[0]] for p in ot.footprint_outline(shape, width, height)])
    inside = Path(poly).contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    return float(np.mean(_div_true(name, GX, GY)[inside]))


def _planefit_div(name, offsets):
    """Plane-fit divergence [1/s] a sampling recovers: sample U,V analytically at the
    offsets and fit u=a+bx+cy, v=a+bx+cy. Uses osse_tools._planefit_slope_weights (the
    production projection) so this matches compute_w_planefit / planefit_divergence_stencil."""
    offs = np.asarray(offsets, float)                       # (N,2) = (dlat, dlon)
    U, V = _UV(name, offs[:, 1], offs[:, 0])
    wx, wy = ot._planefit_slope_weights(offsets, LAT0)
    return float(wx @ U + wy @ V)


def _werr(name, kind, width=WIDTH, height=HEIGHT, depth=DEPTH_M):
    """w error (m/day) of 'dense' or 'edge' sampling vs the closed-form area-mean truth."""
    truth = _areamean_div(name, width / 2, height / 2)
    offs = _dense_offsets(width, height) if kind == 'dense' else _edge_offsets(width, height)
    return depth * (_planefit_div(name, offs) - truth) * 86400.0


def compute_cases(width=WIDTH, height=HEIGHT):
    """For every case: the closed-form truth divergence, the dense/edge plane-fit
    divergences, and their w errors (m/day)."""
    out = {}
    for name in CASE_ORDER:
        truth = _areamean_div(name, width / 2, height / 2)
        dd = _planefit_div(name, _dense_offsets(width, height))
        ee = _planefit_div(name, _edge_offsets(width, height))
        out[name] = dict(truth=truth, dense_div=dd, edge_div=ee,
                         werr_dense=DEPTH_M * (dd - truth) * 86400.0,
                         werr_edge=DEPTH_M * (ee - truth) * 86400.0)
    return out


def print_table(res):
    print(f'\n  synthetic footprint w-error  (square4 {WIDTH:g}x{HEIGHT:g} deg, {DEPTH_M:g} m, '
          f'dense sampling at model dx={MODEL_DX:.4f} deg):')
    print(f'    {"case":8s} {"truth_div":>12s} {"dense_div":>12s} {"edge_div":>12s}'
          f' {"wERR dense":>11s} {"wERR edge":>11s}  winner')
    for name in CASE_ORDER:
        r = res[name]
        win = 'dense' if abs(r['werr_dense']) < abs(r['werr_edge']) - 1e-6 else (
              'edge' if abs(r['werr_edge']) < abs(r['werr_dense']) - 1e-6 else 'tie')
        print(f'    {name:8s} {r["truth"]:12.3e} {r["dense_div"]:12.3e} {r["edge_div"]:12.3e}'
              f' {r["werr_dense"]:+11.3f} {r["werr_edge"]:+11.3f}  {win}')


# ------------------------------------- gridded fields (for the pipeline check + caveat)
def _grid_field(name, dx=MODEL_DX, lon_lim=(217.5, 222.5), lat_lim=(-2.5, 2.5)):
    """The analytic field sampled onto a staggered mean-dict grid, for exercising the
    production pipeline (footprint_w_error / planefit_divergence_stencil)."""
    lon = np.round(np.arange(lon_lim[0], lon_lim[1] + dx / 2, dx), 8)
    lat = np.round(np.arange(lat_lim[0], lat_lim[1] + dx / 2, dx), 8)
    LON, LAT = np.meshgrid(lon, lat)
    U2d, V2d = _UV(name, LON - LON0, LAT - LAT0)
    U = xr.DataArray(U2d, dims=('YC', 'XG'), coords={'YC': lat, 'XG': lon})
    V = xr.DataArray(V2d, dims=('YG', 'XC'), coords={'YG': lat, 'XC': lon})
    return {'U': U, 'V': V}, lon, lat


# --------------------------------------------------------------------------- tests
def test_linear_and_even_exact(res):
    """Constant and odd (canceling) divergence: dense AND edge recover the area-mean to ~0."""
    for name in ('linear', 'even'):
        for kind in ('dense', 'edge'):
            e = abs(res[name][f'werr_{kind}'])
            assert e < 1e-6, f'{name} {kind} w_err={e:.2e} m/day, expected ~0'
    print('  (1) linear & even -> dense and edge both exact (w err ~ 0)  OK')


def test_odd_edge_beats_dense(res):
    """Separable cubic: the area-mean divergence is a boundary quantity, so the edge secant
    equals it exactly, while the dense interior fit is biased low (~0.3 m/day). This is the
    SYMMETRIC special case where sparse edge sampling matches/beats dense."""
    r = res['odd']
    assert abs(r['werr_edge']) < 1e-6, f'odd edge w_err {r["werr_edge"]:.2e} not ~0'
    assert abs(r['werr_dense']) > 0.05, f'odd dense w_err {r["werr_dense"]:.2e} not >> 0'
    assert abs(r['werr_edge']) < abs(r['werr_dense'])
    print('  (2) odd (separable): edge exact, dense biased -> edge beats dense  OK')


def test_mixed_dense_beats_edge(res):
    """Non-separable cross-term field: U varies ALONG the edges, so the sparse corners miss
    it and edge-sampling errs MORE than dense-sampling. This is the real sparsity penalty --
    the generic behavior once the field is not symmetric/separable."""
    r = res['mixed']
    assert abs(r['werr_dense']) > 1e-3 and abs(r['werr_edge']) > 1e-3, 'mixed errors too small'
    assert abs(r['werr_dense']) < abs(r['werr_edge']), (
        f'mixed: dense {r["werr_dense"]:.3f} should beat edge {r["werr_edge"]:.3f}')
    assert abs(r['werr_edge']) > 2 * abs(r['werr_dense']), 'mixed penalty not clearly positive'
    print('  (3) mixed (non-separable): dense beats edge (real sparsity penalty)  OK')


def test_pipeline_estimator(res):
    """The production pipeline reproduces these estimates: planefit_divergence_stencil on the
    gridded field, sampled with the edge and dense offsets, matches _planefit_div at the
    center. Ties the analytic story to the code that makes the study's maps."""
    for name in CASE_ORDER:
        means, lon, lat = _grid_field(name)
        U, V, glon, glat = ot.colocate_uv(means['U'], means['V'])
        iy = int(np.argmin(np.abs(glat - LAT0)))
        ix = int(np.argmin(np.abs(glon - LON0)))
        for kind, offs in (('edge', _edge_offsets()), ('dense', _dense_offsets())):
            de = ot.planefit_divergence_stencil(U, V, glon, glat, offs)[iy, ix]
            assert np.isclose(de, res[name][f'{kind}_div'], rtol=1e-6, atol=1e-12), (
                f'{name} {kind}: pipeline {de:.4e} != analytic {res[name][f"{kind}_div"]:.4e}')
    print('  (4) production planefit_divergence_stencil matches the analytic estimates  OK')


def report_mask_alignment_caveat():
    """Demonstrate (not assert) the footprint-mask alignment artifact in the GRID-based
    truth, and confirm it is absent at the real 1/24 deg grid. (The synthetic tests use a
    closed-form truth and are immune; this only concerns the production maps.)"""
    print('\n  mask-alignment caveat (grid-based truth only; synthetic tests use closed form):')
    for dx in (0.025, 0.035):  # 0.025 -> half/dx integer (aligned); 0.035 -> non-integer
        mk, _, _ = ot._shape_cell_mask('square', WIDTH, HEIGHT, dx, dx)
        rowsym = np.array_equal(mk, mk[::-1, :])
        aligned = abs((WIDTH / 2) / dx - round((WIDTH / 2) / dx)) < 1e-6
        print(f'    footprint half/dx={WIDTH/2/dx:6.2f}  aligned={aligned!s:5s} '
              f'mask rows-symmetric={rowsym}')
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            m = pickle.load(f)['means']
        dgrid = ot._grid_spacing_deg(m['VVEL']['XC'].values)
        allsym = True
        for shp in ('hexagon', 'square', 'square4', 'diamond'):
            for w in (0.5, 1.0, 1.5, 2.0):
                mk, _, _ = ot._shape_cell_mask(shp, w, w, dgrid, dgrid)
                allsym &= np.array_equal(mk, mk[::-1, :]) and np.array_equal(mk, mk[:, ::-1])
        print(f'    real grid dx={dgrid:.4f}: all analysis footprint masks '
              f'symmetric = {allsym}  -> study maps unaffected')


# ---------------------------------------------------------------------- figures
C_DENSE, C_EDGE, C_TRUE, C_AREA = '#7570b3', '#d95f02', '#1f77b4', '#2ca02c'


def fig_dense_vs_edge_error(res):
    """Headline: |w err| of dense- vs edge-sampling for all cases (square array). The symmetric
    cases (linear/even/odd) have edge <= dense; the non-separable cases (mixed, fine) have
    dense < edge or both large. Bars below the floor are labeled '~0'."""
    floor = 1e-4
    dense = [max(abs(res[n]['werr_dense']), floor / 3) for n in CASE_ORDER]
    edge = [max(abs(res[n]['werr_edge']), floor / 3) for n in CASE_ORDER]
    top = max(max(dense), max(edge)) * 3.5
    x = np.arange(len(CASE_ORDER)); w = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    b1 = ax.bar(x - w / 2, dense, w, color=C_DENSE, label='dense-sampling (model 1/24°, full interior)')
    b2 = ax.bar(x + w / 2, edge, w, color=C_EDGE, label='edge-sampling (array vertices)')
    ax.set_yscale('log'); ax.set_ylim(floor / 3, top)
    ax.axhline(floor, color='0.6', ls=':', lw=1)
    for bars, key in ((b1, 'werr_dense'), (b2, 'werr_edge')):
        for rect, n in zip(bars, CASE_ORDER):
            v = abs(res[n][key])
            ax.annotate('~0' if v < floor else f'{v:.2f}',
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha='center', va='bottom', fontsize=8.5,
                        xytext=(0, 2), textcoords='offset points')
    for i, n in enumerate(CASE_ORDER):
        win = 'edge' if abs(res[n]['werr_edge']) < abs(res[n]['werr_dense']) - 1e-6 else (
              'dense' if abs(res[n]['werr_dense']) < abs(res[n]['werr_edge']) - 1e-6 else '')
        if win:
            yb = max(dense[i], edge[i]) * 2.0
            ax.annotate(f'{win}\nwins', (i, yb), ha='center', va='center', fontsize=8.5,
                        color='0.25', fontstyle='italic')
    ax.set_xticks(x); ax.set_xticklabels([CASE_LABEL[n] for n in CASE_ORDER], fontsize=8)
    ax.set_ylabel('|w error| vs true area-mean (m day$^{-1}$)')
    ax.set_title('Dense- vs edge-sampling error.  Symmetric fields: edge $\\leq$ dense '
                 '(boundary quantity);\nnon-separable field: dense $<$ edge (the real sparse '
                 'penalty)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9.5)
    ax.grid(True, axis='y', which='both', alpha=0.3)
    fn = os.path.join(FIGDIR, 'dense_vs_edge_error.png')
    fig.savefig(fn, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fn


def _planefit_plane(name, offsets):
    """Full fitted plane U = a + b*x + c*y (x,y in meters) that a sampling recovers -- the
    same least-squares fit whose x/y slopes give the divergence estimate, but returning all
    three coefficients so the plane can be drawn as a surface."""
    offs = np.asarray(offsets, float)                 # (N,2) = (dlat, dlon)
    x_m, y_m = offs[:, 1] * M_PER_DEG, offs[:, 0] * M_PER_DEG
    U = _UV(name, offs[:, 1], offs[:, 0])[0]
    A = np.column_stack([np.ones_like(x_m), x_m, y_m])
    coef, *_ = np.linalg.lstsq(A, U, rcond=None)
    return coef                                       # (a, b, c)


def _cbar(ax, im, label):
    """Attach a slim colorbar to the right of a 2-D map axes."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.06)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def fig_case(name, shape):
    """One figure per test case AND array shape. Two columns -- VELOCITY (left) and DIVERGENCE
    (right) -- each with three rows:
      (1) the 2-D field (color), with the array footprint polygon and its gliders (green);
      (2) the same field as a 3-D surface (height = the value that is color in row 1);
      (3) the dense (purple) and edge (orange) plane fits over the TRUE field surface (gray).
          For VELOCITY that is all we show -- the mean velocity is irrelevant; what matters is
          how the plane captures the field/slope. For DIVERGENCE the plane fit is a single
          constant (a flat plane), and we ALSO draw the true mean (green): the gap between the
          flat plane-fit and the green true-mean plane IS the w error.
    The array `shape` (square4 / hexagon / diamond) sets the gliders (edge sampling), the dense
    interior fill, and the true area-mean. Saved to test_figs/<shape>/. No 1-D transects."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon as MplPolygon

    # shape-specific sampling and truth
    edge_off = list(ot.footprint_offsets(shape, WIDTH, HEIGHT))     # gliders / vertices
    dense_off = _shape_dense_offsets(shape)
    truth = _shape_areamean_div(name, shape)
    dense_div = _planefit_div(name, dense_off)
    edge_div = _planefit_div(name, edge_off)
    wd = DEPTH_M * (dense_div - truth) * 86400.0
    we = DEPTH_M * (edge_div - truth) * 86400.0
    ae, ad = _planefit_plane(name, edge_off), _planefit_plane(name, dense_off)
    outline = [(o[1], o[0]) for o in ot.footprint_outline(shape, WIDTH, HEIGHT)]  # (lon,lat)
    cx = np.array([o[1] for o in edge_off]); cy = np.array([o[0] for o in edge_off])

    half = WIDTH / 2
    pad = 0.15
    DS = 1e6                                              # show divergence in 1e-6 s^-1
    gx = np.linspace(-half - pad, half + pad, 141)        # 2-D map grid (padded)
    GX, GY = np.meshgrid(gx, gx)
    gg = np.linspace(-half, half, 41)                     # 3-D surface grid (footprint box)
    GXX, GYY = np.meshgrid(gg, gg)
    XM, YM = GXX * M_PER_DEG, GYY * M_PER_DEG

    U2d = _UV(name, GX, GY)[0]
    D2d = _div_true(name, GX, GY) * DS
    Us = _UV(name, GXX, GYY)[0]
    Ds = _div_true(name, GXX, GYY) * DS
    Uc = _UV(name, cx, cy)[0]; Dc = _div_true(name, cx, cy) * DS
    Pe = ae[0] + ae[1] * XM + ae[2] * YM                  # edge-fit U plane
    Pd = ad[0] + ad[1] * XM + ad[2] * YM                  # dense-fit U plane

    # taller 3-D rows and a bigger figure make the cubes larger without cropping the gliders
    fig = plt.figure(figsize=(12.5, 17.5))
    gs = fig.add_gridspec(3, 2, left=0.10, right=0.92, top=0.905, bottom=0.04,
                          height_ratios=[1.0, 1.32, 1.32], hspace=0.16, wspace=0.26)
    axV2 = fig.add_subplot(gs[0, 0]); axD2 = fig.add_subplot(gs[0, 1])
    axVa = fig.add_subplot(gs[1, 0], projection='3d')
    axDa = fig.add_subplot(gs[1, 1], projection='3d')
    axVb = fig.add_subplot(gs[2, 0], projection='3d')
    axDb = fig.add_subplot(gs[2, 1], projection='3d')

    # ---- row 1: 2-D fields (no transect line) ---------------------------------
    vmax = max(float(np.nanmax(np.abs(U2d))), 1e-12)     # rows 1 and 2 share these limits
    _cbar(axV2, axV2.pcolormesh(gx, gx, U2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                shading='auto'), 'U (m s$^{-1}$)')
    dmax = max(float(np.nanmax(np.abs(D2d))), 1e-12)
    _cbar(axD2, axD2.pcolormesh(gx, gx, D2d, cmap='PuOr_r', vmin=-dmax, vmax=dmax,
                                shading='auto'), 'div (10$^{-6}$ s$^{-1}$)')
    for ax2 in (axV2, axD2):
        ax2.add_patch(MplPolygon(outline, closed=True, fill=False, ec='k', lw=1.3))
        ax2.scatter(cx, cy, s=60, c='lime', ec='k', lw=1.0, zorder=5)
        ax2.set_xlim(gx[0], gx[-1]); ax2.set_ylim(gx[0], gx[-1]); ax2.set_aspect('equal')
        ax2.set_xlabel('lon offset (deg)'); ax2.set_ylabel('lat offset (deg)')
    axV2.set_title('velocity', fontsize=13, fontweight='bold')
    axD2.set_title('divergence', fontsize=13, fontweight='bold')

    def _style3d(a, zlabel):
        a.set_xlabel('lon (deg)', fontsize=8); a.set_ylabel('lat (deg)', fontsize=8)
        a.set_zlabel(zlabel, fontsize=8); a.tick_params(labelsize=7)
        a.view_init(elev=22, azim=-67)      # ~15 deg counter-clockwise of the default view

    # ---- row 2: bare 3-D surfaces (same color limits as row 1) ----------------
    axVa.plot_surface(GXX, GYY, Us, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      linewidth=0, antialiased=True)
    axVa.scatter(cx, cy, Uc, c='lime', edgecolor='k', s=45, depthshade=False)
    _style3d(axVa, 'U (m s$^{-1}$)')
    axDa.plot_surface(GXX, GYY, Ds, cmap='PuOr_r', vmin=-dmax, vmax=dmax,
                      linewidth=0, antialiased=True)
    axDa.scatter(cx, cy, Dc, c='lime', edgecolor='k', s=45, depthshade=False)
    _style3d(axDa, 'div (10$^{-6}$ s$^{-1}$)')

    # ---- row 3: plane fits over the true field --------------------------------
    # Velocity: the dense/edge plane fits over the TRUE velocity surface (the mean velocity is
    # not a quantity we care about -- what matters is how the plane captures the field/slope).
    axVb.plot_surface(GXX, GYY, Us, color='0.6', alpha=0.25, linewidth=0)
    axVb.plot_wireframe(GXX, GYY, Pe, color=C_EDGE, rstride=8, cstride=8, lw=2.0)
    axVb.plot_wireframe(GXX, GYY, Pd, color=C_DENSE, rstride=8, cstride=8, lw=2.0)
    axVb.scatter(cx, cy, Uc, c='lime', edgecolor='k', s=45, depthshade=False)
    _style3d(axVb, 'U (m s$^{-1}$)')

    # Divergence: same patterns, but here the plane fit is a flat constant, so the gap between
    # it and the true-mean plane (green) IS the w error -- this is the quantity we care about.
    axDb.plot_surface(GXX, GYY, Ds, color='0.6', alpha=0.25, linewidth=0)
    axDb.plot_wireframe(GXX, GYY, np.full_like(Ds, edge_div * DS), color=C_EDGE, rstride=8, cstride=8, lw=2.0)
    axDb.plot_wireframe(GXX, GYY, np.full_like(Ds, dense_div * DS), color=C_DENSE, rstride=8, cstride=8, lw=2.0)
    axDb.plot_surface(GXX, GYY, np.full_like(Ds, truth * DS), color=C_AREA, alpha=0.55, linewidth=0)
    axDb.scatter(cx, cy, Dc, c='lime', edgecolor='k', s=45, depthshade=False)
    _style3d(axDb, 'div (10$^{-6}$ s$^{-1}$)')

    edge_h = Line2D([0], [0], color=C_EDGE, lw=2, marker='s', markerfacecolor='none',
                    markersize=7, label='edge plane fit')
    dense_h = Line2D([0], [0], color=C_DENSE, lw=2, marker='s', markerfacecolor='none',
                     markersize=7, label='dense plane fit')
    true_h = Patch(fc='0.6', alpha=0.35, label='true field')
    mean_h = Patch(fc=C_AREA, alpha=0.55, label='true mean')
    glider_h = Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                      markeredgecolor='k', label='gliders')
    axVb.legend(handles=[true_h, edge_h, dense_h, glider_h], fontsize=8,
                loc='upper left', bbox_to_anchor=(-0.04, 1.03))
    axDb.legend(handles=[true_h, edge_h, dense_h, mean_h, glider_h], fontsize=8,
                loc='upper left', bbox_to_anchor=(-0.04, 1.03))

    # row labels down the left margin
    for ax, lab in [(axV2, '2-D field'), (axVa, '3-D field'), (axVb, '3-D + plane fit')]:
        p = ax.get_position()
        fig.text(0.032, p.y0 + p.height / 2, lab, rotation=90, ha='center', va='center',
                 fontweight='bold', fontsize=11)

    # case header (name + equation), plus the array shape and the actual w errors
    win = ('edge' if abs(we) < abs(wd) - 1e-6 else
           'dense' if abs(wd) < abs(we) - 1e-6 else 'tie')
    head = CASE_LABEL[name].replace('\n', '   —   ')
    tag = 'tie' if win == 'tie' else f'{win} wins'
    fig.text(0.5, 0.965, head, ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.935, f'{SHAPE_DIR[shape]} array   ·   w error:  dense {wd:+.2f} · '
             f'edge {we:+.2f} m/day   ({tag})', ha='center', va='center', fontsize=10.5)

    subdir = os.path.join(FIGDIR, SHAPE_DIR[shape])
    os.makedirs(subdir, exist_ok=True)
    fn = os.path.join(subdir, f'mechanism_{name}.png')
    fig.savefig(fn, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fn


# --------------------------------------------- rotation-alignment demo (odd field)
# The odd field U=C x^3, V=C y^3 is separable ALONG lon/lat. Rotate that SAME field 45 deg
# (with its array) and it is separable along the diagonal instead. Its divergence is the
# ISOTROPIC 3C(x^2+y^2)=3C r^2, so the TRUE area-mean over any shape is identical for both
# rotations -- only the plane-fit estimate moves. A footprint is edge-exact only when the
# field's separability axis lines up with its edges: the square is exact standard / errs
# rotated, the diamond (a rotated square) is the mirror image, and the more isotropic hexagon
# errs the SAME both ways. The point: edge exactness is about field-vs-edge alignment, not shape.
_ROT45 = np.pi / 4.0


def _odd_uv(x, y, rot):
    """Odd velocity (U, V) [m/s]. rot=False: separable along lon/lat (the standard case-3
    field). rot=True: the same field rotated 45 deg, so it is separable along the diagonal."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not rot:
        return C_ODD * x**3, C_ODD * y**3
    cs = sn = np.cos(_ROT45)
    xp, yp = x * cs + y * sn, -x * sn + y * cs        # coords in the rotated frame
    up, vp = C_ODD * xp**3, C_ODD * yp**3             # separable in the rotated frame
    return up * cs - vp * sn, up * sn + vp * cs       # rotate the vector back to lon/lat


def _odd_div_rot(x, y, rot):
    """Divergence [1/s] of _odd_uv. The trace is rotation invariant, so this is 3C(x'^2+y'^2)
    in the rotated coords -- the isotropic 3C r^2, hence IDENTICAL for rot=False/True."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not rot:
        return (3 * C_ODD * x**2 + 3 * C_ODD * y**2) / M_PER_DEG
    cs = sn = np.cos(_ROT45)
    xp, yp = x * cs + y * sn, -x * sn + y * cs
    return (3 * C_ODD * xp**2 + 3 * C_ODD * yp**2) / M_PER_DEG


def _odd_truth(shape, rot, width=WIDTH, nfine=481):
    """True area-mean divergence [1/s] over the shape polygon (fine masked average)."""
    from matplotlib.path import Path
    xs = np.linspace(-width / 2, width / 2, nfine)
    GX, GY = np.meshgrid(xs, xs)
    poly = np.array([[p[1], p[0]] for p in ot.footprint_outline(shape, width, width)])
    inside = Path(poly).contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    return float(np.mean(_odd_div_rot(GX, GY, rot)[inside]))


def _odd_planefit_div(offsets, rot):
    """Plane-fit divergence [1/s] the sampling recovers for the (rotated) odd field."""
    offs = np.asarray(offsets, float)
    U, V = _odd_uv(offs[:, 1], offs[:, 0], rot)
    wx, wy = ot._planefit_slope_weights(offsets, LAT0)
    return float(wx @ U + wy @ V)


def _odd_planefit_plane(offsets, rot, comp=0):
    """Fitted plane coefficients (a, b, c) of velocity component `comp` (0=U, 1=V) for
    drawing the plane-fit surface. The divergence estimate uses b of U and c of V."""
    offs = np.asarray(offsets, float)
    x_m, y_m = offs[:, 1] * M_PER_DEG, offs[:, 0] * M_PER_DEG
    F = _odd_uv(offs[:, 1], offs[:, 0], rot)[comp]
    A = np.column_stack([np.ones_like(x_m), x_m, y_m])
    coef, *_ = np.linalg.lstsq(A, F, rcond=None)
    return coef


def _fmt_err(v):
    """Format a w error, snapping grid-noise-level values to +0.00 (avoids a bare -0.00)."""
    return f'{0.0 if abs(v) < 5e-3 else v:+.2f}'


def fig_case_rotation(shape, width=WIDTH):
    """Rotation-alignment demo for ONE array shape. SIX columns = two odd-field experiments --
    standard (separable ∥ lon/lat) and the same field rotated 45 deg (separable ∥ diagonal) --
    each drawn as a THREE-column block (U velocity | V velocity | divergence) over three rows
    (2-D field, 3-D field, 3-D + plane fit), like fig_case but with both velocity components
    shown. The U and V component maps are frame-dependent, so they look different between the two
    experiments even though the flow is the SAME field rigidly rotated; the divergence is the
    isotropic 3C r^2 and is IDENTICAL across the experiments, so only the flat plane-fit moves.
    Edge sampling is exact only where the field's separability axis lines up with the footprint
    edges: the square is exact standard / errs rotated, the diamond (rotated square) is the mirror
    image, and the isotropic hexagon errs the same both ways. `width` sets the footprint box
    (deg); the diamond is drawn at width=1.5*sqrt2 so it is a TRUE 45 deg rotation of the 1.5 deg
    square (same size), which makes its two errors the square's flipped. Saved to
    test_figs/<shape>/mechanism_odd_rotation.png."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon as MplPolygon

    half = width / 2
    pad = 0.15
    DS = 1e6                                              # divergence shown in 1e-6 s^-1
    gx = np.linspace(-half - pad, half + pad, 141); GX, GY = np.meshgrid(gx, gx)
    gg = np.linspace(-half, half, 41); GXX, GYY = np.meshgrid(gg, gg)
    XM, YM = GXX * M_PER_DEG, GYY * M_PER_DEG
    # shared color limits across both experiments AND both velocity components
    vmax = max(float(np.nanmax(np.abs(_odd_uv(GX, GY, r)[c])))
               for r in (False, True) for c in (0, 1))
    dmax = max(float(np.nanmax(np.abs(_odd_div_rot(GX, GY, r) * DS))) for r in (False, True))

    edge_off = list(ot.footprint_offsets(shape, width, width))
    dense_off = _shape_dense_offsets(shape, width, width)
    outline = [(o[1], o[0]) for o in ot.footprint_outline(shape, width, width)]  # (lon,lat)
    cx = np.array([o[1] for o in edge_off]); cy = np.array([o[0] for o in edge_off])

    fig = plt.figure(figsize=(30.0, 16.8))
    gs = fig.add_gridspec(3, 6, left=0.045, right=0.975, top=0.885, bottom=0.035,
                          height_ratios=[1.0, 1.32, 1.32], hspace=0.16, wspace=0.32)

    def _style3d(a, zlabel):
        a.set_xlabel('lon (deg)', fontsize=8); a.set_ylabel('lat (deg)', fontsize=8)
        a.set_zlabel(zlabel, fontsize=8); a.tick_params(labelsize=7)
        a.view_init(elev=22, azim=-67)

    def _vel_column(col, rot, comp, title):
        """A velocity-component block (comp: 0=U, 1=V): 2-D map, 3-D surface, 3-D + plane fits."""
        F2 = _odd_uv(GX, GY, rot)[comp]
        Fs = _odd_uv(GXX, GYY, rot)[comp]
        Fc = _odd_uv(cx, cy, rot)[comp]
        cE = _odd_planefit_plane(edge_off, rot, comp)
        cD = _odd_planefit_plane(dense_off, rot, comp)
        Pe = cE[0] + cE[1] * XM + cE[2] * YM
        Pd = cD[0] + cD[1] * XM + cD[2] * YM
        zlab = 'U (m s$^{-1}$)' if comp == 0 else 'V (m s$^{-1}$)'
        a2 = fig.add_subplot(gs[0, col])
        aa = fig.add_subplot(gs[1, col], projection='3d')
        ab = fig.add_subplot(gs[2, col], projection='3d')
        _cbar(a2, a2.pcolormesh(gx, gx, F2, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                shading='auto'), zlab)
        a2.add_patch(MplPolygon(outline, closed=True, fill=False, ec='k', lw=1.3))
        a2.scatter(cx, cy, s=55, c='lime', ec='k', lw=1.0, zorder=5)
        a2.set_xlim(gx[0], gx[-1]); a2.set_ylim(gx[0], gx[-1]); a2.set_aspect('equal')
        a2.set_xlabel('lon offset (deg)'); a2.set_ylabel('lat offset (deg)')
        a2.set_title(title, fontsize=12, fontweight='bold')
        aa.plot_surface(GXX, GYY, Fs, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        linewidth=0, antialiased=True)
        aa.scatter(cx, cy, Fc, c='lime', edgecolor='k', s=42, depthshade=False)
        _style3d(aa, zlab)
        ab.plot_surface(GXX, GYY, Fs, color='0.6', alpha=0.25, linewidth=0)
        ab.plot_wireframe(GXX, GYY, Pe, color=C_EDGE, rstride=8, cstride=8, lw=2.0)
        ab.plot_wireframe(GXX, GYY, Pd, color=C_DENSE, rstride=8, cstride=8, lw=2.0)
        ab.scatter(cx, cy, Fc, c='lime', edgecolor='k', s=42, depthshade=False)
        _style3d(ab, zlab)
        return a2, aa, ab

    def _div_column(col, rot):
        """The divergence block, and the dense/edge w errors."""
        truth = _odd_truth(shape, rot, width)
        dense_div = _odd_planefit_div(dense_off, rot)
        edge_div = _odd_planefit_div(edge_off, rot)
        D2 = _odd_div_rot(GX, GY, rot) * DS
        Ds = _odd_div_rot(GXX, GYY, rot) * DS
        Dc = _odd_div_rot(cx, cy, rot) * DS
        a2 = fig.add_subplot(gs[0, col])
        aa = fig.add_subplot(gs[1, col], projection='3d')
        ab = fig.add_subplot(gs[2, col], projection='3d')
        _cbar(a2, a2.pcolormesh(gx, gx, D2, cmap='PuOr_r', vmin=-dmax, vmax=dmax,
                                shading='auto'), 'div (10$^{-6}$ s$^{-1}$)')
        a2.add_patch(MplPolygon(outline, closed=True, fill=False, ec='k', lw=1.3))
        a2.scatter(cx, cy, s=55, c='lime', ec='k', lw=1.0, zorder=5)
        a2.set_xlim(gx[0], gx[-1]); a2.set_ylim(gx[0], gx[-1]); a2.set_aspect('equal')
        a2.set_xlabel('lon offset (deg)'); a2.set_ylabel('lat offset (deg)')
        a2.set_title('divergence', fontsize=12, fontweight='bold')
        aa.plot_surface(GXX, GYY, Ds, cmap='PuOr_r', vmin=-dmax, vmax=dmax,
                        linewidth=0, antialiased=True)
        aa.scatter(cx, cy, Dc, c='lime', edgecolor='k', s=42, depthshade=False)
        _style3d(aa, 'div (10$^{-6}$ s$^{-1}$)')
        ab.plot_surface(GXX, GYY, Ds, color='0.6', alpha=0.25, linewidth=0)
        ab.plot_wireframe(GXX, GYY, np.full_like(Ds, edge_div * DS), color=C_EDGE,
                          rstride=8, cstride=8, lw=2.0)
        ab.plot_wireframe(GXX, GYY, np.full_like(Ds, dense_div * DS), color=C_DENSE,
                          rstride=8, cstride=8, lw=2.0)
        ab.plot_surface(GXX, GYY, np.full_like(Ds, truth * DS), color=C_AREA,
                        alpha=0.55, linewidth=0)
        ab.scatter(cx, cy, Dc, c='lime', edgecolor='k', s=42, depthshade=False)
        _style3d(ab, 'div (10$^{-6}$ s$^{-1}$)')
        wd = DEPTH_M * (dense_div - truth) * 86400.0
        we = DEPTH_M * (edge_div - truth) * 86400.0
        return (a2, aa, ab), wd, we

    def _experiment(col0, rot, label, xmid):
        vel = _vel_column(col0, rot, 0, 'U velocity')
        _vel_column(col0 + 1, rot, 1, 'V velocity')
        div_axes, wd, we = _div_column(col0 + 2, rot)
        win = ('edge' if abs(we) < abs(wd) - 1e-6 else
               'dense' if abs(wd) < abs(we) - 1e-6 else 'tie')
        fig.text(xmid, 0.955, label, ha='center', va='center', fontsize=13, fontweight='bold')
        fig.text(xmid, 0.915, f'w error:  dense {_fmt_err(wd)} · edge {_fmt_err(we)} m/day   '
                 f'({"tie" if win == "tie" else win + " wins"})',
                 ha='center', va='center', fontsize=11)
        return vel, div_axes

    vel0, _ = _experiment(0, False, 'standard odd:  U = C x³,  V = C y³\nseparable ∥ lon/lat', 0.265)
    vel3, div3 = _experiment(3, True, 'rotated odd:  same field turned 45°\nseparable ∥ diagonal', 0.735)

    # legends once: velocity on the rotated V-velocity row-3 axis, divergence on its row-3 axis
    edge_h = Line2D([0], [0], color=C_EDGE, lw=2, marker='s', markerfacecolor='none',
                    markersize=7, label='edge plane fit')
    dense_h = Line2D([0], [0], color=C_DENSE, lw=2, marker='s', markerfacecolor='none',
                     markersize=7, label='dense plane fit')
    true_h = Patch(fc='0.6', alpha=0.35, label='true field')
    mean_h = Patch(fc=C_AREA, alpha=0.55, label='true mean')
    glider_h = Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                      markeredgecolor='k', label='gliders')
    vel3[2].legend(handles=[true_h, edge_h, dense_h, glider_h], fontsize=8,
                   loc='upper left', bbox_to_anchor=(-0.04, 1.03))
    div3[2].legend(handles=[true_h, edge_h, dense_h, mean_h, glider_h], fontsize=8,
                   loc='upper left', bbox_to_anchor=(-0.04, 1.03))

    # row labels down the far-left margin (from the standard-experiment U column)
    for ax, lab in ((vel0[0], '2-D field'), (vel0[1], '3-D field'), (vel0[2], '3-D + plane fit')):
        p = ax.get_position()
        fig.text(0.014, p.y0 + p.height / 2, lab, rotation=90, ha='center', va='center',
                 fontweight='bold', fontsize=11)

    fig.add_artist(Line2D([0.508, 0.508], [0.03, 0.9], color='0.8', lw=1.3, ls='--'))
    size_note = (f'   (diamond = the 1.5° square rotated 45°: diagonal {width:.2f}°, so its two '
                 'errors are the square’s flipped)' if shape == 'diamond' else '')
    fig.text(0.5, 0.988, f'{SHAPE_DIR[shape]} array — odd field, standard vs 45° rotation:  '
             'edge exactness follows the field-to-edge alignment, not the shape' + size_note,
             ha='center', va='center', fontsize=15, fontweight='bold')

    subdir = os.path.join(FIGDIR, SHAPE_DIR[shape])
    os.makedirs(subdir, exist_ok=True)
    fn = os.path.join(subdir, 'mechanism_odd_rotation.png')
    fig.savefig(fn, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fn


def fig_error_vs_width(widths=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5)):
    """|w err| vs footprint width (log-log, slope-2 guide). For the mixed (non-separable)
    field EDGE-sampling stays above DENSE at every size (the persistent sparse penalty); the
    odd (separable) dense curve is shown for reference. Odd edge is ~0 (exact) at every width,
    so it is off-scale below and noted in the title rather than drawn as a floor line."""
    widths = np.array(widths, float)
    curves = {('mixed', 'edge'): ('--s', C_EDGE, 'mixed (non-separable): edge-sampling'),
              ('mixed', 'dense'): ('-s', C_DENSE, 'mixed (non-separable): dense-sampling'),
              ('odd', 'dense'): ('-o', '#9e9ac8', 'odd (separable): dense-sampling')}
    fig, ax = plt.subplots(figsize=(7.8, 6.0), constrained_layout=True)
    for (name, kind), (ls, col, lab) in curves.items():
        y = np.array([abs(_werr(name, kind, w, w)) for w in widths])
        ax.plot(widths, y, ls, color=col, label=lab)
    ref = np.array([abs(_werr('mixed', 'edge', w, w)) for w in widths])
    ax.plot(widths, ref[0] * (widths / widths[0])**2, ':', color='k', lw=1.4,
            label='slope 2 (width$^2$) guide')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('footprint width = height (deg)')
    ax.set_ylabel('|w error| (m day$^{-1}$)')
    ax.set_title('Error grows as curvature x width$^2$. For the non-separable field,\n'
                 'edge-sampling stays above dense at every size (odd-field edge $\\approx$ 0)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    fn = os.path.join(FIGDIR, 'error_vs_width.png')
    fig.savefig(fn, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fn


def fig_real_field_shrink():
    """Demonstration (not a strict unit test): on the real TPOSE24 3-month mean field, shrink
    the footprint toward the grid scale. The honest result is NOT error -> 0. It falls to an
    O(0.1 m/day) floor at a few grid cells (~0.2-0.4 deg), then RISES once the footprint is
    ~2 cells wide and the plane fit degenerates. On the real field the 4-corner EDGE sampling
    is consistently WORSE than DENSE (the real sparse-sampling penalty) -- the along-edge
    structure the corners miss is exactly the mixed-case effect, now on real data."""
    if not os.path.exists(CACHE):
        print(f'  SKIP real-field demo: cache not found at {CACHE}')
        return None
    with open(CACHE, 'rb') as f:
        m = pickle.load(f)['means']
    U = m['UVEL'].sel(YC=slice(-5, 5), XG=slice(208, 234))
    V = m['VVEL'].sel(YG=slice(-5, 5), XC=slice(208, 234))
    means = {'U': U, 'V': V}
    dgrid = ot._grid_spacing_deg(V['XC'].values)

    widths = np.array([2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0833])
    ev_lat, ev_lon = slice(-2, 2), slice(212, 230)
    # 'tier1'/'tier2' are the production pipeline's names for dense/edge sampling
    rows = {'dense': [], 'edge': []}
    tier = {'dense': 'tier1', 'edge': 'tier2'}
    print(f'\n  real-field footprint shrink (grid ~{dgrid:.3f} deg), '
          'median & RMS |w err| over 2S-2N, 212-230E:')
    print(f'    {"width":>7s} {"kind":6s} {"median":>11s} {"rms":>11s}  (m/day)')
    for w in widths:
        for kind in ('dense', 'edge'):
            we = ot.footprint_w_error(means, SHAPE, float(w), float(w), DEPTH_M, tier=tier[kind])
            v = np.abs(we.sel(YC=ev_lat, XC=ev_lon).values)
            med = float(np.nanmedian(v)); rms = float(np.sqrt(np.nanmean(v**2)))
            rows[kind].append((med, rms))
            print(f'    {w:7.3f} {kind:6s} {med:11.3e} {rms:11.3e}')

    fig, ax = plt.subplots(figsize=(7.8, 5.8), constrained_layout=True)
    for kind, c in (('edge', C_EDGE), ('dense', C_DENSE)):
        ax.plot(widths, [r[0] for r in rows[kind]], '-o', color=c, label=f'{kind} median')
        ax.plot(widths, [r[1] for r in rows[kind]], '--s', color=c, alpha=0.6, label=f'{kind} RMS')
    ax.axvspan(dgrid, 3 * dgrid, color='gray', alpha=0.15, label='plane fit degenerates (<~3 cells)')
    ax.axvline(dgrid, color='gray', ls=':', label=f'grid scale ~{dgrid:.3f} deg')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('footprint width = height (deg)')
    ax.set_ylabel('|w error| over 2S-2N, 212-230E (m day$^{-1}$)')
    ax.set_title('Real TPOSE24 mean field: 4-corner edge sampling is worse than dense\n'
                 '(real sparse penalty); error floors then rises as the fit degenerates (NOT -> 0)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8.5)
    fn = os.path.join(FIGDIR, 'real_field_shrink.png')
    fig.savefig(fn, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fn


# ------------------------------------------------------------------------- main
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print('Computing synthetic dense- vs edge-sampling errors (closed-form truth)...')
    res = compute_cases()
    print_table(res)

    print('\nAsserting the mechanism:')
    failed = 0
    for label, fn in [
        ('linear & even exact', lambda: test_linear_and_even_exact(res)),
        ('odd: edge beats dense', lambda: test_odd_edge_beats_dense(res)),
        ('mixed: dense beats edge', lambda: test_mixed_dense_beats_edge(res)),
        ('pipeline estimator', lambda: test_pipeline_estimator(res)),
    ]:
        try:
            fn()
        except AssertionError as e:
            print(f'  FAILED [{label}]: {e}')
            failed += 1

    report_mask_alignment_caveat()

    print('\nWriting figures to test_figs/ ...')
    print('  ', fig_dense_vs_edge_error(res))
    for shape in FIG_SHAPES:
        for name in CASE_ORDER:
            print('  ', fig_case(name, shape))
        # diamond is scaled to sqrt2*width so it is a TRUE 45 deg rotation of the square
        print('  ', fig_case_rotation(shape, WIDTH * np.sqrt(2) if shape == 'diamond' else WIDTH))
    print('  ', fig_error_vs_width())
    print('  ', fig_real_field_shrink())

    if failed:
        print(f'\n{failed} assertion group(s) failed')
        sys.exit(1)
    print('\nAll assertions passed; figures in test_figs/')


if __name__ == '__main__':
    main()
