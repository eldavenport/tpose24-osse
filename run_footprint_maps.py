"""
Footprint plane-fit error maps: how much a candidate glider footprint aliases the
depth-/time-mean vertical velocity across the domain.

For each footprint shape (hexagon, square, diamond) and size, place the footprint at
every grid point, estimate w at the base of the 0..d m layer by the same plane-fit /
divergence / integrate chain the OSSE uses, and map the error against the like-computed
true area-mean w. The footprint is an ELLIPSE of semi-axes width/2 x height/2 (a CIRCLE
when width == height, matching the symhex/symsq/symdia config convention): the shape's
vertices sit ON that ellipse and the truth is the ellipse area-mean -- so all shapes at a
given size are scored against the SAME truth. Because the estimator and truth are linear
and time-averaging commutes through them, the error in the 3-month-mean w is evaluated
directly on the time-and-depth-mean field -- so these reuse means['UVEL','VVEL'] from the
run_domain_maps cache (no model read).

Two tiers of the estimate are produced:
  Tier 2 -- the discrete glider stencil (6 points for hexagon, 4 for square/diamond),
            exactly what a real array samples. Its error bundles the field's
            nonlinearity with the penalty of sparse, specifically-placed points.
  Tier 1 -- the plane fit to EVERY grid cell inside the shape's (on-ellipse) hull. Its
            error isolates the plane-fit-to-shape (nonlinearity) part alone; tier2 -
            tier1 is the cost of sparse sampling that shape.

Two figures split by height (1deg, 2deg) per tier; 12 panels each (3 shapes x 4 widths).
Both tiers share ONE symmetric color scale (pooled across tiers) so they are directly
comparable. Full-domain and equatorial-crop views, under domain/footprint/tier1/ and
domain/footprint/tier2/. Only the shallowest layer (0-70 m, the w error at the 70 m
layer base via depth-mean-divergence continuity) and the full 3-month window are
produced.

Also `symhex_compare` (see symhex_compare): one figure comparing the plane-fit w-error of
the symmetric REGULAR hexagons at 4 diameters (0.3/0.5/0.75/1.0deg, 4 columns), rows = the
two tiers, into domain/footprint/symhex_compare/.

Usage:
    python run_footprint_maps.py
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import cmocean.cm as cmo

import osse_tools as ot
from plotting_tools import plot_domain_grid, plot_footprint_profiles

OUTDIR = os.path.join(os.path.dirname(__file__), 'domain', 'footprint')
CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'
CROP_LON = slice(217, 223)
CROP_LAT = slice(-3, 3)

TIERS = ('tier1', 'tier2')          # tier1 = dense hull fit, tier2 = glider stencil
TIER_LABEL = {'tier1': 'Tier 1', 'tier2': 'Tier 2'}

SHAPES = ('hexagon', 'square4', 'diamond')   # rows of each figure
SHAPE_TITLE = {'hexagon': 'hexagon', 'square4': 'square', 'diamond': 'diamond'}
SHAPE_COLOR = {'hexagon': '#1b9e77', 'square4': '#7570b3', 'diamond': '#e7298a'}
WIDTHS = (0.5, 1.0, 1.5, 2.0)      # zonal extent (deg), columns of each figure
HEIGHTS = (1.0, 2.0)               # meridional extent (deg), one figure each

# profile summaries: reduce the maps to "which shape is best where" line plots.
# vs-latitude   -> median |w-error| over the design longitude band, plotted vs lat
# vs-longitude  -> median |w-error| over the equatorial lat band, plotted vs lon
LON_AVG = slice(217, 223)   # design longitude band for the latitude profiles
LAT_VIEW = slice(-4, 6)     # latitude range shown (equator + TIW band)
LAT_AVG = slice(-2, 2)      # equatorial band for the longitude profiles
LON_VIEW = slice(211, 229)  # longitude range shown (edges trimmed)

PERIODS = {                        # matches run_domain_maps; only the 3-month window
    '3mo': ('', 'Oct 11-Dec 2012'),
}


def _cache_path(d, pkey):
    return os.path.join(CACHE_DIR, f'domain_maps_{d}m_{pkey}.pkl')


def _panels(das, titles):
    """(values, lon, lat, title) tuples for the full-domain and equatorial-crop views."""
    full, crop = [], []
    for da, t in zip(das, titles):
        full.append((da.values, da['XC'].values, da['YC'].values, t))
        c = da.sel(XC=CROP_LON, YC=CROP_LAT)
        crop.append((c.values, c['XC'].values, c['YC'].values, t))
    return full, crop


def _shared_vmax(das, pct=98, trim=3):
    """Pooled symmetric limit from the interior (trims `trim` border cells so edge
    artifacts of the finite-difference / convolution don't set the color scale)."""
    vals = [np.abs(da.values[trim:-trim, trim:-trim]) for da in das]
    return float(np.nanpercentile(np.concatenate([v.ravel() for v in vals]), pct))


def _profile(da, avg, view, reduce_axis, stat):
    """median/mean |w-error| of a map over the `avg` band on one axis, at each point
    of the `view` range on the other. reduce_axis: 'XC' (-> vs latitude) or 'YC'."""
    f = np.nanmedian if stat == 'median' else np.nanmean
    if reduce_axis == 'XC':
        sub = da.sel(XC=avg, YC=view)
        return f(np.abs(sub.values), axis=1), sub['YC'].values
    sub = da.sel(XC=view, YC=avg)
    return f(np.abs(sub.values), axis=0), sub['XC'].values


def _render_tier(maps, tier, d, suf, plabel, vmax):
    """Render one tier's w-error maps (two height figures) and the two 'best shape
    where' profile summaries. `maps` is keyed (shape, h, w) for this tier; `vmax` is
    the shared symmetric limit (pooled across both tiers) so the tiers are comparable."""
    shapes, labels = SHAPES, SHAPE_TITLE
    tier_dir = os.path.join(OUTDIR, tier)
    full_dir = os.path.join(tier_dir, 'full_domain')
    crop_dir = os.path.join(tier_dir, 'crop_140')
    summ_dir = os.path.join(tier_dir, 'summary')
    for dd in (full_dir, crop_dir, summ_dir):
        os.makedirs(dd, exist_ok=True)

    # --- w-error maps (no suptitle; identity is in the filename) ---------------
    for h in HEIGHTS:
        das = [maps[(shape, h, w)] for shape in shapes for w in WIDTHS]
        titles = [f'{labels[shape]}  {w:g}°×{h:g}°'
                  for shape in shapes for w in WIDTHS]
        full, crop = _panels(das, titles)
        base = f'footprint_werr_{tier}_h{h:g}_{d}m{suf}'
        kw = dict(cbar_label='w error (m day$^{-1}$)', cmap=cmo.balance,
                  diverging=True, ncols=len(WIDTHS), vlim=(-vmax, vmax), suptitle=None)
        plot_domain_grid(full, fname=os.path.join(full_dir, f'{base}.png'), **kw)
        plot_domain_grid(crop, fname=os.path.join(crop_dir, f'crop_{base}.png'), **kw)

    # --- 'best shape where' profile summaries (median rows, then mean rows) -----
    labellist = [labels[s] for s in shapes]
    colors = {labels[s]: SHAPE_COLOR[s] for s in shapes}
    rows = [('median', h) for h in HEIGHTS] + [('mean', h) for h in HEIGHTS]
    ttl = f'{TIER_LABEL[tier]} · 0–{d} m · {plabel}'

    prof_lat, prof_lon = {}, {}
    for stat in ('median', 'mean'):
        for h in HEIGHTS:
            for shape in shapes:
                for w in WIDTHS:
                    pl, lat = _profile(maps[(shape, h, w)], LON_AVG, LAT_VIEW, 'XC', stat)
                    po, lon = _profile(maps[(shape, h, w)], LAT_AVG, LON_VIEW, 'YC', stat)
                    prof_lat[(stat, labels[shape], h, w)] = pl
                    prof_lon[(stat, labels[shape], h, w)] = po
    plot_footprint_profiles(
        prof_lat, lat, rows, WIDTHS, labellist, colors, xlabel='Latitude (°N)',
        legend_title=f'best shape vs latitude — {ttl} (over 217–223°E)',
        fname=os.path.join(summ_dir, f'footprint_shape_by_lat_{tier}_{d}m{suf}.png'),
        ref_x=0.0)
    plot_footprint_profiles(
        prof_lon, lon, rows, WIDTHS, labellist, colors, xlabel='Longitude (°E)',
        legend_title=f'best shape vs longitude — {ttl} (over 2°S–2°N)',
        fname=os.path.join(summ_dir, f'footprint_shape_by_lon_{tier}_{d}m{suf}.png'),
        ref_x=220.0)


def run_config(means, d, suf, plabel):
    """All footprint diagnostics for one depth/period, for BOTH tiers: the w-error
    maps (two height figures each) plus the two 'best shape where' profile summaries.
    Tier 1 (dense hull fit) and Tier 2 (glider stencil) share ONE symmetric color scale
    pooled across both, so the sparse-sampling penalty (tier2 - tier1) reads directly."""
    # compute every (tier, shape, height, width) map once; reuse for maps and profiles
    maps = {(tier, shape, h, w): ot.footprint_w_error(means, shape, w, h, d, tier=tier)
            for tier in TIERS for h in HEIGHTS for shape in SHAPES for w in WIDTHS}
    vmax = _shared_vmax(list(maps.values()))     # pooled across tiers -> comparable
    for tier in TIERS:
        tmaps = {(shape, h, w): maps[(tier, shape, h, w)]
                 for shape in SHAPES for h in HEIGHTS for w in WIDTHS}
        _render_tier(tmaps, tier, d, suf, plabel, vmax)
    sys.stderr.write(f'  tier1+tier2 {suf or "3mo"} {d}m  vmax={vmax:.3g} m/day\n')


# --------------------------------------------------------------------------- #
# Symmetric REGULAR-hexagon comparison: one shape, four diameters (4 columns)  #
# --------------------------------------------------------------------------- #
SYMHEX_DIAMS = (0.3, 0.5, 0.75, 1.0)      # E-W diameter (deg) == 2 * lon offset
SYMHEX_DIR = os.path.join(OUTDIR, 'symhex_compare')


def regular_hex_wh(d):
    """(width, height) box for a REGULAR hexagon of circle diameter d: isotropic
    (width == height == d), so the footprint ellipse is a circle of radius d/2 and the
    6 vertices land on it -- identical to the symhex config of diameter d."""
    return d, d


def symhex_compare(means, d_m, suf, plabel):
    """Domain w-error maps comparing REGULAR hexagons at diameters 0.3/0.5/0.75/1.0°
    (4 columns), rows = the two tiers (dense hull fit vs the 6-glider stencil). One
    shared symmetric color scale across all panels; full-domain and equatorial-crop
    views into domain/footprint/symhex_compare/."""
    os.makedirs(SYMHEX_DIR, exist_ok=True)
    maps = {(tier, d): ot.footprint_w_error(means, 'hexagon', *regular_hex_wh(d),
                                            d_m, tier=tier)
            for tier in TIERS for d in SYMHEX_DIAMS}
    vmax = _shared_vmax(list(maps.values()))
    das = [maps[(tier, d)] for tier in TIERS for d in SYMHEX_DIAMS]
    titles = [f'{TIER_LABEL[tier]} · {d:g}° hexagon' for tier in TIERS for d in SYMHEX_DIAMS]
    full, crop = _panels(das, titles)
    base = f'footprint_symhex_werr_{d_m}m{suf}'
    kw = dict(cbar_label='w error (m day$^{-1}$)', cmap=cmo.balance, diverging=True,
              ncols=len(SYMHEX_DIAMS), vlim=(-vmax, vmax), suptitle=None)
    plot_domain_grid(full, fname=os.path.join(SYMHEX_DIR, f'{base}.png'), **kw)
    plot_domain_grid(crop, fname=os.path.join(SYMHEX_DIR, f'crop_{base}.png'), **kw)
    sys.stderr.write(f'  symhex w-error compare  vmax={vmax:.3g} m/day\n')


def main(depth=70, pkey='3mo'):
    os.makedirs(OUTDIR, exist_ok=True)
    suf, plabel = PERIODS[pkey]
    with open(_cache_path(depth, pkey), 'rb') as f:
        m = pickle.load(f)['means']
    means = {'U': m['UVEL'], 'V': m['VVEL']}
    run_config(means, depth, suf, plabel)
    symhex_compare(means, depth, suf, plabel)
    sys.stderr.write(f'DONE {pkey} {depth}m\n')


if __name__ == '__main__':
    main()
