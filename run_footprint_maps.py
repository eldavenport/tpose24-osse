"""
Footprint plane-fit error maps: how much a candidate glider footprint aliases the
depth-/time-mean vertical velocity across the domain.

For each footprint shape (hexagon, square, diamond) and size, place the footprint at
every grid point, estimate w at the base of the 0..d m layer by the same plane-fit /
divergence / integrate chain the OSSE uses, and map the error against the like-computed
true area-mean w. Because the estimator and truth are linear and time-averaging
commutes through them, the error in the 3-month-mean w is evaluated directly on the
time-and-depth-mean field -- so these reuse means['UVEL','VVEL'] from the
run_domain_maps cache (no model read).

  Tier 2 -- discrete glider stencil (what the array actually samples).
  Tier 1 -- filled footprint (dense-sampling floor: the field's intrinsic nonlinearity).

Two figures per (tier, depth, period), split by height (1deg, 2deg); 16 panels each
(4 shapes x 4 widths) on ONE shared symmetric color scale for cross-footprint
comparison. Full-domain and equatorial-crop views, under domain/footprint/.

Usage:
    python run_footprint_maps.py [depths_csv] [periods_csv] [tiers_csv]
    e.g. python run_footprint_maps.py 70,120,250 3mo,1mo 1,2
         python run_footprint_maps.py 70 3mo 2
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

SHAPES = ('hexagon', 'square', 'square4', 'diamond')   # rows of each figure
SHAPE_TITLE = {'hexagon': 'hexagon', 'square': 'square (6)',
               'square4': 'square (4)', 'diamond': 'diamond'}
SHAPE_COLOR = {'hexagon': '#1b9e77', 'square': '#d95f02',
               'square4': '#7570b3', 'diamond': '#e7298a'}
WIDTHS = (0.5, 1.0, 1.5, 2.0)      # zonal extent (deg), columns of each figure
HEIGHTS = (1.0, 2.0)               # meridional extent (deg), one figure each
TIER_LABEL = {1: 'Tier 1 (filled footprint, dense-sampling floor)',
              2: 'Tier 2 (discrete glider stencil)'}

# profile summaries: reduce the maps to "which shape is best where" line plots.
# vs-latitude   -> median |w-error| over the design longitude band, plotted vs lat
# vs-longitude  -> median |w-error| over the equatorial lat band, plotted vs lon
LON_AVG = slice(217, 223)   # design longitude band for the latitude profiles
LAT_VIEW = slice(-4, 6)     # latitude range shown (equator + TIW band)
LAT_AVG = slice(-2, 2)      # equatorial band for the longitude profiles
LON_VIEW = slice(211, 229)  # longitude range shown (edges trimmed)

PERIODS = {                        # matches run_domain_maps
    '3mo': ('', 'Oct 11-Dec 2012'),
    '1mo': ('_1mo', 'Oct 11-Nov 11 2012'),
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


def _shapes_labels(tier):
    """(shapes, labels) for a tier. Tier 1 fills the footprint, so it depends only on
    the outline: square4 duplicates square and the glider-count labels are meaningless
    -> show the 3 distinct filled shapes with plain names. Tier 2 uses all 4 stencils."""
    if tier == 1:
        shapes = ('hexagon', 'square', 'diamond')
        return shapes, {s: s for s in shapes}
    return SHAPES, SHAPE_TITLE


def _profile(da, avg, view, reduce_axis, stat):
    """median/mean |w-error| of a map over the `avg` band on one axis, at each point
    of the `view` range on the other. reduce_axis: 'XC' (-> vs latitude) or 'YC'."""
    f = np.nanmedian if stat == 'median' else np.nanmean
    if reduce_axis == 'XC':
        sub = da.sel(XC=avg, YC=view)
        return f(np.abs(sub.values), axis=1), sub['YC'].values
    sub = da.sel(XC=view, YC=avg)
    return f(np.abs(sub.values), axis=0), sub['XC'].values


def run_config(means, d, suf, plabel, tier):
    """All footprint diagnostics for one depth/period/tier: the w-error maps (two
    height figures on a shared color scale) plus the two 'best shape where' profile
    summaries (vs latitude and vs longitude), each with median and mean rows."""
    shapes, labels = _shapes_labels(tier)
    # compute every (height, shape, width) map once; reuse for maps and profiles
    maps = {(shape, h, w): ot.footprint_w_error(means, shape, w, h, d, tier=tier)
            for h in HEIGHTS for shape in shapes for w in WIDTHS}
    vmax = _shared_vmax(list(maps.values()))

    tier_dir = os.path.join(OUTDIR, f'tier{tier}')
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
        base = f'footprint_werr_tier{tier}_h{h:g}_{d}m{suf}'
        kw = dict(cbar_label='w error (m day$^{-1}$)', cmap=cmo.balance,
                  diverging=True, ncols=len(WIDTHS), vlim=(-vmax, vmax), suptitle=None)
        plot_domain_grid(full, fname=os.path.join(full_dir, f'{base}.png'), **kw)
        plot_domain_grid(crop, fname=os.path.join(crop_dir, f'crop_{base}.png'), **kw)

    # --- 'best shape where' profile summaries (median rows, then mean rows) -----
    labellist = [labels[s] for s in shapes]
    colors = {labels[s]: SHAPE_COLOR[s] for s in shapes}
    rows = [('median', h) for h in HEIGHTS] + [('mean', h) for h in HEIGHTS]
    ttl = f'Tier {tier} · 0–{d} m · {plabel}'

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
        fname=os.path.join(summ_dir, f'footprint_shape_by_lat_tier{tier}_{d}m{suf}.png'),
        ref_x=0.0)
    plot_footprint_profiles(
        prof_lon, lon, rows, WIDTHS, labellist, colors, xlabel='Longitude (°E)',
        legend_title=f'best shape vs longitude — {ttl} (over 2°S–2°N)',
        fname=os.path.join(summ_dir, f'footprint_shape_by_lon_tier{tier}_{d}m{suf}.png'),
        ref_x=220.0)
    sys.stderr.write(f'  tier{tier} {suf or "3mo"} {d}m  vmax={vmax:.3g} m/day\n')


def main(depths, periods, tiers):
    os.makedirs(OUTDIR, exist_ok=True)
    for pkey in periods:
        suf, plabel = PERIODS[pkey]
        for d in depths:
            with open(_cache_path(d, pkey), 'rb') as f:
                m = pickle.load(f)['means']
            means = {'U': m['UVEL'], 'V': m['VVEL']}
            for tier in tiers:
                run_config(means, d, suf, plabel, tier)
            sys.stderr.write(f'DONE {pkey} {d}m\n')


if __name__ == '__main__':
    depths = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [70, 120, 250]
    periods = sys.argv[2].split(',') if len(sys.argv) > 2 else ['3mo', '1mo']
    tiers = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [1, 2]
    main(depths, periods, tiers)
