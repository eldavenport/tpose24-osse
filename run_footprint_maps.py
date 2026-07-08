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
from plotting_tools import plot_domain_grid

OUTDIR = os.path.join(os.path.dirname(__file__), 'domain', 'footprint')
CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'
CROP_LON = slice(217, 223)
CROP_LAT = slice(-3, 3)

SHAPES = ('hexagon', 'square', 'square4', 'diamond')   # rows of each figure
SHAPE_TITLE = {'hexagon': 'hexagon', 'square': 'square (6)',
               'square4': 'square (4)', 'diamond': 'diamond'}
WIDTHS = (0.5, 1.0, 1.5, 2.0)      # zonal extent (deg), columns of each figure
HEIGHTS = (1.0, 2.0)               # meridional extent (deg), one figure each
TIER_LABEL = {1: 'Tier 1 (filled footprint, dense-sampling floor)',
              2: 'Tier 2 (discrete glider stencil)'}

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


def run_config(means, d, suf, plabel, tier):
    """All footprint maps for one depth/period/tier: one map per (shape, width,
    height), split into two figures by height on a shared color scale."""
    head = f'TPOSE24 {plabel}, 0-{d} m'
    # compute every (height, shape, width) map first so both height figures share a scale
    maps = {h: [] for h in HEIGHTS}
    titles = {h: [] for h in HEIGHTS}
    for h in HEIGHTS:
        for shape in SHAPES:
            for w in WIDTHS:
                maps[h].append(ot.footprint_w_error(means, shape, w, h, d, tier=tier))
                titles[h].append(f'{SHAPE_TITLE[shape]}  {w:g}°×{h:g}°')
    vmax = _shared_vmax([m for h in HEIGHTS for m in maps[h]])

    full_dir = os.path.join(OUTDIR, f'tier{tier}', 'full_domain')
    crop_dir = os.path.join(OUTDIR, f'tier{tier}', 'crop_140')
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    for h in HEIGHTS:
        full, crop = _panels(maps[h], titles[h])
        base = f'footprint_werr_tier{tier}_h{h:g}_{d}m{suf}'
        supt = (f'{head} — footprint w-error, {TIER_LABEL[tier]}, height {h:g}°'
                '\n(base-of-layer w, mean over period; shared scale across heights)')
        kw = dict(cbar_label='w error (m day$^{-1}$)', cmap=cmo.balance,
                  diverging=True, ncols=len(WIDTHS), vlim=(-vmax, vmax), suptitle=supt)
        plot_domain_grid(full, fname=os.path.join(full_dir, f'{base}.png'), **kw)
        plot_domain_grid(crop, fname=os.path.join(crop_dir, f'crop_{base}.png'), **kw)
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
