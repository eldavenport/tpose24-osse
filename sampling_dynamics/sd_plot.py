"""
Shared plotting + statistics helpers for the sampling_dynamics figures.

Figure style follows the project convention: legends across the TOP (no suptitles),
bold/large axis labels, American spelling, diameters coloured light->dark with size,
array estimate solid + model truth dotted.  Every diagnostic script loads the cache
files written by run_sample.py through the loaders here and reuses these builders.
"""

import os

import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common as C
import osse_tools as ot

SEC_PER_DAY = C.SEC_PER_DAY

# --------------------------------------------------------------------------- style
mpl.rcParams.update({
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 110, 'savefig.dpi': 130,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25,
    # scientific ×10ⁿ offset instead of long decimal strings (N², fluxes) so
    # small/large-number x-axes don't overlap
    'axes.formatter.limits': (-3, 4), 'axes.formatter.use_mathtext': True,
})
from matplotlib.ticker import MaxNLocator


def tidy_x(ax, n=5):
    """Thin a linear x-axis to <=n ticks (pairs with the scientific-offset rcParam)."""
    ax.xaxis.set_major_locator(MaxNLocator(n))

TRUTH_KW = dict(ls=':', lw=2.4)          # model truth = dotted
ARRAY_KW = dict(ls='-', lw=2.0)          # array estimate = solid


def outdir(sub):
    d = os.path.join(C.HERE, sub)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- loaders
def load_array(diam):
    return xr.open_dataset(os.path.join(C.CACHE_DIR, f'{C.config_name(diam)}_array.nc'))


def load_hull(diam):
    return xr.open_dataset(os.path.join(C.CACHE_DIR, f'{C.config_name(diam)}_hull.nc'))


def load_cloud(diam):
    return xr.open_dataset(os.path.join(C.CACHE_DIR, f'{C.config_name(diam)}_cloud.nc'))


def have_cache():
    return all(os.path.exists(os.path.join(C.CACHE_DIR, f'{C.config_name(d)}_array.nc'))
               for d in C.DIAMETERS)


# --------------------------------------------------------------------------- units
def to_display(var, values):
    """Scale a quantity to its display units (w -> m/day; else unchanged)."""
    if var in ('W', 'w_est', 'w_est_mid', 'wbar'):
        return values * SEC_PER_DAY
    return values


# --------------------------------------------------------------------------- array reductions
def array_profile(arr, var, gliders_only=False):
    """Array's estimate of a mean field: average the sampled var over the platforms.

    gliders_only drops the mooring (is_mooring), i.e. the 6-glider footprint mean.
    """
    da = arr[var]
    if gliders_only and 'is_mooring' in arr.coords:
        da = da.isel(glider=~arr.is_mooring.values)
    return da.mean('glider')


def w_array_series(arr):
    """Plane-fit w on the obs axis (time, obs_depth), in m/s."""
    return arr['w_est_mid']


# --------------------------------------------------------------------------- statistics
def time_stats(series, dim='time'):
    """mean/median/std/min/max of a series along a dim -> dict of DataArrays."""
    return dict(mean=series.mean(dim), median=series.median(dim),
                std=series.std(dim), min=series.min(dim), max=series.max(dim))


def moments(x):
    """(mean, std, skew) of a 1-D sample, ignoring NaNs."""
    from scipy.stats import skew
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    return float(x.mean()), float(x.std()), float(skew(x))


def js_wass(obs, truth, bins=60):
    """Jensen-Shannon distance + Wasserstein distance between two 1-D samples."""
    from scipy.stats import wasserstein_distance
    o = np.asarray(obs); o = o[np.isfinite(o)]
    t = np.asarray(truth); t = t[np.isfinite(t)]
    if o.size < 5 or t.size < 5:
        return np.nan, np.nan
    lo = min(o.min(), t.min()); hi = max(o.max(), t.max())
    edges = np.linspace(lo, hi, bins + 1)
    js = ot._js_distance(o.reshape(-1, 1), t.reshape(-1, 1), [edges])
    return float(js), float(wasserstein_distance(o, t))


def js_wass_null(obs, truth, n_obs, bins=60, ndraw=40, seed=0):
    """Random-placement null for js/wass: draw n_obs points at random from truth."""
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(seed)
    t = np.asarray(truth); t = t[np.isfinite(t)]
    if t.size < max(5, n_obs):
        return np.nan, np.nan
    lo, hi = t.min(), t.max()
    edges = np.linspace(lo, hi, bins + 1)
    jsn, wn = [], []
    for _ in range(ndraw):
        sub = rng.choice(t, size=n_obs, replace=False)
        jsn.append(ot._js_distance(sub.reshape(-1, 1), t.reshape(-1, 1), [edges]))
        wn.append(wasserstein_distance(sub, t))
    return float(np.mean(jsn)), float(np.mean(wn))


# --------------------------------------------------------------------------- depth avg
def depth_average(da, dim='obs_depth'):
    """Simple mean over the obs-depth axis (uniform 2 m spacing)."""
    return da.mean(dim)


# --------------------------------------------------------------------------- legends
def top_legend(fig, diam=True, method=True, extra=None, ncol=None, y=1.04):
    """One combined legend across the top: diameter colours + array/truth line styles.

    Keeps a single non-overlapping legend per figure (project convention: top legend,
    no suptitle).  `diam` adds the E-W diameter colour key; `method` adds the array
    (solid) vs model truth (dotted) key.
    """
    handles = []
    if diam:
        handles += [Line2D([0], [0], color=C.diam_color(d), lw=2.6, label=f'{d:g}$^\\circ$')
                    for d in C.DIAMETERS]
    if method:
        handles += [Line2D([0], [0], color='0.25', **ARRAY_KW, label='array estimate'),
                    Line2D([0], [0], color='0.25', **TRUTH_KW, label='model truth')]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc='upper center', ncol=ncol or len(handles),
               frameon=False, bbox_to_anchor=(0.5, y))


def finish(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path
