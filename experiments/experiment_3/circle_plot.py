"""
Shared plotting style + cache loaders for the experiment_3 (circling glider) figures.

Follows the project figure conventions: legends across the TOP (no suptitles),
bold/large labels, American spelling.  Two colour keys are used consistently:
  * number of gliders N -> light->dark  (more gliders = darker)
  * circle diameter     -> light->dark  (bigger circle = darker), matching
    sampling_dynamics' DIAM_COLOR.
Array estimate = solid, model (disk) truth = dotted.
"""

import os

import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import circle_common as C

mpl.rcParams.update({
    'axes.labelsize': 13, 'axes.labelweight': 'bold',
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 110, 'savefig.dpi': 130,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.formatter.limits': (-3, 4), 'axes.formatter.use_mathtext': True,
})

FIG_DIR = os.path.join(C.HERE, 'figs')

# number of gliders -> colour (light -> dark with N)
N_COLOR = {3: '#bae4b3', 4: '#74c476', 5: '#31a354', 6: '#006d2c'}
# circle diameter -> colour (light -> dark with size)
DIAM_COLOR = {0.3: '#a6cee3', 0.5: '#4292c6', 0.75: '#2171b5', 1.0: '#08306b'}

TRUTH_KW = dict(ls=':', lw=2.4)
ARRAY_KW = dict(ls='-', lw=2.0)
SEC_PER_DAY = C.SEC_PER_DAY


def n_color(n):
    return N_COLOR.get(n, '#333333')


def diam_color(d):
    return DIAM_COLOR.get(d, '#333333')


def outdir(sub=None):
    d = FIG_DIR if sub is None else os.path.join(FIG_DIR, sub)
    os.makedirs(d, exist_ok=True)
    return d


def tidy_x(ax, n=5):
    ax.xaxis.set_major_locator(MaxNLocator(n))


def finish(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- loaders
def load_array(n, diam):
    return xr.open_dataset(os.path.join(C.CACHE_DIR, f'{C.config_name(n, diam)}_array.nc'))


def load_disk(diam):
    return xr.open_dataset(os.path.join(C.CACHE_DIR, f'circle_d{diam}_disk.nc'))


def load_cloud(diam):
    return xr.open_dataset(os.path.join(C.CACHE_DIR, f'circle_d{diam}_cloud.nc'))


def load_meanw():
    """The 2-D post-spin-up time-mean model w field over the bbox (m/day); see
    run_meanw_field.py."""
    return xr.open_dataset(os.path.join(C.CACHE_DIR, 'meanw_field.nc')).Wmean


def load_metrics():
    import pandas as pd
    return pd.read_csv(os.path.join(C.DATA_DIR, 'metrics.csv'))


# --------------------------------------------------------------------------- legends
def n_legend(fig, method=True, y=1.03, ncol=None):
    """Top legend: one entry per glider count, plus array/truth line-style key."""
    handles = [Line2D([0], [0], color=n_color(n), lw=2.8, label=f'{n} gliders')
               for n in C.N_GLIDERS]
    if method:
        handles += [Line2D([0], [0], color='0.25', **ARRAY_KW, label='array estimate'),
                    Line2D([0], [0], color='0.25', **TRUTH_KW, label='disk truth')]
    fig.legend(handles=handles, loc='upper center', ncol=ncol or len(handles),
               frameon=False, bbox_to_anchor=(0.5, y))
