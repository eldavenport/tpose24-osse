#!/usr/bin/env python
"""
make_sym_1day_figs.py — W-skill summary figures for the symmetric regular-shape
(symhex/symdia/symsq) sweep, built from the **1-day-averaged** data produced by
run_experiment_2_1day.py.

Reuses the same summary_fig_tools builders as summary.ipynb's sym_sweep section,
but points them at data_1day/ and writes into summary_figs/sym_sweep_1day/.
W only (no heat flux). Run run_experiment_2_1day.py first.

Run:  python experiment_2/make_sym_1day_figs.py
"""
import os
import sys

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import summary_fig_tools as sft  # noqa: E402

DATA   = os.path.join(HERE, "data_1day")
SYMDIR = os.path.join(HERE, "summary_figs", "sym_sweep_1day")
os.makedirs(SYMDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 120
sft.apply_style()

m = pd.read_csv(os.path.join(DATA, "metrics.csv"))
print(f"{len(m)} sym cells across {m.config.nunique()} configs (1-day averages)")


def load_cell(row):
    """Open the saved w_est/w_model/bias arrays for one metrics row."""
    return xr.open_dataset(os.path.join(HERE, row.nc_path))


for fam in sft.SYM_FAMILIES:                        # symhex, symdia, symsq
    sft.make_sym_summary(m, SYMDIR, family=fam)     # w-skill vs diameter, line per centre
    sft.make_sym_w_scatter(DATA, SYMDIR, fam)       # est-vs-true w scatter (rows=centre, cols=diam)
    sft.make_fig7_sym(m, SYMDIR, load_cell, family=fam, diam=1.0)   # <w> +/-95% CI per centre
sft.make_sym_w_scatter_deepest(DATA, SYMDIR)        # deepest-level est-vs-true <w>, 3 shape panels
for c in sft.SYMHEX_CENTERS:                        # shape comparison at each centre latitude
    sft.make_sym_shape_summary(m, SYMDIR, center=c)                 # w-skill vs diameter, line per shape
    sft.make_fig7_sym_shapes(m, SYMDIR, load_cell, center=c, diam=1.0)  # <w> +/-95% CI, shape columns

print(f"Wrote sym 1-day figures -> {SYMDIR}")
