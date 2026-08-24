#!/usr/bin/env python
"""
make_sym_1day_heat_figs.py — vertical-heat-flux (w·∂T/∂z) summary figures for the
symmetric regular-shape (symhex/symdia/symsq) sweep, built from the 1-day-averaged
data produced by run_heat_flux_1day.py + run_glider_heat_flux_1day.py.

Reuses the same heat_flux_fig_tools builders as heat_flux_summary.ipynb's sym_sweep
cell, but points them at data_heat_1day/ and writes into heat_flux_figs/sym_sweep_1day/.
Run run_heat_flux_1day.py and run_glider_heat_flux_1day.py first.

Run:  python experiment_2/make_sym_1day_heat_figs.py
"""
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import heat_flux_fig_tools as hft  # noqa: E402

# heavy .nc caches live on /data (not /home); see project_sym_disk_truth memory
DATA = os.path.join('/data/SO3/edavenport/tpose24-osse/cache', 'experiment_2', 'data_heat_1day')
SYMDIR = os.path.join(HERE, "heat_flux_figs", "sym_sweep_1day")
os.makedirs(SYMDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 120
hft.apply_style()


def savesym(fig, name):
    fig.savefig(os.path.join(SYMDIR, name), bbox_inches="tight", dpi=150)
    return fig


for fam in hft.SYM_FAMILIES:                       # symhex, symdia, symsq
    # summary: area-average heating error vs diameter, one line per centre latitude
    savesym(hft.make_error_scaling_symhex(DATA, family=fam), f"error_scaling_{fam}.png")
    # structural components (columns = diameter, line per centre) + W/m2 duplicate
    sx = hft.symhex_components_xlims(DATA, family=fam)
    sxf = hft.symhex_components_xlims(DATA, family=fam, flux=True)
    savesym(hft.make_area_components_symhex(DATA, family=fam, xlims=sx), f"area_components_{fam}.png")
    savesym(hft.make_area_components_symhex(DATA, family=fam, flux=True, xlims=sxf), f"area_components_{fam}_flux.png")
    # reducible plane-fit [w]/[dT/dz] errors at 1.0 deg AND 0.5 deg; point stand-in at 1.0 deg
    savesym(hft.make_area_reducible_symhex(DATA, diam=1.0, family=fam), f"area_reducible_{fam}.png")
    savesym(hft.make_area_reducible_symhex(DATA, diam=0.5, family=fam), f"area_reducible_{fam}_d0.5.png")
    savesym(hft.make_point_flux_symhex(DATA, family=fam), f"point_flux_{fam}.png")

# shape comparison (hexagon vs diamond vs square), one figure per centre latitude
for c in hft.SYMHEX_CENTERS:
    savesym(hft.make_error_scaling_sym_shapes(DATA, center=c), f"shape_compare_error_scaling_c{c:+.1f}.png")

print(f"Wrote sym 1-day heat-flux figures -> {SYMDIR}")
