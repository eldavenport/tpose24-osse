# tpose24-osse

Observing System Simulation Experiment (OSSE) for a wave-glider array in the
equatorial Pacific, using TPOSE24 model output as truth.

## Modules

- **`osse_tools.py`** — computational core (sampling, plane-fit w, skill metrics,
  distributions). No plotting.
- **`plotting_tools.py`** — `plot_w_comparison`, `plot_velocity_map`, `plot_field_pdfs`, … plus the
  per-config experiment experiment figures used in `make_experiment_figs` notebooks. 
- **`summary_fig_tools.py`** — shared style, helpers, and `make_fig*` builders for
  the W-skill summary figures; used by `experiment_*/summary.ipynb` and
  `compare_exp_1_2` (each notebook just loads `metrics.csv` and calls the builders).

`osse_tools.py` provides three workflows:

- **Vertical velocity:** sample U, V at the glider array, fit a plane to get the
  horizontal divergence, integrate to estimate w, and compare against the model's
  area-mean WVEL (`sample_fields` → `compute_w_planefit` → `sample_model_w` →
  `plot_w_comparison`).
- **Distributions:** sample T, S, U, V at the gliders and over the full model field
  inside the array, then compare PDFs and joint PDFs of the fields, density,
  eddy heat fluxes (v'T', u'T'), Reynolds stress (u'v'), and divergence
  (`sample_fields` / `model_region` → `add_density` → `eddy_anomalies` →
  `plot_field_pdfs` / `plot_joint_compare`).
    - I chose a somewhat arbitrary metric to measure similarity between the PDFs.
    - Jensen-Shannon Divergence (slightly different from KL divergence) is 0 if the
      two PDFs are identical and 1 if they are disjoint (we want it to be small).
- **Footprint error maps (array design):** spatial patterns of the errors of a particular footprint
  (`footprint_offsets`: hexagon/square/square4/diamond, width×height) at every grid
  point and map the error it makes estimating the depth-/time-mean w — plane-fit
  divergence vs. like-computed true area-mean divergence, ×H → w error in m/day
  (`footprint_w_error`; Tier 2 = discrete stencil, Tier 1 = filled footprint). The
  estimator and truth are linear and time-averaging commutes through them, so the
  3-month-mean-w error is evaluated directly on the mean field. 
  `run_footprint_maps.py` reuses the cached `means` from `run_domain_maps.py`.

# Misc Notes

Functions like `_convex_hull_mask` can be used to find all the points inside any polygon (for the model "truth" averages). 

Array configurations can live in `experiment_1/configs/` as lists of (lat, lon) positions. There are some examples in hex/, rectangle/ etc. 
  - the .json files define `positions` which is a union of all points in the array, and `cells` which are
    individual cells to compute W over. (you could realistically have 2 or 3 cells total, depending on the configuration).

There are some tests that can be run with: `python test_osse_tools.py`

required packages: numpy, xarray, matplotlib, cmocean, gsw, xmitgcm

tested with Python 3.12 (so it's at least compatible with this version)
