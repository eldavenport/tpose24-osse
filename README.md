# tpose24-osse

Observing System Simulation Experiment (OSSE) for a wave-glider array in the
equatorial Pacific, using TPOSE24 model output as truth.

`osse_tools.py` provides two workflows:

- **Vertical velocity:** sample U, V at the glider array, fit a plane to get the
  horizontal divergence, integrate to estimate w, and compare against the model's
  area-mean WVEL (`sample_fields` → `compute_w_planefit` → `sample_model_w` →
  `plot_w_comparison`).
- **Distributions:** sample T, S, U, V at the gliders and over the full model field
  inside the array, then compare PDFs and joint PDFs of the fields, density,
  eddy heat fluxes (v'T', u'T'), Reynolds stress (u'v'), and divergence
  (`sample_fields` / `model_region` → `add_density` → `eddy_anomalies` →
  `plot_field_pdfs` / `plot_joint_compare`).

Functions like `_convex_hull_mask` can be used to find all the points inside any polygon (for the model "truth" averages). 

Array configurations can live in `configs/*.json` as lists of (lat, lon) positions. There are some examples in hex/, rectangle/ etc. 

There are some tests that can be run with: `python test_osse_tools.py`

required packages: numpy, xarray, matplotlib, cmocean, gsw, xmitgcm

tested with Python 3.12 (so it's at least compatible with this version)
