# experiment_3 — a CIRCLING glider array

Experiments 1–2 (and the `sampling_dynamics/` virtual mooring) treat the wave gliders
as a small **stationary mooring array**: they sit at fixed positions and a plane fit to
their currents estimates the footprint-mean vertical velocity and heat flux.

experiment_3 asks a different question. The **TAO mooring stays put at 0°N, 140°W**, and
the wave gliders **orbit it**: N gliders spread evenly around a circle, all travelling
**clockwise at 2 knots through the water**. Because the through-water speed is fixed, a
smaller circle is orbited faster (period = circumference / speed), so each diameter
samples its footprint at a different angular cadence.

> **What circle diameter and how many gliders best recover the footprint truth?**

## Design

| choice | value |
|---|---|
| centre | 0°N, 140°W (the TAO mooring, fixed) |
| gliders per array `N` | 3, 4, 5, 6 |
| circle diameters | 0.3, 0.5, 0.75, 1.0° (the symhex diameters; **radius = diameter/2**) |
| glider speed | 2 kt through water, **clockwise** |
| start layout | a regular `N`-gon with one vertex due north; the whole array rotates rigidly |
| model run + w method | **identical to experiment_2**: `oct2012_3month_transp_cons`, 3-hourly, plane-fit w with the shear extrapolated to the surface, 8–80 m, w = 0 at the surface |

At `N = 6, 4, ...` the t = 0 layout is exactly the symhex / symsq / … regular polygon, so
this experiment is the "moving" analogue of those configs — but the configs themselves
are **not** used; the geometry is generated directly (`circle_common.glider_positions`).

### Truth = the fixed disk
The estimate is compared against the **fixed circular disk** of that diameter centred at
0°N,140°W — the area-mean over every model grid point inside the disk. The centre is on
the equator, where 1° lon = 1° lat in metres, so a circle in degrees is a circle in
metres. The disk truth is the **same target for every `N`** at a given diameter, so skill
differences isolate the number of gliders and the orbit, not the truth region. This is the
circular analogue of experiment_2's convex-hull truth, except it does not rotate.

### Moving-glider sampling (efficient, one model read)
Everything comes from **one in-memory read** of the array bounding box
(`circle_common.load_bbox_memory`). The moving glider samples use xarray's *diagonal*
advanced interpolation: the glider positions are a `(time, glider)` array that shares the
model `time` coordinate, so each timestep is interpolated at that timestep's positions
(`sample_moving`). The plane fit rebuilds its design matrix per timestep from the
instantaneous positions (`compute_w_planefit_moving`). The disk truth and the distribution
cloud come from a tracer-centred region over the same bbox (`region_bbox` → `disk_select`).

## Files

* `circle_common.py` — geometry, bbox loader, moving sampler, moving plane-fit, disk truth.
* `run_experiment_3.py` — COMPUTE: one bbox read → per-diameter disk truth + cloud, per
  `(N, diameter)` moving array samples + plane-fit w + heat-flux inputs → `cache/`, and the
  w-skill / heat-flux-skill `data/metrics.csv`.
* `run_meanw_field.py` — a lightweight, SUBSAMPLED read of just `WVEL` → `cache/meanw_field.nc`,
  the post-spin-up time-mean vertical velocity field over the bbox (for the demonstration
  figure; the smooth time mean needs far fewer timesteps than the moving-sampler compute).
* `circle_plot.py` — shared style + cache loaders (N coloured light→dark, diameter
  light→dark, array solid / disk truth dotted).
* `make_figs.py` — w-skill and heat-flux-skill figures → `figs/`.

The **distribution** figures (sampling_dynamics-style `fit_distributions`) live at the repo
top level in `circling_gliders/` (`run_distributions.py`), reading these same caches.

## Outputs

* `cache/circle_d{diam}_disk.nc` — fixed-disk truth: `w_true` (interfaces) and, on the obs
  axis, `eddy_flux` ⟨w′T′⟩, `adv_total` ⟨w dT/dz⟩, `adv_eddy`, `wbar`, `Tbar`, plus the
  disk-point-mean eddy-flux series `wT/wU/wV/uT/vT/uv` for the distribution fits.
* `cache/circle_d{diam}_cloud.nc` — the disk point cloud (U,V,T,W), time-subsampled ×3.
* `cache/circle_n{n}_d{diam}_array.nc` — moving-glider U,V,T, the plane-fit `w_est`
  (interfaces) + `w_est_mid` (obs axis), and the array-mean glider T `Tbar_glider`.
* `data/metrics.csv` — one row per `(N, diameter)`: w-skill (`w_corr`, `w_norm_rms`,
  `w_std_ratio`, `w_frac_mean_bias`, …) and heat-flux skill (`flux_corr_{25,50,70}`,
  `adv_corr_{…}`, std ratios, JS) against the disk truth.
* `cache/meanw_field.nc` — `Wmean(depth, YC, XC)` (m/day), the time-mean model `WVEL` over
  the bbox (background for the demonstration figure).
* `figs/`:
  * `geometry` — orbit geometry per `(N × diameter)` (start `N`-gon, circle, one track, disk).
  * `tracks_meanw` — **demonstration**: the orbits drawn over the mean-`w` field per `(N × diameter)`.
  * `w_skill_summary`, `w_skill_heatmap`, `w_profiles` — w-skill vs diameter / over the grid / profiles.
  * `w_fig7_d1` — time-mean ⟨w⟩(z) with autocorrelation-aware **95 % CI**, one column per `N`
    (+ total-over-depth forest), at the 1° diameter (summary_figs fig 7 analogue).
  * `w_scatter` — estimated-vs-true `w` scatter (all times & depths), rows = `N`, cols = diameter.
  * `heat_skill_summary`, `heat_skill_heatmap`, `heat_profiles` — heat-flux skill vs diameter / grid / profiles.
  * `heat_area_components`(`_flux`) — structural decomposition (true area mean vs estimated
    mean, estimate error, sub-array covariance), columns = diameter, lines coloured by `N`.
  * `heat_area_reducible_d1` — reducible plane-fit / sampling errors (`w`, ∂T/∂z, heating, heat
    flux) at the 1° diameter, true vs estimate + estimate error, lines coloured by `N`.
  * `heat_scatter` — estimated-vs-true eddy heat flux `w'T'` scatter, rows = `N`, cols = diameter.

Re-running skips caches already on disk; delete `cache/` (and `data/metrics.csv`) to redo.

Run (from the `tpose` env):
```
python experiments/experiment_3/run_experiment_3.py   # compute (one model read)
python experiments/experiment_3/run_meanw_field.py    # mean-w background (subsampled read)
python experiments/experiment_3/make_figs.py          # w + heat figures
python circling_gliders/run_distributions.py          # distribution fits (top level)
```
