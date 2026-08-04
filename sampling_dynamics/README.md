# sampling_dynamics — the virtual mooring

Evaluating a "virtual mooring" = a **symmetric hexagon of 6 wave gliders + 1 TAO mooring at the
center**. Quantify the characteristics of vertical velocity `w`, KPP mixing, and
the vertical/lateral transport of heat and momentum, and ask how well the
array samples each quantity versus the model truth (every grid point inside the array
footprint), comparing across symmetric hexagons of increasing diameter.

## Model run

Model output: `oct2012_3mo_dt60_AB3` (ΔT = 60 s, 3-hourly output, ~80 days after
spin-up)

| diag | fields |
|--------|--------|
| `diag_state` | THETA, SALT, UVEL, VVEL, WVEL, PHIHYD, DRHODR |
| `diag_surf`  | ETAN, KPPhbl (boundary-layer depth) |
| `diag_kpp`   | KPPviscA (ν), KPPdiffT (κ_T) |

## Configurations

The `symhex` family from `experiments/experiment_1/configs/symhex`, center +0.0°N, 
E-W diameter 0.3 / 0.5 / 0.75 / 1.0°, 6 gliders each. Add the TAO mooring at the center;
the convex hull (the truth footprint) is unchanged. Diameters are colored light→dark
with size in every figure.

- **Array estimate** — `w`: plane-fit from the array U,V (`compute_w_planefit` defaults,
  extrapolate-to-surface, 8–80 m, like experiment_2). Mean fields: mean over
  the 6 gliders + mooring. Eddy fluxes: vertical uses the plane-fit w′ with the
  array-mean tracer′; lateral uses the glider/mooring mean of a′b′.
- **Model truth** — array-area-mean over every grid point inside the hexagon (mean fields, w),
  and the area-area-point-mean instantaneous eddy-flux series (vertical w′φ′, lateral a′b′).
  Primes = deviations from each point's time mean.

## Pipeline

1. **`common.py`** — shared config, the dt60 loader, co-located bbox/point/hull samplers
   (reuse osse_tools; KPP + DRHODR registered so `sample_fields`/`model_region` place
   them like WVEL), N² helper, styling.
2. **`run_sample.py`** (compute) — reads the array bounding box into memory once
   (`load_bbox_memory`, ~4 min), then for each diameter writes to `cache/`:
   - `<name>_array.nc` — fields at the 6 gliders + mooring `(time, glider, obs_depth)`,
     `hbl(time,glider)`, and the plane-fit `w_est` / `w_est_mid`.
   - `<name>_hull.nc` — hull-mean profile of every field `(time, obs_depth)`, `hbl(time)`,
     and the instantaneous hull-mean eddy-flux series (vertical `wT`/`wU`/`wV`, lateral
     `uT`/`vT`/`uv`).
   - `<name>_cloud.nc` — the full hull point cloud, time-subsampled ×3, for the
     distribution comparisons.
3. Diagnostic scripts (each reads only the cache; no model access):

| script | subfolder | what it shows |
|--------|-----------|---------------|
| `run_w.py` | `vertical_velocity/` | w(z) mean/std/`r`, time-mean ±95% CI (autocorr-aware), w Hovmöller, max/min/mean/median stats |
| `run_mixing.py` | `mixing/` | κ_T, ν, N² profiles; KPP boundary-layer depth time series + distribution; κ_T Hovmöller with hbl; intermittency stats |
| `run_transport.py` | `transport/` | vertical (w′T′, w′u′, w′v′) + lateral (u′T′, v′T′, u′v′) eddy-flux mean profiles, array-vs-truth correlation r(z), w′T′ Hovmöller |
| `run_distributions.py` | `distributions/` | array-vs-truth PDFs (JS + Wasserstein vs a random-placement null), JS summary vs diameter, moment recovery, skew-normal fits |
| `run_spectra.py` | `spectra/` | vertical-wavenumber × frequency PSD of area-mean w and w′T′, true vs array (diurnal + TIW marked) |
| `run_animations.py` | `animations/` | the field advecting through the footprint at 30 m; w & κ_T profiles evolving in time |

## Run order

```bash
conda activate tpose
python run_sample.py          # compute -> cache/ (reads the model once)
python run_w.py               # then the diagnostics, in any order / in parallel
python run_mixing.py
python run_transport.py
python run_distributions.py
python run_spectra.py
python run_animations.py
```
