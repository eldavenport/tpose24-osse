# experiment_2 — W-estimate skill across array configurations

same as experiment_1 but with extrapolation of the velocity shear to the surface and w=0 at the surface. 
(this is the default option in `compute_w_planefit`)

Sampled vertical extent is 8–80 m (`MAX_DEPTH = 80`).

use configs from experiment_1/configs/

**Symmetric regular-shape sweep** (`experiment_1/configs/{symhex,symdia,symsq}/`): a family
of geometrically **regular** (isotropic, least plane-fit bias on a curved field) footprints —
**hexagon** (`symhex`, 6 gliders), **diamond** (`symdia`, 4) and **square** (`symsq`, 4) — at
E-W **diameter** 0.3/0.5/0.75/1.0° (diameter = 2 × glider lon offset, so the 1.0° member is
the 0.5°-offset shape; hexagon half-height = diameter/√3), centred at 0.0/0.5/−0.5°N, with
**no moorings** (evaluated on the shape gliders alone). All three families run through the
whole pipeline below: `run_experiment_2.py` (w skill, picked up automatically by the config
glob), `run_heat_flux.py` + `run_glider_heat_flux.py` + `run_gradient_maps.py` (their `SCOPE`
lists include the 12 configs per family = 36), and the domain footprint maps. The summary
figures compare each family across diameter (one line per centre latitude) and compare the
three shapes against each other (see below).

2. `python run_experiment_2.py` — opens the model once, samples U,V at every glider
   position, computes plane-fit `w_est` and hull-mean model-truth `w_model` per
   cell, and writes:
   - `data/<config>__cell_<center>.nc` — `w_est`, `w_model`, `bias` (dims time, depth)
   - `data/metrics.csv` — one row per cell with skill stats + config metadata
   Re-running skips cells whose `.nc` already exists (delete `data/` to recompute).
3. `summary.ipynb` — the 6 presentation figures (adds `fig6_metric_heatmaps`, the colored-square metric grid, plus `fig6a–6e` breaking it out one method per figure like `fig2a–2e`/`fig5a–5e`; `2e`/`5e`/`6e` are the `shift_hex` array) → `summary_figs/`. Under `summary_figs/sym_sweep/`, the symmetric regular-shape (hexagon/diamond/square) set: `fig_<fam>_skill_vs_diameter.png` (each family's w-skill in figure-2 metric order vs E-W diameter, one line per centre latitude) and `fig_sym_shape_skill_vs_diameter_c<lat>.png` (the shape comparison at each centre, one line per shape); `fig_<fam>_w_scatter.png` (estimated-vs-true `w`, rows = centre latitude × cols = diameter, coloured by depth, with 1:1 line + correlation) plus `fig_sym_w_scatter_deepest.png` (one figure, three shape subplots, **time-mean** estimated-vs-true `⟨w⟩` at the deepest observed level — one point per config, all centres × diameters overlaid, colour hue = centre latitude, shading light→dark = diameter); `fig7_<fam>_d1.png` and `fig7_sym_shapes_c<lat>_d1.png` (time-mean `⟨w⟩` ±95% CI depth profiles — reusing the fig7 machinery — comparing centre latitude within a shape and comparing shapes at a centre); and `sym_layouts.png` (geometry diagram of all 36 regular arrays: rows = shape, cols = diameter, colour = centre — from `run_shape_demo.py`).
4. `make_experiment_figs.ipynb` — per-config diagnostics
   (`w_comparison` Hovmöllers + `velocity_map`) → `experiment_figs/<config>/`.

### 1-day-averaged sym sweep (`run_experiment_2_1day.py` / `make_sym_1day_figs.py`)

`run_experiment_2_1day.py` reruns **only** the symmetric regular-shape families
(`symhex`/`symdia`/`symsq`, 36 cells) at **1-day** temporal resolution — one w
estimate per day instead of every 3 hours — to mimic a realistic wave-glider
sampling cadence. Same physics as `run_experiment_2.py`; the only change is that the
sampled U,V (and the hull-mean model-truth WVEL) are averaged into 1-day means before
the plane fit. Because the plane-fit estimator and the hull-mean truth are both linear,
this daily mean commutes through them, so averaging the inputs equals daily-averaging
the per-timestep w. Writes `data_1day/<config>__cell_<center>.nc` + `data_1day/metrics.csv`
(81 daily steps → the skill metrics, `n_eff`/`tau`, and autocorrelation-aware CIs all
reflect the coarser sampling). `make_sym_1day_figs.py` then rebuilds the same sym-sweep
W figures as step 3 (`fig_<fam>_skill_vs_diameter`, `fig_sym_shape_skill_vs_diameter_c<lat>`,
`fig_<fam>_w_scatter`, `fig_sym_w_scatter_deepest`, `fig7_<fam>_d1`, `fig7_sym_shapes_c<lat>_d1`)
from `data_1day/` into `summary_figs/sym_sweep_1day/`.

The **heat-flux** analogue is `run_heat_flux_1day.py` + `run_glider_heat_flux_1day.py`
→ `data_heat_1day/`, then `make_sym_1day_heat_figs.py` → `heat_flux_figs/sym_sweep_1day/`.
Advective heating `w·∂T/∂z` and eddy flux `w'T'` are **nonlinear**, so the daily mean does
NOT commute through them; instead these scripts resample the whole model to daily means
(`ds.resample(time='1D').mean()`) BEFORE sampling, so the estimate, the resolved hull
components and the truth pointwise products are all formed from the same daily fields —
one self-consistent "daily-sampled world" in which the `⟨[w·∂T/∂z]⟩ = ⟨[w][∂T/∂z]⟩ + sub-array`
decomposition still holds exactly (the daily analogue of the w rerun, which also
daily-averaged the truth). Sub-daily covariance is therefore excluded from the truth as
well. The figure set mirrors the 3-hourly sym heat-flux sweep (`error_scaling_<fam>`,
`area_components_<fam>{,_flux}`, `area_reducible_<fam>{,_d0.5}`, `point_flux_<fam>`,
`shape_compare_error_scaling_c<lat>`).

## Vertical heat transport

Augments the same plane-fit `w` method to study the **vertical advective heating**
`w·∂T/∂z` — a heating *rate* (°C/day) that integrates vertically to a flux
(ρ₀cₚ∫w∂ᵤT dz). Writing `[·]` for a spatial average over the array footprint and
`⟨·⟩` for a time average, the array's estimate `⟨[w][∂T/∂z]⟩` (product of the two
footprint means) differs from the truth `⟨[w·∂T/∂z]⟩` (footprint mean of the
pointwise product) in two ways: an **irreducible** within-footprint spatial
covariance the array can never see, and **reducible** errors in the plane-fit `[w]`
and the sampled `[∂T/∂z]`. The `w`-skill results above are unchanged. (`run_heat_flux`
also stores the eddy flux `w'T'`, but the summary figures focus on advective heating.)

5. `python run_heat_flux.py` — for a focused set of configs samples U,V,T, forms
   the plane-fit estimate (`w_est` paired with `T̄_glider`) and the model truth
   (with the area-mean components `w̄_hull`, `T̄_hull` as an intermediate). Scope:
   (a) equator single-cell **shape** sweep — diamond/hexagon/square at 1° and 2°
   cell height, each at diameters 0.5/1.0/1.5° (lon offsets 0.25/0.5/0.75);
   (b) off-equator **latitude** sweep — 1° `shift` diamonds, hexagons and squares at
   ±0.5, ±1.5 (configs `shift_w0.5`, `shift_hex_w0.5`, `shift_sq_w0.5`) plus the
   interleaved −1/0/+1 (`shift_w0.5_mid`, `shift_hex_w0.5_mid`, `shift_sq_w0.5_mid`).
   Writes:
   - `data_heat/<config>__cell_<center>.nc` — series (dims time, depth):
     `w_est_mid`, `Tbar_glider`, `wbar_hull`, `Tbar_hull`, `F_true_hull_eddy`
     (eddy flux ⟨w'T'⟩), `A_true_total`/`A_true_eddy` (advective heating ⟨w∂ᵤT⟩)
   - `data_heat/heat_flux_metrics.csv` — per-cell skill (mean, variability ratio,
     correlation, JS/KS of the temporal PDFs at key depths) for both quantities
6. `python run_glider_heat_flux.py` — cheap per-glider companion: samples model
   THETA and WVEL at each glider point (not the hull) so the advective heating can
   be estimated AT each glider by pairing the array's area-mean `w` with that
   glider's OWN `dT/dz`. Writes `data_heat/<config>__cell_<center>__glider.nc`
   (`T_glider`, `w_true_glider` on the 2 m obs axis; plus `T_glider_native`, the same
   THETA columns at the model's *native* vertical resolution — dims time, Zc, glider).
   The 2 m and native columns let us contrast the `dT/dz` a glider profile would
   estimate against the true model `dT/dz` at that point. Note the equator
   single cells include a centre observation at 0°N,140°W — the TAO mooring — used in
   both the plane fit and the array-mean T, so those results are already *with* mooring.
7. `heat_flux_summary.ipynb` — presentation figures for the advective heating
   `w·∂T/∂z`, built directly from the quantities as computed (no term I/II/III
   notation). `[·]` = footprint spatial average, `⟨·⟩` = time average. Every figure has
   an `a` = **equator shapes** variant and a `b` = **shift diamond / hexagon / square**
   variant (shape colours match the equator experiments: diamond blue, hexagon green,
   square red; the shift per-latitude figures share x-limits for comparison).
   Two sets of figures + context maps → `heat_flux_figs/`:
   - **Area-average — components**: three time-mean rows separating the *observable*
     estimate error from the *unobservable* sub-array. (1) **true mean** `⟨[w][∂T/∂z]⟩_true`
     (solid) vs **estimated mean** `⟨[w][∂T/∂z]⟩_est` (dashed); (2) **estimate error** =
     true mean − est mean; (3) **sub-array (unobservable)** = true total − true mean =
     `⟨[w·∂T/∂z]⟩ − ⟨[w][∂T/∂z]⟩_true`, the within-footprint covariance no single footprint
     value can see. Identity: true total = est mean + estimate error +
     sub-array. The equator variant is split by cell height →
     `area_components_equator_{1deg,2deg}.png` (3 shapes each, columns = diameter); the
     shift variant `area_components_shift.png` has one column per centre latitude across
     −1.5…+1.5° (the interleaved −1/0/+1° come from the `*_mid` configs). Each of the three
     is also written as a `_flux` duplicate (`area_components_{equator_1deg,equator_2deg,
     shift}_flux.png`) with every profile vertically integrated into a heat flux
     `ρ₀cp∫·dz` (W/m²) — same decomposition, units only. All figures in the group share one
     set of per-row x-limits (`components_xlims`, computed over every equator diameter/shape
     and shift latitude; °C/day and W/m² spans separately) so any two are directly comparable.
   - **Area-average — reducible**: the plane-fit / sampling errors. Rows = `⟨[w]⟩`,
     `⟨[∂T/∂z]⟩`, the advective heating `⟨[w][∂T/∂z]⟩`, and a 4th row = that heating
     integrated into a heat flux `ρ₀cp∫[w][∂T/∂z] dz` (W/m²); left column = true (solid) vs
     estimate (dashed), right column = **estimate error** (true − estimate). `[w]_est` is
     the plane fit at the moorings, `[∂T/∂z]_est` the glider-average gradient, and the
     heating-row error `⟨[w][∂T/∂z]⟩_true − ⟨[w][∂T/∂z]⟩_est` is identical to the
     estimate-error row of the components figure (the flux row is its depth-integral).
     Equator split by height →
     `area_reducible_equator_{1deg,2deg}.png` (diameter 1°); shift rendered once per
     centre latitude → `area_reducible_shift_{+0.0,+0.5,+1.5}deg.png` (latitude printed in
     the column titles). All figures in the group share one set of per-row x-limits
     (`reducible_xlims`, over the equator shapes + every shift latitude).
   - **Point** (`point_flux_equator_{1deg,2deg}.png`, `point_flux_shift_{+0.0,+0.5,+1.5}deg.png`):
     the array's single area-mean `w` used as a stand-in for the *local* `w` at each glider,
     applied to that glider's own `∂T/∂z`, then averaged over the gliders. Notation here:
     `⟨·⟩` averages over time **and** the gliders (one profile), and `[·]` is the plane-fit
     footprint average — only on the stand-in `w`; local per-glider values stay bare, so the
     truth has no `[·]`. Four columns: truth `⟨w·∂T/∂z⟩` vs estimate `⟨[w]·∂T/∂z⟩`;
     **truth − estimate** (total point error); the w-substitution error; and the gradient
     error (2 m glider sampling vs model native). w-substitution + gradient = the exact
     point error; columns 3–4 share x-limits (w-substitution dominates; gradient is small).
     A 2nd row repeats all four columns with the heating integrated into a heat flux
     `ρ₀cp∫·dz` (W/m²). `point_flux_shift_+0.0deg_mooring.png` is the same but the single
     "glider" is *just* the 0°N,140°W TAO mooring at the cell centre (no glider averaging) —
     each shape's `[w]` against the local mooring flux; the truth is the same for all shapes.
     All figures in the group (equator, shift, and mooring) share one set of per-column
     x-limits (`point_xlims`, °C/day and W/m² rows separately).
   - **Shift summary** (`error_scaling_shift.png`): the **area-average** error vs centre
     latitude, one line per shape — three panels: total area error
     `⟨[w·∂T/∂z]⟩ − ⟨[w][∂T/∂z]⟩_est` and its parts, the estimate error (reducible) and the
     sub-array (unobservable); total = estimate error + sub-array. Two rows: the 0–80 m
     depth-mean heating error (°C/day) and the 0–80 m depth-integrated heat flux error
     `ρ₀cp∫·dz` (W/m²).
   - **Context maps** (`context_maps_reliability.png`): mean `w`, mean advective heating,
     and the temporal variability (std) of `w` and of `∂T/∂z` over the equatorial box at
     25/50/75 m and a 0–80 m depth average, with a representative 1° footprint overlaid —
     showing where a single footprint-average value is least representative. Built from
     `subcell_maps.pkl`.
   - **Symmetric regular-shape sweep** (all figures under `heat_flux_figs/sym_sweep/`): the
     `symhex`/`symdia`/`symsq` families compared across E-W diameter (0.3/0.5/0.75/1.0°), one
     line per centre latitude (0.0/±0.5°N). Per family `<fam>`: `error_scaling_<fam>.png` is
     the summary (area-average heating error vs diameter — total = estimate error + sub-array,
     depth-mean °C/day and depth-integrated W m⁻² rows), analogous to `error_scaling_shift.png`
     but vs diameter; `area_components_<fam>{,_flux}.png` is the structural decomposition
     (columns = diameter); `area_reducible_<fam>{,_d0.5}.png` and `point_flux_<fam>.png` are
     the reducible and point sets at the representative 1.0° diameter (reducible also at 0.5°).
     `shape_compare_error_scaling_c<lat>.png` compares the three shapes at each centre latitude
     (one line per shape: hexagon green / diamond blue / square red).
   Advective heating is °C/day, `w` is m/day, `∂T/∂z` is °C/m. The `W/m²` heat-flux
   panels vertically integrate the heating with `ρ₀cp∫·dz` (`ρ₀=1027 kg/m³`,
   `cp=3994 J/kg/°C`, matching the model's `rhonil` and MITgcm's default `Cp`; from
   `osse_tools.RHO0/CP`), accumulated from the top of the sampled column downward.

8. `python run_subcell_diag.py` — box-wide fields behind the context maps and the
   sub-array covariance. On the same model as run_heat_flux, over the equatorial box
   (217–223°E, −3–3°N), it caches (`subcell_maps.pkl`, per depth 25/50/75 m plus a
   `depthavg` 0–80 m average) the time-mean `w`, `T` and advective heating `⟨w·∂T/∂z⟩`,
   and the temporal variance of
   `w′`, `T′`, `∂T/∂z′`. `heat_flux_summary.ipynb`'s context maps read this cache
   (mean `w`, mean heating, std `w`, std `∂T/∂z`) with the footprint overlaid to show
   where within-footprint structure makes the spatial average least reliable. (The
   script also writes `subcell_scatter.pkl`, no longer rendered.)

## Horizontal-gradient maps

9. `python run_gradient_maps.py [compute|plot|all]` — per-array maps of the plane-fit
   horizontal gradients against the true model field, for the same heat-flux config
   set. Each cell gets four 3×3 figures (depth-average 0–80 m and levels 25/50/70 m).
   Rows are ∂U/∂x, ∂V/∂y and their sum (horizontal divergence δ); columns are **True**
   (the spatially varying model gradient over the footprint), **Estimate** (the single
   plane-fit slope the array reports — constant over the cell, the slope
   `compute_w_planefit` integrates) and **Difference** (True − Est, where the linear
   fit departs). All panels are 101-level filled contours on `cmo.balance`, so the
   Difference column diverges about 0; True/Est share a row scale, Difference its own.
   Because the plane fit and time/depth-averaging are linear, the mean gradients equal
   the gradients of the mean field, so the model is read only to cache the small
   time-mean U, V (`data_grad/gradient_means.pkl`); every figure derives from that
   cache. Split compute/plot like `run_domain_maps.py`; PLOT skips existing figures.
   Writes `experiment_figs/<config>/gradient_maps_{depthavg,25m,50m,70m}_cell_<center>.png`.
   The `SCOPE` also includes the 12 symhex cells (the cached region covers them, so PLOT
   alone suffices — no model re-read). (The symdia/symsq families are not in the gradient-map
   `SCOPE`.)

## Sub-array-scale flux maps

10. `python run_subarray_flux_maps.py [compute|plot|all]` — 2-D maps over the equatorial
    box (217–223°E, −3–3°N) of the **sub-array-scale** vertical advective heat flux,
    defined (as in the summary figures) as the true total minus the flux from mean
    advection, `⟨[w·∂ᵤT]⟩ − ⟨[w][∂ᵤT]⟩` (the within-footprint spatial covariance no single
    footprint-average value can see). Three **1° footprints** — hexagon (`equator_hex1deg`),
    square (`equator_sq1deg`) and diamond (`equator_1deg`) — are each slid to be centred at
    every grid point; the flux is a depth integral `ρ₀cp∫·dz` over 0–75 m (W m⁻²) — from the
    surface (where w=0, so the integrand vanishes) to 75 m — read straight from the model
    truth (no plane fit). One figure, **four panels** (2×2): **at each
    point** (`⟨w·∂ᵤT⟩ − ⟨[w][∂ᵤT]⟩`, the local pointwise flux minus the footprint
    mean-advection — no shape overlay, own colour scale), then **over the hexagon / square /
    diamond array** (`⟨[w·∂ᵤT]⟩ − ⟨[w][∂ᵤT]⟩`, the footprint average of the pointwise field —
    each with its shape overlaid at the box centre, the three sharing one symmetric colour
    scale so the shapes compare directly). Footprint means are formed by convolving with a
    fixed normalised kernel per shape on the model grid (the box is loaded with a 0.5° margin
    so every displayed centre has full coverage); COMPUTE reads the model once and caches the
    maps (`cache/subarray_flux_maps.pkl`), PLOT renders. Writes
    `heat_flux_figs/subarray_flux_maps.png`.

    The same run also writes a **companion figure** `heat_flux_figs/model_flux_decomp_maps.png`
    — the pointwise **temporal** (Reynolds-in-time) decomposition of the advective heat flux
    straight from the model fields, **no array stencil**: three maps (same 0–75 m `ρ₀cp∫·dz`,
    W m⁻²) of the total `⟨w·∂ᵤT⟩`, the mean advection `⟨w⟩⟨∂ᵤT⟩`, and their difference the
    eddy covariance `⟨w·∂ᵤT⟩ − ⟨w⟩⟨∂ᵤT⟩ = ⟨w′·∂ᵤT′⟩` (the time analog of the spatial sub-array
    term). Total & mean-advection share one colour scale; the eddy panel gets its own.

## Domain footprint maps — regular-hexagon diameter comparison

The repo-root `run_footprint_maps.py` and `run_footprint_heat_maps.py` (which slide a
footprint over every grid point and map the plane-fit w-error / advective-heat-flux-error)
each also write a **4-column** comparison of the symmetric REGULAR hexagons at diameters
0.3/0.5/0.75/1.0° (rows = the two tiers: dense hull fit vs the 6-glider stencil), reusing the
existing domain caches (no model read). Output under
`domain/footprint/symhex_compare/footprint_symhex_{werr,heaterr}_70m*.png` (full-domain and
equatorial-crop views). See `run_footprint_maps.symhex_compare` /
`run_footprint_heat_maps.symhex_compare_heat`.

## Footprint distribution sampling — how well does a shape sample its own footprint?

11. `python run_pdf_sampling.py [compute|plot|all]` — for the symmetric REGULAR shape
    sweep at 0°N/140°W — **hexagon** (`symhex`, 6 gliders), **square** (`symsq`, 4) and
    **diamond** (`symdia`, 4), each at E-W **diameter** 0.3/0.5/0.75/1.0° — asks how
    faithfully the handful of glider points represents the TRUE field distribution
    inside the array's convex hull. The square/diamond configs
    (`experiment_1/configs/symsq`, `…/symdia`) are the shape analogues of `symhex`,
    generated by `generate_configs.py` at the same diameters (regular, isotropic on the
    equator, no moorings). At **25/50/75 m and a 0–80 m depth average** it compares the
    array-sampled distribution (`sample_fields`) against the hull truth
    (`model_region`) for the 1-D marginals `T,S,U,V,W,σ₀` and the correlated pairs the
    array is meant to capture — the horizontal Reynolds stress `u'v'`, the vertical
    fluxes of horizontal momentum `u'w'`, `v'w'`, the eddy heat fluxes `u'T'`, `v'T'`,
    `w'T'`, and `T–S`. `W` is the model's true `WVEL` at the
    glider points (spatial-sampling adequacy for `w'T'`), NOT the plane-fit `w`. Metrics:
    **Jensen-Shannon distance** (1-D and 2-D joints) and the **Wasserstein** (earth-mover)
    distance on the marginals in each field's own units (replaces the old KS statistic),
    plus the **correlation/covariance** the array reports vs the truth. Because a 4–6-point
    array has a sparse empirical PDF whose raw JS/W is inflated by glider **count**, each
    distance carries a **random-placement null** (`js_null`, `js2d_null`, `w_null`) — the
    metric from the same number of gliders dropped at random points in the same hull
    (mean/95th pct over draws); the array beats its null when its geometry samples better
    than random placement, and since the shapes differ in count (hex 6 vs sq/dia 4) the
    heatmaps plot **metric − null** for a fair shape comparison.
    COMPUTE reads the model once, samples every config and writes
    `data/pdf_sampling_metrics.csv` (+ `_pairs.csv`) and `cache/pdf_sampling.pkl` (also
    the flagship-1.0° scatter arrays); PLOT renders `pdf_figs/` (no model read). Figures
    (shape colours diamond=blue, hexagon=green, square=red): the JS-vs-diameter grids
    `summary_js_{fields,pairs}.png` (with the null overlaid), the Wasserstein-vs-diameter
    grid `summary_w_fields.png` (per-field panels, own y-axis units, `w_null` dotted),
    `summary_std_ratio.png` (std-dev ratio array/truth, variance capture; note `σ₀` =
    potential density is a field, distinct from this std-dev `σ`), `summary_corr_recovery.png`
    (the dimensionless correlation coefficient `r` of each pair over the hull truth vs the
    array sample — the gap is the array's sampling error in `r`), `summary_js_heatmap.png`
    (**JS − JS_null** for every quantity × config × depth, 1-D marginals and 2-D pair joints
    on separate symmetric `cmocean balance` scales, blue = below the random floor / red =
    above), `summary_w_heatmap.png` (the Wasserstein analogue: one heatmap **per variable**,
    cells `W − W_null` in the field's units, config × depth, own colourbar each),
    `summary_corr_recovery_heatmap.png` (the SIGNED complement: `Δr = r_array − r_true`
    per pair × config × depth on a diverging scale, blue = array under-states / red =
    over-states the true correlation — shows the *direction* of the error JS discards),
    plus per-shape/per-depth detail `field_pdfs_*` and `joint_*` at the flagship 1.0°
    diameter. See `pdf_figs/README.md`. Key result: the 1-D marginals are sampled well
    (low JS, σ-ratio ≈ 1), but the joint eddy-flux distributions (especially `w'T'`) are
    much harder — JS rises with diameter and the array's reported flux correlation
    departs from the truth, worst for the 4-glider square/diamond and best for the
    6-glider hexagon.