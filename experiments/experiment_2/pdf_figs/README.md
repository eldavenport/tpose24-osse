# pdf_figs — how well does an array sample the true field distribution inside its footprint?

Built by `../run_pdf_sampling.py`. For the symmetric REGULAR shape sweep at
0°N/140°W — **hexagon** (`symhex`, 6 gliders), **square** (`symsq`, 4 gliders) and
**diamond** (`symdia`, 4 gliders), each at E-W diameter **0.3 / 0.5 / 0.75 / 1.0°** —
this compares, at **25 / 50 / 75 m** and a **0–80 m depth average**:

- the distribution the **array** sees (model fields interpolated to the handful of
  glider positions, `sample_fields`), against
- the **truth** inside the footprint (model fields at every grid point in the convex
  hull of the array, `model_region`).

The square/diamond configs (`experiment_1/configs/symsq`, `…/symdia`) are the
shape analogues of `symhex`, generated at the same diameters (diameter = E-W width =
2 × lon offset; both isotropic on the equator). All are at 0°N/140°W with no
moorings, so this is a clean regular-hexagon / square / diamond × diameter comparison.

## Quantities

- **1-D marginals** for `T`, `S`, `U`, `V`, `W`, `σ₀`. `W` is the model's true `WVEL`
  sampled at the glider points (spatial-sampling adequacy for `w'T'`), **not** the
  plane-fit `w` estimate — that estimator error is the subject of the heat-flux
  figures.
- **Correlated pairs** the array is meant to capture: the horizontal Reynolds stress
  `u'v'`, the vertical fluxes of horizontal momentum `u'w'`, `v'w'`, the eddy heat
  fluxes `u'T'`, `v'T'`, `w'T'`, and the `T`–`S`
  relation. Primes are temporal-eddy anomalies (value minus its own time mean at each
  location, after the depth reduction), pooled over time and space. 
  
  NOTES FROM ELLEN: *** we don't want pooled over space *** we want the true underlying distribution of the model. the only "pooled" over space is w', which we will have to substitute the plane fit for in the array estimate. but, taken from the model as truth, these quantities should not be averaged in time or space. 

## Metrics

- **Jensen–Shannon distance** (0 identical, 1
  disjoint) on the 1-D marginals and the 2-D joint distributions.
- **Wasserstein** (earth-mover) distance on the 1-D marginals, in **each field's own
  physical units** — tail/spread-sensitive and directly interpretable, so it ranks the
  patterns within a variable. It replaces the old KS statistic in every figure.
- **Correlation coefficient / covariance** the array reports vs the truth — the
  quantity of interest for the eddy fluxes (does the 4–6-point array recover the sign
  and strength of `⟨w'T'⟩`, `⟨u'v'⟩`, …?).
- **Random-placement null**, now for BOTH distances (`js_null`, `js2d_null`, `w_null`,
  each with its 95th pct): the metric you would get from the SAME number of gliders
  dropped at RANDOM points in the same hull (mean over 40 draws). Because an array only
  has 4–6 fixed points, its empirical PDF is inherently sparse and its raw JS/W is
  inflated by glider **count** alone; the null is that noise floor. **Metric at or below
  its own null ⇒ the geometry samples the footprint as well as (or better than) random
  placement of the same glider count.** The shapes differ in count (hexagon 6 vs
  square/diamond 4), so **metric − null** is the fair shape-to-shape comparison — the
  heatmaps plot exactly that.

## Figures

Shape colours: **diamond = blue, hexagon = green, square = red** (matching the
heat-flux experiments).

Summaries (rows = depth 25/50/75 m + depth-avg, columns = quantity, x = diameter,
one line per shape):

- `summary_js_fields.png` — JS of each 1-D field vs diameter; dotted line = that
  shape's random-placement null.
- `summary_js_pairs.png` — 2-D JS of each correlated pair vs diameter; dotted = null.
- `summary_w_fields.png` — Wasserstein distance of each 1-D field vs diameter, one line
  per shape, dotted = that shape's `w_null`. Each column is a field with its OWN y-axis
  (physical units differ), so read pattern-vs-pattern within a panel, not across.
- `summary_std_ratio.png` — std-dev ratio, std(array)/std(truth) per field (1.0 = the
  array captures the full spatial+temporal spread; <1 = it under-samples the variance).
  This `σ` is a standard deviation, distinct from the `σ₀` potential-density *field*.
- `summary_corr_recovery.png` — the dimensionless correlation coefficient `r` the array
  reports (dashed, open markers) vs the truth (solid) for each pair; how faithfully the
  array recovers `r[w'T']`, `r[u'v']`, …. The solid–dashed gap is the sampling error in `r`.
- `summary_js_heatmap.png` — **JS − JS_null** per quantity × config, one panel-block per
  depth, split into the 1-D field marginals and the 2-D pair joints on **separate
  symmetric colour scales** (the pair JS runs higher). Diverging `cmocean` `balance`
  about 0: **blue = below the random-placement floor** (geometry beats random of the same
  glider count), white = at the floor, **red = above it** (worse than random). Subtracting
  the null makes the hexagon-6 vs square/diamond-4 comparison fair; limits clipped to each
  block's robust |value| spread.
- `summary_w_heatmap.png` — the Wasserstein analogue: **one heatmap per variable** (units
  differ, so each field gets its own colourbar). Cells = `W − W_null` in the field's units,
  rows = the 12 shape×diameter configs, cols = depth, same diverging `balance` about 0.
  This is the "which pattern samples each variable best" view.
- `summary_corr_recovery_heatmap.png` — the SIGNED complement to the JS heatmap: the
  correlation-recovery error `Δr = r_array − r_true` per pair × config, per depth. JS is
  an unsigned distance; this shows the *direction* of the error — a diverging `cmocean`
  `balance` scale about 0 (**blue = array under-states the true correlation, white =
  recovers `r` exactly, red = over-states**), symmetric limits clipped to the robust
  |Δr| spread. Pairs only (correlation recovery is undefined for the 1-D fields).

Detail (flagship 1.0° diameter, one figure per shape × depth):

- `field_pdfs_<shape>_d1.0_<depth>.png` — the six 1-D marginals, array (red step) over
  truth (grey filled), annotated with JS and Wasserstein (W, field units), each shown
  against its random-placement null.
- `joint_<shape>_d1.0_<depth>.png` — the seven joint scatters, array points over the
  truth cloud, annotated with array vs truth correlation and the 2-D JS against its null.
  **Each axis is
  standardized by the truth mean/std** (units of `σ`, a common transform for both
  clouds) so the correlation shape is legible despite the pairs' very different physical
  scales (e.g. `w'`~10⁻⁴ m/s vs `T'`~0.5 °C would otherwise flatten the momentum-flux
  clouds to a line); `r` and JS are invariant under this rescaling, and the array cloud
  stays narrower than the truth where it under-samples the variance.

## Data

`../data/pdf_sampling_metrics.csv` (1-D fields) and `…_pairs.csv` (pairs) hold every
number behind the figures; `../cache/pdf_sampling.pkl` also stores the flagship
scatter arrays so `run_pdf_sampling.py plot` re-renders without touching the model.

Sampling: 8–80 m at 4 m resolution, time subsampled by 3 (~216 steps). `W` in m/s,
`T` in °C, `S` in g/kg, `σ₀` in kg/m³.
