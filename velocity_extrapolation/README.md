# velocity_extrapolation — does constant-shear surface extrapolation make sense?

`compute_w_planefit` samples U, V from `min_depth = 8` m downward and, before fitting a
plane, extrapolates into the unsampled 0–8 m with `extrapolate_currents_to_surface`: it measures the
vertical shear ∂U/∂z, ∂V/∂z between the two shallowest sampled levels (~9 and 11 m) and
assumes that shear is constant all the way to the surface. This folder tests that
assumption against TPOSE24 truth in the equatorial box 2°S–2°N, 142–138°W
(218–222°E), top ~20 m, over the 3-month run.

## What is computed

"Shear" = vertical shear of the horizontal currents: ∂U/∂z, ∂V/∂z (per meter) and the
magnitude |S| = √((∂U/∂z)² + (∂V/∂z)²). U (on XG) and V (on YG) are interpolated to the
tracer centers (XC, YC); shear is a centered difference between adjacent Z cell-centers,
placed at the interface midpoint. The top 12 m are 1-m layers, so near-surface shear is
well resolved. Run with:

```
python run_shear_analysis.py [compute|plot|all]
```

`compute` reads the model once, caches the box's U, V to `uv_region.nc` (so re-runs are
instant) and the reduced statistics to `shear_cache.pkl`; `plot` renders `figs/` from the
cache. Band-profile spread (± std) is taken over (daily means × horizontal points)
(day-to-day + spatial variability, not sub-daily).

**Native model levels, no interpolation.** The production estimator samples on a 2 m
`dz_obs` grid, but here we evaluate the constant-shear assumption directly on the model's
own levels (1 m thick in the top 12 m). The constant-shear extrapolation covers the levels shallower than 8 m
(−0.5 … −7.5 m); it is anchored at the shallowest level at/below 8 m (−8.5 m) and uses
the native top shear between −8.5 and −9.5 m (i.e. ∂U/∂z at −9 m). The "deep" shear
alternative is the mean native shear over 8.5–20 m.

## Figures (`figs/`)

- **shear_profiles_latbands / _lonbands** — |S|, ∂U/∂z, ∂V/∂z vs depth (0–20 m), banded
  by 0.5° latitude (1° longitude), mean ± std. The dashed line at −9 m marks the native
  top shear used for the constant-shear extrapolation; the gray band (0–8 m) is what it extrapolates over.
- **velocity_profiles_latbands** — U, V vs depth by band: the curvature the constant-shear
  constant shear ignores.
- **profiles_true_vs_assumed** — box-mean profiles (top 14 m) overlaying the **truth**
  (curved, black) against the **constant shear** (straight, red) for U, V and for the
  shear ∂U/∂z, ∂V/∂z.
- **profiles_errors** — the errors that constant shear produces vs depth: assumed − true shear,
  extrapolated − true velocity (bias ± spatial std), and the resulting horizontal
  divergence — true |div| RMS vs constant-shear |div| RMS (they overlap) with the div-error RMS
  ~15× smaller, i.e. the near-uniform velocity error barely projects onto divergence.
- **profiles_true_vs_assumed_latbands** / **profiles_errors_latbands** — the same two
  figures resolved by 0.5° latitude band (colored). The surface U error is nearly
  latitude-independent (~2.5–3.2 cm/s); the V error grows off-equator (−0.6 → −1.4 cm/s,
  tracking the ∂V/∂z meridional structure); the divergence error RMS is largest right at
  the equator (~0.03×10⁻⁶ s⁻¹) but stays ~10× below the true |div| signal at every band.
- **shear_curvefit** / **shear_curvefit_latbands** — what functional form the model's own
  shear profile follows over 0–20 m. A **power law |S| ∝ d^(−p)** (d = depth) fits with
  R² ≈ 0.99 for all three components; a single exponential A·e^(z/L) only reaches
  R² ≈ 0.81 (L ≈ 6 m). The exponent is p ≈ 0.67–0.73 across 2°S–2°N (slightly smaller at
  the equator), i.e. shear that intensifies toward the surface roughly as an inverse power
  of depth. So to extrapolate the model shear toward the surface, scale it as a power of
  depth, s(z₂) = s(z₁)·(d₂/d₁)^(−p) with p ≈ 0.7 — not a constant and not an exponential.
  For reference each figure also shows the **constant-shear assumption** (the constant-shear
  extrapolation's actual behaviour): a vertical dotted line at the −9 m shear held fixed to
  the surface (`shear_curvefit`), and the p = 0 reference line on the exponent panel
  (`shear_curvefit_latbands`) — making explicit that constant shear assumes p = 0 while the
  model wants p ≈ 0.7. The `shear_curvefit_latbands` R² panel reports the fit
  quality for ∂U/∂z and ∂V/∂z separately (power law ≈ 0.99, exponential ≈ 0.80–0.88 at
  every band), since those components are extrapolated independently rather than combined
  into |S|.
- **maps_depthstats** — depth-mean of time-mean |S|, and the signed depth-mean ∂U/∂z and
  ∂V/∂z, over 0–20 m.
- **maps_8m_timestats** — time-mean / -max / -min |S| at 8 m.
- **maps_shear_differences** — how shear changes with depth: |S|(8 m)−|S|(surf),
  |S|(20 m)−|S|(8 m), |S|(20 m)−|S|(surf), and the ∂U/∂z, ∂V/∂z surface-vs-8 m versions.
- **maps_extrap_error** — mean-state surface-velocity error (cm/s) of the constant shear using the
  constant −9 m shear vs. an 8.5–20 m mean shear, and their difference.
- **maps_uv_error** — spatial structure of the 0–8 m-mean U, V error from the constant shear, with
  each panel's spatial **mean** and **std** annotated. Divergence sees only the gradients,
  so the std (structure) is what matters, not the mean (uniform part).
- **maps_divergence_w_error** — the actual w consequence: the true near-surface w(8 m)
  from 0–8 m convergence, and the w error without the constant shear (skip the layer, w=0 at
  8 m) vs. with the constant shear, plus where constant shear helps. Area-RMS annotated.

### Time evolution & a statistical extrapolation rule

Everything above collapses time to a single mean. These figures keep time, to ask two
things: do the snapshots look like the mean,
and is there a rule that maps the 8 m current to the shallower current at each
observation — specifically a per-lat/lon regression of u(z), v(z) across time onto the
8 m currents u(8 m), v(8 m). All computed from the cached `uv_region.nc`.

- **shear_hovmoller** — box-mean |S|, ∂U/∂z, ∂V/∂z vs (time, depth), 0–20 m, over the
  3-month run. Shows the surface intensification is a persistent feature: the top 1–2 m
  is the strongest-sheared layer at essentially every time step. ∂U/∂z is persistently
  negative near the surface; ∂V/∂z is surface-intensified but intermittent, arriving in
  ~2-week bursts. The dashed line is the 8 m sampling limit — the 0–8 m band above it
  is what the estimator must extrapolate.
- **profiles_snapshots** — each daily box-mean profile (thin) over the time mean (black),
  with the 10–90%-of-days envelope, for U, V, ∂U/∂z, ∂V/∂z. The snapshots are offset /
  rescaled copies of the same curved shape: the magnitude varies day to day but the
  curvature (surface intensification) is there every day. So the mean profile is
  representative, and a fixed-shape extrapolation is at least self-consistent.
All regressions here are local: at each lat/lon a column's 0–8 m curve is predicted from
that column's own 8 m time series (per-point coefficients). (A single pooled "global" rule
gives essentially identical skill — the rule is nearly uniform in space — but local is the
framing of interest: predict the curve from its own 8 m measurements.)

- **regression_rule** — the fitted rule vs depth (box-mean over points): the gains
  b (u←u(8 m), v←v(8 m)), the cross gains (veering), and the held-out R². The gains
  are ≈ 1.00–1.01 (the surface current amplifies the 8 m fluctuation by only ~1 %),
  the cross gains are ≈ 0 (negligible rotation with depth over 0–8 m), and R² ≈ 0.98–1.0.
  So the rule collapses to u(z) ≈ u(8 m) + a(z) — a fixed depth-dependent mean
  offset (the intercept: ≈ −5.8 cm/s in U, +1.9 cm/s in V at the surface). The 8 m current
  explains essentially all of the *fluctuation*; the only thing to add is the steady mean
  shear.
- **regression_skill** — **block-CV held-out** (train one contiguous half of the record, score
  the other, both ways) RMS reconstruction error of the 0–8 m current vs depth for five schemes:
  persistence (hold the 8 m value), the instantaneous constant shear, and the regression fit
  onto u(8 m),v(8 m) only, + the measured 8 m shear, and + u,v,shear at four depths.
  The u(8 m),v(8 m)-only rule roughly halves the persistence error (removing the mean
  offset) but is beaten by constant shear at every depth (no curvature information). Adding
  the measured 8 m shear beats constant shear at every depth — cutting the surface error ~3×
  (U 3.05→1.01, V 1.27→0.66 cm/s), because it both projects the instantaneous shear *and* learns
  to amplify it. Adding u,v,shear at four depths (8.5–14.25 m) is worse, not better under
  this fair split (U surface 1.69, V 1.70): the 17-predictor model overfits (see Finding 9).
- **regression_matrix_maps** — one figure for the whole shear-augmented rule: the surface
  (−0.5 m) spatial map of every component of its 2×4 transfer (gain) matrix — targets (u,v)
  × predictors (u(8 m), v(8 m), ∂u/∂z, ∂v/∂z) — all on one shared colorscale so the
  component magnitudes are directly comparable. Diagonal "gain" terms ≈ 1 dominate; off-diagonal
  "cross gain" terms ≈ 0. The two shear columns are shown as the ratio to the geometric
  constant-shear distance (1 = plain constant shear): the ∂U/∂z gain is ≈ 1.17 ± 0.06
  (uniform → a single "scaled constant shear" works for U), the ∂V/∂z gain ≈ 1.41 ± 0.09
  and latitude-dependent (strongest just south of the equator).
- **regression_r2_maps** — reconstruction skill (vector R² of U,V, averaged over 0–8 m) per
  method, one shared colorscale. All methods sit at R² ≈ 0.985–1.000: because u(8 m) explains
  almost all of the variance, R² is a *saturated, weakly discriminating* metric here — which is
  exactly why the cumulative-error map is the more useful one.
- **regression_cumerr_maps** — the cumulative (0–8 m depth-integrated) velocity error per
  method: time-RMS of |∫₀⁸(δu,δv)dz| (m² s⁻¹). This is the metric that matters for w — the layer
  errors *integrate* (a sign-coherent bias accumulates, a zero-mean wiggle cancels), and this
  depth-integrated transport error is what the horizontal divergence turns into a w error.
  Area-means: persistence 0.264, constant shear 0.082, u(8 m),v(8 m)-only regression 0.148
  (worse than constant shear — its residual is shear-shaped and coherent), + 8 m shear 0.038
  (best), + 4 depths 0.054 (overfit). So the integrated metric reorders the middle of the pack
  even where the depth-averaged R² barely separates them.

## Findings

1. **Shear intensifies strongly toward the surface.** |S| grows from ~0.4×10⁻² s⁻¹ at
   10–20 m to ~2×10⁻² s⁻¹ in the top 1–2 m — a factor of ~4–5. Within 0–20 m the depth-max
   and depth-min of |S| differ by ~8×. The near-surface velocity profile is clearly
   curved, not linear (velocity_profiles figure). A single constant shear
   cannot represent this column.

2. **So the constant shear systematically under-estimates near-surface shear**, and
   therefore under-estimates surface current magnitude. In the mean state the reconstructed
   surface velocity is off by ~2.8–3.5 cm/s (maps_extrap_error). |S|(8 m) is already
   ~1.3×10⁻² s⁻¹ weaker than |S|(surface) everywhere (maps_shear_differences), and the
   −9 m shear the constant shear uses is weaker still.

3. **It fails about the same at all latitudes.** The near-surface intensification of
   ∂U/∂z is remarkably latitude-independent across 2°S–2°N (the band curves collapse), and
   the |S| deficit and surface-velocity error are fairly uniform (slightly larger just
   south of the equator). ∂V/∂z *does* carry latitude structure (it changes sign across the
   band), but that is a horizontal-gradient signal, not a change in how well the
   extrapolation works.

4. **Averaging shear over a longer/deeper range does NOT help — it makes it worse.**
   Using the 8.5–20 m mean shear instead of the −9 m shear increases the surface-velocity
   error everywhere (the difference panel is entirely positive). Because shear increases
   toward the surface, a deeper average is even weaker and under-shoots more. The only
   corrections that would help go the other way: extrapolate the rising shear trend
   rather than a constant.

### Time evolution and a statistical rule

5. **The snapshots look like the mean.** The surface-intensified, curved profile is present
   at essentially every time step (shear_hovmoller, profiles_snapshots): day-to-day the
   magnitude varies (∂V/∂z arrives in ~2-week TIW bursts) but the shape does not. The
   time-mean profile the earlier figures use is therefore representative of the individual
   observations, and a fixed-shape extrapolation is at least self-consistent in time.

6. **A regression of u(z),v(z) on u(8 m),v(8 m) reduces to "add a mean offset."** At every
   lat/lon the fitted gain is ≈ 1 (surface amplifies the 8 m fluctuation by only ~1 %),
   the cross gains ≈ 0 (no meaningful veering over 0–8 m), and the held-out R² ≈ 0.98–1.0 —
   the 8 m current already explains essentially all of the *temporal variance* of the
   shallower current. The only thing the rule adds is the intercept: a steady depth-dependent
   **mean offset** (≈ −5.8 cm/s U, +1.9 cm/s V at the surface).

7. **But that statistical rule does NOT beat the instantaneous constant shear** the estimator
   already uses (regression_skill). Removing the mean offset roughly halves the persistence
   error, yet constant shear has lower reconstruction RMS at every depth (U at −4.5 m: 0.5 vs
   1.4 cm/s; V everywhere). The reason: with gain ≈ 1 a fixed rule can only correct the
   *mean* shear, whereas the near-surface curvature is *time-varying*, and constant shear
   captures it by measuring ∂u/∂z at each instant. Given only u(8 m),v(8 m) there is no
   rule that does better than a mean-offset correction — the useful extra information is the
   locally *measured* shear, not any fixed function of the 8 m velocity.

   All skill numbers below are **block cross-validated** (fit on one contiguous half of the
   record, scored on the other, both ways) so they test generalization to an independent
   period, not interpolation between autocorrelated 3-hourly neighbours.

8. **Regression on measured 8 m shear beats constant shear at every depth**
   (regression_skill). Regressing u(z),v(z) onto u(8 m),v(8 m) **and** ∂u/∂z,∂v/∂z at 8 m
   cuts the surface reconstruction error ~3× below constant shear (U 3.05→1.01, V
   1.27→0.66 cm/s) and drives R² to ≈ 1.000. It works for two reasons: it keeps constant
   shear's instantaneous, time-varying shear, and it learns to amplify it — the fitted
   gain on the measured shear is ≈ 1.06–1.17× the geometric extrapolation distance, which
   is exactly the correction for the near-surface intensification that plain constant shear
   (gain = 1×distance) omits. So the deployable "rule at each observation" is a scaled
   constant shear: u(z) ≈ u(8 m) + (mean offset) + ~1.1·(z+8.5)·∂u/∂z|₈.

9. **Adding more depths overfits — the block-CV test exposes it.** Extending the predictors to
   u,v and ∂u/∂z,∂v/∂z at four sampled depths (8.5–14.25 m; 17 predictors) looked marginally
   *better* under an interleaved even/odd split (surface U 0.94→0.91), but under the fair
   contiguous-block split it is worse. The extra depths add parameters that fit period-specific noise and
   do not generalize; the anchor velocity + the 8 m shear already carry all the robust
   information. 

10. **For w, the depth-integrated error is what matters, and it reorders the methods**
   (regression_cumerr_maps vs regression_r2_maps). Depth-averaged R² is near-saturated for every
   method (0.985–1.000) because u(8 m) explains almost all the variance, so it barely
   discriminates. But w is built by integrating the layer velocities, so a sign-coherent bias
   accumulates while a zero-mean error cancels. Scoring the time-RMS of the depth-integrated
   error |∫₀⁸(δu,δv)dz| gives area-means of: persistence 0.264, constant shear 0.082,
   u(8 m),v(8 m)-only regression 0.148 (worse than constant shear — its residual is
   shear-shaped and coherent, so it integrates), + 8 m shear 0.038 (best), + 4 depths 0.054
   (m² s⁻¹). So the integrated metric both confirms the +shear rule as best and reveals that the
   velocity-only regression, which looks competitive by R², is actually worse than constant
   shear for the divergence/w purpose.

## The divergence consequence — why the constant-shear extrapolation is still worth doing

The plane-fit w depends on the horizontal divergence of the extrapolated velocity, not
its absolute value, so any spatially uniform part of the U/V bias contributes nothing to
w. This turns out to rescue the constant shear:

11. **The velocity error is ~90% spatially uniform** (maps_uv_error). The 0–8 m-mean U
   error is ≈ +0.9 cm/s but with a spatial std of only ~0.07 cm/s; the V error is
   ≈ −0.3 ± 0.10 cm/s. Only the ~0.1 cm/s *structured* part can project onto divergence.

12. **So the constant-shear extrapolation nearly eliminates the near-surface w error** (maps_divergence_w_error).
   The true w built from the 0–8 m convergence has area-RMS ≈ 0.22 m/day. **Skipping** the
   layer (w = 0 at 8 m, the `extrapolate_to_surface=False` path) leaves that entire
   0.22 m/day as error — and because it is an offset at 8 m, it contaminates the whole w
   profile *below* 8 m too. The **constant shear** cuts that to ≈ 0.01 m/day, a ~20×
   reduction: even with the wrong shear magnitude, the extrapolated field reproduces the
   near-surface *divergence* almost exactly.

**Summary:** As a reconstruction of the near-surface *velocity profile* the
constant-shear assumption is poor (Findings 1–4: it underestimates surface shear ~4–5×,
and a deeper shear average is worse, not better). But for the estimator's actual job —
getting the horizontal divergence, and hence w, right — it is reasonably justified, because its
error is dominated by a spatially uniform bias that divergence discards. Turning the constant-shear extrapolation
off is clearly worse than leaving it on. The residual structured error (~0.01 m/day
here, largest just off the equator where ∂V/∂z has meridional structure) is small; whether
it matters for a specific array is best judged per footprint (the shear varies with
latitude, so an array straddling the equator sees the most). A purely statistical
alternative helps only if it is given the right predictor (Findings 5–9): regressing the
shallower current on the 8 m velocity alone just re-adds the mean shear (gain ≈ 1) and is
out-performed by constant shear; but regressing on the 8 m velocity plus the measured 8 m
shear beats constant shear at every depth (surface error ~3× smaller), because it scales
the measured shear up by ≈ 1.1× to correct the surface intensification constant shear
ignores. Adding predictors from deeper than 8 m saturates. So the best deployable rule is a
scaled constant shear — but note this only sharpens the near-surface velocity
reconstruction; for the estimator's actual job (divergence → w) plain constant shear is
already within ~0.01 m/day (Findings 10–11).

## Power-law shear & a regression extrapolation — do either improve w? (`power_law_extrap.py`)

Constant shear underestimates the mean near-surface shear because the mean profile
intensifies toward the surface roughly as a **power law** |S| ∝ d^(−p) (d = depth; the
mean-profile fit gives **p ≈ 0.70**, so a power-law fill amplifies the measured 8 m shear by
1/(1−p) ≈ **3.4×** at the surface; p = 0 recovers constant shear). Two questions this
companion script answers, reading the same cached `uv_region.nc`:

- **powerlaw_mean_vs_var** — (a) the **mean** surface-ward velocity increment follows the
  power law almost exactly (model 6.3 cm/s at 0.5 m vs power-law 6.7; constant shear gives
  only 3.3 — it undershoots the *mean* profile ~2×). (b) But the regression gain
  ∂u(z)/∂u(8 m) ≈ **1.00** at every depth and the Ekman cross-terms ≈ 0, so the
  fluctuations do not surface-intensify — the power law's ×3.4 amplification is a
  mean-profile property only.
- **powerlaw_w_scatter** — instantaneous w_error vs w_true, per scheme (true instantaneous
  w RMS ≈ 2.1 m/day; all schemes reach skill 1−σ²ₑ/σ²ₜ ≈ 1.00, so the fill perturbs w only
  slightly). The error RMS ranks: **constant shear 0.046 < regression+shear 0.062 < power
  law 0.098 < regression(u,v-only) 0.102 m/day** — power law is ~2× worse than constant
  shear, and the (u,v)-only regression (gain ≈ 1 ⇒ no shear-driven convergence) is worst.
- **w_error_maps** — time-mean cumulative upwelling w(z) = −∫₀ᶻ∇·u dz′ (truth) and each
  scheme's mean w-error, by depth (rows: 0.5/2.5/4.5/6.5 m + the full 0–8 m) × scheme
  (columns). The truth grows with depth as the convergence integrates (area-RMS 0.029 → 0.221
  m/day); the error columns share one scale. For the mean, every scheme is tiny (0.004–0.013
  m/day at 8 m vs the 0.22 m/day signal); power law (0.0057) does not beat constant shear
  (0.0052).
- **w_error_profiles** — vertical profiles of the cumulative-w error for each scheme: (left)
  the true cumulative w(z) magnitude for context (RMS grows 0.27 → 2.1 m/day over 0–8 m), then
  (middle) the w-error RMS in time √⟨e²⟩ₜ (box row also over space; not the RMS of the
  time-mean map that w_error_maps shows — this keeps the fluctuating part, so it is ~10× larger)
  and (right) the w-error bias (time mean) vs depth, one line per
  scheme. RMS grows monotonically with depth and the last point (7.5 m) equals w at 8 m; the
  ranking (constant shear < regression+shear < power law < regression(u,v)) holds at every
  depth, and RMS ≫ |bias| for all schemes (the error is structured fluctuation, not a mean bias).
  **Rows**: the box mean, then single grid points at **0°N, +0.5°N, −0.5°N (all 140°W)** — the
  same ranking holds point-by-point; single-point RMS runs a bit higher than the box mean (no
  spatial averaging), largest just off the equator.
- **w_error_contours** — the same cumulative w-error binned by the 8 m flow state (u8, v8),
  **rows = depth (0.5/2.5/4.5/6.5 m), cols = scheme**, one shared scale. Shows *where in
  flow-state space* each fill injects convergence error: constant shear stays flattest, the
  power law tilts hardest (it over-amplifies the fluctuating shear of strong flows).
- **extrap_error_contours_by_method** — velocity error vs the 8 m flow state (u8, v8) at
  0.5 m for **all four schemes** (constant / power / regression / regression+shear), on a
  fixed shared ±2.5 cm/s scale. Constant shear carries a strong state-dependent surface bias
  (RMS 2.8 cm/s across the (u8,v8) plane — it undershoots strong flows); power law and
  regression flatten it (0.8, 0.6, 0.4 cm/s). That systematic velocity error is
  ~divergence-neutral, so flattening it does not help w (see the scatter).
- **extrap_error_contours_{const,power,reg,regsh}** — one figure per scheme: its velocity
  error vs (u8,v8) at 0.5/2.5/4.5/6.5 m (rows u/v, cols depth), same fixed ±2.5 cm/s scale.
  The regression residual is small and only weakly state-dependent, i.e. a single linear
  (u8,v8) rule captures the instantaneous relationship well (little Ekman turning or
  nonlinearity to exploit); constant shear shows the strongest surface tilt, fading toward 8 m.

**Summary:**

13. **The power law describes the mean profile but not the variability.** The mean 0–8 m
    increment matches |S| ∝ d^(−0.70) to a few %, while the instantaneous regression gain is
    ≈ 1.0 with negligible Ekman turning. So near-surface *fluctuations* are nearly
    depth-uniform over 0–8 m even though the *mean* is strongly surface-intensified.

14. **Power-law extrapolation does not improve the sampled upwelling** — it is ~2× worse
    instantaneously and no better for the mean.** Because it applies the mean's 3.4× shear
    amplification to every snapshot, it over-extrapolates the fluctuating shear and injects
    structured convergence error (w-error RMS 0.098 vs constant shear 0.046 m/day).
    Constant shear remains the best scheme for w.

15. **Velocity-profile accuracy ≠ w accuracy.** Power law and regression greatly reduce the
    near-surface *velocity* bias (contour maps), yet leave w unchanged or worse — because
    that bias is largely spatially uniform and so divergence-neutral. The practical lesson:
    for upwelling, do **not** optimize the velocity reconstruction; the simple constant-shear
    fill is already near-optimal for the divergence, and the physically "better" power law is
    actively counter-productive for w.

Run with `python power_law_extrap.py [compute|plot|all]` (caches to `powerlaw_cache.pkl`).
