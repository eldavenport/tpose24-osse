# sampling_dynamics_control_volume — the OCV flux budget

A wave-glider array as a **finite-volume flux budget**, not a point sampler. We ask how
faithfully the array reproduces the *time-evolving, space-averaged* ocean, keeping spatial
and temporal variability separate: we reduce the high-res truth to a few space-averaged
quantities first, then compare only their **temporal** PDFs. The headline question:
**at what diameter D do the obs PDFs converge to the truth?**

(The sibling `sampling_dynamics_point_cloud/` study pools every grid point × every time into
one histogram, which mixes the two kinds of variability; this study replaces that.)

## The observation control volume (OCV)

The OCV is a polygonal prism centred at 0°N, 140°W, extruded over the 8–80 m column in
dz = 2 m layers. The **circle of diameter D is inscribed in the polygon**: one glider sits
at the **centre of each face**, on the circle, and the polygon vertices lie *outside* at
circumradius `(D/2)/cos(π/n)`. A TAO mooring sits at the centre. The geometry is built for a
general regular n-gon: we sample **symhex** (hexagon, 6 faces), **symsq** (square, 4 faces)
and **symdia** (diamond, 4 faces — the square rotated 45°) at D = 0.3 / 0.5 / 0.75 / 1.0°.
The enlarged vertices are derived in code by scaling the shared config vertices outward by
`1/cos(π/n)`; the shared config JSONs are left untouched. Cross-shape figures colour the
shapes consistently: **diamond blue, square red, hexagon green** (`C.SHAPE_COLOR`).

## Method / theory

**Area-averaged w — the divergence theorem.** The Boussinesq ocean is incompressible,
`∇·u = 0`. Integrate over one layer of the OCV prism (horizontal area `A`, thickness `dz`)
and apply the divergence theorem, `∮_∂V u·n dA = 0`. The lateral faces carry only the
horizontal flow; the top and bottom carry `w`:

```
Σ_faces ⟨u·n_out⟩_i · L_i · dz  +  A·(⟨w⟩_top − ⟨w⟩_bot)  =  0
⇒  ⟨w⟩_bot = ⟨w⟩_top + (dz/A) · Σ_i ⟨u·n_out⟩_i · L_i ,   marched down from ⟨w⟩ = 0 at the surface.
```

So `w` exists **only as an average over the horizontal cross-section A** — the budget never
resolves its spatial structure. There is **no plane fit**: the divergence is a geometric flux
sum around the polygon (`div = (1/A)·Σ ⟨u·n⟩_i L_i`, the area-mean divergence), integrated
by the same cumulative-sum continuity as the old plane-fit `w`.

**Vertical heat flux — the same budget, with storage.** For advective heat conservation
`∂T/∂t + ∇·(uT) = 0`,

```
(wT)_bot = (wT)_top + dz·∂⟨T⟩/∂t + (dz/A)·Σ_i ⟨u·n T⟩_i · L_i ,   from (wT) = 0 at the surface,
```

where `∂⟨T⟩/∂t` (the **heat storage** term) is the time tendency of the OCV volume-mean T.
This is advective-only (ignores diffusion and surface forcing) — a documented approximation
applied **identically** to truth and obs, so the comparison is consistent.

**Heat diagnostic = advective HEATING, not a flux.** An advective heat *flux* `ρcp·w·T` is
reference-dependent (it changes with T's zero point, and with net upwelling carries
`ρcp·T_ref·⟨w⟩`). We instead report the advective **heating rate** `u·∇T` [°C/day] — the
quantity the rest of the repo uses (`osse_tools.advective_heating`; `sample_model_heat_flux`
explicitly refuses to return the reference-dependent `⟨wT⟩`). Because it depends on temperature
**gradients**, it is reference-**independent** (a constant added to T drops out of any
derivative). Two reference-free components, both from cached fields (`cv_plot.adv_heating_*`):
- **vertical** `w·∂T/∂z` = `⟨w⟩·∂⟨T⟩/∂z` (continuity w × gradient of the OCV-mean T); ~+0.04 °C/day
  at depth (upwelling into stratification → cooling tendency).
- **horizontal** `u·∇ₕT`, obtained by the divergence theorem as the face flux of the temperature
  **anomaly**: `(1/A)Σ_faces ⟨u_n(T−⟨T⟩)⟩ L = (1/A)Σ (unT − Tbar·un) L`. Subtracting `⟨T⟩` removes
  the reference-heat carried by the net mass flux — the same term that made the flux frame-
  dependent — so it too is reference-free. ~−0.03 °C/day (warm advection).
- **total** `u·∇T` = horizontal + vertical.

`budget_closure.png` checks the advective tendency `−u·∇T` against the actual `∂⟨T⟩/∂t`: they do
**not** match (advective warming vs actual cooling), the gap being the non-advective residual
(KPP turbulent mixing + surface forcing) — the expected equatorial cold-tongue balance. The
per-face lateral heat term is likewise the reference-free anomaly flux `u_n(T−⟨T⟩)`
(`cv_plot.face_anom_flux_da`). (The old heat-budget flux `wT` remains in the cache, unused.)

**Truth vs obs.**
- *Truth* averages the outward-normal current/flux over each **full face** (every high-res
  model point along the edge) and takes the OCV volume-mean over the polygon interior. With
  the true face-averages the budget is **exact** to incompressibility, so the truth-integrated
  `w` should equal the model's directly-diagnosed area-mean `⟨WVEL⟩` — the **validation
  overlay** (verified: r = 1.000, magnitudes within ~2–4%).
- *Obs* replaces each face-average by the **single glider at the face centre** — the
  **midpoint rule**, exact only if `u·n` is linear along the edge, with error
  `~ (L²/24)·∂²(u·n)/∂ℓ²` that grows with face length `L` (i.e. with D) and along-face flow
  curvature, then accumulates through the vertical integral. The OCV volume-mean is the
  average of the face gliders + mooring.

**The D tradeoff.** Small D → short faces (accurate midpoint) but weak, noisy signal; large
D → strong signal but the single face-centre obs misses along-face structure (TIW meanders,
eddies). The convergence diameter is where the obs temporal PDFs meet the truth. (This is a
milder assumption than the plane fit: it needs `u·n` linear along each edge, not `u,v` linear
across the whole interior, and it respects the true polygon geometry.)

## Model run

`oct2012_3mo_dt60_AB3` (ΔT = 60 s, 3-hourly output) — the only tpose24 run carrying the KPP
diagnostics, so the loaders match the point-cloud study.

## Pipeline

1. **`common.py`** — config, dt60 loader (`load_bbox_memory`, `region_bbox`), and the OCV
   geometry: `ocv_vertices` (azimuth-sorted, scaled to inscribe the circle),
   `face_geometry` (face centres, outward unit normals, edge lengths, area A),
   `glider_positions`/`array_positions` (face-centre gliders + mooring),
   `edge_sample_positions` (dense truth edge sampling), `select_polygon` (interior mask).
   Reuses `osse_tools` for the metre projection, C-grid sampling, and surface extrapolation.
2. **`run_ocv.py`** (compute) — reads the model bbox once, then per config writes
   `cache/<name>_ocv.nc` (run `python run_ocv.py` for all shapes, or
   `run_ocv.main(shapes=[...])` for a subset):
   - per-face outward current/heat flux `un_{true,obs}` / `unT_{true,obs}` `(time, face, obs_depth)`,
   - OCV volume means `{T,S,U,V,W}bar_true` / `{T,S,U,V}bar_obs` `(time, obs_depth)`,
   - budgets `w_{true,obs}` / `wT_{true,obs}` `(time, depth)` and the divergence diagnostics.
3. Figure scripts (read the cache only). Each writes **per-shape** figures to
   `<category>/<shape>/` and **cross-shape** comparisons to `<category>/compare/`:

| script | per-shape (`<cat>/<shape>/`) | cross-shape (`<cat>/compare/`) |
|--------|------------------------------|--------------------------------|
| `run_faces.py`  | `layout.png` (inscribed circle, polygon, gliders, normals, **numbered faces**); per-face current & anomaly-heat-flux `u_n(T−⟨T⟩)` profiles, all D overlaid; `midpoint_error.png` = per-face normalized **RMSE (row 1) + signed bias (row 2)** vs D | `midpoint_error.png` = **4 rows** × (current, anomaly heat flux): RMSE face-mean, RMSE per-face, bias face-mean, bias per-face. Shapes green/red/blue; individual faces by **line opacity** |
| `run_budget.py` | `w_profiles.png`; `heating_profiles.png` = advective heating (**horizontal / vertical / total** panels) [°C/day], truth vs obs, all D; `budget_closure.png` = advective tendency vs actual `∂ₜ⟨T⟩` | `w_profiles.png` & `heating_profiles.png` (3 shapes overlaid); `error_vs_D.png` = normalized **RMSE + bias** of integrated w / total advective heating (obs vs truth, observed 8–80 m) vs D, one line per shape |
| `run_pdfs.py`   | `w_pdfs.png` / `heating_pdfs.png` (panel per D); `convergence.png` (JS vs D, base+mid depth); `volmean_pdfs_{30m,70m}.png` (T,S,U,V rows × D cols); `face_pdfs_{30m,70m}.png` (face rows × D cols) | `convergence.png` (JS vs D, one line per shape); `w_pdfs.png` / `heating_pdfs.png` (truth + one fit per shape, panel per D) |

**PDF style:** the truth is drawn as **grey filled shading**, each estimate as a **solid
histogram line**, and every distribution's **dashed skew-normal fit**; a single upper-left box
lists the fit moments (mean μ, std σ, skew γ) for each series, truth included. Heat fluxes are
the reference-free advective heating `w·∂ₓT` [°C/day] (see Method), ~0.01–0.05 °C/day at
depth. Diameters are coloured light→dark with D; shapes are
coloured **diamond blue / square red / hexagon green**. Faces are numbered consistently per
shape (see each `layout.png`).

**Near-surface / integration boundary:** the face-averaged currents & fluxes are
shear-extrapolated from 8 m to the surface (`ot.extrapolate_currents_to_surface`), so the
continuity/heat integration starts from **⟨w⟩ = ⟨wT⟩ = 0 at z = 0** and marches down to −80 m
(interfaces 0…−80 m). Error metrics use the **observed 8–80 m** range (0–8 m is extrapolated,
not observed); the per-depth std normalization stays stable near the surface because error and
std shrink together.

## Run order

```bash
conda activate tpose
python run_ocv.py          # compute -> cache/ (reads the model once, ~5 min)
python run_faces.py        # then the figures, in any order
python run_budget.py
python run_pdfs.py
```
