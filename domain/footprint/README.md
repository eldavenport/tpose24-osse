Footprint maps show error in the depth-/time-mean w (m day⁻¹) that a footprint of a given shape and size makes at every grid point, plane-fit divergence vs. the same true area-mean divergence, ×H. 

Tier 2 — discrete glider stencil (the realistic one). Fits the plane to only the actual glider positions (6 points for hexagon/square, 4 for square4/diamond). This is exactly what a real array does, so its error bundles two things: the field's nonlinearity and the penalty of sparse, specifically-placed sampling (few points, particular geometry, conditioning).

Summary fig, averages errors zonally/meridionally to compare which shape is best at each latitude/longitude. 

## Two products

`footprint_werr_*` — the vertical-velocity error described above (m day⁻¹), from `run_footprint_maps.py`.

`footprint_heaterr_*` — the same footprint framework applied to the **vertically-integrated (0–70 m) advective heat flux** ρ₀cp ∫ w ∂T/∂z dz (W m⁻²), from `run_footprint_heat_maps.py`. Estimate = plane-fit continuity w × array-mean ∂T/∂z; truth = model w × ∂T/∂z multiplied point-wise then footprint-averaged (so it keeps the sub-footprint covariance). Same Tier 1/Tier 2 meaning, same shared-scale panels and best-shape summaries.

Only the shallowest layer (0–70 m) and the full 3-month window are produced for both products.

**heat maps:** advective heating is a *product* of two fields (w and ∂T/∂z), so unlike w it is not linear in the model field — time-averaging does not commute through it (⟨w ∂zT⟩ = ⟨w̄⟩⟨∂zT̄⟩ + ⟨w′(∂zT)′⟩). A map built from time-mean fields can only carry the **resolved / mean-advection** part ⟨w̄⟩⟨∂zT̄⟩; the temporal eddy-covariance term ⟨w′(∂zT)′⟩ is absent. The spatial footprint operations stay linear per depth level, which is what lets the same convolution/stencil machinery be reused.