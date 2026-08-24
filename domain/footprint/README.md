Footprint maps show error in the depth-/time-mean w (m day⁻¹) that a footprint of a given shape and size makes at every grid point, plane-fit divergence vs. the same true area-mean divergence, ×H.

## Two tiers

**Tier 2** (`tier2/`) — discrete glider stencil (the realistic one). Fits the plane to only the actual glider positions (6 points for hexagon/square, 4 for square4/diamond). This is exactly what a real array does, so its error bundles two things: the field's nonlinearity and the penalty of sparse, specifically-placed sampling (few points, particular geometry, conditioning).

**Tier 1** (`tier1/`) — dense fit. Fits the plane to *every grid cell inside the footprint hull*, so it removes the sparse-sampling penalty and its error is the plane-fit-to-shape (nonlinearity) part alone — the best that shape could do given the full field under it. 

Both tiers are rendered on one shared color scale (pooled across the two tiers), so tier1 vs tier2 panels are directly comparable.

Summary fig, averages errors zonally/meridionally to compare which shape is best at each latitude/longitude. 

`footprint_werr_*` — the vertical-velocity error described above (m day⁻¹), from `run_footprint_maps.py`.