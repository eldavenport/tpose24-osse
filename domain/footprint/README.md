Both footprint tiers map the same quantity — the error in the depth-/time-mean w (m day⁻¹) that a footprint of a given shape and size makes at every grid point, plane-fit divergence vs. the same true area-mean divergence, ×H. The only difference is how the velocity field is sampled before the plane fit:

Tier 2 — discrete glider stencil (the realistic one). Fits the plane to only the actual glider positions (6 points for hexagon/square, 4 for square4/diamond). This is exactly what a real array does, so its error bundles two things: the field's nonlinearity and the penalty of sparse, specifically-placed sampling (few points, particular geometry, conditioning).

Tier 1 — filled footprint (the best-case floor). Fits the plane to every model grid point inside the footprint (hundreds of points), not just the gliders. That removes the sparse-sampling penalty and isolates the footprint's intrinsic nonlinearity at that size — the irreducible aliasing floor. No real array can beat it.

So Tier 1 answers "how much does a footprint this size unavoidably alias, even with a perfect dense array?" and Tier 2 answers "how much does THIS actual 4-/6-glider arrangement alias?" The gap between them is the cost of sparse sampling.

Summary fig, averages errors zonally/meridionally to compare which shape is best at each latitude/longitude. 