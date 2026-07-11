# experiment_4a — W-estimate skill across array configurations (Ri5 model)

The default Ri_inf threshold is 0.7 -- the model run in this experiment uses Ri_inf = 0.5.

Identical to **experiment_4** but pulls truth from the `oct2012_TP6Vel_3month_Ri5`
run (`RUN_DIR = /data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month_Ri5`) instead of
`oct2012_3month_transp_cons`. Run with `python run_experiment_4a.py`.

same as experiment_2 (extrapolation of the velocity shear to the surface and w=0 at
the surface, the default option in `compute_w_planefit`) but over a deeper 8–120 m
sampling range instead of 8–70 m.

use configs from experiment_1/configs/

2. `python run_experiment_4.py` — opens the model once, samples U,V at every glider
   position, computes plane-fit `w_est` and hull-mean model-truth `w_model` per
   cell, and writes:
   - `data/<config>__cell_<center>.nc` — `w_est`, `w_model`, `bias` (dims time, depth)
   - `data/metrics.csv` — one row per cell with skill stats + config metadata
   Re-running skips cells whose `.nc` already exists (delete `data/` to recompute).
3. `summary.ipynb` — the 6 presentation figures (adds `fig6_metric_heatmaps`, the colored-square metric grid, plus `fig6a–6e` breaking it out one method per figure like `fig2a–2e`/`fig5a–5e`; `2e`/`5e`/`6e` are the `shift_hex` array) → `summary_figs/`.
4. `make_experiment_figs.ipynb` — per-config diagnostics
   (`w_comparison` Hovmöllers + `velocity_map`) → `experiment_figs/<config>/`.
