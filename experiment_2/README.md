# experiment_2 — W-estimate skill across array configurations

same as experiment_1 but with extrapolation of the velocity shear to the surface and w=0 at the surface. 
(this is the default option in `compute_w_planefit`)

use configs from experiment_1/configs/

2. `python run_experiment_2.py` — opens the model once, samples U,V at every glider
   position, computes plane-fit `w_est` and hull-mean model-truth `w_model` per
   cell, and writes:
   - `data/<config>__cell_<center>.nc` — `w_est`, `w_model`, `bias` (dims time, depth)
   - `data/metrics.csv` — one row per cell with skill stats + config metadata
   Re-running skips cells whose `.nc` already exists (delete `data/` to recompute).
3. `summary.ipynb` — the 5 presentation figures → `summary_figs/`.
4. `make_experiment_figs.ipynb` — per-config diagnostics
   (`w_comparison` Hovmöllers + `velocity_map`) → `experiment_figs/<config>/`.