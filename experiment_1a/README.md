# experiment_1a — W-estimate skill across array configurations (Ri5 model)

The default Ri_inf threshold is 0.7 -- the model run in this experiment uses Ri_inf = 0.5.

Identical to **experiment_1** but pulls truth from the `oct2012_TP6Vel_3month_Ri5`
run (`RUN_DIR = /data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month_Ri5`) instead of
`oct2012_3month_transp_cons`. Configs are read from `experiment_1/configs/` (no
`generate_configs.py` / `configs/` of its own). Run with `python run_experiment_1a.py`.
