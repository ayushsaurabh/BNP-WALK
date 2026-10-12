# BNP-WALK
Bayesian NonParametric Walk Analysis and Latent-state Kinetics


[Read the Manual](./BNP-WALK_manual.pdf)

## Run the parallel sampler from Python

Edit `INPUT_FILES`, `OUTPUT_DIRECTORY`, `DATA_RANGE`, and `PARAMETERS` in
`run_parallel.py`, then run:

```bash
python3 run_parallel.py
```

The launcher creates a temporary TOML configuration and invokes
`src_parallel/main.jl`. Julia starts one worker process per input file, and each
worker writes `mcmc_output_<input-stem>.h5` in `OUTPUT_DIRECTORY`. Input file
basenames must therefore be unique.

By default, Julia is located through `PATH`. Set `JULIA_EXECUTABLE` when a
specific executable is required:

```bash
JULIA_EXECUTABLE=/path/to/julia python3 run_parallel.py
```
