# Model-independent EFTofLSS: analyses

| folder | what | start at |
|---|---|---|
| [`desi_fsbao/`](desi_fsbao/) | **The MI analysis of DESI DR1 full shape + BAO**, fiducial model `Ag245F`: the linear spectrum, growth and distances are fitted freely, then projected onto ΛCDM, w0waCDM and EDE | [`desi_fsbao/README.md`](desi_fsbao/README.md) |
| [`fisher/`](fisher/) | Fisher forecasts for a DESI-Y6-like mock: how much the bispectrum adds to the MI reconstruction (P(k) vs P(k)+B(k)) | [`fisher/README.md`](fisher/README.md) |

The parameter-recovery summary of the fiducial model is
[`../results/Ag245F_parameter_recovery.pdf`](../results/Ag245F_parameter_recovery.pdf).

Each folder follows the same rule: scripts are the source of truth. A builder writes each notebook, and an `exec_*.sbatch`
launcher executes it in place, so a committed notebook is an executed record. Chains, caches and figures go to
`output/`, which is not in git.

The first version of the pipeline was retired from this folder on 2026-10-05. That was the 106-parameter model with 60
nodes and h_conv (`mi_model.py`, `setups.py`), together with its mock / BOSS / DESI notebooks, robustness tests and the
first paper draft. It was never committed and is kept on the analysis machine, in `archive/2026-10-05_cleanup/`.
