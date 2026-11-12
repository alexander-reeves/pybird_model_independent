# P(k) vs P(k)+B(k) Fisher forecast (`../../notebooks/5_fisher_pk_bk.ipynb`)

What the tree-level bispectrum adds to the model-independent reconstruction on a noiseless DESI-Y6-like mock, in the legacy MI model (ln a on 60 nodes over the 80 emulator knots, the HMC priors): the linear P(k) band, f(z), ΛCDM direct vs through the MI model, the prior audit, then BAO with a free template, with the smoothness prior and with the BAO phase prior.

| file | what |
|---|---|
| `fisher.py` | the Fisher system (60-node basis, priors, phase prior, g1/g2 projection of the data), the ΛCDM projection, and the notebook steps `data_summary`, `likelihood_summary`, `growth`, `prior_audit`, `stage_a`, `bao` (figures to `output/fisher_pb/meeting/`) |
| `rebuild.py` | one runner: `jacobians`, `cosmo_map`, `robust`, `se07` rebuild the caches, `notebook [--execute]` writes (and runs) the notebook; `python rebuild.py -h` |
| `run.sbatch` | `ARGS="<subcommand>" sbatch run.sbatch` (CPU; one marked block of cluster settings; log `output/fisher_pb/pk_bk_<jobid>.log`) |

**Notebook:** `ARGS="notebook --execute" sbatch run.sbatch` from this folder (about a minute), or `python rebuild.py notebook --execute` anywhere with the cached inputs.

**Cached inputs** in `output/fisher_pb/` (each subcommand takes `--out`; default = the cache):

| file | subcommand | cost |
|---|---|---|
| `jacobians_pb.npz` (+ `jacobians_pb.log`, which carries the mock nbar) | `jacobians` | 5-10 min |
| `cosmo_map_pb.npz` (ΛCDM map + Jacobian gate) | `cosmo_map` (needs the Jacobians) | ~3 min |
| `robust_pb_results.npz` (BAO stage A) | `robust` | ~1 min |
| `se07_forecast.npz` (standard-BAO reference) | `se07` (nbar from the log) | seconds |

**pybird-dev required** for `jacobians` and `cosmo_map`: the bispectrum JAX code is only in `pybird-dev`, which `rebuild.py` puts first on `sys.path` (default: a `pybird-dev` checkout next to this repository; override with `PYBIRD_DEV`) and asserts that `pybird` resolves there. They write the mock (`fake_desi_y6_fisher_pb.h5` + `_config.yaml`) next to `--out`. The notebook itself needs only numpy, scipy and getdist.
