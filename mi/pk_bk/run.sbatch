#!/bin/bash
# One rebuild.py step as a SLURM job, on the CPU (clean derivatives). Submit from this folder:
#     ARGS="notebook --execute" sbatch run.sbatch      a few minutes (the cached inputs exist)
#     ARGS=jacobians sbatch run.sbatch                 5-10 min (needs pybird-dev); then cosmo_map ~3 min, robust ~1 min, se07
#
# ---- cluster settings: the only lines to edit on another machine ---------------------------------------------------
#SBATCH --job-name=pk_bk
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml           # CSCS container engine: delete this line elsewhere
#SBATCH --output=../../output/fisher_pb/%x_%j.log
ENV_SETUP='source $HOME/pybird/jax_env/bin/activate'      # python with jax, getdist, nbclient (+ pybird-dev for jacobians)
# -----------------------------------------------------------------------------------------------------------------------
eval "$ENV_SETUP"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
echo "rebuild.py ${ARGS:-} ($(hostname), $(date))"
python -u rebuild.py ${ARGS:-}
