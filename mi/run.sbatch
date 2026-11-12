#!/bin/bash
# One pipeline step as a SLURM job. Submit from this folder:
#     ARGS="chain --template agnostic --seed 1" sbatch run.sbatch        (any run.py step)
#     SCRIPT=checks.py ARGS="map" sbatch run.sbatch                      (a check)
# pipeline.sh submits every step at once, each waiting for its inputs.
#
# ---- cluster settings: the only lines to edit on another machine ---------------------------------------------------
#SBATCH --job-name=mi
#SBATCH --account=a0158
#SBATCH --partition=normal          # CSCS: 'debug' starts faster but allows 1h30 and 2 jobs per user (too few for pipeline.sh)
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml           # CSCS container engine: delete this line elsewhere
#SBATCH --output=../output/desi_dr1/logs/%x_%j.log
#   one GPU:      #SBATCH --gpus=1                         HARDWARE=""
#   CPU only:     #SBATCH --cpus-per-task=64               HARDWARE="--platform cpu --cpus 16"   (16 groups of 2 chains)
ENV_SETUP='source $HOME/pybird/jax_env/bin/activate'      # python with jax, blackjax, optax, cosmopower_jax, getdist, pybird
HARDWARE=""                                                 # run.py hardware options (see python run.py -h)
# -----------------------------------------------------------------------------------------------------------------------
eval "$ENV_SETUP"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -z "$SCRIPT" ]; then CMD="python -u run.py $HARDWARE $ARGS"; else CMD="python -u $SCRIPT $ARGS"; fi
echo "$CMD   ($(hostname), $(date))"
$CMD
