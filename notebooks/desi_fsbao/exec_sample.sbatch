#!/bin/bash
#SBATCH -A a0158
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao/logs/sample_%j.log
# Direct + MI NUTS chains of one variant (settings.VARIANTS), 32 chains on the 4 GPUs of the node.
#   VARIANT=baseline sbatch exec_sample.sbatch        (~1 h 45 min)
#   VARIANT=w0wa7 sbatch exec_sample.sbatch           (direct chain only; needs baseline's best fit)
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/desi_fsbao
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "VARIANT=${VARIANT:-baseline} EXTRA_ARGS='${EXTRA_ARGS:-}' ($(date))"
python -u sample.py --variant "${VARIANT:-baseline}" ${EXTRA_ARGS:-} 2>&1 | grep --line-buffered -v -i warning
