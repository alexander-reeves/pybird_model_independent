#!/bin/bash
#SBATCH -A a0158
#SBATCH -p preemptable
#SBATCH --requeue
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao/logs/seq_%j.log
# Several direct-only variants one after the other on all 4 GPUs (cached stages are skipped):  VARS="lcdm5 lcdm5_cmb"
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/desi_fsbao
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for V in ${VARS}; do echo "=== ${V} ($(date))"; python -u sample.py --variant ${V} 2>&1 | grep --line-buffered -v -i warning; done
