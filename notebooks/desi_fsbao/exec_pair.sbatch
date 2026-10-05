#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao/logs/pair_%j.log
# Two direct-only variants side by side, 2 GPUs each:  V1=lcdm5 V2=lcdm5_cmb sbatch exec_pair.sbatch
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/desi_fsbao
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "V1=${V1} V2=${V2} ($(date))"
CUDA_VISIBLE_DEVICES=0,1 python -u sample.py --variant ${V1} 2>&1 | grep --line-buffered -v -i warning | sed -u "s/^/[${V1}] /" &
CUDA_VISIBLE_DEVICES=2,3 python -u sample.py --variant ${V2} 2>&1 | grep --line-buffered -v -i warning | sed -u "s/^/[${V2}] /" &
wait
echo "done ($(date))"
