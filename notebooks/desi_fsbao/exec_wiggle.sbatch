#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao/logs/wiggle_%j.log
# Wiggle/no-wiggle MI models (wiggle_*.py):
#   SCRIPT=wiggle_gates.py sbatch exec_wiggle.sbatch                      gates + design study (~30 min, 1 GPU)
#   SCRIPT=wiggle_sample.py ARGS="--wiggle B" sbatch exec_wiggle.sbatch   MI best fit + chain (4 GPUs)
#   SCRIPT=wiggle_recovery.py ARGS="--wiggle B" sbatch exec_wiggle.sbatch the recovery tests
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/desi_fsbao
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "SCRIPT=${SCRIPT} ARGS='${ARGS:-}' ($(date))"
python -u ${SCRIPT} ${ARGS:-} 2>&1 | grep --line-buffered -v -i warning
echo "exit ${PIPESTATUS[0]} ($(date))"
