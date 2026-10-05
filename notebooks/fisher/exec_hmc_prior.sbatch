#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/fisher_pb/hmc_prior_bands.log
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/fisher
python3 -u hmc_prior_bands.py
