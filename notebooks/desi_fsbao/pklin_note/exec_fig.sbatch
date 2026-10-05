#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao/logs/pklin_fig_%j.log
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/desi_fsbao/pklin_note
python -u make_fig.py
