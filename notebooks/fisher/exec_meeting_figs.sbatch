#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:25:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/fisher_pb/meeting/figs.log
# Rebuilds every figure the two presentation notebooks show (~2 min, from the cached
# jacobians_pb.npz). The notebooks re-run the last two themselves; this is for iterating on
# the plots alone. cosmo_map_pb.py needs the emulator, the rest is numpy + getdist.
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/fisher
python3 -u cosmo_map_pb.py
python3 -u bao_se07_forecast.py
echo "=================== meeting_growth_pb.py ==================="
python3 -u meeting_growth_pb.py
echo "=================== meeting_bao_pb.py ==================="
python3 -u meeting_bao_pb.py
