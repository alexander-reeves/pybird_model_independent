#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/fisher_pb/exec_robust_nb.log
# Executes 07_robust_pk_bk.ipynb in place (reuses jacobians_pb.npz if present; ~5 min).
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/fisher
python - << 'PYEOF'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('07_robust_pk_bk.ipynb', as_version=4)
NotebookClient(nb, timeout=1500, kernel_name='python3',
               resources={'metadata': {'path': '.'}}).execute()
nbformat.write(nb, '07_robust_pk_bk.ipynb')
print('NOTEBOOK EXECUTED AND SAVED')
PYEOF
