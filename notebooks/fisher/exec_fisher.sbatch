#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/fisher_v3/exec_nb.log
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/fisher
python - << 'PYEOF'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('01_fisher_v3.ipynb', as_version=4)
client = NotebookClient(nb, timeout=1500, kernel_name='python3')
client.execute()
nbformat.write(nb, '01_fisher_v3.ipynb')
print('NOTEBOOK EXECUTED AND SAVED')
PYEOF
