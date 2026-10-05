#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/fisher_pb/meeting/exec_nbs.log
# Executes the two presentation notebooks in place (~2 min: the Jacobians are cached).
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/fisher
python - << 'PYEOF'
import nbformat
from nbclient import NotebookClient
for name in ('08_growth_pk_bk.ipynb', '09_bao_pk_bk.ipynb'):
    nb = nbformat.read(name, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name='python3',
                   resources={'metadata': {'path': '.'}}).execute()
    nbformat.write(nb, name)
    nerr = sum(1 for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error')
    print(f'EXECUTED {name}: {len(nb.cells)} cells, {nerr} errors')
PYEOF
