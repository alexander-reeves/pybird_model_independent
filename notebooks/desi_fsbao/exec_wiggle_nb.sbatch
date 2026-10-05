#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao/logs/wiggle_nb_%j.log
# wiggle_analysis.py -> wiggle_analysis.ipynb -> executed in place (nbclient); printed output copied to
# output/desi_fsbao/wiggle_figs/analysis.log, figures to output/desi_fsbao/wiggle_figs/.
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/desi_fsbao
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=/capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao/wiggle_figs; mkdir -p $OUT
python build_wiggle_nb.py
python - << PYEOF
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('wiggle_analysis.ipynb', as_version=4)
try:
    NotebookClient(nb, timeout=3600, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
finally:
    nbformat.write(nb, 'wiggle_analysis.ipynb')
    with open('${OUT}/analysis.log', 'w') as f:
        for c in nb.cells:
            for o in c.get('outputs', []):
                if o.get('output_type') == 'stream': f.write(''.join(o.get('text', '')))
                elif o.get('output_type') == 'error': f.write('\n'.join(o.get('traceback', [])) + '\n')
print('EXECUTED wiggle_analysis.ipynb')
PYEOF
