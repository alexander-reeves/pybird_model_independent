#!/bin/bash
#SBATCH -A a0158
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH -o /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/fisher_pb/exec_nb.log
# P(k) vs P(k)+B(k) model-independent Fisher. The bispectrum code lives in pybird-dev, which
# run_fisher_pb.py puts first on sys.path (PYBIRD_DEV overrides the location). The Hessians
# are computed on the CPU on purpose: GPU autodiff reductions are non-deterministic and the
# sector marginals divide by the block being marginalized, which amplifies that noise.
source $HOME/pybird/jax_env/bin/activate
cd /capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/notebooks/fisher
export PYBIRD_DEV=${PYBIRD_DEV:-/capstor/store/cscs/swissai/a0158/areeves/pybird-dev}
mkdir -p ../../output/fisher_pb
python3 build_fisher_pb_nb.py
python - << 'PYEOF'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('06_fisher_pk_bk.ipynb', as_version=4)
client = NotebookClient(nb, timeout=3300, kernel_name='python3')
client.execute()
nbformat.write(nb, '06_fisher_pk_bk.ipynb')
print('NOTEBOOK EXECUTED AND SAVED')
PYEOF
