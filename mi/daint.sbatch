#!/bin/bash
# run.sbatch for Daint (alps-daint, project lp184; one node = 4 GH200): same interface, same environment variables.
#     ssh daint 'cd <this folder> && ARGS="chain --template bao_template_no_rs --seed 1" sbatch [-p debug -t 00:30:00] daint.sbatch'
#     ssh daint 'cd <this folder> && SCRIPT=checks.py ARGS="exact_repr w0wa" sbatch daint.sbatch'
#     CMD="<shell line>" sbatch daint.sbatch      anything else, e.g. several 1-GPU runs: CUDA_VISIBLE_DEVICES=0 python ... & ... ; wait
# daint_queue.sh feeds job files to the debug partition one after the other.
# Daint rejects --environment on the command line when the script sets one: keep it here. debug: 30 min, 1 running job.
#SBATCH --job-name=mi
#SBATCH --account=lp184
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --environment=/users/areeves/.edf/pybird-jax.toml
#SBATCH --output=../output/desi_dr1/logs/%x_%j.log
source $HOME/pybird/jax_env/bin/activate
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -n "$CMD" ]; then :; elif [ -z "$SCRIPT" ]; then CMD="python -u run.py $ARGS"; else CMD="python -u $SCRIPT $ARGS"; fi
echo "$CMD   ($(hostname), $(date))"
bash -c "$CMD"
