#!/bin/bash
# Submit the whole pipeline as SLURM jobs, each starting when its inputs exist; independent steps (the three reference
# fits, the chains of the two templates and their seeds) run in parallel on different nodes. Cluster settings: run.sbatch.
#     bash pipeline.sh [seeds per template, default 3]
# Without SLURM, run the same steps in this order by hand: python run.py <step> ...
# Not submitted here: the template growth_from_h, the models mnu and `reference --sound-horizon`, the projections of ede and mnu,
# sectors with theta_s, the planck step, the checks of notebooks 6, 7, 9 (checks.py) and the mocks (MI_MOCK). See README.md.
set -e
cd "$(dirname "$0")"; mkdir -p ../output/desi_dr1/logs
NSEED=${1:-3}

sub() {  # job name, the job ids it waits for (space-separated, may be empty), then the arguments of run.py (or checks.py)
  local name=$1 deps; deps=$(echo $2 | tr -s ' ' ':'); shift 2
  ARGS="$*" sbatch --parsable --job-name="$name" ${deps:+--dependency=afterok:$deps} run.sbatch
}

ref_lcdm=$(sub ref_lcdm "" reference --model lcdm)                 # the other fits start from its best fit
ref_w0wa=$(sub ref_w0wa "$ref_lcdm" reference --model w0wa)
ref_ede=$(sub ref_ede "$ref_lcdm" reference --model ede)
knots=""; [ -f ../data/mi_chains/knot_fisher.npz ] || knots=$(sub knots "$ref_lcdm" knots)
done_templates=""
for t in agnostic bao_template bao_template_no_rs; do
  first=$(sub chain_${t}_1 "$ref_lcdm $knots" chain --template $t --seed 1)       # also writes the shared best fit
  seeds=$first
  for s in $(seq 2 $NSEED); do seeds="$seeds $(sub chain_${t}_$s "$first" chain --template $t --seed $s)"; done
  rec=$(sub recovery_$t "$seeds $ref_w0wa $ref_ede" recovery --template $t)        # fits the score and flow densities the next two reuse
  prj=$(sub project_$t "$rec" project --template $t)
  sec=$(sub sectors_$t "$rec" sectors --template $t)
  done_templates="$done_templates $seeds $rec $prj $sec"
done
map=$(SCRIPT=checks.py sub map_test "$ref_w0wa $ref_ede $knots" map)
fisher=$(sub fisher "$done_templates" fisher)
bands=$(sub bands "$done_templates" bands)
exp=$(sub export "$done_templates $map $fisher $bands" export)
sub notebooks "$exp" notebooks --execute > /dev/null
echo "submitted the pipeline ($NSEED seeds per template); squeue -u $USER, logs in output/desi_dr1/logs/"
