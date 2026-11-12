#!/bin/bash
# Daint's debug partition gives 30 minutes and one running job per user. This runs the job files of output/desi_dr1/queue/
# (*.sh, shell lines executed in this folder inside a daint.sbatch allocation) one after the other, in name order, and moves
# each to queue/done/ or queue/failed/. Add or remove job files at any time; `touch queue/STOP` ends the loop.
#     ssh daint 'cd <this folder> && setsid nohup bash daint_queue.sh > ../output/desi_dr1/queue/driver.log 2>&1 &'
cd "$(dirname "$0")"; Q=../output/desi_dr1/queue; mkdir -p $Q/done $Q/failed
while [ ! -e $Q/STOP ]; do
  f=$(ls $Q/*.sh 2>/dev/null | head -1)
  if [ -z "$f" ]; then sleep 30; continue; fi
  n=$(basename "$f" .sh); echo "$(date '+%F %T') submit $n"
  CMD="bash $f" sbatch --wait --partition=debug --time=00:30:00 --job-name="$n" daint.sbatch > $Q/$n.submit 2>&1; rc=$?
  if ! grep -q 'Submitted batch job' $Q/$n.submit; then sleep 60; continue; fi          # not accepted (2 jobs per user): try again
  if [ $rc -eq 0 ]; then mv "$f" $Q/done/; echo "$(date '+%F %T') done   $n"
  else mv "$f" $Q/failed/; echo "$(date '+%F %T') FAILED $n ($(grep -o 'Submitted batch job.*' $Q/$n.submit))"; fi
done
