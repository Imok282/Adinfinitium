#!/bin/sh
# Rover v3.1: stages A, B, C of the final evaluation. Needed because the first launcher's `xargs -I{}` refused the job lines ("command line cannot be
# assembled, too long": macOS limits -I replacements to 255 bytes), so NO stage A/B/C job started and the launcher went straight on to stage D.
# This script runs exactly the job lines the first launcher wrote to jobs.txt, once stage D has finished (7 processes at most at any time). Refuses to run twice.
set -u
ROOT=$HOME/adinfinitium-rover31
OUT=$ROOT/runs/final
export ADI_DATA=$ROOT/data ADI_RUNS=$ROOT/runs PYTHONPATH=$ROOT/code PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT/code" || exit 1
[ -e "$OUT/ABC_LAUNCHED" ] && { echo "stages A/B/C were already launched once; refusing"; exit 2; }
ls "$OUT"/A_*.json "$OUT"/B_*.json "$OUT"/C_*.json >/dev/null 2>&1 && { echo "stage A/B/C outputs exist; refusing"; exit 2; }
date > "$OUT/ABC_LAUNCHED"
while [ ! -e "$OUT/FINISHED" ]; do sleep 5; done
date > "$OUT/ABC_STARTED"
mkdir -p "$OUT/jobs"
i=0
while IFS= read -r line; do
  i=$((i + 1))
  printf '%s\n' "$line" > "$OUT/jobs/job_$(printf %02d $i).sh"
done < "$OUT/jobs.txt"
ls "$OUT"/jobs/job_*.sh | xargs -P 7 -n 1 sh
date > "$OUT/ABC_DONE_REAL"
