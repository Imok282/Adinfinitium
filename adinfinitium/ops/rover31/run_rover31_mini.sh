#!/bin/sh
# Rover v3.1: the SINGLE final evaluation on the Mac mini (ROVER_V31_PLAN.md section 5 + amendments). Refuses to run twice. CPU only, 7 processes, 1 thread each.
#   sh ops/rover31/run_rover31_mini.sh final [extra args for stage D, e.g. --no-weak]
set -u
ROOT=$HOME/adinfinitium-rover31
CODE=$ROOT/code
OUT=$ROOT/runs/final
export ADI_DATA=$ROOT/data ADI_RUNS=$ROOT/runs PYTHONPATH=$CODE PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=$HOME/adinfinitium-run/adinfinitium/.venv/bin/python
UNTOUCHED="LM2 LM3 LM4 LM5"
SEEN="DM1 LM7 NPA NPB NPC NPD Site23 Site42"
CHANGES="${ROVER31_CHANGES:-blocked_route dusty_panels late_light early_light weak_battery}"
cd "$CODE" || exit 1
mkdir -p "$OUT"
[ "${1:-}" = "final" ] || { echo "usage: $0 final [--no-weak] [--no-early]"; exit 1; }
shift
[ -e "$OUT/LAUNCHED" ] && { echo "the final evaluation was already launched once ($(cat "$OUT/LAUNCHED")); refusing to run it again"; exit 2; }
date > "$OUT/LAUNCHED"
$PY -c "import worldmodel; print('module path:', worldmodel.__file__)"
git -C "$CODE" rev-parse HEAD 2>/dev/null || cat "$CODE/ROVER31_COMMIT.txt"
JOBS=$OUT/jobs.txt
: > "$JOBS"
for s in $UNTOUCHED; do echo "$PY -m worldmodel.rover3.run eval --sites $s --n 512 --seed 1234 --threads 1 --out $OUT/A_$s.json > $OUT/A_$s.log 2>&1" >> "$JOBS"; done
for s in $UNTOUCHED; do echo "$PY -m worldmodel.rover3.run changed --sites $s --date 2027-06-01 --n 256 --seed 1234 --threads 1 --changes $CHANGES --out $OUT/C_$s.json > $OUT/C_$s.log 2>&1" >> "$JOBS"; done
for s in $SEEN; do echo "$PY -m worldmodel.rover3.run eval --sites $s --n 512 --seed 1234 --threads 1 --out $OUT/B_$s.json > $OUT/B_$s.log 2>&1" >> "$JOBS"; done
tr '\n' '\0' < "$JOBS" | xargs -0 -P 7 -I{} sh -c '{}'
date > "$OUT/ABC_DONE"
$PY -m worldmodel.rover3.run dream31-all --sites $UNTOUCHED --date 2027-06-01 --rounds 2 --out-dir "$OUT" --cap-minutes 30 --concurrency 7 --threads 1 "$@" > "$OUT/D_all.log" 2>&1
date > "$OUT/FINISHED"
