#!/bin/sh
# Rover v3: the SINGLE final evaluation on the Mac mini (ROVER_V3_PLAN.md section 5). Refuses to run twice.
#   sh ops/rover3/run_rover3_mini.sh worlds     build the fresh-site worlds (light + hazard caches; terrain only, no rover episode)
#   sh ops/rover3/run_rover3_mini.sh final N    stages A + B (N rovers per world), then C, then D (Amendment 2)
set -u
ROOT=$HOME/adinfinitium-rover3
CODE=$ROOT/code
OUT=$ROOT/runs/final
export ADI_DATA=$ROOT/data ADI_RUNS=$ROOT/runs PYTHONPATH=$CODE PYTHONDONTWRITEBYTECODE=1
PY=$HOME/adinfinitium-run/adinfinitium/.venv/bin/python
FRESH="DM1 LM7 NPA NPB NPC NPD"
cd "$CODE" || exit 1
mkdir -p "$OUT"
$PY -c "import worldmodel; print('module path:', worldmodel.__file__)"
case "${1:-}" in
worlds)
  for s in $FRESH; do
    $PY -c "
import sys
from contracts.mission import load_mission
from worldmodel.hazard.lola import regions_for_site
from worldmodel.rover.world import build_world
cfg = load_mission()
for d in ('2027-03-01', '2027-06-01', '2027-09-01', '2027-12-01'):
    w = build_world(regions_for_site('$s')[0], d, cfg)
    print(w.region, d, 'hazard:', w.hazard_source, '| targets:', w.ice_source, len(w.targets_rc), flush=True)
" > "$OUT/worlds_$s.log" 2>&1 &
  done
  wait
  cat "$OUT"/worlds_*.log
  ;;
final)
  N=${2:-512}
  [ -e "$OUT/LAUNCHED" ] && { echo "the final evaluation was already launched once ($(cat "$OUT/LAUNCHED")); refusing to run it again"; exit 2; }
  date > "$OUT/LAUNCHED"
  for s in $FRESH; do
    $PY -m worldmodel.rover3.run eval --sites $s --n $N --seed 1234 --threads 2 --out "$OUT/A_$s.json" > "$OUT/A_$s.log" 2>&1 &
  done
  for s in Site23 Site42; do
    $PY -m worldmodel.rover3.run eval --sites $s --n $N --seed 1234 --threads 2 --out "$OUT/B_$s.json" > "$OUT/B_$s.log" 2>&1 &
  done
  wait
  date > "$OUT/AB_DONE"
  for s in $FRESH; do
    $PY -m worldmodel.rover3.run changed --sites $s --date 2027-06-01 --n 256 --seed 1234 --threads 3 --out "$OUT/C_$s.json" > "$OUT/C_$s.log" 2>&1 &
  done
  wait
  date > "$OUT/C_DONE"
  $PY -m worldmodel.rover3.run dream-all --out-dir "$OUT" --cap-minutes 40 --concurrency 6 --threads 3 > "$OUT/D_all.log" 2>&1
  date > "$OUT/FINISHED"
  ;;
*) echo "usage: $0 worlds | final [N]"; exit 1 ;;
esac
