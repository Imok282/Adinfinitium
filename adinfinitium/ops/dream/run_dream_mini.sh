#!/bin/bash
# EXPERIMENTAL ADDENDUM: the rover keeps learning by itself after training has ended. Run on the Mac mini AFTER the main overnight run has frozen.
# From THIS (separate) folder, never from the main run folder:   bash ops/dream/run_dream_mini.sh [max_minutes=140] [run_id=selflearn1]
# Running the same command again RESUMES the run (accepted drivers, saved worlds and strategy are kept; a run is never overwritten).
#   MAIN  main run folder (default ~/adinfinitium-run/adinfinitium): its data/ is READ (tiles, hazard checkpoint, frozen rover driver); nothing is written there.
# To start from a driver trained AFTER the freeze (kept outside the main run folder, not in FROZEN.json; the report names it "starting driver", never frozen):
#   START_DRIVER_FILE=$HOME/adinfinitium-rover2/data/models/rover_driver_v2.pt bash ops/dream/run_dream_mini.sh 140 selflearn_v2
#   START_DRIVER_LABEL  optional, how the report names it (default "rover v2, trained after the freeze"). Tiles and the hazard checkpoint still come from MAIN/data.
#   Use a new run id: a run that began from one starting driver refuses to resume from another.
# Writes only under ./runs/_dream/. The python entry point refuses to start while run_everything.sh / ops.overnight.run_all is alive, before the main
# run's freeze step, or if the simulator files here differ from the main run's.
cd "$(dirname "$0")/../.." || exit 1
MAXMIN="${1:-140}"
RID="${2:-selflearn1}"
MAIN="${MAIN:-$HOME/adinfinitium-run/adinfinitium}"
case "$MAXMIN" in ''|*[!0-9]*) echo "REFUSED: max_minutes must be a whole number, got '$MAXMIN'"; exit 1;; esac
case "$RID" in ''|[!A-Za-z0-9]*|*[!A-Za-z0-9_.-]*) echo "REFUSED: run id may only contain letters, digits, _ . - and must start with a letter or digit"; exit 1;; esac
MAXMIN=$((10#$MAXMIN))                  # "08" is eight minutes, not a bad octal number
HERE="$(pwd -P)"
[ -x "$MAIN/.venv/bin/python" ] || { echo "REFUSED: no python at $MAIN/.venv/bin/python"; exit 1; }
case "$HERE/" in "$(cd "$MAIN" && pwd -P)/"*) echo "REFUSED: this folder is inside the main run folder $MAIN"; exit 3;; esac
START=()
if [ -n "${START_DRIVER_FILE:-}" ]; then
  [ -f "$START_DRIVER_FILE" ] || { echo "REFUSED: START_DRIVER_FILE is set but there is no file at $START_DRIVER_FILE"; exit 1; }
  START=(--start-driver-file "$START_DRIVER_FILE")
  [ -n "${START_DRIVER_LABEL:-}" ] && START+=(--start-driver-label "$START_DRIVER_LABEL")
  echo "starting driver: ${START_DRIVER_LABEL:-rover v2, trained after the freeze} ($START_DRIVER_FILE), NOT the main run's frozen driver"
fi
mkdir -p runs/_dream
export ADI_DATA="$MAIN/data" ADI_RUNS="$HERE/runs" PYTHONPATH="$HERE" PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
HARD=$(( (MAXMIN + 10) * 60 ))          # hard stop 10 min after the budget: a hung step can't run into the team's morning
echo "self-learning run $RID: budget $MAXMIN min (hard stop at $((MAXMIN + 10))), log runs/_dream/$RID.log, report runs/_dream/$RID/report.md"
exec caffeinate -dimsu perl -e 'alarm shift; exec @ARGV' "$HARD" "$MAIN/.venv/bin/python" -m worldmodel.dream.run \
  --run-id "$RID" --max-minutes "$MAXMIN" --resume --main-run-dir "$MAIN" --frozen-name rover_driver_mini --hazard-name hazard_unet_big ${START[@]+"${START[@]}"} \
  >> "runs/_dream/$RID.log" 2>&1
