#!/bin/bash
# ROVER V2 (trained AFTER the main run's freeze; not part of the frozen results). Run on the Mac mini from THIS separate folder, never from the main run folder:
#     bash ops/rover2/run_rover2_mini.sh [train_minutes=110]
# Steps: preflight -> data overlay -> train + select on Site11 (time-bounded) -> ONE held-out evaluation with the unchanged evaluator -> report.
# Running the same command again continues where it stopped: while no driver exists yet, training resumes from rover_driver_v2_last.pt inside the same
# total budget; once training is closed (a driver was handed to the held-out evaluation) it is never resumed; a complete held-out evaluation is never repeated.
#   MAIN  main run folder (default ~/adinfinitium-run/adinfinitium): its data/ and .venv are READ through ./data symlinks; nothing is written there.
# Writes only under this folder: data/models/rover_driver_v2*.pt, data/rover_cache/ (new Site11 caches), runs/_rover2/.
# Exit codes: 0 done; 1 or 3 refused / cannot start; 2 no driver was produced; 5 the held-out evaluation is missing or incomplete; 6 the report crashed.
# Worst case wall time: train_minutes + 8 (hard stop) + 25 (evaluation hard stop) + about 2.
# Refuses while run_everything.sh / ops.overnight.run_all / any worldmodel stage is alive, before the main run's freeze step and results zip, or if the
# simulator files here differ from the main run's. Plan and rules: ROVER_V2_PLAN.md.
cd "$(dirname "$0")/../.." || exit 1
TRAIN_MIN="${1:-110}"
MAIN="${MAIN:-$HOME/adinfinitium-run/adinfinitium}"
NAME=rover_driver_v2
OUTNAME=_rover2
# Test hooks for a smoke run on a laptop ONLY (smaller network, other sites). Never set on the mini. A run with any of them is NOT the planned run: it
# gets its own driver name and output folder (rover_driver_v2_smoke, runs/_rover2_smoke), so it can never be picked up as the real result.
ROVER2_TEACH_ARGS="${ROVER2_TEACH_ARGS:-}"; ROVER2_EVAL_ARGS="${ROVER2_EVAL_ARGS:-}"
if [ -n "$ROVER2_TEACH_ARGS$ROVER2_EVAL_ARGS${ROVER2_NO_ZIP_CHECK:-}" ]; then NAME=rover_driver_v2_smoke; OUTNAME=_rover2_smoke; fi
case "$TRAIN_MIN" in ''|*[!0-9]*) echo "REFUSED: train_minutes must be a whole number, got '$TRAIN_MIN'"; exit 1;; esac
TRAIN_MIN=$((10#$TRAIN_MIN))
HERE="$(pwd -P)"
PY="$MAIN/.venv/bin/python"
[ -x "$PY" ] || { echo "REFUSED: no python at $PY"; exit 1; }
case "$HERE/" in "$(cd "$MAIN" && pwd -P)/"*) echo "REFUSED: this folder is inside the main run folder $MAIN"; exit 3;; esac
OUT="$HERE/runs/$OUTNAME"
mkdir -p "$OUT"
export ADI_DATA="$HERE/data" ADI_RUNS="$OUT" PYTHONPATH="$HERE" PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
LOG="$OUT/launcher.log"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z') $*" | tee -a "$LOG"; }
timed() { local secs="$1"; shift; perl -e 'alarm shift; exec @ARGV' "$secs" "$@"; }       # hard stop: SIGALRM kills the step after `secs`
complete() { "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" "$1" 2>/dev/null; }

"$PY" -m worldmodel.rover2.preflight --main-run-dir "$MAIN" ${ROVER2_NO_ZIP_CHECK:+--no-zip-check} | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 3
bash ops/rover2/make_overlay.sh "$MAIN/data" "$HERE/data" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 3
[ -e "$ADI_DATA/models/hazard_unet_big.pt" ] || { say "REFUSED: no hazard_unet_big.pt in the main run's models"; exit 1; }
caffeinate -dimsu -w $$ &
say "rover v2 start: train budget $TRAIN_MIN min, code $(cat ROVER2_COMMIT.txt 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo unknown), main $MAIN"
NOTES=()
if [ "$OUTNAME" != "_rover2" ]; then
  NOTES+=(--note "NOT THE PLANNED RUN: launcher overrides were set (teach: '$ROVER2_TEACH_ARGS' eval: '$ROVER2_EVAL_ARGS' no-zip-check: '${ROVER2_NO_ZIP_CHECK:-}')")
fi

# 1. train + select on Site11. Closed for good once the held-out evaluation has started (TRAINING_CLOSED): no training after a look at the test sites.
if [ -e "$OUT/TRAINING_CLOSED" ]; then
  say "training already closed ($(cat "$OUT/TRAINING_CLOSED")); not resumed"
else
  # One hard deadline for all attempts: budget + 8 min. At most 3 attempts: a crash (not the alarm, exit 142) is resumed with --resume: the trainer keeps
  # its used minutes in the checkpoint, so retries stay inside the same budget, and the test sites have not been looked at yet.
  END=$(( $(date +%s) + (TRAIN_MIN + 8) * 60 ))
  TRY=0; RC=1
  while [ "$TRY" -lt 3 ]; do
    TRY=$((TRY + 1)); LEFT=$(( END - $(date +%s) ))
    [ "$LEFT" -gt 60 ] || { say "no time left for training attempt $TRY"; break; }
    timed "$LEFT" "$PY" -m worldmodel.rover2.teach --max-minutes "$TRAIN_MIN" --resume --name "$NAME" \
      --arch former --d 384 --layers 6 --heads 8 --hazard-name hazard_unet_big $ROVER2_TEACH_ARGS >> "$OUT/train.log" 2>&1
    RC=$?
    say "training attempt $TRY ended with exit code $RC"
    if [ "$RC" -eq 0 ] || [ "$RC" -eq 142 ] || [ "$RC" -eq 4 ]; then break; fi
    [ "$TRY" -lt 3 ] && NOTES+=(--note "training attempt $TRY crashed (exit code $RC) and was resumed; see train.log")
  done
  if [ "$RC" -ne 0 ]; then NOTES+=(--note "training was CUT SHORT or crashed (exit code $RC; 142 = the hard time limit); the driver is the best candidate Site11 had selected by then"); fi
  if [ ! -e "$ADI_DATA/models/$NAME.pt" ]; then
    say "FAILED: no $NAME.pt (no candidate was ever evaluated on Site11). No held-out evaluation is run. Run the same command again to resume."
    "$PY" -m worldmodel.rover2.report --main-run-dir "$MAIN" --name "$NAME" "${NOTES[@]}" --note "NO DRIVER: training ended (exit code $RC) before any Site11 candidate existed" | tee -a "$LOG"
    exit 2
  fi
  echo "closed $(date '+%Y-%m-%d %H:%M:%S %Z') driver sha256 $(shasum -a 256 "$ADI_DATA/models/$NAME.pt" | cut -d' ' -f1) training exit code $RC" > "$OUT/TRAINING_CLOSED"
  [ -s "$OUT/TRAINING_CLOSED" ] || { say "FAILED: could not write $OUT/TRAINING_CLOSED (disk full?); no held-out evaluation is run"; exit 1; }
fi
# The driver evaluated must be the one training was closed with (also on a relaunch).
CLOSED_SHA="$(sed -n 's/.*driver sha256 \([0-9a-f]*\).*/\1/p' "$OUT/TRAINING_CLOSED")"
NOW_SHA="$(shasum -a 256 "$ADI_DATA/models/$NAME.pt" | cut -d' ' -f1)"
[ -n "$CLOSED_SHA" ] && [ "$CLOSED_SHA" = "$NOW_SHA" ] || { say "FAILED: $NAME.pt (sha256 $NOW_SHA) is not the driver training was closed with ($CLOSED_SHA)"; exit 1; }
if [ "${RC:-0}" -eq 0 ]; then grep -q "exit code 0$" "$OUT/TRAINING_CLOSED" || NOTES+=(--note "training did not end normally: $(cat "$OUT/TRAINING_CLOSED")"); fi

# 2. the ONE held-out evaluation: the main run's evaluator, unchanged, same arguments as the main rover_eval stage.
if [ -e "$OUT/eval.json" ] && complete "$OUT/eval.json"; then
  say "held-out evaluation already complete; not repeated"
else
  say "evaluation starts (planned run: Site23 + Site42 x 4 dates x 512 rovers; extra arguments: '${ROVER2_EVAL_ARGS}')"
  timed $(( 25 * 60 )) "$PY" -m worldmodel.rover.evaluate --name "$NAME" --n 512 --hazard-name hazard_unet_big $ROVER2_EVAL_ARGS >> "$OUT/eval.log" 2>&1
  say "held-out evaluation ended with exit code $?"
  [ -e "$OUT/rover/eval.json" ] && cp "$OUT/rover/eval.json" "$OUT/eval.json"
fi
if [ ! -e "$OUT/eval.json" ]; then NOTES+=(--note "the held-out evaluation produced no file"); elif ! complete "$OUT/eval.json"; then NOTES+=(--note "the held-out evaluation is INCOMPLETE (stopped at its 25 min limit or crashed)"); fi

# 3. report
"$PY" -m worldmodel.rover2.report --main-run-dir "$MAIN" --name "$NAME" "${NOTES[@]}" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "FAILED: the report step crashed; eval.json and selection.json are in $OUT"; exit 6; }
say "rover v2 done: $OUT/ROVER_V2_REPORT.md, $OUT/eval.json, driver $ADI_DATA/models/$NAME.pt sha256 $NOW_SHA"
complete "$OUT/eval.json" || exit 5
