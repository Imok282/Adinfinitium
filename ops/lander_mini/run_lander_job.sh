#!/bin/bash
# Lander-pilot improvement job for the Mac mini: steps 1 to 6 of ops/lander_mini/START_HERE.md, one at a time (they share one GPU).
#
#   nohup bash ops/lander_mini/run_lander_job.sh > /dev/null 2>&1 &      # whole job; progress in runs/lander/logs/job.log
#   bash ops/lander_mini/run_lander_job.sh --from 4                      # continue at step 4 after fixing something
#   bash ops/lander_mini/run_lander_job.sh --only 6                      # one step
#
# Each step logs to runs/lander/logs/<step>.log. The chain STOPS at the first failed step, except:
#   step 2 (mining): on failure the shipped pool data/pilot_dream/failure_pool.npz is used and the chain goes on;
#   step 4 (training rounds): a failure, a skip (past the time limit) or a time-out never stops the chain.
# Step 6 (Site23 final exam) runs ONCE: the script refuses to start it again after it has printed a result row.
# All clock times are IST. Limits can be moved with the variables below, never by editing the gate or the seeds.
set -u -o pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
export TZ=Asia/Kolkata PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1

STEP4_LATEST_START="${STEP4_LATEST_START:-1415}"   # HHMM IST: step 4 is skipped if it would start later than this
STEP4_END_BY="${STEP4_END_BY:-1505}"               # HHMM IST: the number of rounds is cut so that step 4 ends about here
STEP4_ROUNDS="${STEP4_ROUNDS:-5}"
STEP4_MINUTES="${STEP4_MINUTES:-12}"               # training minutes per round
STEP4_GATE_MIN="${STEP4_GATE_MIN:-4}"              # guessed minutes for the Site11 gate after each round (set it from the step 3 timing)
TARGET_DONE="${TARGET_DONE:-1600}"                 # HHMM IST: results zip wanted by then (code freeze 17:00)

FROM=1; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; FROM="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

# keep the Mac awake for as long as this script runs
if [ -z "${ADI_AWAKE:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export ADI_AWAKE=1
  exec caffeinate -i -m -s bash "$0" --from "$FROM" ${ONLY:+--only "$ONLY"}
fi

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "no .venv: run  bash ops/lander_mini/setup.sh  first"; exit 1; }
LOGS="runs/lander/logs"
mkdir -p "$LOGS"
JOB="$LOGS/job.log"
PROD="data/models/lander_pilot.pt"
POOL_BIG="data/pilot_dream/failure_pool_big.npz"
POOL_SHIPPED="data/pilot_dream/failure_pool.npz"

say()    { echo "[$(date '+%H:%M:%S') IST] $*" | tee -a "$JOB"; }
tomin()  { echo $(( 10#$1 / 100 * 60 + 10#$1 % 100 )); }                 # HHMM -> minutes since midnight
nowmin() { echo $(( 10#$(date '+%H') * 60 + 10#$(date '+%M') )); }
sha()    { shasum -a 256 "$1" | awk '{print $1}'; }
wants()  { [ "$1" -ge "$FROM" ] && { [ -z "$ONLY" ] || [ "$1" = "$ONLY" ]; }; }
late()   { [ "$(nowmin)" -gt "$(tomin "$TARGET_DONE")" ] && say "WARNING: it is past the $TARGET_DONE IST target; finish the remaining steps and collect (code freeze 17:00 IST)"; return 0; }

# run_step <log name> <command...>: logs, times, returns the command's exit code
run_step() {
  local name="$1"; shift
  local log="$LOGS/$name.log" t0 rc
  t0=$(date +%s)
  say "START $name: $*"
  { echo "### $(date '+%Y-%m-%d %H:%M:%S') IST  $*"; } >> "$log"
  "$@" >> "$log" 2>&1
  rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$dt" "$(date '+%H:%M:%S')" >> "$LOGS/timings.tsv"
  if [ $rc -eq 0 ]; then say "OK    $name in ${dt}s"; else say "FAIL  $name (exit $rc) after ${dt}s; last lines of $log:"; tail -n 15 "$log" | tee -a "$JOB"; fi
  return $rc
}
stop() { say "CHAIN STOPPED at $1. Fix the cause (never the gate, seeds, simulator or mission.yaml), then:  bash ops/lander_mini/run_lander_job.sh --from $2"; exit 1; }

say "==== lander job: from step $FROM${ONLY:+ (only $ONLY)} in $ROOT ===="
[ -f "$LOGS/timings.tsv" ] || printf 'step\texit\tseconds\tended_ist\n' > "$LOGS/timings.tsv"
# the hash of the production pilot as shipped: written once, never overwritten by a re-run
[ -f "$LOGS/pilot_sha_start.txt" ] || sha "$PROD" > "$LOGS/pilot_sha_start.txt"
say "production pilot sha256 at job start: $(cat "$LOGS/pilot_sha_start.txt")"

# ---- step 1: smoke (DM2, 32 missions, nominal only; this is NOT Site23) -------------------------------------------------------
if wants 1; then
  run_step 1_smoke "$PY" -m worldmodel.lander.benchmark --sites DM2 --n 32 --seeds 1 --harsh 0 --out smoke_bench || stop "step 1 (smoke)" 1
  say "smoke seconds: $(awk -F'\t' '$1=="1_smoke"{s=$3} END{print s}' "$LOGS/timings.tsv") (32 missions x 2 pilots on DM2)"
fi

# ---- step 2: mine a bigger failure pool on the 5 TRAINING maps -----------------------------------------------------------------
if wants 2; then
  if run_step 2_mine "$PY" -m worldmodel.lander.analysis mine --sites Site01 Site04 Site06 Site07 Site20 --n 512 --seeds 2 --out failure_pool_big.npz \
     && [ -s "$POOL_BIG" ]; then
    echo "$POOL_BIG" > "$LOGS/pool_used.txt"
  else
    say "step 2 failed: falling back to the shipped pool $POOL_SHIPPED (say so in RESULTS_SUMMARY.md)"
    echo "$POOL_SHIPPED" > "$LOGS/pool_used.txt"
  fi
fi
if [ -f "$LOGS/pool_used.txt" ]; then POOL="$(cat "$LOGS/pool_used.txt")"; elif [ -s "$POOL_BIG" ]; then POOL="$POOL_BIG"; else POOL="$POOL_SHIPPED"; fi
say "scenario pool: $POOL"

# ---- step 3: solve-then-learn (authority rules, gated on Site11) ---------------------------------------------------------------
if wants 3; then
  run_step 3_solve "$PY" -m worldmodel.lander.solve --pool "$POOL" --try 3 --gate-n 256 --gate-seeds 2 || stop "step 3 (solve)" 3
  sha "$PROD" > "$LOGS/pilot_sha_after_step3.txt"
  grep -E "^solve round|nothing goes to the gate" "$LOGS/3_solve.log" | tee -a "$JOB"
fi

# ---- step 4: targeted training rounds (never stops the chain) ------------------------------------------------------------------
if wants 4; then
  # snapshot of the changelog as it was after step 3 (rsi_targeted appends to the same file)
  [ -f runs/lander/rsi_changelog.md ] && cp runs/lander/rsi_changelog.md runs/lander/rsi_changelog_after_solve.md
  NOW=$(nowmin)
  if [ "$NOW" -gt "$(tomin "$STEP4_LATEST_START")" ]; then
    say "SKIP  step 4: it is $(date '+%H:%M') IST, later than the $STEP4_LATEST_START limit. Report it as skipped for time."
    printf '%s\t%s\t%s\t%s\n' "4_rsi_targeted" "skipped_late" "0" "$(date '+%H:%M:%S')" >> "$LOGS/timings.tsv"
  else
    PER=$(( STEP4_MINUTES + STEP4_GATE_MIN ))
    FIT=$(( ( $(tomin "$STEP4_END_BY") - NOW - STEP4_GATE_MIN ) / PER ))
    ROUNDS=$STEP4_ROUNDS
    [ "$FIT" -lt "$ROUNDS" ] && ROUNDS=$FIT
    [ "$ROUNDS" -lt 1 ] && ROUNDS=1
    [ "$ROUNDS" -lt "$STEP4_ROUNDS" ] && say "step 4: only $ROUNDS of $STEP4_ROUNDS rounds fit before $STEP4_END_BY IST (say so in RESULTS_SUMMARY.md)"
    cp "$PROD" data/models/lander_pilot_before_step4.pt
    run_step 4_rsi_targeted "$PY" -m worldmodel.lander.rsi_targeted --pool "$POOL" --rounds "$ROUNDS" --minutes "$STEP4_MINUTES" \
      --gate-n 256 --gate-seeds 2 --envs 512 --mb-size 256 || say "step 4 failed: the chain goes on with the production pilot as it is now"
    grep -E "time budget .* requested units fit|^RSI round" "$LOGS/4_rsi_targeted.log" | tee -a "$JOB"
    # a crash in the middle of a save must not leave a broken production pilot
    if ! "$PY" -c "import torch; torch.load('$PROD', map_location='cpu', weights_only=False)['state_dict']" >> "$LOGS/4_rsi_targeted.log" 2>&1; then
      say "production pilot does not load after step 4: restoring the copy made before step 4"
      cp data/models/lander_pilot_before_step4.pt "$PROD"
    fi
  fi
  sha "$PROD" > "$LOGS/pilot_sha_after_step4.txt"
fi

# ---- step 5: unseen benchmark with the final production pilot (same seeds as the shipped benchmark) ------------------------------
if wants 5; then
  late
  [ -f runs/lander/benchmark_before.json ] || cp runs/lander/benchmark.json runs/lander/benchmark_before.json
  [ -f runs/lander/benchmark_before.md ]   || cp runs/lander/benchmark.md   runs/lander/benchmark_before.md
  S0="$(cat "$LOGS/pilot_sha_start.txt")"; S1="$(sha "$PROD")"
  if [ "$S0" = "$S1" ]; then
    say "SKIP  step 5: the production pilot is unchanged (sha256 $S1), so runs/lander/benchmark.json already describes it (same seeds would give the same missions)"
    printf '%s\t%s\t%s\t%s\n' "5_benchmark" "skipped_pilot_unchanged" "0" "$(date '+%H:%M:%S')" >> "$LOGS/timings.tsv"
  else
    say "production pilot changed ($S0 -> $S1): re-running the unseen benchmark"
    run_step 5_benchmark "$PY" -m worldmodel.lander.benchmark --n 256 --seeds 3 --harsh 3 --out benchmark || stop "step 5 (benchmark)" 5
  fi
fi

# ---- step 6: Site23 final exam, ONCE -------------------------------------------------------------------------------------------
if wants 6; then
  late
  FLOG="$LOGS/6_final_site23.log"
  if [ -f runs/lander/benchmark_site23_final.json ] || { [ -f "$FLOG" ] && grep -qE "^Site23 " "$FLOG"; }; then
    say "REFUSED step 6: the Site23 final exam has already printed results. It is never flown again. Use the existing results."
  else
    if [ -f "$FLOG" ]; then
      say "an earlier final-exam attempt crashed before any result row: keeping its log and running once more"
      mv "$FLOG" "$LOGS/6_final_site23.crashed_$(date '+%H%M%S').log"
    fi
    sha "$PROD" > "$LOGS/pilot_sha_final_exam.txt"
    say "FINAL EXAM on Site23 with production pilot sha256 $(cat "$LOGS/pilot_sha_final_exam.txt"). No training, tuning or gating after this."
    run_step 6_final_site23 "$PY" -m worldmodel.lander.benchmark --final --n 512 --seeds 2 --harsh 3 --out benchmark_site23_final \
      || stop "step 6 (final exam; re-run ONLY if $FLOG has no line starting with 'Site23 ')" 6
  fi
fi

say "==== chain finished. Next: write RESULTS_SUMMARY.md in $ROOT, then  bash ops/lander_mini/collect_results.sh ===="
column -t -s "$(printf '\t')" "$LOGS/timings.tsv" | tee -a "$JOB"
