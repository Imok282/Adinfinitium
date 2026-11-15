#!/bin/bash
# ONE command, ONE access to the mini: setup -> checks -> smoke -> overnight run -> verify -> freeze -> results zip.
# Unattended; safe to re-run after an interruption (it resumes). Run from the package root:  bash ops/macmini/run_everything.sh
# Optional env: ADI_RUN_ID (default macmini_night), ADI_DEADLINE (default 2026-11-15T08:00+05:30; training budgets are capped to end by then).
cd "$(dirname "$0")/../.." || exit 1
if [ -z "$ADI_CAFFEINATED" ] && command -v caffeinate >/dev/null 2>&1; then
  export ADI_CAFFEINATED=1
  exec caffeinate -dimsu bash ops/macmini/run_everything.sh "$@"     # the whole script keeps the Mac awake, not only the training
fi
RID="${ADI_RUN_ID:-macmini_night}"
DEADLINE="${ADI_DEADLINE:-2026-11-15T08:00+05:30}"
LOG=runs/run_everything.log; mkdir -p runs
LOCK=runs/run_everything.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  OLD="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -n "$OLD" ] && ps -p "$OLD" -o command= 2>/dev/null | grep -q run_everything; then      # a reused pid of another program is not us
    echo "run_everything.sh is ALREADY RUNNING (pid $OLD). Not starting a second one. Watch: tail -f $LOG" | tee -a "$LOG"; exit 3
  fi
  echo "== $(date) stale lock from pid ${OLD:-?} (not running): taking over" >> "$LOG"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
exec >> "$LOG" 2>&1
echo "== $(date) run_everything start (pid $$, run id $RID, deadline $DEADLINE)"
echo "== $(date) setup"
if [ ! -f .venv/.setup_complete ]; then      # a half-made .venv has no marker, so setup runs again
  bash ops/macmini/setup_mac.sh || { echo "SETUP FAILED"; exit 1; }
fi
. .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "== $(date) verify before"; python -m ops.verify before
echo "== $(date) unit tests (recorded, never blocks the run)"; perl -e 'alarm 900; exec @ARGV' python -m pytest -q 2>&1 | tail -n 15      # 15 min cap: a hung test must not stall the night
echo "== $(date) smoke (tiny, ~5-25 min, outputs in runs/_smoke; stages that already passed are skipped; a failure is recorded, the real run still starts)"
python -m ops.overnight.run_all --profile smoke --run-id smoke1 --resume
echo "== $(date) real overnight run"
python -m ops.overnight.run_all --profile macmini --run-id "$RID" --resume --deadline "$DEADLINE"
if [ $? -eq 3 ]; then echo "== $(date) another runner owns run $RID: NOT verifying, freezing or collecting beside it"; exit 3; fi
echo "== $(date) verify after"
if python -m ops.verify after --run-id "$RID"; then
  rm -f runs/VERIFY_FAILED.txt
  echo "== $(date) freeze"; python -m ops.freeze && python -m ops.freeze --check
else
  echo "#####################################################################"
  echo "## VERIFY AFTER FAILED: NOT FROZEN. Read the FAIL lines above,      ##"
  echo "## report them in RESULTS_FOR_TEAM.md. Results are still collected. ##"
  echo "#####################################################################"
  echo "verify after FAILED at $(date); nothing was frozen. See runs/run_everything.log" > runs/VERIFY_FAILED.txt
fi
echo "== $(date) collect"; bash ops/macmini/collect_results.sh
echo "== $(date) ALL DONE. Now write RESULTS_FOR_TEAM.md, then run:  bash ops/macmini/collect_results.sh --add-report"
