#!/bin/bash
# ONE command, ONE access to the mini: setup -> checks -> smoke -> overnight run -> verify -> freeze -> results zip.
# Unattended; safe to re-run after an interruption (it resumes). Run from the package root:  bash ops/macmini/run_everything.sh
cd "$(dirname "$0")/../.."
LOG=runs/run_everything.log; mkdir -p runs
{
  echo "== $(date) setup"; [ -d .venv ] || bash ops/macmini/setup_mac.sh || { echo "SETUP FAILED"; exit 1; }
  . .venv/bin/activate
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  echo "== $(date) verify before"; python -m ops.verify before
  echo "== $(date) smoke (tiny; a failure here is recorded, the real run still starts)"
  python -m ops.overnight.run_all --profile smoke --run-id smoke1 --resume
  echo "== $(date) real overnight run (caffeinate keeps the Mac awake)"
  caffeinate -dimsu python -m ops.overnight.run_all --profile macmini --run-id macmini_night --resume
  echo "== $(date) verify after"; python -m ops.verify after
  echo "== $(date) freeze"; python -m ops.freeze && python -m ops.freeze --check
  echo "== $(date) collect"; bash ops/macmini/collect_results.sh
  echo "== $(date) ALL DONE (see RESULTS_FOR_TEAM.md to write, then tell the user)"
} 2>&1 | tee -a "$LOG"
