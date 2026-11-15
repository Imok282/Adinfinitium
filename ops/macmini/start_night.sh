#!/bin/bash
# Start the overnight run detached from the terminal; keeps the Mac awake. Safe to re-run: it resumes.
set -e
cd "$(dirname "$0")/../.."
. .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
RID="${1:-macmini_night}"
nohup caffeinate -dimsu python -m ops.overnight.run_all --profile macmini --run-id "$RID" --resume > "runs/${RID}.out" 2>&1 &
echo "started run $RID (pid $!). Watch:  cat runs/overnight/$RID/status.json   |   tail -f runs/${RID}.out"
