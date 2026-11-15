#!/bin/bash
# Step 7: pack what goes back by AirDrop ->  <package root>/adinfinitium_lander_results_macmini.zip
#   runs/lander/ (reports, json, changelog, logs), data/models/lander_pilot*.pt, data/pilot_dream/, RESULTS_SUMMARY.md,
#   plus a diff of every code file changed on this machine (against the untouched copy that setup.sh kept in .pristine/).
set -eu -o pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
export TZ=Asia/Kolkata
OUT="adinfinitium_lander_results_macmini.zip"
LOGS="runs/lander/logs"
mkdir -p "$LOGS"

if [ ! -s RESULTS_SUMMARY.md ]; then
  echo "RESULTS_SUMMARY.md is missing in $ROOT. Write it first (START_HERE.md, section 'What to report back'), then run this again."
  exit 1
fi

# code changes made on this machine
if [ -d .pristine ]; then
  : > "$LOGS/code_changes.diff"
  for d in worldmodel contracts config sim_jax tests ops pyproject.toml; do
    diff -ruN -x __pycache__ -x '*.pyc' -x package_manifest.sha256 ".pristine/$d" "$d" >> "$LOGS/code_changes.diff" || true
  done
  grep -E '^diff -ruN' "$LOGS/code_changes.diff" | awk '{print $NF}' > "$LOGS/code_changed_files.txt" || true
  echo "code files changed on this machine: $(wc -l < "$LOGS/code_changed_files.txt" | tr -d ' ') (list: $LOGS/code_changed_files.txt, diff: $LOGS/code_changes.diff)"
  if grep -qE '^config/mission.yaml$|^worldmodel/lander/env.py$' "$LOGS/code_changed_files.txt"; then
    echo "WARNING: the mission config or the simulator was edited. The hard rules forbid that for better numbers: explain it in RESULTS_SUMMARY.md."
  fi
else
  echo "note: no .pristine/ copy (setup.sh was not used), so no code diff is included; list the changed files by hand in RESULTS_SUMMARY.md"
fi
{ echo "# sha256 of the pilot checkpoints at collection, $(date '+%Y-%m-%d %H:%M:%S') IST"; shasum -a 256 data/models/lander_pilot*.pt; } > "$LOGS/pilot_sha_collect.txt"

rm -f "$OUT"
zip -qr "$OUT" RESULTS_SUMMARY.md runs/lander data/pilot_dream -x '*/__pycache__/*' -x '*.DS_Store'
# every pilot checkpoint (production, the copies kept before each accepted round, candidates), not the training-resume state
for f in data/models/lander_pilot*.pt; do
  case "$f" in *_cand_last.pt) continue ;; esac
  zip -q "$OUT" "$f"
done
unzip -l "$OUT" | tail -n 1
echo "results zip: $ROOT/$OUT"
echo "size: $(du -h "$OUT" | awk '{print $1}')   sha256: $(shasum -a 256 "$OUT" | awk '{print $1}')"
