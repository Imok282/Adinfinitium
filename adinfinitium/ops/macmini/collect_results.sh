#!/bin/bash
# Pack what the team needs (target < 1.5 GB): reports, logs, evals, figures and the BEST checkpoint of each trained model.
# The resumable *_last.pt files (weights + optimiser, ~2x the size) go into a separate, optional zip.
#   bash ops/macmini/collect_results.sh               build both zips
#   bash ops/macmini/collect_results.sh --add-report  add RESULTS_FOR_TEAM.md to the existing zip (run after writing it)
set -e
cd "$(dirname "$0")/../.."
OUT="adinfinitium_results_macmini.zip"
LAST="adinfinitium_results_macmini_last_checkpoints_optional.zip"
if [ "$1" = "--add-report" ]; then
  [ -f RESULTS_FOR_TEAM.md ] || { echo "RESULTS_FOR_TEAM.md does not exist yet: write it first"; exit 1; }
  [ -f "$OUT" ] || { echo "$OUT does not exist: run this script without arguments first"; exit 1; }
  zip -q "$OUT" RESULTS_FOR_TEAM.md && echo "added RESULTS_FOR_TEAM.md to $OUT" && ls -lh "$OUT"
  exit 0
fi
rm -f "$OUT" "$LAST"
zip -qr "$OUT" runs -x 'runs/*/__pycache__/*' 'runs/run_everything.lock/*'
for f in CHANGES_BEFORE_RUN.md RESULTS_FOR_TEAM.md data/tiles/excluded_windows.json; do
  [ -f "$f" ] && zip -q "$OUT" "$f"
done
for n in hazard_unet_big light_skyline_big rover_driver_mini terrain_diffusion_mini; do
  if [ -f "data/models/$n.pt" ]; then zip -q "$OUT" "data/models/$n.pt"; else echo "NOTE: data/models/$n.pt is missing (not in the zip)"; fi
  [ -f "data/models/${n}_last.pt" ] && zip -q "$LAST" "data/models/${n}_last.pt"
done
if [ -d data/light ]; then
  MB=$(du -sm data/light | cut -f1)
  if [ "$MB" -le 300 ]; then zip -qr "$OUT" data/light; else echo "NOTE: data/light is ${MB} MB: left out of the zip (it can be rebuilt with the points stage)"; fi
fi
ls -lh "$OUT"; [ -f "$LAST" ] && ls -lh "$LAST"
MB=$(du -m "$OUT" | cut -f1)
[ "$MB" -le 1500 ] || echo "WARNING: $OUT is ${MB} MB, above the 1.5 GB target"
[ -f RESULTS_FOR_TEAM.md ] || echo "RESULTS_FOR_TEAM.md is not in the zip yet: write it, then run  bash ops/macmini/collect_results.sh --add-report"
