#!/bin/bash
# Pack everything worth bringing back (small): reports, logs, evals, figures, checkpoints of the trained models.
set -e
cd "$(dirname "$0")/../.."
OUT="adinfinitium_results_macmini.zip"
rm -f "$OUT"
zip -qr "$OUT" runs -x 'runs/*/__pycache__/*'
for n in hazard_unet_big light_skyline_big rover_driver_mini terrain_diffusion_mini; do
  [ -f "data/models/$n.pt" ] && zip -q "$OUT" "data/models/$n.pt" "data/models/${n}_last.pt" 2>/dev/null || true
done
[ -d data/light ] && zip -qr "$OUT" data/light
ls -lh "$OUT"
