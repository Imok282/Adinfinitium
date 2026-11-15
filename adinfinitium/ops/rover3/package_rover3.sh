#!/bin/sh
# Rover v3: build ~/adinfinitium_addendum3_rover_v3.zip on the mini (plan, report, eval json, stage files, code, brain config). Writes nothing in the frozen folder.
set -eu
ROOT=$HOME/adinfinitium-rover3
CODE=$ROOT/code
PK=$ROOT/package/adinfinitium_addendum3_rover_v3
export ADI_DATA=$ROOT/data PYTHONPATH=$CODE PYTHONDONTWRITEBYTECODE=1
PY=$HOME/adinfinitium-run/adinfinitium/.venv/bin/python
rm -rf "$ROOT/package"
mkdir -p "$PK/code/worldmodel" "$PK/code/ops" "$PK/code/tests" "$PK/final" "$PK/validation" "$PK/development_smoke"
cp "$CODE/ROVER_V3_PLAN.md" "$PK/"
[ -f "$ROOT/ROVER_V3_REPORT.md" ] && cp "$ROOT/ROVER_V3_REPORT.md" "$PK/"
cp "$ROOT/runs/final/eval.json" "$ROOT/runs/final/tables.md" "$PK/"
cp "$ROOT"/runs/final/*.json "$ROOT"/runs/final/*.log "$ROOT"/runs/final/LAUNCHED "$ROOT"/runs/final/FINISHED "$PK/final/"
cp "$ROOT"/runs/val/*.json "$ROOT"/runs/val/*.log "$PK/validation/"
cp "$ROOT"/runs/dev/* "$PK/development_smoke/" 2>/dev/null || true
cp -R "$CODE/worldmodel/rover3" "$PK/code/worldmodel/"
cp -R "$CODE/ops/rover3" "$PK/code/ops/"
cp "$CODE/tests/test_rover3.py" "$PK/code/tests/"
cp "$CODE/ROVER3_COMMIT.txt" "$PK/code/"
[ -f "$ROOT/rover_v3.diff" ] && cp "$ROOT/rover_v3.diff" "$PK/code/"
find "$PK" -name __pycache__ -type d -prune -exec rm -rf {} +
cd "$CODE" && $PY -c "
import dataclasses, json
from worldmodel.rover3.brain import Theta, candidate_table
t = Theta(); c = candidate_table(t, 'cpu')
json.dump({'what': 'rover v3 brain as deployed: scripted planner + scripted executor; no neural network, so there are no weights. This file and code/worldmodel/rover3 are everything needed to run it.',
           'theta': dataclasses.asdict(t), 'candidates': [{'index': i, 'wait_hours': int(c['wait'][i]), 'charge_lo': float(c['lo'][i]), 'charge_hi': float(c['hi'][i]),
           'kind': ['strategy', 'baseline naive', 'baseline cautious', 'baseline cautious_wait_light', 'safe harbour'][int(c['kind'][i])]} for i in range(len(c['kind']))],
           'maps_used': 'hazard U-Net hazard_unet_big.pt output, LOLA slope, modelled light (SPICE + skyline), NASA PSR-edge targets'}, open('$PK/rover_brain_v3_config.json', 'w'), indent=1)
"
cd "$ROOT/package" && rm -f "$HOME/adinfinitium_addendum3_rover_v3.zip" && zip -qr "$HOME/adinfinitium_addendum3_rover_v3.zip" adinfinitium_addendum3_rover_v3
ls -la "$HOME/adinfinitium_addendum3_rover_v3.zip"
unzip -l "$HOME/adinfinitium_addendum3_rover_v3.zip" | tail -1
