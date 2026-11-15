#!/bin/sh
# Rover v3.1: build ~/adinfinitium_addendum5_rover_v31.zip on the mini (plan, report, eval json, stage files, code, diff, brain config). Writes nothing in frozen folders.
set -eu
ROOT=$HOME/adinfinitium-rover31
CODE=$ROOT/code
NAME=adinfinitium_addendum5_rover_v31
PK=$ROOT/package/$NAME
export ADI_DATA=$ROOT/data PYTHONPATH=$CODE PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
PY=$HOME/adinfinitium-run/adinfinitium/.venv/bin/python
rm -rf "$ROOT/package"
mkdir -p "$PK/code/worldmodel/dream" "$PK/code/ops" "$PK/code/tests" "$PK/final" "$PK/validation" "$PK/development_smoke" "$PK/diagnosis" "$PK/records"
cp "$CODE/ROVER_V31_PLAN.md" "$PK/"
for f in ROVER_V31_REPORT.md diagnosis.md diagnosis.json; do [ -f "$ROOT/records/$f" ] && cp "$ROOT/records/$f" "$PK/"; done
cp "$ROOT"/records/agents/* "$PK/records/" 2>/dev/null || true
cp "$ROOT/runs/final/eval.json" "$ROOT/runs/final/tables.md" "$PK/"
cp "$ROOT"/runs/final/*.json "$ROOT"/runs/final/*.log "$ROOT"/runs/final/LAUNCHED "$ROOT"/runs/final/FINISHED "$ROOT"/runs/final/jobs.txt "$PK/final/"
cp "$ROOT"/runs/val/*.json "$ROOT"/runs/val/*.log "$PK/validation/"
cp "$ROOT"/runs/dev/* "$PK/development_smoke/" 2>/dev/null || true
cp "$ROOT"/diag/*.py "$ROOT"/diag/*.json "$PK/diagnosis/" 2>/dev/null || true
cp -R "$CODE/worldmodel/rover3" "$PK/code/worldmodel/"
cp "$CODE/worldmodel/dream/changes.py" "$PK/code/worldmodel/dream/"
cp -R "$CODE/ops/rover31" "$PK/code/ops/"
cp "$CODE/tests/test_rover3.py" "$PK/code/tests/"
cp "$CODE/ROVER31_COMMIT.txt" "$CODE/rover_v31.diff" "$PK/code/"
find "$PK" -name __pycache__ -type d -prune -exec rm -rf {} +
cd "$CODE" && $PY -c "
import dataclasses, json
from worldmodel.rover3.brain import Theta, candidate_table, NEVER
t = Theta(v31=True); c = candidate_table(t, 'cpu')
kinds = ['strategy (wait at the lander, then go)', 'baseline naive', 'baseline cautious', 'baseline cautious_wait_light', 'safe harbour', 'shelter on the stay-alive route']
json.dump({'what': 'rover v3.1 brain as evaluated: scripted planner + scripted executor + stay-alive route search + strict-first choice; no neural network, so there are no weights. Theta(v31=True) and code/worldmodel/rover3 are everything needed to run it; Theta() is frozen v3.',
           'theta': dataclasses.asdict(t), 'candidates': [{'index': i, 'wait_hours': (None if int(c['wait'][i]) >= NEVER else int(c['wait'][i])), 'charge_lo': float(c['lo'][i]), 'charge_hi': float(c['hi'][i]),
           'kind': kinds[int(c['kind'][i])]} for i in range(len(c['kind']))],
           'choice_rule': 'rank 3 strict return > rank 2 battery never empty > rank 1 v3 rule; ties as v3',
           'maps_used': 'hazard U-Net hazard_unet_big.pt output, LOLA slope, modelled light (SPICE + skyline), NASA PSR-edge targets'}, open('$PK/rover_brain_v31_config.json', 'w'), indent=1)
"
cd "$ROOT/package" && rm -f "$HOME/$NAME.zip" && zip -qr "$HOME/$NAME.zip" "$NAME"
ls -la "$HOME/$NAME.zip"
unzip -l "$HOME/$NAME.zip" | tail -1
