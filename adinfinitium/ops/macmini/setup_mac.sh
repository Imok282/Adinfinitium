#!/bin/bash
# Mac mini setup: python venv + libraries + seed data. Run from the package root:  bash ops/macmini/setup_mac.sh
set -e
cd "$(dirname "$0")/../.."
PY=""
for c in python3.12 python3.11 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "Need Python >= 3.11 (brew install python@3.12), then re-run."; exit 1; fi
echo "using $($PY --version)"
rm -f .venv/.setup_complete
$PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scipy pydantic pyyaml spiceypy rasterio matplotlib pytest
pip install -e . --no-deps
# only the unit tests of the apps / lander sim import these; without them `pytest -q` stops at collection
pip install jax plotly pandas streamlit || echo "WARNING: test-only libraries failed to install; some unit tests will not collect (training is unaffected)"
if [ -f seed/adinfinitium_seed_data.zip ] && [ ! -d data/tiles ]; then
  mkdir -p data
  unzip -q seed/adinfinitium_seed_data.zip -d .      # on its own line so a failed unzip stops setup (no marker)
  echo "seed data unpacked into data/"
fi
mkdir -p runs
touch .venv/.setup_complete
echo "setup done. Next:  . .venv/bin/activate && python -m ops.verify before"
