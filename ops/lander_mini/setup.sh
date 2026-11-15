#!/bin/bash
# Step 0 (setup) of the lander-pilot job on the Mac mini. Run from anywhere:   bash ops/lander_mini/setup.sh
# Creates .venv (Python 3.12 or 3.11), installs the project with torch, checks the package files, the two test files and MPS.
# Safe to run again. Stops at the first problem and says what to do.
set -eu -o pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
export TZ=Asia/Kolkata
mkdir -p runs/lander/logs
LOG="runs/lander/logs/0_setup.log"
say() { echo "[$(date '+%H:%M:%S') IST] $*" | tee -a "$LOG"; }

say "setup in $ROOT"

# 1. files of the package are complete and unchanged (manifest written when the package was built)
MAN="ops/lander_mini/package_manifest.sha256"
if [ -f "$MAN" ] && [ ! -f .pristine/.done ]; then
  if shasum -a 256 --quiet -c "$MAN" >> "$LOG" 2>&1; then
    say "package files: all match the manifest ($(wc -l < "$MAN" | tr -d ' ') files)"
  else
    say "FAIL: some package files are missing or differ from the manifest (see $LOG). Unzip the package again."
    exit 1
  fi
fi
for t in dm2 shoemaker site01 site04 site06 site07 site11 site20 site23 site42; do
  [ -f "data/tiles/lola5m_$t.npz" ] || { say "FAIL: data/tiles/lola5m_$t.npz is missing"; exit 1; }
done
for f in data/models/lander_pilot.pt data/pilot_dream/failure_pool.npz config/mission.yaml runs/lander/benchmark.json; do
  [ -f "$f" ] || { say "FAIL: $f is missing"; exit 1; }
done

# 2. untouched copy of the code, so every later fix can be shown as a diff (the package has no git history)
if [ ! -f .pristine/.done ]; then
  rm -rf .pristine && mkdir -p .pristine
  cp -R worldmodel contracts config sim_jax tests ops pyproject.toml .pristine/
  find .pristine -name __pycache__ -type d -prune -exec rm -rf {} +
  touch .pristine/.done
  say "kept an untouched copy of the code in .pristine/ (used for the diff in the results zip)"
fi

# 3. Python 3.12 or 3.11
PY=""
for c in python3.12 python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.12 /usr/local/bin/python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  say "FAIL: no Python 3.11 or 3.12 found. Install one, then run this script again:"
  say "   brew install python@3.12        (or)   uv venv --python 3.12 .venv   (or the installer from python.org)"
  exit 1
fi
say "using $PY ($("$PY" --version 2>&1))"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY="$ROOT/.venv/bin/python"

# 4. libraries (torch with MPS comes from the default macOS arm64 wheel)
say "installing libraries (a few minutes; needs internet)"
"$VPY" -m pip install --upgrade pip >> "$LOG" 2>&1
"$VPY" -m pip install -e ".[worldmodel,dev]" >> "$LOG" 2>&1 || { say "FAIL: pip install failed, see $LOG"; exit 1; }

# 5. checks
"$VPY" - <<'EOF' 2>&1 | tee -a "$LOG"
import platform, sys, torch
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  machine {platform.machine()}  macOS {platform.mac_ver()[0]}")
ok = torch.backends.mps.is_available()
print("MPS available:", ok)
if not ok:
    sys.exit("FAIL: torch cannot use the Apple GPU (MPS). Check that this is an arm64 Python (not Rosetta) and macOS >= 12.3.")
x = torch.randn(256, 256, device="mps")
print("MPS compute check:", float((x @ x).abs().mean()) > 0)
from contracts.mission import load_mission
from contracts.paths import data_root, runs_root
load_mission()
print("mission.yaml loads; data ->", data_root(), "; runs ->", runs_root())
EOF
say "running the two test files"
"$VPY" -m pytest tests/test_mission_brain.py tests/test_lander_torch.py -q 2>&1 | tee -a "$LOG"

sysctl -n machdep.cpu.brand_string hw.memsize 2>/dev/null | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"
df -h . | tail -1 | tee -a "$LOG"
say "SETUP OK. Next:  nohup bash ops/lander_mini/run_lander_job.sh > /dev/null 2>&1 &   then   tail -f runs/lander/logs/job.log"
