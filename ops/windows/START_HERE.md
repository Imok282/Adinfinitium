# START HERE: 4090 PC, 2-hour run of the pilot brain + mission planner (Adinfinitium)

> **One command, about 2 hours, unattended:** `powershell -ExecutionPolicy Bypass -File ops\windows\run_brain.ps1`
> It sets up the PC (first time only), trains the two models, runs their held-out exams, verifies, freezes (SHA-256) and zips the results.
> **After the freeze nothing changes:** no re-training, no tuning, no edits. This is written for whoever operates the PC: a person, or an agent with remote hands.

## 1. Context (the project in 6 lines)
- Adinfinitium, NASA Space Apps 2026, *Lunar Landing Site Navigator*: an autonomous lunar lander + rover that choose where and when to land so the mission survives.
- Framing: **physics-informed ML** on real NASA data (LOLA terrain, SPICE DE440 light, LRO permanent-shadow map). Every model ships next to a physics backup and is examined on sites it never saw.
- Test site: **Site23** (never trained on; Site42 tiles are not in this package, so it is skipped with a note). Validation: Site11. Training: Site01, 04, 06, 07, 20.
- Judges score validity: honest held-out numbers, failures included.
- Results are wanted before **Sun 15 Nov 12:00 IST** (cut rule: anything not working ships as its physics backup).
- Disclose AI-written code (Claude) in the submission; that is handled by the team.

## 2. The two models trained here
| model | what it does | size | physics backup it must beat (on Site23) |
|---|---|---|---|
| **pilot brain** `lander_pilot` | lands the 1.5 t, 5-engine lander: a vision transformer that sees the sensors + a camera image of the ground and adds a correction (residual) to classic ZEM/ZEV guidance. Untrained it flies exactly like the backup | ~15 M | classic ZEM/ZEV guidance (already ~95-98 % success with engine-out, navigation drift and blind-camera failures injected) |
| **meta layer** `planner_set` (+ `planner_logistic` baseline) | picks the landing cell and hour with the best simulated chance of the full mission (land, rover reaches the ice edge, rover returns); writes the plan + decision log | ~0.6 M (data are small: a bigger net would memorise) | the exact Navigator ranking |

How the 2 hours are used: the pilot trains on the GPU for ~95 min **while** 3 processes simulate the planner's labels (lander flights + rover drives per landing cell); then the exams (~10 min), the planner training (~2 min) and a demo plan.

## 3. Steps
1. Copy `adinfinitium_4090_package.zip` to the PC (pen drive), right-click, Extract All, open PowerShell in the extracted `adinfinitium` folder.
2. Make sure the NVIDIA driver works: `nvidia-smi` must print the RTX 4090.
3. Run: `powershell -ExecutionPolicy Bypass -File ops\windows\run_brain.ps1`. First time it installs Python 3.12 (winget), CUDA PyTorch and the libraries (~10 min, needs internet).
4. Leave it. Progress: `type runs\overnight\brain_4090\status.json`, logs in `runs\overnight\brain_4090\logs\`. If the PC restarts or the window closes, run the same command again: it resumes.
5. When it prints `ALL DONE`, bring back `adinfinitium_results_4090.zip` (pen drive) and write down anything that failed.

## 4. Verification (the script runs these; you can re-run them)
```powershell
.venv\Scripts\python.exe -m ops.verify before
.venv\Scripts\python.exe -m ops.verify after --names lander_pilot,planner_set,planner_logistic --families lander,planner
.venv\Scripts\python.exe -m ops.freeze --check --names lander_pilot,planner_set,planner_logistic --families lander,planner
```
**DONE means:** `status.json` says `"stage": "done"`; `verify after` shows 0 FAIL; `freeze --check` says ALL UNCHANGED; `runs\lander\eval.json` and `runs\planner\eval.json` exist with Site23 numbers;
`runs\demo\plan_000.json` and `decisions_000.csv` exist; the results zip is built. A model that does NOT beat its backup is a valid, reported result (the backup ships), not a failure to hide.

## 5. If something goes wrong
| symptom | do |
|---|---|
| `No CUDA build of PyTorch can see the GPU` | update the NVIDIA driver, reboot, re-run |
| `CUDA out of memory` in the `brain` stage | edit `ops\overnight\profiles.py` (brain4090): `--envs 512` -> `256`, re-run (it resumes); note it |
| a label process fails | the others continue; the planner trains on what exists; note which sites are missing |
| PowerShell blocks scripts | use the exact `powershell -ExecutionPolicy Bypass -File ...` form |
| the PC slept | re-run; it resumes. The script sets "never sleep on AC power" |
Never: train on Site23, change model code to improve numbers, delete `data` or `runs`, or edit anything after the freeze.
