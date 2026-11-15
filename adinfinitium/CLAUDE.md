# START HERE: you are the overnight trainer for Adinfinitium (Mac mini)

> **ONE ACCESS, ONE RUN.** The team gets a single chance on this machine. Do everything in this one session, unattended, and leave nothing to a later visit.
> The whole job is one command: `bash ops/macmini/run_everything.sh` (setup, checks, smoke test, the overnight run, verification, freeze, results zip). Read this file, run that command, check on it every 30 to 60 minutes, and fix only genuine crashes.
> **After training finishes, NOTHING is changed:** the models, evaluations and code are frozen with SHA-256 (`python -m ops.freeze`); never re-train, re-tune or edit afterwards.

You are a Claude Code session on a different subscription from the team's. You have never seen this project. This file is all the context you need.
Your job: **set up this machine, train the project's models overnight on real NASA data, verify them, and hand back results and an honest record of everything that went wrong.** You are not asked to improve the models or add features.

## 1. The project in 10 lines
- **Adinfinitium** is a NASA Space Apps 2026 entry (*Lunar Landing Site Navigator*, Pune, online): an autonomous lunar lander + rover that picks where and when to land so the mission survives.
- Judges score Impact, Creativity, Validity, Relevance, Presentation. **Validity means honest, held-out, real-data results.** No invented numbers.
- Framing: **physics-informed ML**. Every network is trained on real NASA data and always shipped next to a plain physics backup; we report whether the network beats that backup on sites it never saw.
- Data is real NASA only: LOLA 5 m terrain (PGDA), SPICE DE440 Sun/Earth directions (checked against JPL Horizons), LRO permanently-shadowed-region map.
- **Test sites Site23 and Site42 are NEVER trained on.** Validation = Site11. Training = Site01, 04, 06, 07, 20, Haworth, Shoemaker, DM2 (Site20v2 is excluded: leakage).
- The four models you will train (all PyTorch, `cuda`/`mps`/`cpu` picked automatically):
  1. **hazard** attention U-Net (~60 M params): recovers a hazard map from the ~10% of pixels the laser really measured. Label = NASA slope >= 12 deg. Backup = slope rule on a gap-filled surface.
  2. **light** skyline transformer (~38 M): a learned correction on top of an exact ray-cast of coarse terrain, predicting the 360-azimuth skyline; physics turns it into Sun fraction / Earth visibility. Untrained model == physics backup.
  3. **rover** driver (vision transformer, a mini size here): PPO in a vectorised simulator (battery, solar, comms line of sight, ice from the NASA permanently-shadowed map, return home). Compared with naive / cautious / wait-for-light baselines.
  4. **terrain** diffusion (attention U-Net, a mini size here): fills terrain gaps, trained only on laser-measured pixels via hide-and-check. Its outputs are always labelled "GENERATED - NOT OBSERVED".
- Deadlines (IST): results are wanted **by Sun 15 Nov 09:00**; the team's cut rule is Sun 15 Nov 12:00 (anything not working ships as its physics backup).

## 2. Hard rules
1. **Do not modify model or physics code** to make numbers look better. Only fix a real crash (platform bug), keep the fix minimal, and write it in `RESULTS_FOR_TEAM.md` under "code changes".
2. **Record failures.** A crashed stage, an out-of-memory, a model that does not beat its backup: all of it goes in the report. The team shows failures at the showcase.
3. Never train on Site23 / Site42. `python -m ops.verify after` checks this; if it FAILs, stop and report.
4. No credentials: do not ask the user for passwords, do not log in to GitHub, do not push anything. Results go back as a zip file.
5. Do not delete `data/` or `runs/`. Never open more than 3 connections to NASA in total (NASA answers HTTP 503 when hammered); the `tiles` stage is one process with `--workers 3`, so do not start a second download beside it.
8. **Never kill the runner mid-run** to change something. Edits to `ops/overnight/profiles.py` only apply to the next `run_all`; a second launch refuses to start while the first is alive (lock file). To recover from a real crash, let it finish (or confirm it is dead) and re-run `bash ops/macmini/run_everything.sh`.
6. Run modules from the package root as `python -m module` (the runner uses `python -s -m`). Python must be >= 3.11.
7. Use only the project's own data. No pretrained weights, no extra datasets.

## 3. Step by step

**What you actually do:** unzip, then start the single script detached (it takes about 12 hours, longer than one shell call can run):
```bash
unzip adinfinitium_macmini_package.zip && cd adinfinitium
nohup bash ops/macmini/run_everything.sh > /dev/null 2>&1 &
tail -f runs/run_everything.log          # progress; Ctrl-C only stops the tail
```
If the Mac reboots or the script dies, run the same two lines again: it resumes (a second copy refuses to start while the first is alive). `CHANGES_BEFORE_RUN.md` lists every change made to this package before the run and whether it can affect model numbers; include it in your report. Steps 0 to 5 below are what that script does, written out so you can check and fix a failing stage.

### Step 0: unpack
```bash
unzip adinfinitium_macmini_package.zip && cd adinfinitium
bash ops/macmini/setup_mac.sh          # venv, libraries, unpacks seed data into data/
. .venv/bin/activate
```
### Step 1: verify the machine BEFORE training
```bash
python -m ops.verify before            # must end with 0 FAIL. WARN on SPICE kernels is fine (the run fetches them)
pytest -q                              # must pass (a few tests are skipped when data is absent); run_everything.sh also runs it and records the result, without blocking
python -m ops.overnight.run_all --profile smoke --run-id smoke1 --resume     # about 2 min measured on this mini (hard cap 23 min): tiny settings, proves every stage works
```
Smoke trains from scratch under `smoketest_*` names and writes everything under `runs/_smoke/` (never into `runs/<family>/`, which belongs to the real run). Read `runs/_smoke/overnight/smoke1/RUN_REPORT.md`. If a smoke stage failed, fix the cause (see section 6) BEFORE the real run, then re-run the same command (passed stages are skipped).
Also run `python -m ops.overnight.run_all --profile macmini --dry-run --deadline 2026-11-15T08:00+05:30` and look at the time plan (about 11.8 h of budget; the stages are time-fitted: each training script sizes its own steps to its `--max-minutes`, and `--deadline` caps training budgets so the run ends by then: terrain shrinks first and is skipped below 30 min, rover is never skipped; every cap is in the ledger as `deadline_cap`).

### Step 2: start the real run (detached, keeps the Mac awake)
```bash
bash ops/macmini/start_night.sh macmini_night
```
Stages, in order: env_check, kernels (NASA SPICE), tiles (NASA LOLA, budget 90 min, the slowest; the seed only has the centre window of Site01/04/06/07/11/20/23, so Haworth, Shoemaker, DM2, Site42 and every grid window had to download: 50 of the 89 windows were already fetched on this mini before the run with the same command, the other 39 download tonight, about 60 min), psr, spatial_split (drops training/validation windows that touch a test tile), points, light (60 min), light_eval, hazard (120 min), hazard_eval, rover (150 min), rover_eval, terrain (125 min), terrain_eval (50 min: 25 moved from terrain training after measuring the eval on this mini).
A failed stage does NOT stop the others; it is written to the ledger with its error tail.

### Step 3: monitor (check every 30-60 min, cheap)
```bash
cat runs/overnight/macmini_night/status.json       # current stage, last log line, disk
tail -n 5 runs/overnight/macmini_night/logs/*.log
tail -n 3 runs/overnight/macmini_night/ledger.jsonl
```
If the process died (reboot, sleep, crash): re-run `bash ops/macmini/start_night.sh macmini_night`. It resumes: finished stages are skipped, training continues from `*_last.pt`. A training stage that ended (ok or timeout) WITHOUT its named checkpoint is recorded as `failed` and is run again, once immediately and again on every resume.
Healthy signs: the loss in `runs/<family>/train_log.jsonl` goes down; MPS memory use stays under about 40 GB; free disk stays above 10 GB.

### Step 4: verify AFTER training
```bash
python -m ops.verify after --run-id macmini_night
```
It checks the ledger of exactly that run id: every stage finished, every checkpoint loads and builds its network, every held-out evaluation exists, belongs to tonight's checkpoint (not a leftover file) and used Site23/Site42, nothing trained on them or on a window touching them, and it says clearly when Site42 tiles are missing (then every held-out number rests on Site23 only). If it FAILs, `run_everything.sh` does NOT freeze; it still collects the logs. It prints the headline numbers (network vs physics backup). **It does not require the network to win**; it only requires honest evidence. If the network loses to its backup on a held-out site, that is a valid finding: report it.

Do NOT re-run any `evaluate` command by hand after the run: it would overwrite the frozen `eval.json`. The eval stages already wrote the figures (`runs/hazard/compare_*.png`); just look at them.
A rover that scores 0% reach and 100% dead is not "broken code" by itself: the naive/cautious baselines also die often in this environment. Report the numbers as they are.

### Step 5: hand back
```bash
bash ops/macmini/collect_results.sh        # -> adinfinitium_results_macmini.zip (reports, logs, evals, figures, BEST checkpoints; <1.5 GB) + an optional zip of the *_last.pt files
# ... write RESULTS_FOR_TEAM.md (below), then put it INTO the zip:
bash ops/macmini/collect_results.sh --add-report
```
Write `RESULTS_FOR_TEAM.md` in the package root with: (a) the table of stages with status and minutes (copy from `RUN_REPORT.md`), (b) the headline held-out numbers per model: network vs physics backup, (c) every failure and what you did about it, (d) code changes (ideally none), (e) the machine (chip, RAM, free disk, torch version, device used) and the real measured speed (steps per minute from the logs), because the team is planning a 4090 run from it. Tell the user the zip's path.


## 4. The plan: what runs on the mini, in what order, what the fallback is, and when you are DONE

| # | model | size | time budget | why it matters | if it fails or loses to its backup |
|---|---|---|---|---|---|
| 1 | **light** skyline transformer | ~38 M | 60 min | feeds the Navigator's sunlight / Earth-link timeline for every landing site | ship the physics ray-cast (already works; untrained == backup) |
| 2 | **hazard** attention U-Net | ~60 M | 120 min | the hazard map the lander uses | ship the slope rule on the gap-filled surface |
| 3 | **rover** vision-transformer driver (mini) | ~9 M | 150 min | the autonomous rover brain | ship the planner (Dijkstra) driver; report that PPO did not beat it |
| 4 | **terrain** diffusion (mini) | ~50 M | 125 min | fills terrain gaps; shown only as "GENERATED - NOT OBSERVED" | ship the gap-fill + noise backup |

Data first (kernels, tiles, psr, points: about 2 h, most of it NASA download), then models 1 to 4. Total budget about 11.8 h (about 7.6 h of training). **Models 1, 2 and 3 (light, hazard, rover) are equally important must-haves**; terrain (4) is the one to shorten if time runs short (`--skip terrain,terrain_eval` or `--budget-scale 0.6` for the whole run). Never skip rover. The parameter counts above are estimates; `python -m ops.verify after` prints the real ones.

**Fallback rule (team rule, cut at Sun 15 Nov 12:00 IST):** every model keeps a physics backup that already works. A model that crashes, diverges or does not beat its backup on the held-out sites is NOT hidden and NOT tuned until it looks good: it is reported honestly and the team ships the backup.

**Definition of DONE (you may tell the user "training finished" only when all hold):**
1. `runs/overnight/macmini_night/status.json` says `"stage": "done"` and `RUN_REPORT.md` exists.
2. Every stage is `ok` or `timeout` in the ledger (a training stage that timed out without a checkpoint is recorded as `failed`, not `timeout`; `skipped_deadline` on terrain means the deadline rule dropped it); any `failed` stage is explained in `RESULTS_FOR_TEAM.md`.
3. `python -m ops.verify after --run-id macmini_night` prints 0 FAIL.
4. Each of the 4 checkpoints exists in `data/models/` with the name used by the profile (`light_skyline_big`, `hazard_unet_big`, `rover_driver_mini`, `terrain_diffusion_mini`) and loads.
5. Each family has a held-out `eval.json` (Site23/Site42) with network vs backup numbers, plus the figures.
6. `python -m ops.freeze` wrote `runs/FROZEN.json` and `python -m ops.freeze --check` says ALL UNCHANGED.
7. `adinfinitium_results_macmini.zip` is built, `RESULTS_FOR_TEAM.md` is written and added to it (`collect_results.sh --add-report`).
If 1 to 7 hold for light, hazard and rover but not terrain, say "core models done, terrain fell back to physics" and why.

**After training, on a normal laptop:** inference runs there (these are 9 to 60 M-parameter networks, 40 to 250 MB each; a hazard map or skyline takes seconds on CPU). Terrain diffusion is slower, so its tiles are generated once and stored. You do not need to do anything for this; just keep the checkpoints in the results zip.

## 5. What good looks like
| model | held-out target | what to report |
|---|---|---|
| hazard | F1 and missed-hazard rate on Site23/Site42 vs the slope-rule backup | both numbers, plus the picture |
| light | skyline MAE (deg) on Site23/Site42 vs the coarse-terrain ray-cast baseline (earlier small model: 0.54 deg vs 0.74 deg baseline on val Site11) | MAE, p95 |
| rover | reached-ice / returned / died fraction vs naive, cautious, wait-for-light on Site23 and Site42 (Site42 only if its tiles downloaded) | table, plus `hazard_source`, `selection` and `worlds_without_psr_edge_targets` from `runs/rover/eval.json` |
| terrain | hide-and-check error on held-out pixels vs the gap-fill backup | table; never show output as observed |

## 6. Known problems and what to do
- **MPS out of memory**: lower `--batch` (hazard 12 -> 6; terrain 32 -> 16; rover `--envs` 512 -> 256) by editing `ops/overnight/profiles.py` for that stage, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` only if nothing else works, and record it as a failure with the new setting.
- **`randperm` returning garbage on MPS**: already worked around in the code (CPU permutation). If you see index errors, report them.
- **NASA HTTP 503 / slow downloads**: normal. The fetcher retries and uses range reads. If the `tiles` stage times out, the models still train on the tiles that exist; note which sites are missing (Site42 is test-only; the training sites matter most).
- **A stage times out at its budget**: the run continues. If the training stage has its checkpoint it is kept as `timeout` and is NOT trained further on resume; if it has none it is recorded `failed`, retried once at once, and again on resume. An eval killed at its budget still leaves an `eval.json` (test sites are scored first; it says `"complete": false`).
- **`ModuleNotFoundError`**: the venv is not activated, or `pip install -e . --no-deps` was skipped.
- **Python 3.9**: too old, install 3.12 (`brew install python@3.12`).
- Disk almost full: the seed data is about 0.3 GB; the grid tiles add a few GB. Check `df -h .` before the run.

## 7. Known limitations to state in the results (do not hide them)
- **light**: the skyline only sees terrain within 1.28 km (the ray-cast range), in the truth as well as in the model, so more distant mountains are ignored. The "0.54 deg vs 0.74 deg" figure is a VALIDATION number (Site11, earlier small model), not a held-out result.
- **rover**: the best checkpoint is picked on TRAINING rollouts (there is no validation world); if no rollout reached 50 finished episodes the named checkpoint is simply the last one (`selection` in the checkpoint and in `eval.json` says so). `reached_ice` may count plain science targets in worlds with no shadow edge (`worlds_without_psr_edge_targets`). The rover sees tonight's hazard map (`hazard_source`).
- **terrain**: the validation score during training is not a true hide-and-check (it is a denoising error at fixed noise levels); only `terrain_eval` hides real pixels and checks them.
- **hazard**: the truth (NASA slope map) is about 90% interpolated. `eval.json` now also has the same metrics on laser-measured pixels only (`*_measured_px`); those pixels are the model's own input, so they do not measure gap recovery either. State both.
- **splits**: sites are split by name; `spatial_split` additionally drops training/validation windows whose tile, grown by 1.28 km, touches a test tile (`data/tiles/excluded_windows.json`). If Site42 did not download, held-out evidence is Site23 only.

## 8. If you are unsure
Do not guess a number. Say what you ran, what the log says, and what you did not manage to check.
