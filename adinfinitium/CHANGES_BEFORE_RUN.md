# Changes made to this package before the overnight run (Sat 14 Nov 2026, 16:20-18:30 IST)

Made after an independent review of the package, before the single Mac mini run. Rule followed: no change to model
architecture, training hyperparameters, labels or physics. Only pipeline/ops bug fixes, and evaluation/reporting ADDITIONS
(new numbers beside the existing ones; nothing removed or replaced). One git commit per fix group in the working copy
(`git log`), nothing pushed.

"Numbers?" says whether the change can alter any model number compared with the unchanged package.

## P0: the run would break or misreport without these

| # | What changed | Why | Where | Numbers? |
|---|---|---|---|---|
| 1 | `verify after` reads the ledger of the run id it is given (`--run-id`, default `macmini_night`), never "the last one on disk". It checks each `eval.json` belongs to tonight's checkpoint (name and checkpoint id), that the run reached `run_end`, and says "held-out rests on Site23 only" when Site42 tiles are missing. | `sorted(glob)[-1]` picked the `smoke1` ledger; a leftover smoke `eval.json` passed. | `ops/verify.py:90`, `:152`, `:165` | No |
| 2 | `run_everything.sh`: a FAILED `verify after` no longer freezes (logs are still collected, a banner and `runs/VERIFY_FAILED.txt` say so); lock dir + pid so a second launch exits; the whole script runs under `caffeinate`; setup is skipped only when `.venv/.setup_complete` exists; passes `--run-id` and `--deadline`; runs `pytest -q` and records it (never blocks). `run_all` also refuses to start beside a live runner of the same run id. | Freeze after a failed verify; two runners side by side; sleep during setup/smoke; a half-made `.venv` skipped setup forever. | `ops/macmini/run_everything.sh:8`, `:13`, `:26`, `:43`; `ops/overnight/run_all.py:241`; `ops/macmini/setup_mac.sh:18` (marker, test-only libraries jax/plotly/pandas/streamlit so `pytest -q` collects) | No |
| 3 | Training budget now counts setup time (light's ray-cast data build ran before the clock started). A training stage that ends (ok or timeout) without its named checkpoint is recorded `failed`, retried once at once, and again on `--resume` (it used to be skipped as "finished"). Checkpoints are written to a temp file and renamed. Earlier attempt logs are kept (`*.attemptN.log`). | The runner could kill light at 60 min before its first save; `timeout` was never retried; a kill mid-write could leave a truncated checkpoint. | `worldmodel/budget.py:17`; `worldmodel/light/train.py:115`, `hazard/train.py:126`, `terrain/train.py:128`; `*/model.py` `save_checkpoint`; `ops/overnight/run_all.py:88`, `:192`, `:298` | Indirect: a long setup now leaves fewer training steps inside the same wall-clock budget (the budget is honest). No change to steps, LR or batch settings. |
| 4 | Tiles stage is ONE process with `--workers 3` (3 NASA connections in total, was 3 processes x 4 threads = 12). Each tile is built right after its window is fetched. Order: centre windows of all sites first, then grid windows round-robin (Site42, Site23, Site11, Haworth, Shoemaker, DM2, rest). An unreachable site is logged and skipped. A rebuild keeps an attached PSR mask; tiles also store `centre_xy_m`. | NASA 503s; tiles only existed after a whole group finished; one bad site killed its group; rebuild wiped `psr_mask`. | `worldmodel/hazard/lola.py:54`, `:175`, `:249`; `ops/overnight/profiles.py:49` | Tile pixels are identical for the same window. Numbers can move only because a different set of tiles exists when training starts. |
| 5, P0-b | Smoke trains from scratch under `smoketest_*` names (the seed's `light_smoke_last.pt` / `rover_smoke_last.pt` are never resumed), total budget 23 min, and everything it writes goes under `runs/_smoke/` (ledger, `eval.json`, `train_log.jsonl`). | Smoke could train 0 steps, take ~2 h, and leave `runs/<family>/eval.json` to be verified and frozen as the real result. | `ops/overnight/profiles.py:57`; `ops/overnight/run_all.py:231`, `:286` | No (smoke only) |
| 6, P0-d | Results zip keeps the 4 best checkpoints, `runs/`, this file and `excluded_windows.json`; `*_last.pt` go to a separate optional zip; `data/light` only if <= 300 MB; warns above 1.5 GB. `collect_results.sh --add-report` adds `RESULTS_FOR_TEAM.md` after it is written. | Zip was ~2.5 GB and was built before the report existed. | `ops/macmini/collect_results.sh:9`, `:10` | No |
| 7 | Report prints held-out TEST sites first and whole (only long per-direction lists are summarised), then validation Site11, labelled as validation and shortened; top-level caveats and sources are printed; says when Site42 is absent. | 2500-char cut showed Site11 and hid Site23/Site42. | `ops/overnight/report.py:84`, `:115` | No |
| 8 | `run_all --deadline <ISO>` (run_everything passes Sun 15 Nov 08:00 +05:30): before each training stage its budget is capped to what still fits with every later eval budget reserved. Terrain shrinks first and is skipped below 30 min (its eval too, if there is no terrain checkpoint); remaining shortfall is shared by the other training stages (never below 5 min; rover is never skipped). Every cap is a `deadline_cap` ledger event; `--dry-run --deadline` shows the plan. | No rule existed for a late start or an overrun. | `ops/overnight/run_all.py:94`; `ops/overnight/profiles.py:124` | Only when the deadline bites: fewer training minutes (logged). |
| P0-a | Rover always writes `<name>.pt` (from the first update). Until a rollout meets the unchanged "best" rule (>= 50 finished episodes) it holds the latest weights and says so: `selection = "last, not best: ..."` in the checkpoint, the log and `eval.json`. Rover `--max-minutes` counts world-building time. | No `rover_driver_mini.pt` meant baselines-only eval and a verify FAIL. | `worldmodel/rover/train.py:32`, `:40` | How "best" is chosen is unchanged. When no best exists, a model is now evaluated where before there was none. |
| P0-c | START_HERE no longer tells the operator to re-run `evaluate` by hand after the freeze. | Those commands overwrite frozen `eval.json` files. | `START_HERE.md` step 4 | No |
| P0-e | PSR stage logs a failing tile and continues; exit 1 only if all fail. | First failure left every later tile without shadow data. | `worldmodel/rover/psr.py:93` | Only which tiles carry a real PSR mask (rover worlds). |
| P0-f | Evaluators (hazard, light, terrain; rover per world) score the TEST sites first and rewrite `eval.json` after every window (`complete`, `windows_planned`); verify WARNs on a cut-short eval. Eval budgets were measured on the mini with the real architectures (see "Measured on the mini"). | Eval budgets were unmeasured; a timeout left no `eval.json` at all. | `worldmodel/hazard/evaluate.py:98`, `light/evaluate.py:113`, `terrain/evaluate.py:109`, `rover/evaluate.py` `write()`; budgets `ops/overnight/profiles.py` macmini: terrain_eval 25 -> 50 min with terrain training 150 -> 125, rover_eval 20 -> 30 | Evals: no (same per-window seeds; only the order and when the file is written). Terrain training: 25 fewer budget minutes, about 7,700 of the 8,000 requested steps instead of all 8,000 at the measured speed. |
| 4b | NASA windows are read in 128-row bands, and a site is dropped for the run after 2 failed windows in a row. | On the mini the Haworth centre window failed 6 times in a row (~7 min) as one 1024-row read; the banded read worked on the next try (117 s). Not proven whether banding or a passing NASA fault made the difference. Each failed window costs ~7 min of retries, so a dead site would eat the stage. | `worldmodel/hazard/lola.py` `_read_window`, `main` | No (same pixels) |
| review | From an independent read-only review of these changes: tiles are saved to a temp file and renamed (a `psr`/`tiles` stage killed at its budget mid-write would leave a truncated tile that crashes every trainer); `run_everything.sh` stops without verifying/freezing/collecting if another runner owns the run; its `pytest` is capped at 15 min; a failed `unzip` stops setup; `verify after` treats a deadline-skipped stage's missing files as WARN, checks held-out coverage against the windows really scored (rover's `test_sites` list used to satisfy it), and requires `run_end` after the last launch; pid locks only honour a live runner (pid reuse after a reboot). | Found by review and by a sandboxed rehearsal of the real profile on the mini. | `contracts/tiles.py` `save_tile`; `ops/macmini/run_everything.sh`; `ops/macmini/setup_mac.sh`; `ops/verify.py`; `ops/overnight/run_all.py:241` | No |
| crash | Smoke terrain used `--depth 2`, which puts self-attention on 64x64 tokens; validation then asks MPS for 32 GiB and crashes. Smoke now uses `--depth 3` like the real profile. Found by the first smoke on the mini. | Genuine platform crash in the smoke settings. | `ops/overnight/profiles.py` (smoke terrain stage) | No (smoke only; the real profile was already depth 3 and ran on MPS in the probe) |

## P1: honesty additions

| # | What changed | Why | Where | Numbers? |
|---|---|---|---|---|
| 9 | Hazard `eval.json` adds `unet_measured_px`, `backup_slope_rule_measured_px`, `measured_fraction` per window and a `caveat`. Existing keys unchanged. | The all-pixel truth is ~90% NASA interpolation. Note the limit: laser pixels are the model's own input, so these rows do not measure gap recovery. | `worldmodel/hazard/evaluate.py:45` | No (reporting only) |
| 10 | Rover takes `--hazard-name` (profile passes `hazard_unet_big` to train and eval; slope-rule backup if that file is missing). The hazard source is recorded in the checkpoint and `eval.json`, with a real `trained_on` list and a warning if train and eval used different hazard maps. | `world.py` hard-coded the small seed `hazard_unet`. | `worldmodel/rover/world.py:44`; `ops/overnight/profiles.py:119` | **Yes: the rover (and the baselines' observations) now see tonight's hazard map instead of the seed model's.** No reward, physics or hyperparameter changed. |
| 11 | New stage `spatial_split` (after tiles/psr, before any training): any training/validation window whose tile, grown by 1.28 km, touches a test tile is listed in `data/tiles/excluded_windows.json` and dropped by `regions_for_site`. `verify after` FAILs if a checkpoint trained on such a window. | Splits were by site name only. | `ops/spatial_split.py:29`; `worldmodel/hazard/lola.py:64`; `ops/verify.py:130` | **Yes if anything is excluded** (it only removes training/validation windows). On the seed tiles nothing is excluded: nearest train-test centres are 73.6 km apart (Site20-Site23), val-test 112.6 km. |
| 12, P1-b | START_HERE: real budgets (rover 150, terrain 150), smoke duration and location, timeout semantics, rover eval covers Site23 and Site42, never kill the runner, profile edits apply to the next run only, `--add-report`, and a "Known limitations to state in the results" section. | Stale lines. | `START_HERE.md` (copies: `CLAUDE.md`, `ops/macmini/START_HERE.md`) | No |
| P1-a | Rover `eval.json` records `ice_source` per world, `worlds_without_psr_edge_targets` and a caveat. | `reached_ice` can count plain science targets where a world has no shadow edge. | `worldmodel/rover/evaluate.py:89` | No |
| P1-c | `verify after` builds each network from its checkpoint and loads the weights. | Counting tensors in the file does not prove it loads. | `ops/verify.py:72` | No |

## Tests added (all run by `pytest -q`)
`tests/test_overnight_runner.py` (deadline table, no-checkpoint retry, smoke isolation, report order),
`tests/test_tiles_pipeline.py`, `tests/test_spatial_split.py`, `tests/test_prerun_fixes_rover_hazard.py`.

## Seed vs tonight's downloads
The seed holds only the centre 1024 px window of Site01, 04, 06, 07, 20 (train), Site11 (validation) and Site23 (test), plus
the SPICE kernels and light files for those. NASA's files give 89 windows in total: Site23 13, Haworth 22, every other site 6.
Haworth, Shoemaker, DM2, Site42 and all grid windows must download.

## Measured on the mini before the run (Apple M5 Pro, 64 GB, torch MPS, python 3.11)
Short probes with the REAL architectures under throwaway names (`probe_*`, deleted afterwards; logs in
`~/adinfinitium-run/probe.out` and `~/adinfinitium-run/probe_outputs/`). They are not training results.

| stage | measured | what it means for tonight |
|---|---|---|
| smoke (12 stages) | about 2 min in total, device mps, all ok after the terrain depth fix | cap is 23 min |
| hazard train (60.4 M) | 200 steps + validation in 127 s (about 95 steps/min) | the 40 requested epochs take about 85 min of the 106 min training budget |
| light train (38.3 M) | 113 steps in 101 s (about 67 steps/min); data build 8 s for 11 windows | about 3,500 of the 6,000 requested steps fit in 53 min |
| terrain train (49.9 M) | 138 steps in 117 s (about 70 steps/min), no out-of-memory | about 7,700 of 8,000 steps in 110 min |
| rover train (mini) | world build 123 s for 20 worlds; 3 updates fit in the following minute | a few hundred of the 800 requested updates in 132 min (rough) |
| hazard eval | 19 s for 4 windows incl. figures | 25 windows: about 2 min (budget 15) |
| light eval | 44 s for 7 windows | 25 windows: about 3 min (budget 15) |
| terrain eval | 361 s for 4 windows (about 95 s each) | 25 windows: about 40 min, the 19 test windows first: about 30 min (budget now 50) |
| rover eval | 631 s for 8 test worlds x 512 rovers | budget now 30 |
| NASA download | 87-117 s per window with 3 connections | 89 windows in total: see below |

## Tiles fetched before the run
With 3 connections one window takes about 100 s, so the 81 windows missing from the seed would need about 2.3 h, more than
the 90 min tiles budget. To leave tonight's stage a smaller job, the SAME command the profile runs
(`worldmodel.hazard.lola all --grid --workers 3`, one process, 3 connections) was run on the mini from 16:45 to 17:45 IST
and then stopped between two tiles. Its log is `~/adinfinitium-run/prefetch_tiles2.out` (first attempt, with the Haworth
failure: `prefetch_tiles.out`). Tonight's tiles stage skips what is on disk and fetches the rest; the PSR and spatial_split
stages then run over every tile. Result: 50 of the 89 windows are on disk and all load (DM2 5, Haworth 5, Shoemaker 5, Site11 5, Site23 5, Site42 5, Site01/04/06/07/20 4 each); 39 remain, about 60 min at the measured speed, inside the 90 min budget if NASA keeps that speed. No window failed in the second attempt. The spatial check over these 50 tiles excludes nothing: the closest training tile to a test tile is 37.6 km away (Haworth r0c3 to Site23 r0c0), the closest validation tile 93.5 km.

## Rehearsal
The real `macmini` profile was run once in a sandbox on the mini (separate models and runs folders, real tiles, budgets x 0.03,
a deadline 15 min away; outputs in `~/adinfinitium-run/rehearsal_outputs/`). It behaved as designed: evals killed at their
budget left partial `eval.json` files with the test sites first, hazard timed out with its checkpoint, rover built 32 worlds
and wrote its checkpoint, terrain and its eval were `skipped_deadline` in the ledger, the report and `verify after` ran.
Nothing from it is left in `data/models/` or `runs/`.

## Not done / not verified
- If a tile file is unreadable for another reason than a kill mid-write, `spatial_split` and the trainers that load it still
  stop (not guarded).
- If terrain is skipped by the deadline but an older terrain checkpoint exists, `terrain_eval` still runs its 50 min.
- The real `run_everything.sh` was not run end to end (that is the overnight run). Its lock, failed-verify and collect paths
  were run with a stubbed `python`; `verify after` was run against the smoke sandbox (49 checks, 0 FAIL, 1 WARN).
- The deadline rule is unit-tested and shown by `--dry-run --deadline`; it has not capped a real stage yet.
- The spatial check has only seen whatever tiles were on disk; tonight's stage recomputes it over the full set.
- Rover with the real `hazard_unet_big` map is untested (the checkpoint does not exist until tonight); the probe used a
  1.5-minute hazard model of the real size.
- `info["best_score"]` of the rover still counts rollouts under 50 episodes when resuming (left as is: part of how best is chosen).
- One commit per fix group, not strictly per fix: the rover files (P0-a, 10, P1-a) and `lola.py` (4 and the hook of 11) each
  carry more than one fix.
