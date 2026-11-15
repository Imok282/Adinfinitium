# RESULTS FOR THE TEAM: Adinfinitium overnight run `macmini_night`

NASA Space Apps 2026, Lunar Landing Site Navigator. Written Sun 15 Nov 2026 after the run was frozen.
Every number below names its file. Paths are relative to `~/adinfinitium-run/adinfinitium/` on the Mac mini (the same paths are inside the results zip).
Numbers are rounded to 3 decimals (or 1 decimal for percentages); the full values are in the `eval.json` files.

## 1. Summary

- **What ran:** one unattended run of 14 stages that downloaded NASA LOLA 5 m terrain, trained 4 networks, and scored each against its physics backup on two sites it never saw (Site23, Site42).
- **When:** launched Sat 14 Nov 17:55:47 IST (checks, unit tests and a smoke test of 2 min 38 s first). Real run 17:58:48 Sat to 02:01:28 Sun 15 Nov. Verified, frozen and zipped by 02:02:29 (`runs/run_everything.log`).
- **Machine:** Mac mini, Apple M5 Pro, 18 cores, 64 GB, macOS 27.0.1, PyTorch 2.15.0 on the Apple GPU (MPS). The machine was shared with other work.
- **All 14 stages ended `ok` on the first attempt.** No timeout, no retry. `verify after`: "87 checks, 0 FAIL, 0 WARN".
- **Hazard network: beats its backup.** F1 over all windows of Site23 0.965 vs 0.899, Site42 0.917 vs 0.567. On laser-measured pixels only: 0.949 vs 0.832 and 0.926 vs 0.677.
- **Light network: beats its backup.** Skyline error (mean over windows) Site23 0.534 deg vs 0.675 deg, Site42 0.270 deg vs 0.342 deg. About 21% lower on both.
- **Terrain diffusion network: beats its backup on error, but its uncertainty band is far too narrow.** Hide-and-check RMSE Site23 1.84 m vs 17.92 m, Site42 2.82 m vs 24.82 m. Its "90%" band holds the truth only 22% and 37% of the time.
- **Rover network: FAILED.** On the 8 test worlds (centre window of each test site, 4 dates) it reached ice in 1.7% of drives, never returned (0.0%), and died in 95.5%. The simple planners reached ice in 6.9% to 19.6% and returned in 6.0% to 9.3%. The planner is the rover driver that works from this run.
- Only part of the requested training fitted the time budgets: light 3,258 of 6,000 steps, hazard 37 of 40 epochs, rover 144 of 800 updates, terrain 7,671 of 8,000 steps.
- Nothing was changed after the freeze. The rover numbers are simulation only.

## 2. Stages of run `macmini_night`

Source: `runs/overnight/macmini_night/ledger.jsonl` (times converted from UTC to IST), minutes as printed in `RUN_REPORT.md`, training amounts from `logs/<stage>_0.log`.

| stage | status | start (IST) | end (IST) | minutes | budget min | how much fitted |
|---|---|---|---|---|---|---|
| env_check | ok | 17:58:48 | 17:58:49 | 0.0 | 2 | - |
| kernels | ok | 17:58:49 | 17:58:50 | 0.0 | 10 | 5 SPICE files ok |
| tiles | ok | 17:58:50 | 19:00:19 | 61.5 | 90 | 85 of 89 windows on disk (50 before the run, 35 built tonight, 4 missing) |
| psr | ok | 19:00:19 | 19:01:27 | 1.1 | 15 | shadow mask on 85 tiles, 0 failed |
| spatial_split | ok | 19:01:27 | 19:01:28 | 0.0 | 5 | 0 windows excluded |
| points | ok | 19:01:28 | 19:01:36 | 0.1 | 30 | 132 light files (11 centre tiles x 12 months) |
| light | ok | 19:01:36 | 19:55:03 | 53.5 | 60 | **3,258 of 6,000 steps (54%)**. The log first estimated 3,538, then stopped at 3,258 |
| light_eval | ok | 19:55:03 | 19:58:15 | 3.2 | 15 | 25 of 25 windows |
| hazard | ok | 19:58:15 | 21:29:31 | 91.3 | 120 | **37 of 40 epochs** (200 steps each) |
| hazard_eval | ok | 21:29:31 | 21:32:05 | 2.6 | 15 | 25 of 25 windows |
| rover | ok | 21:32:05 | 23:33:38 | 121.5 | 150 | **144 of 800 updates (18%)** |
| rover_eval | ok | 23:33:38 | 23:41:35 | 8.0 | 30 | 8 of 8 worlds |
| terrain | ok | 23:41:35 | 01:23:54 | 102.3 | 125 | **7,671 of 8,000 steps (96%)** |
| terrain_eval | ok | 01:23:54 | 02:01:28 | 37.6 | 50 | 25 of 25 windows |

Total budget 717 min; used 482.7 min (8 h 02 min 40 s wall clock). The 08:00 deadline rule never had to cut a stage (no `deadline_cap` event in the ledger).
"ok" for the rover stage means the program finished. It does not mean the rover learned (section 4).

## 3. Held-out results: network vs physics backup

Test sites Site23 (13 windows) and Site42 (6 windows) were never trained on. Each window is 1024 x 1024 pixels at 5 m.
"All windows" rows are computed by us from the per-window entries of `eval.json`. "Centre window" rows are the entries `lola5m_site23` and `lola5m_site42`; these are the lines `verify after` printed, so they are one window, not the whole site.

### 3.1 Hazard (`runs/hazard/eval.json`, complete, 25 of 25 windows)

Metric: a pixel is "unsafe" when NASA's slope is 12 deg or more. F1 combines how many unsafe pixels were found and how many alarms were right (1.0 is perfect). Missed-hazard rate is the share of unsafe pixels called safe.
Backup: the slope rule on a gap-filled surface. "Measured px" keeps only pixels with a real laser return.

| site | pixels | U-Net F1 | backup F1 | U-Net missed | backup missed |
|---|---|---|---|---|---|
| Site23, all 13 windows (counts summed) | all | 0.965 | 0.899 | 2.9% | 13.7% |
| Site23, all 13 windows (counts summed) | measured | 0.949 | 0.832 | 4.2% | 26.6% |
| Site23, centre window | all | 0.926 | 0.814 | 7.2% | 20.9% |
| Site23, centre window | measured | 0.913 | 0.737 | 7.7% | 38.6% |
| Site42, all 6 windows (counts summed) | all | 0.917 | 0.567 | 5.3% | 56.9% |
| Site42, all 6 windows (counts summed) | measured | 0.926 | 0.677 | 5.9% | 45.1% |
| Site42, centre window | all | 0.876 | 0.556 | 9.0% | 55.7% |
| Site42, centre window | measured | 0.890 | 0.595 | 9.3% | 53.0% |

- Per-window F1 range, all pixels: Site23 U-Net 0.899 to 0.999, backup 0.764 to 0.963. Site42 U-Net 0.876 to 0.973, backup 0.484 to 0.618.
- Per-window F1 range, measured pixels: Site23 U-Net 0.886 to 0.999, backup 0.643 to 0.940. Site42 U-Net 0.890 to 0.979, backup 0.581 to 0.796.
- The U-Net has the higher F1 in all 19 test windows, on both pixel sets.
- Not all in the network's favour: on Site23 measured pixels the backup has the higher precision (0.961 vs 0.940 summed). It raises fewer false alarms, but it misses far more hazards.
- Laser-measured pixels are 6.2% to 14.2% of a Site23 window and 3.7% to 4.5% of a Site42 window.
- Four Site23 windows are 96% to 99.8% unsafe. A high F1 is easy there, so the top of the range says little.
- Validation (Site11, 6 windows, summed): U-Net 0.946 vs backup 0.862. This is a validation number.

### 3.2 Light (`runs/light/eval.json`, complete, 25 of 25 windows)

Metric: skyline MAE in degrees = mean absolute error of the predicted horizon height over all compass directions, at 1,000 sampled cells per window. Truth is an exact ray-cast of the 5 m grid.
Backup ("baseline"): the same ray-cast on terrain averaged 4x. Lower is better.

| site | network MAE | backup MAE | network p95 | backup p95 |
|---|---|---|---|---|
| Site23, mean of 13 windows | 0.534 deg | 0.675 deg | - | - |
| Site23, centre window | 0.474 deg | 0.628 deg | 1.99 deg | 2.65 deg |
| Site42, mean of 6 windows | 0.270 deg | 0.342 deg | - | - |
| Site42, centre window | 0.350 deg | 0.469 deg | 1.35 deg | 2.01 deg |

- Per-window MAE range: Site23 network 0.305 to 0.711, backup 0.378 to 0.906. Site42 network 0.197 to 0.350, backup 0.250 to 0.469.
- The network has the lower MAE in all 19 test windows.
- Not all in the network's favour: the median (p50) error is lower for the network in only 9 of 13 Site23 windows and 1 of 6 Site42 windows. The gain is in the larger errors, not the typical one.
- The worst single error stays large for both. Centre windows: 14.7 deg (network) vs 15.9 deg (backup) on Site23, 16.7 vs 17.2 on Site42. Worst over all test windows: 61.5 vs 63.8 deg on Site23, 23.4 vs 25.8 on Site42.
- Sun-fraction error after the physics step, centre windows: Site23 0.022 vs 0.029, Site42 0.014 vs 0.020. Sunlit-hours error: 14.0 h vs 21.2 h and 8.6 h vs 13.9 h.
- Validation (Site11, mean of 6 windows): 0.381 deg vs 0.491 deg. This is a validation number.

### 3.3 Terrain diffusion (`runs/terrain/eval.json`, complete, 25 of 25 windows)

The evaluation was not cut short: `"complete": true`, `windows_planned` 25, all scored in 37.6 of the 50 budget minutes.
Metric (hide-and-check): in 96 patches per window, half of the laser-measured pixels are hidden. The model makes 6 versions from the other half. RMSE and MAE compare the mean of the 6 versions with the hidden real heights, in metres. "Coverage" is how often the hidden height falls inside the model's 5% to 95% band.
Backup: gaussian gap-fill of the shown pixels plus noise from LOLA's error map. Output of this model is always labelled "GENERATED - NOT OBSERVED".

| site | hidden pixels | network RMSE | backup RMSE | network MAE | backup MAE | network coverage | backup coverage |
|---|---|---|---|---|---|---|---|
| Site23, all 13 windows (weighted by hidden pixels) | 265,141 | 1.84 m | 17.92 m | 1.12 m | 8.03 m | 21.9% | 17.8% |
| Site23, centre window | 21,767 | 1.31 m | 10.26 m | 0.88 m | 4.37 m | 21.2% | 19.1% |
| Site42, all 6 windows (weighted by hidden pixels) | 49,150 | 2.82 m | 24.82 m | 1.57 m | 17.76 m | 37.3% | 4.9% |
| Site42, centre window | 9,174 | 3.15 m | 21.00 m | 1.64 m | 14.82 m | 33.4% | 5.7% |

- Per-window RMSE range: Site23 network 1.31 to 2.92 m, backup 8.30 to 37.07 m. Site42 network 2.46 to 3.21 m, backup 19.71 to 34.23 m.
- The network has the lower RMSE and MAE in all 19 test windows.
- **Failure to report:** the band should hold the truth about 90% of the time. It holds it 16.2% to 33.3% of the time on Site23 windows and 33.4% to 41.5% on Site42. The model is much too sure of itself. Do not show its band as a real uncertainty.
- Our own arithmetic, not from a file: with only 6 versions even a perfect generator would score about 66% with this band (at most 71%), not 90%. The measured coverage is still far below that.
- On coverage the backup beats the network in 4 of the 13 Site23 windows. The network is ahead in the other 9 and in all 6 Site42 windows.
- The backup's error is very large. We did not check whether its gap-fill settings are the reason. A stronger interpolator would narrow the gap, so treat the size of the win with care.
- Validation (Site11, 6 windows, weighted): RMSE 1.90 m vs 17.92 m. This is a validation number.

### 3.4 Rover (`runs/rover/eval.json`, complete, 8 of 8 worlds, simulation only)

8 held-out worlds = centre window of Site23 and Site42 on 4 dates, 512 simulated rovers each.
Each cell is: reached ice % / returned home % / died % / mean return (the simulator's score, higher is better).
"Reached ice" means the rover got to a dark-and-cold shadow-edge target. It is not confirmed ice.
The three planners all follow a Dijkstra shortest-path plan: naive, cautious, and cautious that waits for light.

| world | network (PPO) | naive planner | cautious planner | cautious, waits for light |
|---|---|---|---|---|
| site23 @ 2027-03 | 2.9 / 0.0 / 97.9 / -44.7 | 18.2 / 4.9 / 95.1 / -13.8 | 0.6 / 0.4 / 59.4 / -28.8 | 0.0 / 0.0 / 89.5 / -45.3 |
| site23 @ 2027-06 | 2.5 / 0.0 / 100.0 / -42.4 | 13.3 / 0.4 / 99.6 / -25.4 | 0.0 / 0.0 / 17.8 / -10.9 | 0.0 / 0.0 / 89.5 / -45.5 |
| site23 @ 2027-09 | 2.5 / 0.0 / 92.2 / -48.5 | 27.5 / 13.5 / 86.5 / +2.6 | 9.0 / 7.4 / 74.0 / -15.2 | 4.3 / 3.7 / 86.1 / -32.1 |
| site23 @ 2027-12 | 2.1 / 0.0 / 96.1 / -46.0 | 30.9 / 16.2 / 83.8 / +8.6 | 11.5 / 8.2 / 76.2 / -14.2 | 4.7 / 3.7 / 86.5 / -32.2 |
| site42 @ 2027-03 | 0.6 / 0.0 / 100.0 / -42.5 | 13.3 / 4.3 / 89.6 / +1.9 | 5.5 / 4.3 / 59.8 / -65.8 | 1.0 / 0.8 / 68.2 / -28.9 |
| site42 @ 2027-06 | 0.0 / 0.0 / 100.0 / -41.2 | 6.6 / 2.5 / 92.8 / -8.4 | 8.2 / 7.6 / 55.7 / -50.6 | 3.9 / 3.9 / 68.6 / -23.1 |
| site42 @ 2027-09 | 1.0 / 0.0 / 90.4 / -73.3 | 19.9 / 12.5 / 76.6 / +27.3 | 23.2 / 19.3 / 44.9 / +15.3 | 19.7 / 16.8 / 52.3 / +25.7 |
| site42 @ 2027-12 | 2.1 / 0.0 / 87.1 / -103.2 | 27.1 / 19.9 / 70.1 / +44.9 | 26.0 / 23.0 / 46.5 / +22.7 | 21.5 / 18.9 / 50.6 / +28.1 |
| **Site23, mean of 4** | 2.5 / 0.0 / 96.5 / -45.4 | 22.5 / 8.7 / 91.3 / -7.0 | 5.3 / 4.0 / 56.8 / -17.3 | 2.2 / 1.9 / 87.9 / -38.8 |
| **Site42, mean of 4** | 0.9 / 0.0 / 94.4 / -65.0 | 16.7 / 9.8 / 82.3 / +16.4 | 15.7 / 13.6 / 51.7 / -19.6 | 11.5 / 10.1 / 59.9 / +0.4 |
| **all 8 (file summary)** | 1.7 / 0.0 / 95.5 / -55.2 | 19.6 / 9.3 / 86.8 / +4.7 | 10.5 / 8.8 / 54.3 / -18.4 | 6.9 / 6.0 / 73.9 / -19.2 |

- The network never returned home in any world. It has the lowest mean return of the four drivers in 4 of 8 worlds.
- For completeness: on Site23 the network's reached-ice (2.5%) is above the wait-for-light planner (2.2%) and, in March and June, above the cautious planner. It still never returned.
- The planners are not good either. The best one for returning (naive, 9.3%) still dies in 86.8% of drives. The cautious planner dies least (54.3%).
- `hazard_source`: "hazard U-Net hazard_unet_big.pt" in training and evaluation. `worlds_without_psr_edge_targets`: 0 of 8.

## 4. Rover: what happened

- **It never succeeded once in training.** `runs/rover/train_log.jsonl` has 144 updates. In every one, reached-ice is 0.0, returned is 0.0 and `best_score` is 0.0. In 15 updates no drive even finished.
- **The policy collapsed to "stop".** A read-only diagnosis during training (`records/agents/rover-diagnosis.md`) rolled out the live checkpoint at updates 94 and 96. It chose "stop" with probability 1.00 and never left the 60 m start disc. Rovers then die only when the battery runs out.
- **The evaluated checkpoint is the network after update 1.** The rule keeps a new "best" only when a score is above the old best (`worldmodel/rover/train.py:211`). Update 1 scored 0.0 with 227 finished drives and was saved. Nothing ever scored above 0.0, so it was never replaced. File times agree: `data/models/rover_driver_mini.pt` 21:36:07, `rover_driver_mini_last.pt` 23:33:37.
- **So the label is misleading.** `eval.json` and `verify after` say `selection: "best by training success (rollout with >= 50 finished episodes)"`. In truth the scored network is almost untrained. The table in 3.4 is the update-1 network, not the result of 144 updates. The final network (update 144) was not scored on the test sites.
- **The worlds can be solved.** The planners reach ice on the same test worlds (table above). In the diagnosis, the best planner reached ice in 0.32 to 0.49 of drives on 5 of the 6 training worlds checked (on the sixth, site20 in March, every planner got 0.00). Random driving reached 0.00.
- **The hazard-map switch is not the cause.** Tonight's hazard map and the older seed map agree on 94.9% (Site01) and 97.2% (Site07) of cells in the central 512 pixels of two training windows. On the one world tested with both maps (Site01, March) the planner scores are in the same band. Reward and physics code are unchanged from the original package (same diagnosis file).
- Likely reasons, from the diagnosis, not proven: only 18% of the planned updates fitted; targets are 330 to 1,600 m away and random driving never finds one, so the learner never saw a reward for reaching ice.
- **What to ship from this run:** the planner. Team rule: a model that does not beat its backup ships as the backup.

**After the freeze, a separate rover v2 attempt and a self-learning addendum were started; they are NOT part of these frozen results and are reported separately in their own files.**

## 5. Failures and deviations (complete list)

1. **Rover network failed** (sections 3.4 and 4). Its "best" label is wrong in effect.
2. **Terrain uncertainty band fails** (coverage 16.2% to 41.5% where about 90% is claimed, section 3.3).
3. **4 of 89 planned tile windows are missing, all training windows** (`logs/tiles_0.log`). Site06 r1c0, Haworth r0c4 and Haworth r1c4: NASA read failed ("Read failed. See previous exception for details"). **The underlying error was not captured in the log, so the root cause is unknown.** Haworth r4c3: downloaded, then rejected because 0.01% of its pixels were not finite. All validation (6) and test (13 + 6) windows are present.
4. **Less training than requested** on all four models (section 2). Light stopped at 3,258 steps after first estimating 3,538.
5. **Earlier smoke failure, before the run** (`runs/_smoke/overnight/smoke1_1638/`): at 16:39 IST the terrain smoke stage crashed with "MPS backend out of memory ... Tried to allocate 32.00 GiB". Cause: the smoke setting `--depth 2` puts attention on 64 x 64 tokens. Fix: smoke uses `--depth 3`, like the real profile. `terrain_eval` also failed then because no checkpoint existed. The fresh smoke at launch (17:56 to 17:58) passed all 12 stages.
6. **Unproven claim by the fix lead:** its log says a 16:30 pytest of the untouched package on the mini failed with "5 jax collection errors". The pre-launch verifier searched three logs on the mini and found no pytest output in them. The 5-error runs that are on record happened on the MacBook. Treat the claim as not evidenced.
7. **Pytest count discrepancy (167 vs about 30), not formally resolved.** The fix lead logged "167 passed, 7 skipped, 1 deselected". The verifier counted 26 passed, 4 skipped. The launch log (`runs/run_everything.log`) shows 174 result marks: 167 dots and 7 "s", no failure mark; its last line alone has 26 dots and 4 "s", which matches the verifier's count. This points to 167 being right, but the "N passed" summary line was never captured and nobody re-ran it to settle it.
8. **Pre-run download trouble:** the Haworth centre window failed 6 times in a row in the first prefetch. Reads were then split into 128-row bands and it worked. Whether banding or a passing NASA fault fixed it is not proven.
9. **A bug caught before the run:** the first version of the rover evaluator change would have crashed at run time (a function was defined after the loop that calls it). Review caught it; the unit tests had not (`records/agents/fix-lead.md`, 17:02).
10. **Corrected statements in the monitor's record** (`RUN_LOG.md`): 11 earlier claims are listed there as corrected (C1 to C11). During the run: "86 tiles, 3 unexplained" was corrected to 85 tiles, 4 missing (the 86th file is a synthetic demo tile). The 19:30 audit note "3,538 of 6,000 light steps (59%)" was an estimate; the final count is 3,258 (54%). Some entry times were estimates and were corrected.
11. **Small record mismatches:** the plan said the rover network has about 9 M parameters; it has 11,377,354. Row 12 of `CHANGES_BEFORE_RUN.md` still says "terrain 150" min; the real budget was 125. The copy of `run_everything.log` inside the zip was taken up to 59 s before the end, so it lacks the last 4 lines (the zip sizes and "ALL DONE"). Changed files were not linted (no linter installed).
12. **Stage timeouts or retries in the real run: none.** Every stage has `attempt: 1` and status `ok` in the ledger.

## 6. Code changes before the run

Source: `CHANGES_BEFORE_RUN.md` (in the zip). 21 change entries, made in 24 commits between 16:29:57 and 17:45:51 IST on 14 Nov, after four pre-run reviews (`records/reviews/`). Final code version: commit `07eb0f0`.
Rule followed: no change to model architecture, training settings, labels or physics.

- **15 pipeline and safety fixes (P0):** verify reads the right run; no freeze after a failed verify; one runner at a time; training time counts setup; a stage with no checkpoint is retried; one tile downloader with 3 NASA connections; smoke test kept apart from real results; smaller results zip; deadline rule; evaluators score test sites first.
- **6 honesty additions (P1):** hazard metrics on laser-measured pixels; the rover's hazard map is named; the spatial split check; rover ice source; verify builds each network; updated instructions.
- **Changes that can move a held-out number:**
  - #10: the rover and the planners see tonight's hazard map (`hazard_unet_big`) instead of the old seed model's. Marked "Yes" in the file.
  - #11: the spatial check can drop training windows near a test tile. Tonight it dropped 0, so no effect.
  - #3 and P0-f: setup time now counts in the budget, and 25 budget minutes moved from terrain training to terrain evaluation. Fewer training steps fit.
  - #4 and P0-e: which tiles exist and carry a shadow mask when training starts.
  - P0-a: the rover always leaves a checkpoint, so a network is evaluated where before there might have been none.
  - #8: the deadline rule. It did not act tonight.
- **Code changes during the run: none. After the freeze: none.** At 02:01:30 IST the run log says: "frozen 9 items" and "FROZEN check: ALL UNCHANGED". The check was run again at 02:31:03 IST, just before this report was copied to the mini:

```
$ .venv/bin/python -m ops.freeze --check
FROZEN check: ALL UNCHANGED
```

It was run once more after the report was added to the zip. That output cannot be inside this file; it is in `records/agents/results-writer.md`.

## 7. Machine and measured speed

Source: `logs/env_check_0.log`, `logs/<stage>_0.log`, the ledger. Use these to plan the 4090 run.

| item | value |
|---|---|
| chip / memory / OS | Apple M5 Pro, 18 cores, 64 GB (env_check reports 68.7 GB decimal), macOS 27.0.1; chip and cores from `sysctl` on the mini at 02:20 IST |
| software | Python 3.11.17, torch 2.15.0, device `mps` (no CUDA) |
| free disk | 350.8 GB at start, 324.6 GB at end (`status.json`). The run's own `data/` is 4.8 GB and the two zips 2.4 GB (decimal); the rest of the drop is other work on the shared machine (not checked) |

| stage | size | measured speed |
|---|---|---|
| light train | 38,279,809 params | 3,258 steps in 3,138 s = 0.96 s per step (62 steps/min), validation included; "data ready in 34s" (`logs/light_0.log`, line 1) |
| hazard train | 60,361,561 params | 145 to 156 s per epoch of 200 steps with validation (mean 147 s, about 0.73 s per step) |
| rover train | 11,377,354 params | about 49 s per update (512 rovers x 32 steps); world build 192 s for 32 worlds |
| terrain train | 49,943,681 params | 7,671 steps in 6,134 s = 0.80 s per step (75 steps/min), validation included |
| light eval | 25 windows | 191.8 s (7.7 s per window) |
| hazard eval | 25 windows with figures | 153.5 s (6.1 s per window) |
| rover eval | 8 worlds x 512 rovers x 4 drivers | 477.6 s (60 s per world) |
| terrain eval | 25 windows x 96 patches x 6 versions | 2,253.9 s (90 s per window) |
| NASA tiles | 36 windows fetched, 3 connections | 59 to 145 s per window (mean 89 s); stage 61.5 min |

The mini was shared: other sessions and jobs ran on it all night (1-minute load 0.2 to 1.5 at the 11 status checks in `records/checks/checks.log`, 5.95 at 02:17 after the run, from other work). Speeds on an idle machine may be better.

## 8. Known limits of validity

Each was checked against the final code (`07eb0f0`) or tonight's files and still applies.

- **Hazard truth is mostly interpolated.** NASA's slope map is built from a surface where only 3.7% to 14.2% of the pixels in a test window hold a laser return. The all-pixel score mostly measures agreement with NASA's interpolation. The measured-pixel score is beside it in 3.1, but those pixels are the network's own input, so it does not measure gap recovery either.
- **The hazard comparison leans towards the network.** The backup's slope is one of the network's inputs, and so is NASA's error map (`worldmodel/hazard/features.py`).
- **Haworth is 19 of the 60 training tiles (32%).** One crater dominates training.
- **One validation site.** Site11 (6 windows) picked the best hazard, light and terrain checkpoints.
- **Splits are by site, with a spatial check.** 60 train + 6 validation tiles were checked against 19 test tiles with a 1,280 m margin: 0 excluded. The closest training tile to a test tile is 28.8 km away (Haworth r0c3 to Site23 r3c0); the closest validation tile 90.1 km (`data/tiles/excluded_windows.json`).
- **Light: Sun and Earth directions are modelled** (SPICE, marked `[MODELLED]`) and exist for centre tiles only. In the evaluation, the 17 non-centre test windows borrowed the DM2 January directions for the sun-fraction numbers. Skyline MAE does not depend on this.
- **Light: ray range limit.** Truth and model both ignore terrain beyond 1,280 m, so far mountains are missing. The input is the same grid averaged 4x, not an independent coarse product.
- **Terrain: the validation score in training is not hide-and-check.** It is a denoising error at fixed noise levels. Only `terrain_eval` hides real pixels. Hidden pixels sit next to shown pixels on the same laser track, which flatters both model and backup.
- **Shadow (PSR) mask is NASA's 20 m map upsampled 4x** to 5 m. Position may be off by up to 10 m (half a 20 m pixel, stated in `worldmodel/rover/psr.py`).
- **"Reached ice" means reaching a dark-and-cold shadow-edge target.** It is not confirmed ice.
- **The rover is simulation only.** Its "best" checkpoint is picked on training rollouts (no validation world), and tonight that rule kept the update-1 network (section 4). The rover also sees NASA's official slope as an input.
- **Earlier figure "0.54 deg vs 0.74 deg" for light** is a validation number from an older small model. It is not a result of this run.

## 9. Files

On the Mac mini, folder `~/adinfinitium-run/adinfinitium/`:

- **`adinfinitium_results_macmini.zip`** (637,147,760 bytes = 637 MB before this report was added; `ls -lh` prints 608M). Contains `runs/` (ledger, `RUN_REPORT.md`, stage logs, 4 `eval.json`, 4 `train_log.jsonl`, 25 hazard pictures, training curves, `FROZEN.json`, the smoke runs under `runs/_smoke/`), `CHANGES_BEFORE_RUN.md`, `data/tiles/excluded_windows.json`, the 4 evaluated checkpoints (`data/models/hazard_unet_big.pt`, `light_skyline_big.pt`, `rover_driver_mini.pt`, `terrain_diffusion_mini.pt`), the 132 light files in `data/light/`, and this report.
- **`adinfinitium_results_macmini_last_checkpoints_optional.zip`** (1,777,693,553 bytes = 1.78 GB; `ls -lh` prints 1.7G). Contains only the 4 resumable `*_last.pt` files (weights plus optimiser). Needed only to continue training. `rover_driver_mini_last.pt` is the update-144 rover network.
- **`runs/FROZEN.json`**: SHA-256 of the 4 checkpoints, the 4 `eval.json` files and the code, frozen at 2026-11-14T20:31:30 UTC (02:01:30 IST Sun 15 Nov). Check with `.venv/bin/python -m ops.freeze --check`.
- **Full attempt record** (MacBook): `~/Documents/Ojas/Hackathon/records/RUN_LOG.md` (time-ordered, with failures and corrected claims), `records/agents/*.md` (one file per agent), `records/reviews/*.md` (the four pre-run reviews), `records/checks/checks.log` (every status check), and `~/Documents/Ojas/Hackathon/TRACKER.md`.
- Code at the run's version: `~/Documents/Ojas/Hackathon/work/adinfinitium`, commit `07eb0f0`.

Do not re-run any `evaluate` command by hand: it would overwrite a frozen `eval.json`.
