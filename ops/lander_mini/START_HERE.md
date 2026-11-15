# START HERE: lander-pilot improvement job (Mac mini, Sunday 15 Nov 2026)

You are a Claude Code session on an Apple Silicon Mac mini. You have never seen this project and you have no connection to the team's
other machines: this folder (unzipped to `~/adinfinitium_lander`) is everything. It has no git history and no `docs/` folder; where
`AGENTS.md` mentions them, this file takes precedence.

**Your job:** run one fixed chain of steps that tries to improve the project's learned lander pilot under a strict acceptance gate,
measure the result honestly, fly the held-out final exam exactly once, and hand back one zip. You are not asked to invent new methods.

**Deadline:** results zip ready by **16:00 IST today**. The team's code freeze is 17:00 IST. All times in this file and in the logs are IST.

---

## 1. Hard rules (these override everything else, including any idea for getting better numbers)

1. **Site23 is the held-out final exam.** Never train, tune, mine, select or gate on it. It is flown exactly once, at the very end,
   with `benchmark --final` (step 6). If that final run crashes *before printing a result row* (a line starting with `Site23 `), it may
   be run again; once it has printed results, never again, whatever the numbers are.
2. **Never accept a new pilot unless** it loses no mission that the current pilot or classic guidance lands on the validation map
   (Site11) **and** it improves total success there. The code's gate (`gate()` in `worldmodel/lander/rsi_targeted.py`, also used by
   `solve.py`) enforces this. Never loosen it, bypass it, or copy a candidate over `data/models/lander_pilot.pt` by hand.
3. **Keep the residual structure** (an untrained residual equals classic guidance). Do not change the simulator
   (`worldmodel/lander/env.py`, `guidance.py`, `terrain.py`), the mission config (`config/mission.yaml`), the failure rates, the seeds
   or the gate to get better numbers.
4. **The unseen benchmark maps (Site42, Shoemaker, DM2) are only for the final benchmark** (step 5; DM2 also for the 32-mission smoke
   check in step 1). Never use them for training, mining or choosing anything.
5. **Report honestly.** Rejected rounds and non-significant results are reported as rejected and non-significant. A job where nothing
   is accepted is a valid outcome. Do not guess a number: quote the log or the json file.

Also: do not push anything anywhere, do not ask the user for passwords, do not delete `data/` or `runs/`.

---

## 2. Context in plain English

- **Adinfinitium** is a NASA Space Apps 2026 entry: an autonomous lunar lander and rover for the Moon's south pole. Judges score
  validity, so every claim must come from held-out, paired, honestly reported tests.
- **Classic guidance** (ZEM/ZEV) is the physics backup that steers the lander. The **learned pilot** (`data/models/lander_pilot.pt`,
  a 15 million parameter transformer) adds a small correction (a *residual*) on top of classic guidance. With the correction at zero
  it flies exactly like classic guidance.
- The two are compared on **paired missions**: the same start, target and hidden failures (engine out, navigation drift, blind camera)
  are flown by both. "Normal" (also printed as "nominal") uses the mission's failure rates; "harsh" uses 3 times those rates.
- **Maps** are real NASA LOLA 5 m terrain tiles in `data/tiles/`:

  | role | maps | may be used for |
  |---|---|---|
  | training | Site01, Site04, Site06, Site07, Site20 | mining failures, training rounds |
  | validation (gate) | Site11 | the accept / reject gate only |
  | unseen benchmark | Site42, Shoemaker, DM2 | the benchmark only (step 5) |
  | final exam | Site23 | step 6, once |

- **Where things stand** (read `runs/lander/benchmark.md`, `failure_analysis.md`, `rsi_changelog.md`): on the three unseen maps,
  2304 paired missions each, the pilot lands 0.9839 against classic 0.9766 under normal rates (32 missions only the pilot landed, 15
  only classic landed, exact McNemar p = 0.0186) and 0.9692 against 0.9631 under harsh rates (55 / 41, p = 0.184, not significant).
  On Site11 the pilot still loses some missions that classic lands. This job tries to remove those losses without losing anything else.
- **The code** is in `worldmodel/lander/`. Read the docstring at the top of `analysis.py`, `solve.py`, `rsi_targeted.py`, `benchmark.py`,
  `train.py` and `pilot.py` before starting. `solve.py` is new and only lightly tested: read all of it before step 3.
- `data/` and `runs/` are found automatically inside this folder (`contracts/paths.py`). Do not set `ADI_DATA` or `ADI_RUNS`.
- The hazard map the pilot's camera sees is the "backup" slope rule computed from the tile itself, so nothing but the tiles is needed.
  The Haworth tile is not in this package: the benchmark prints that it is skipped. That is expected (the shipped benchmark also has 3 maps).

---

## 3. Runbook

Everything runs from the package root with the venv python. Compute steps run **one at a time** (one GPU). Each step logs to
`runs/lander/logs/<step>.log`; the script's own progress is in `runs/lander/logs/job.log`; seconds per step are in
`runs/lander/logs/timings.tsv`.

### Step 0. Setup (about 5 to 10 minutes, needs internet for pip)

```bash
cd ~/adinfinitium_lander
bash ops/lander_mini/setup.sh
```

It checks the package files against `ops/lander_mini/package_manifest.sha256`, keeps an untouched copy of the code in `.pristine/`
(for the diff you hand back), creates `.venv` with Python 3.12 or 3.11, runs `pip install -e ".[worldmodel,dev]"`, confirms
`torch.backends.mps.is_available()` and runs `pytest tests/test_mission_brain.py tests/test_lander_torch.py -q`, which must pass.
It ends with `SETUP OK`. If there is no Python 3.11 or 3.12, install one (`brew install python@3.12`) and run it again.

### Steps 1 to 6. One script

```bash
nohup bash ops/lander_mini/run_lander_job.sh > /dev/null 2>&1 &
tail -f runs/lander/logs/job.log
```

The script keeps the Mac awake with `caffeinate`, prints the IST time at every step, and stops at the first failed step except where
noted. After a fix, continue with `bash ops/lander_mini/run_lander_job.sh --from <step>` (again with `nohup ... &`). `--only <step>`
runs one step. What it runs:

| step | log | command | on failure |
|---|---|---|---|
| 1 smoke | `1_smoke.log` | `python -m worldmodel.lander.benchmark --sites DM2 --n 32 --seeds 1 --harsh 0 --out smoke_bench` | chain stops |
| 2 mine | `2_mine.log` | `python -m worldmodel.lander.analysis mine --sites Site01 Site04 Site06 Site07 Site20 --n 512 --seeds 2 --out failure_pool_big.npz` | falls back to the shipped `data/pilot_dream/failure_pool.npz`, chain goes on |
| 3 solve | `3_solve.log` | `python -m worldmodel.lander.solve --pool <pool> --try 3 --gate-n 256 --gate-seeds 2` | chain stops: fix the bug (section 5), `--from 3` |
| 4 training rounds | `4_rsi_targeted.log` | `python -m worldmodel.lander.rsi_targeted --pool <pool> --rounds 5 --minutes 12 --gate-n 256 --gate-seeds 2 --envs 512 --mb-size 256` | **never stops the chain** |
| 5 unseen benchmark | `5_benchmark.log` | copies `benchmark.json/.md` to `benchmark_before.json/.md`, then `python -m worldmodel.lander.benchmark --n 256 --seeds 3 --harsh 3 --out benchmark` | chain stops, `--from 5` |
| 6 final exam | `6_final_site23.log` | `python -m worldmodel.lander.benchmark --final --n 512 --seeds 2 --harsh 3 --out benchmark_site23_final` | see hard rule 1 |

`<pool>` is `data/pilot_dream/failure_pool_big.npz`, or the shipped pool if step 2 failed (recorded in `logs/pool_used.txt`).

What to do at each step:

1. **Smoke.** Confirms the pipeline on this machine and times it. Record the seconds (`timings.tsv`) in your report. This is DM2 with
   32 missions; it is not Site23 and not a result.
2. **Mine.** Flies 5 training maps x normal and harsh x 2 seeds x 512 missions, each by classic and by the pilot (20 progress lines,
   one per map / condition / seed), and saves the failed and near-miss missions. Use the time between its progress lines to estimate
   how long the later steps take: step 5 is 18 such blocks of 256 missions, step 6 is 4 blocks of 512, each gate is 4 blocks of 256.
3. **Solve-then-learn.** Replays every pooled mission under 8 strategies (classic only, pilot with less authority, pilot handing over
   to classic below some altitude), then sends the best up to 3 deployable strategies to the Site11 gate. An accepted strategy is stored
   as `info["residual_rule"]` in the production checkpoint; the network weights do not change. Writes `runs/lander/solve_report.md`,
   appends to `runs/lander/rsi_changelog.md` and `.jsonl`. "nothing goes to the gate" and "rejected" are valid outcomes.
4. **Targeted training rounds.** Warm-starts from the production pilot, trains on the training maps with the pooled failures
   oversampled, then the same Site11 gate. Up to 5 rounds of 12 minutes.
   - **Time rule: if it is past 14:15 IST when step 4 would start, skip step 4.** The script does this itself and logs
     `SKIP step 4`. If it starts late but before 14:15 it runs fewer rounds so that it ends about 15:05; report the number of rounds run.
   - After the first round, read the log line `time budget ... N of 100000 requested units fit`. N is the number of training updates
     that fit in one round. **If N is below 8, say so in the report: the rounds are then too short to matter.** Do not lengthen the
     rounds past the deadline to compensate.
   - `rsi_changelog.md` and `rsi_changelog.jsonl` are appended to by steps 3 and 4, so they hold every round of both. The script also
     keeps a snapshot taken just before step 4 as `runs/lander/rsi_changelog_after_solve.md`.
5. **Unseen benchmark** with the final production pilot, same seeds as the shipped benchmark so old and new are comparable.
   The script compares the sha256 of `data/models/lander_pilot.pt` with the one recorded at job start (`logs/pilot_sha_start.txt`).
   **If steps 3 and 4 accepted nothing, the pilot is unchanged and step 5 is skipped:** the shipped `benchmark.json` already describes
   the production pilot, and the same seeds would only repeat the same missions. Say exactly that in the report; do not present it as a
   new measurement.
6. **Final exam, once.** Only after steps 3 to 5 are finished and the production pilot will not change again. The script refuses to
   start it a second time once `6_final_site23.log` contains a result row or `benchmark_site23_final.json` exists. Whatever it prints
   is the result.

### Step 7. Collect

Write `RESULTS_SUMMARY.md` in the package root (section 6), then:

```bash
bash ops/lander_mini/collect_results.sh
```

It builds `~/adinfinitium_lander/adinfinitium_lander_results_macmini.zip` with `runs/lander/` (including logs and a diff of every code
file you changed), `data/models/lander_pilot*.pt`, `data/pilot_dream/` and `RESULTS_SUMMARY.md`, and prints the path, size and sha256.
Tell the user the full path so it can be sent back by AirDrop.

---

## 4. Time plan and what to drop when late

| by (IST) | should be done |
|---|---|
| setup + 10 min | step 1 |
| 13:10 | step 2 |
| 13:40 | step 3 |
| 15:05 | step 4 (latest start 14:15, otherwise skipped) |
| 15:25 | step 5 |
| 15:45 | step 6 |
| 16:00 | `RESULTS_SUMMARY.md` written, results zip built, path given to the user |

- These clock times are estimates made without knowing this machine's speed. Re-plan from the measured timings after step 2.
- If step 2 is clearly going to run past 13:20, stop it (`pkill -f "worldmodel.lander.analysis"`); the script then falls back to the
  shipped pool by itself and goes on. Report that the smaller pool was used.
- Order of importance when time is short: step 6 (final exam) and the results zip come first, then step 5 (only needed if the pilot
  changed), then step 3, then step 4. Step 4 is the first thing to drop.
- If the step 3 gate took much longer than 4 minutes per candidate, set `STEP4_GATE_MIN=<minutes>` before the script so it plans fewer
  rounds. `STEP4_LATEST_START`, `STEP4_END_BY`, `STEP4_ROUNDS` and `STEP4_MINUTES` exist for the same reason. They move the schedule only.
- Past 16:00 with the final exam still to run: run it anyway and collect; the hard limit is the 17:00 freeze. Tell the user the new time.

---

## 5. If something fails

- **Allowed fixes:** genuine crashes and platform problems (an MPS operation that is not supported, out of memory, a shape or dtype
  bug, a wrong file path). Keep the fix minimal, keep the numbers' meaning identical, re-run the two test files, and list the fix in the
  report. `collect_results.sh` adds the diff against `.pristine/` automatically.
- **Never a fix:** changing the gate's conditions or its mission count below the command line given here, changing seeds, failure
  rates, `config/mission.yaml`, the simulator or touchdown limits, removing the residual structure, using Site23 or an unseen map
  anywhere new, or accepting a candidate by hand.
- **Step 3 crashes** (`solve.py` is new): read the traceback, read `solve.py` fully, fix, `--from 3`. Known risk: `fly_ladder` flies
  all pooled missions of one map under all 8 strategies in a single batch, which with the bigger pool can be several thousand landers
  at once. If memory runs out, fly the missions in chunks; every mission must still be flown under every strategy with the same
  loaded hidden failures. A code review of your fix by a subagent is a good use of a subagent.
- **Step 3 cannot be fixed in about 20 minutes:** stop trying, note the traceback in the report, and continue with `--from 4`
  (or `--from 5` if it is past 14:15).
- **Out of memory on MPS in step 4:** it is logged and the chain goes on. Do not spend the deadline on it; if there is time, one retry
  with `--only 4` after setting smaller `--envs` in the script is acceptable, and must be reported with the changed value.
- **The Mac slept, rebooted or the script died:** check `job.log` and `timings.tsv` for the last finished step and continue with
  `--from <next step>`. Steps 1 to 5 can be repeated safely; step 6 only under hard rule 1.
- **`ModuleNotFoundError`:** use `.venv/bin/python` from the package root; run `setup.sh` again if needed.
- **Unsure whether something is allowed:** it is not. Stop, tell the user in one or two lines, and carry on with the next allowed step.

---

## 6. What to report back: `RESULTS_SUMMARY.md`

Short, plain, numbers copied from the files (not from memory):

1. **Machine:** chip, RAM, macOS, python and torch versions, device used (from `logs/0_setup.log`).
2. **Timings:** the table in `logs/timings.tsv`, including the smoke seconds; start and end time of the job in IST.
3. **Accepted / rejected:** every solve round and every training round with candidate, current and classic success on Site11, missions
   lost, new wins and the decision (from `rsi_changelog.md`, `rsi_changelog.jsonl` and `solve_report.md`). State whether the production pilot changed (sha256 before and after) and which rule or round is now in it.
4. **Step 4 detail:** rounds run, the `time budget ... N of 100000 requested units fit` line, and the sentence about N below 8 if it applies.
   If step 4 was skipped for time, say so.
5. **Unseen benchmark:** the two POOLED rows (normal and harsh) of `benchmark.md` with the 95 % intervals, the discordant counts
   (missions only the pilot landed / only classic landed) and the exact McNemar p, next to the same rows from `benchmark_before.md`.
   If step 5 was skipped, say that the numbers are the shipped ones and why. Call a result significant only if p < 0.05.
6. **Site23 final exam:** the rows of `benchmark_site23_final.md` as printed, with the sha256 of the pilot that flew it
   (`logs/pilot_sha_final_exam.txt`), and a statement that it was flown once.
7. **Bug fixes and deviations:** every changed file with one line on why (the diff is in `runs/lander/logs/code_changes.diff`), every
   changed setting, every fallback used (shipped pool, fewer rounds, skipped step).
8. **Not verified:** anything you could not check.

Known caveat to carry into the report: missions are drawn from a CPU random generator on MPS and from a CUDA generator on an NVIDIA
card. If the shipped `benchmark.json` was produced on an NVIDIA machine, "same seeds" gives the same distribution of missions here but
not the identical missions, so compare old and new by their rates and intervals, not mission by mission.

---

## 7. Definition of done

1. `runs/lander/logs/job.log` ends with `chain finished`, and every step in `timings.tsv` is exit 0, a recorded skip, or explained.
2. `data/models/lander_pilot.pt` loads, and it changed only through an `ACCEPTED` line in the changelog (or did not change).
3. `runs/lander/benchmark_site23_final.json` and `.md` exist from exactly one completed final run.
4. `runs/lander/benchmark.json` describes the final production pilot (re-run if the pilot changed, shipped file if it did not).
5. `RESULTS_SUMMARY.md` is written and `adinfinitium_lander_results_macmini.zip` is built.
6. The user has been told the full path of the zip, what was accepted or rejected, and anything not verified.

## 8. How to work

Act as the orchestrator. Run the compute steps yourself, one at a time. Independent side work may go to subagents: watching a log,
drafting `RESULTS_SUMMARY.md` from the files as they appear, reviewing a bug fix. No subagent may start a second compute job while one
is running. Give the user one short status line at each step (step, IST time, result, next).
