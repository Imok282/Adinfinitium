# What to do on the Mac mini

1. AirDrop `adinfinitium_lander_macmini_package.zip` to the Mac mini.
2. Unzip it into the home folder so that the folder is `~/adinfinitium_lander`:
   `unzip ~/Downloads/adinfinitium_lander_macmini_package.zip -d ~/`
3. Open a terminal there (`cd ~/adinfinitium_lander`), start Claude Code, and paste everything between the two lines below as the first message.
4. When it finishes it tells you the path of `adinfinitium_lander_results_macmini.zip`. AirDrop that file back.

---

You are running a time-boxed job for a NASA Space Apps hackathon project (Adinfinitium, a lunar lander). The project package is unzipped at `~/adinfinitium_lander` on this Apple Silicon Mac mini. This machine has no connection to the team's other machines and the folder has no git history, so that folder is all you have. If the folder is somewhere else, find it (it contains `ops/lander_mini/START_HERE.md`) and use that location throughout.

Before doing anything else, read `~/adinfinitium_lander/ops/lander_mini/START_HERE.md` from start to end. It has the context, the hard rules, the runbook, the time plan, what to do when something fails, the report format and the definition of done. Then read the docstrings at the top of `analysis.py`, `solve.py`, `rsi_targeted.py`, `benchmark.py`, `train.py` and `pilot.py` in `worldmodel/lander/`, and all of `solve.py` (it is new and lightly tested).

Hard rules, which nothing overrides:
1. Site23 is the held-out final exam. Never train, tune, mine, select or gate on it. It is flown exactly once, at the very end, with `benchmark --final`. If that run crashes before printing a result row it may be run again; once it has printed results, never again.
2. Never accept a new pilot unless it loses no mission that the current pilot or classic guidance lands on the validation map (Site11) and it improves total success there. The code's gate enforces this. Never loosen or bypass it.
3. Keep the residual structure (an untrained residual equals classic guidance). Do not change the simulator, the mission config, the failure rates, the seeds or the gate to get better numbers.
4. The unseen benchmark maps (Site42, Shoemaker, DM2) are only for the final benchmark, never for training, mining or choosing anything.
5. Report honestly. Rejected rounds and non-significant results are reported as such. Nothing accepted is a valid outcome.

The job:
- Step 0: `cd ~/adinfinitium_lander && bash ops/lander_mini/setup.sh` (venv with Python 3.12 or 3.11, libraries, the two test files must pass, MPS must be available).
- Steps 1 to 6: `nohup bash ops/lander_mini/run_lander_job.sh > /dev/null 2>&1 &`, then follow `runs/lander/logs/job.log` and the per-step logs in `runs/lander/logs/`. The script keeps the Mac awake with `caffeinate`; if you run any long command outside it, wrap it in `caffeinate -i -m -s` yourself. The chain stops at the first failed step, except that a failed mining step falls back to the shipped pool and a failed training step (step 4) never stops it. After a fix, continue with `--from <step>`.
- Step 4 (training rounds) is skipped if it is past 14:15 IST when it would start. The script enforces this; do not override it. After its first round, read the log line "time budget ... N of 100000 requested units fit" and, if N is below 8, say in the report that the rounds were too short to matter.
- If a step crashes because of a real bug or a platform problem, fix it minimally without weakening the gate or changing what the numbers mean, re-run the two test files, and record the fix. `solve.py` is the most likely place.
- Step 7: write `RESULTS_SUMMARY.md` in the package root as START_HERE.md section 6 describes (what was accepted and rejected, the pooled benchmark rows with 95 % intervals, discordant counts and exact McNemar p for normal and harsh, the Site23 rows, timings, every bug fix and deviation), then run `bash ops/lander_mini/collect_results.sh`.

How to work: act as an orchestrator. Run the compute steps yourself, one at a time, because they share one GPU; never start a second compute job while one is running. You may delegate independent side work to subagents: watching a log, drafting the summary from the result files, reviewing a bug fix. Check on the running step every few minutes rather than waiting blindly. Give me one short status line at each step: step, IST time, result, what is next.

Deadline: everything finished and the results zip built by 16:00 IST today (Sunday 15 November 2026); the team's code freeze is 17:00 IST. If you are running late, drop work in the order START_HERE.md section 4 gives: step 4 first; the Site23 final exam and the results zip matter most.

Do not ask me for passwords, do not push anything to GitHub, do not delete `data/` or `runs/`. End by telling me the full path of the results zip (`~/adinfinitium_lander/adinfinitium_lander_results_macmini.zip`, written out in full), its size, what was accepted or rejected, and anything you could not verify.

---
