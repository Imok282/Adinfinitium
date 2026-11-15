# The rover keeps learning after training ends: Dream-RSI-style design (EXPERIMENTAL ADDENDUM)

Rewritten Sat 14 Nov 2026, 21:50 IST. Nothing here touches tonight's main run, its code or its frozen results.
Code: `work/adinfinitium-dream`, branch `dream-rsi`. Models: `rover_dream_*`. Outputs: `runs/_dream/`.

## 0. The one thing this track must show

**Acceptance criterion (the user's words):** "after the model is complete, and the training data is cut off, it should still recursively self-learn."

What that means here, exactly:
- **Start** = the finished, frozen driver from tonight's run (`rover_driver_mini`), opened read-only and never changed. The training pipeline and its data are not used again. The only new information is what the rover experiences when it drives.
- **Loop**, started by one command, no human in it, running until its time budget ends, resumable:
  drive -> record own steps -> update its world model -> practise in imagination -> test on real episodes -> accept the new driver only if it is clearly better, else keep the old one -> repeat.
- **Recursive** = round N starts from round N-1's accepted driver and model, and the Dream-RSI layer uses the record of all earlier rounds to decide how round N spends its tries.

**How the experiment proves or fails it.** For each situation the run writes one table row and one curve point per round: reached ice, returned home, died, blocked moves, hazard under the rover, mean return, accepted or rejected, real steps used, minutes. Round 0 is the frozen driver. The same score starts are used for every round, for the frozen driver that never adapts, and for the physics planners.
- **MET** in a situation only if at least one round was accepted AND the final driver beats the frozen driver on the score starts by more than 2 standard errors.
- **NOT MET** otherwise, and the report prints that. Rejected rounds and rounds that got worse stay in the table.
- Two situations by default: (i) a world it has never seen (Site23, unchanged), (ii) a changed world (Site42 with a blocked route).

**How the rover keeps learning after training ends (for the presentation).**
> Training gave the rover a driver, and then training stopped. From that moment the rover only has its own driving. Every hour of driving is written to a logbook. From the logbook it builds a small "what happens if I do this" model of the world it is actually in: where it gets blocked, how fast the battery really drains. Inside that model it can try thousands of drives without moving a wheel; that is the "dream". A dream can be wrong, so each idea that looked good in the dream must then prove itself on real drives against the current driver, from the same starting points. Only a clear winner replaces the driver; otherwise the old one stays. Then the loop starts again from the new driver and a fuller logbook. It also keeps a record of which kinds of ideas paid off, and uses that record to spend the next round's tries better: that part follows Google's Dream-RSI. Nobody retrains it and no training data is used. It is a bounded demo in simulation, not a rover that can improve itself without limit.

## a. What Dream-RSI is (read from the three links)

| Link | What is there | Licence |
|---|---|---|
| github.com/google-research/dream-rsi | The code: plain Python, about 10.9k lines, core has no dependencies. Cloned read-only to `work/dream-scratch/thirdparty/`; nothing was run. | Apache-2.0 (task files it downloads are AGPL-3.0 and not bundled) |
| github.com/zhengkid/Dream-RSI | No code. README, paper PDF, images; points to the Google repo. | No LICENSE file |
| dream-rsi.com | Project page for the paper "DREAM-RSI: Recursive Self-Improvement Through Evolving Worlds" (arXiv 2609.14858; Google, Google DeepMind, UMD, UVA) | n/a |

The method, in plain words:
- An LLM coding agent tries to solve a hard problem. Its tries form a tree: a few **branches** (fresh ideas), each **refined** several times. A fixed evaluator scores every try.
- A small Python **policy**, `solve(env, budget)`, decides how the tries are spent: open a new branch, refine which branch, when to stop.
- Each finished round is saved as a **replay world**: the tree with its scores. It is a lookup table of what really happened, not a learned model.
- **Dreaming**: an LLM rewrites the policy code. Each rewrite is replayed on every saved world at several budget levels and scored (how much of the world's best result it reaches for how little budget). The best version is kept; the current one stays if nothing beats it.
- **Recursive**: the kept policy runs the next real round, which adds a new world, which the next dream uses.
- It changes no network weights and has no robot or reinforcement learning in it. The authors report 1.4x to 2.4x savings on coding tasks (their claims, not reproduced by us).

**Blocker:** the code needs a paid LLM service (Gemini CLI with `GEMINI_API_KEYS`; Claude or Codex as alternatives). There is no local, mock or key-free mode. We use no keys, so the original cannot be run as it is.

## b. How it applies to the rover

It cannot drive the rover or replace PPO. Lookup replay only works for tries that were really scored; a new driving policy visits places that were never recorded.
It fits one level up: **how the rover spends its re-learning budget after something changes**. So "adding Dream-RSI" means two layers:

1. **The rover re-learns** (Dreamer-style). The driver is the frozen driver plus a small learned **correction** that starts at exactly zero, so round 0 is the frozen driver. A small world model is learned from the rover's own recorded steps; the correction practises inside it ("imagination"); a test on real episodes decides whether it is accepted.
2. **Dream-RSI decides how the re-learning is done** (the named project, key-free version). Each round is a tree: 3 **branches** (careful, medium, bold recipes) x up to 2 **refinements**. Each node is a candidate driver, scored by real episodes against the current driver on the same starts. A `solve(env, budget)` strategy picks the next node and when to stop. The scored tree is saved as a replay world. Between rounds, strategy variants are replayed on all saved worlds and scored with Dream-RSI's attainment-versus-budget area; the best is kept, the current one stays on a tie. The kept strategy runs the next round.

| | Faithful to Dream-RSI | Only inspired by it |
|---|---|---|
| Kept | record a tree per round; reveal-only replay; `solve(env, budget)` interface; budget sweep (5 levels); area score; incumbent can stay; the winner runs the next round | |
| Changed | | strategy variants come from a fixed grid of 54 settings, not from an LLM rewriting code (needs a key); no parallel tries, so their parallelism penalty is dropped; ties keep the current strategy |
| Added | | the world model and imagination under each node (Dream-RSI has neither) |

## c. How the rover is trained and scored today

- **World** = one (site window, date): the central 2.56 km of a real 5 m LOLA tile with slope, hazard map, shadow mask, hourly sunlight, a lander and 16 ice targets. `worldmodel/rover/world.py:127-149`.
- **Episode**: up to 300 hourly steps; drive to an ice target, then home, before the battery dies. `worldmodel/rover/env.py:104-129`, `config/mission.yaml:108-109`.
- **Observation**: a 6 x 64 x 64 map around the rover and 18 numbers (battery, sunlight now and forecast, goal, lander, phase, time, link). `env.py:148-168`.
- **Action**: 8 headings or stop-and-charge. `env.py:179-223`. **Reward**: +50 ice, +25 home, -50 battery death, small shaping. `env.py:204-216`, `mission.yaml:120-125`.
- **Training**: PPO, 512 rovers per world, train sites only. `worldmodel/rover/train.py:105,151-200`, `ops/overnight/profiles.py:118-120`.
- **Scoring**: on Site23 and Site42 the network and three physics planners start from the same starts. `train.py:47-70`, `worldmodel/rover/evaluate.py:95-110`. The backup planner reads the true map: `worldmodel/rover/planner.py:17-51`.

## d. The experiment

**No base phase and no training data.** The world model starts empty in each situation and learns only from that rover's own steps there (map shrunk to 16 x 16, 128-number latent state, heads for next state, reward, episode end, next 18 numbers; about 0.7 M parameters). A "stand-in" head learns what the frozen driver would do, from the frozen driver's own recorded choices; it replaces the frozen driver inside the dream, where there is no map to show it.

**Situations.** `site:change[:date]`. Changes are applied in memory; no data file is edited.
- `none`: the world as built from NASA data (a site the driver never trained on = a never-seen world).
- `blocked_route`: a ring of impassable ground with 3 gaps around the lander. **Synthetic**, labelled so, never shown as real terrain.
- `tight_slope`: slope limit 20 -> 15 degrees. `dusty_panels`: solar output x 0.6 (nothing in the observation shows it).

**One round (the same sizes for every world; the number of rounds is set only by the clock).**
1. Drive: 128 rovers x 100 steps with the current driver; recorded.
2. The current driver drives 64 test episodes from fresh starts; recorded.
3. The strategy opens and refines up to 6 candidates. A candidate = update the world model on everything recorded, practise in imagination with that branch's recipe (careful, medium or bold), then drive the same 64 test starts; recorded. Its score is the mean gain over the current driver.
4. The best candidate is accepted only if its gain is positive and larger than 2 standard errors. Otherwise the driver stays.
5. The accepted driver is scored on the fixed score starts (256 rovers). The candidate tree is saved as a replay world.
6. Dream-RSI step: replay the 54 strategy variants on all saved worlds, keep the best (the current one stays on a tie). Next round.

**What is compared, on the same score starts:** the frozen driver (never adapts); three planners that read the true map (upper reference); the cautious planner **not told** about the change; the self-learning driver after every round.

**Safety rails.**
- "Held-out" is printed only for a test site with zero experience from it (round 0 and the references). Every round after that is labelled **"online self-learning on that site"**: a different, weaker claim that never replaces the frozen held-out table.
- Score starts are never used for driving, learning, testing candidates or choosing the strategy.
- Every candidate, every round and every skipped step is written down. A run id is continued only with `--resume`, never overwritten.
- The frozen `rover_driver_mini` is only read; its SHA-256 is checked before and after, and a resume refuses if the file changed.
- On the mini it runs from its **own folder** (`ops/freeze.py` hashes the main run folder, one extra file there breaks `freeze --check`). It refuses to start while `run_everything.sh` / `ops.overnight.run_all` is alive, before the freeze step, or if the simulator files differ from the main run; it stops between rounds if the main runner appears. `--max-minutes` budget plus a hard kill 10 minutes later.

## e. Feasibility

- **MacBook tonight:** all code, 27 unit tests, smoke runs on local tiles with the tiny smoke checkpoint standing in for the frozen driver. These show the loop runs, resumes and labels correctly. They are not results. (MacBook trial numbers are in the lead's final report.)
- **Mini (after the main run has frozen and zipped):** one command, 140 minutes plus a 10-minute hard stop = 150 at most. Two situations, about 68 minutes each. A round is estimated at 6 to 10 minutes on the mini (not measured), so about 7 to 10 rounds per situation. If the main run is still alive at 05:30, do not start (it would not finish by 08:00).
- **Not achievable:**
  - The original Dream-RSI with an LLM rewriting the strategy code (needs a paid key).
  - Evidence that the strategy layer helps: about 15 to 20 saved worlds with noisy scores from 64 episodes is too little; the strategy choice can chase noise. Tonight it demonstrates the mechanism only.
  - Open-ended recursive self-improvement. The rover does not change its own learning method or goals; it picks among settings a person wrote, and it only learns a correction on top of the frozen driver.
  - Anything about a real rover: same kinematic, hourly-step simulator. "Real episodes" means simulator episodes, as opposed to imagined ones.
  - A promise that self-learning helps. A 0.7 M-parameter world model built from a few thousand steps is often wrong. "NOT MET" is a likely and reportable outcome.
- **What this is:** bounded, continued online self-learning with a learned world model, a real-episode acceptance test, and a Dream-RSI-style replay loop that tunes how each round's tries are spent.

## f. Recommendation: GO, reduced scope

- Build and test on the MacBook tonight (done; see the final report).
- Run on the mini only after the main run is frozen and zipped, from a separate folder, `bash ops/dream/run_dream_mini.sh 140`, gated by the team lead.
- Present it as an experimental addendum with its own tables and curves and the words "key-free Dream-RSI-style loop, not the original". Show the verdict line as printed, MET or NOT MET.
- If the mini run does not happen, present the design and the MacBook smoke as "built, not yet measured".

## g. Fallback

`--no-rsi` keeps the default strategy (open all three recipes, refine each in turn) for every round. That is the plain world-model self-learning loop and stands on its own.

## h. Running it on the mini (NOT done; the team lead gates this)

Only after `run_everything.sh` has finished, frozen and zipped.

**Operator checklist before launch (on the mini, in `~/adinfinitium-run/adinfinitium`; all must hold):**
- `tail -n 1 runs/run_everything.log` ends with "ALL DONE" (the line starts `== <date> ALL DONE.`). Anything else: the main run is not finished, do not launch.
- `runs/FROZEN.json` is from THIS run: its `frozen_at` (UTC) is after this run's "freeze" line in `runs/run_everything.log`, and `.venv/bin/python -m ops.freeze --check` prints "ALL UNCHANGED". The guard also refuses when the driver hash in FROZEN.json (`sha256["checkpoint/rover_driver_mini"]`) does not match the driver file, which is what a FROZEN.json left over from an older run looks like.
- There is no `runs/VERIFY_FAILED.txt`. If there is one, the main run was NOT frozen: the dream run still starts, and its report says so at the top ("main run was NOT frozen (verify failed); starting driver is the trained checkpoint, not a frozen one") and in `report.json` (`main_run`). Launch in that state only if the team lead accepts that label.
- The guard also refuses while `run_everything.sh`, `ops.overnight.run_all` or a leftover stage process (`python -m worldmodel.<family>.*`, for example `worldmodel.rover.train`) is alive. Check with `ps -axo pid=,command= | rg 'worldmodel\.|run_everything|run_all'` and let it finish or stop it first.

Four copy steps, then one command:
1. `ssh mini "rsync -a --exclude data --exclude runs --exclude .venv --exclude .git --exclude '*.zip' ~/adinfinitium-run/adinfinitium/ ~/adinfinitium-dream/"` (copies the main run's CODE into a new folder; reads the main folder, writes nothing there, so the simulator files match exactly).
2. `rsync -a ~/Documents/Ojas/Hackathon/work/adinfinitium-dream/worldmodel/dream mini:~/adinfinitium-dream/worldmodel/`
3. `rsync -a ~/Documents/Ojas/Hackathon/work/adinfinitium-dream/ops/dream mini:~/adinfinitium-dream/ops/`
4. `git -C ~/Documents/Ojas/Hackathon/work/adinfinitium-dream rev-parse HEAD | ssh mini 'cat > ~/adinfinitium-dream/DREAM_COMMIT.txt'` (records which dream commit was shipped; the copy on the mini has no `.git`, so the report reads the commit from this file and prints it as "dream code commit ... (from DREAM_COMMIT.txt)". Run it AFTER steps 2 and 3, from a clean working tree: `git status --short` prints nothing.)
5. On the mini: `cd ~/adinfinitium-dream && nohup bash ops/dream/run_dream_mini.sh 140 >/dev/null 2>&1 &`

- Budget 140 minutes, hard stop at 150. Run the same command again to resume. `report.md` starts with a status line: "RUNNING, or CUT SHORT" with the last completed round until the run ends by itself, then "finished". A report still saying RUNNING after the process is gone was cut short at that round.
- `report.md` and `report.json` also state the hazard map the world really used (the U-Net, or the slope-rule fallback when `hazard_unet_big.pt` is missing) and that acceptance is best-of-N on the same test starts (optimistic; the score starts are the honest number).
- It reads `~/adinfinitium-run/adinfinitium/data` (tiles, `hazard_unet_big`, `rover_driver_mini`) and uses that folder's `.venv`.
- It writes only under `~/adinfinitium-dream/runs/_dream/`: `selflearn1.log`, `selflearn1/report.md` (tables and verdicts), `report.json`, `curve_*.png`, `ledger.jsonl`, `trace_pool/`, `models/rover_dream_*.pt`, and `_cache/`.
- If it prints REFUSED, read the reason; do not work around it.
