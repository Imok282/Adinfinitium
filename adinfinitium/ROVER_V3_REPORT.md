# Rover v3 report: planner -> executor -> self-learning

Sun 15 Nov 2026. After-freeze experiment; nothing in the frozen main run was changed. Rover v1 (PPO) and rover v2 (imitation) failed and stay reported as failed. The plan (`ROVER_V3_PLAN.md`, two amendments) was committed before any v3 code ran; the final evaluation was launched once.

## Verdict against the pre-registered criteria

| criterion | result |
|---|---|
| PRIMARY: fresh held-out sites, unchanged worlds: v3 returned > best planner, interval above 0, v3 died <= lowest planner died | **PASS** |
| S1: the same on Site23 + Site42 (previously seen held-out; point values, 2 sites, no interval) | MET |
| S2: changed world before any learning, v3 >= best planner that is also not told | MET (blocked route), MET (dusty panels) |
| S2b: changed world before any learning, v3 >= best planner that IS told about the change | MET on the 6-site mean for both; on the blocked route v3 is below the told planner at 2 of 6 sites |
| S3: self-learning in the changed world (blocked route): more than half of 6 situations IMPROVED, none WORSE | **MET**: 6 of 6 IMPROVED, 0 WORSE (one accepted change each, in round 1) |
| S4: self-learning in the unchanged world does no harm (none WORSE) | MET: 0 of 24 rounds accepted, nothing changed |

**Read this before quoting the PASS.** In an unchanged world the v3 planner tries 36 plans per rover inside an exact copy of the simulator (same map, same light forecast, same physics) and picks the best; three of the 36 are the baseline planners themselves. So v3 could not lose to them there if the code is right. PASS means: planning the whole trip with a perfect forecast beats three short-sighted rules. It is not evidence of learning, and not evidence of robustness; the changed-world rows (S2, S3) are the only test of a wrong model.

## Architecture

```
INPUTS    slope map, hazard map (hazard U-Net output), light forecast per cell per hour, PSR-edge target, lander position,
          rover state (position, battery, hour, phase)
   |
PLANNER   scripted physics, recomputed on every world. Route = Dijkstra cost-to-go over the rover's BELIEF map.
          Schedule = for each rover, roll 36 candidate plans to the end of the episode inside the rover's own model and pick one:
          wait 0-240 h at the lander then go (16 waits) x charge on the way or not; the 3 baseline planners; safe harbour.
          Never follows a plan its model predicts to die.
   |
EXECUTOR  scripted controller. Each hour: follow the plan's route field; refuse a step onto ground sensed as too steep (true slope
          one move ahead); go round an unexpected obstacle (wall following); stop to charge; every 6 h compare real battery and
          position with the plan and ask the planner again if they differ.
   |
DREAM-RSI self-learning after deployment: records its own episodes -> corrects the belief (ground it has seen, solar output it has
          measured) and proposes parameter changes -> practises each proposal in imagination (episodes inside the corrected model)
          -> the best one must beat the current brain on 128 paired REAL episodes (gain > 2 SE, no fewer returned, no more lost)
          -> kept or discarded. No training data, no human, no LLM, no key.
```

- **Scripted:** planner and executor. **Learned network:** none. A learned executor was not built: the planner gives a route field on any terrain and following it leaves nothing to learn in this deterministic simulator (rover v2's learned follower agreed with the planner on only 15-26% of steps on unseen terrain).
- **Self-learns (gated):** belief map, belief solar output, energy margin, charge thresholds.
- The original google-research/dream-rsi needs paid LLM keys; none was used. This is the key-free loop (record, imagine, gate, keep-best); its strategy-tree meta-search from the earlier addendum is not used here.

## Why rovers die under the planners (measured, not assumed)

In this simulator the only way to die is an empty battery more than 20 m from the lander (`worldmodel/rover/env.py:215`); hazards cost points but never kill. A rover drives 300 m (8.3 h) in the dark on a full battery; targets are 200-1600 m away. Under the baseline planners on the fresh sites (21,276 deaths in stage A):

| planner | deaths | before reaching the target | in full dark at death | trips none of v3's 36 plans could complete either |
|---|---|---|---|---|
| naive | 8,523 | 86.7% | 97.4% | 58.2% |
| cautious | 4,771 | 95.2% | 85.9% | 46.8% |
| cautious + wait for light | 7,982 | 97.3% | 96.8% | 56.3% |

Every death is in too little light to drive (true by the simulator's definition of death). So: rovers die of energy, mostly on the way out, because the planners set off (or wait where they landed, away from the lander) without checking that the whole trip is survivable. About half of the dead rovers were on trips that none of v3's plans could have completed in that window; the other half died on trips that were possible with better timing. Not hazards, not unreachable targets (all 24 fresh worlds have 16 PSR-edge targets), not the time limit.

## Results: unchanged worlds (v3 as deployed, no self-learning)

All methods in one run, same worlds, same starts, 512 rovers per world, seed 1234. "returned" = reached the target and came back within 20 m of the lander (the simulator's own flag). Brackets: 95% interval from resampling the 6 sites (approximate with 6 sites).

### A. Six fresh held-out sites (PRIMARY): DM1, LM7, NPA, NPB, NPC, NPD x 4 dates = 24 worlds, 12,288 episodes per method

| method | reached target | returned (primary) | died | mean return |
|---|---|---|---|---|
| **v3** | 0.627 [0.497, 0.743] | **0.596** [0.464, 0.714] | **0.000** (0 of 12,288) | +127.7 |
| planner: naive | 0.396 [0.281, 0.497] | 0.299 [0.194, 0.381] | 0.694 [0.610, 0.798] | +54.6 |
| planner: cautious (best by returned) | 0.367 [0.211, 0.508] | 0.329 [0.184, 0.462] | 0.388 [0.261, 0.507] | +59.4 |
| planner: cautious + wait for light | 0.289 [0.186, 0.381] | 0.264 [0.166, 0.348] | 0.650 [0.583, 0.726] | +29.6 |
| reference: safe harbour only (drive to the lander, never leave) | 0.000 | 0.000 | 0.000 | -2.7 |

- Paired difference in returned, v3 minus cautious: **+0.267 [0.191, 0.337]**; v3 higher at 6 of 6 sites (DM1 +0.347, LM7 +0.393, NPA +0.288, NPB +0.230, NPC +0.103, NPD +0.245).
- v3 kept **37.3%** of rovers at the lander (no survivable plan in the window, so no attempt). A rover a baseline returned that v3 did not: 0 of 12,288. Prediction = outcome for 100% of v3 rovers in every world.
- The 0% lost comes from one simulator rule: an empty battery within 20 m of the lander is not a death. The "safe harbour only" row (no planning at all) also loses 0 on these sites. It is not general: on validation Site04, where the ground around the lander is too steep to reach it, v3 and every planner lose the same 55-80%.
- Of v3's 7,327 returned rovers, 3,399 left at once and 3,928 waited at the lander first (waiting for light is where most of the gain is).
- v3 reached the target less often than the naive planner in 6 of 24 worlds: naive reaches targets with rovers that then die.

### B. Site23 + Site42 (previously seen held-out, weaker evidence): 8 worlds, 4,096 episodes per method

| method | reached target | returned | died | mean return |
|---|---|---|---|---|
| **v3** | 0.414 | **0.375** | **0.000** | +85.7 |
| planner: naive (best by returned) | 0.196 | 0.093 | 0.868 | +4.7 |
| planner: cautious | 0.104 | 0.088 | 0.543 | -18.5 |
| planner: cautious + wait for light | 0.069 | 0.060 | 0.738 | -19.0 |
| reference: safe harbour only | 0.000 | 0.000 | 0.000 | -4.5 |

- v3 minus naive in returned: +0.282 (Site23 +0.235, Site42 +0.330). v3 kept 58.6% of rovers at the lander. For comparison on the same worlds: rover v1 returned 0.000, rover v2 0.012.
- These two sites had been seen twice before v3 was designed, so this is weaker evidence than A.
- Baseline rows against the earlier evaluation of the same worlds: naive identical in all 8 worlds (0.196 / 0.093 / 0.868); cautious and wait-for-light within 2 rovers of 512 per world (11 of 24 cells differ). Cause not established (this run was on CPU; a float difference is plausible, not verified).

Per-world numbers for A and B are in `eval.json` (`per_world`).

## Results: changed worlds before any learning (the rover is not told)

Fresh sites, date 2027-06-01, 256 rovers per world, 1,536 episodes per method. "stale" = the planner decides from the original map / physics, like v3; "told" = the planner is given the changed world (reference only).

| change | method | reached target | returned | died |
|---|---|---|---|---|
| blocked route (synthetic ring 90-135 m round the lander, 3 gaps) | **v3** | 0.420 | **0.299** | **0.074** |
| | best stale planner (cautious) | 0.093 | 0.066 | 0.255 |
| | best told planner (cautious) | 0.270 | 0.237 | 0.426 |
| dusty panels (solar output x 0.6, invisible to the rover) | **v3** | 0.475 | **0.413** | **0.139** |
| | best stale = told planner (naive) | 0.316 | 0.202 | 0.797 |

- With a wrong model v3 **does lose rovers**: 7.4% (blocked route) and 13.9% (dusty panels), against 0% in unchanged worlds.
- The stale planners have no obstacle avoidance, so S2 mostly shows that the executor gets round the ring. Against the told planner on the blocked route v3 is below at DM1 (0.203 vs 0.227) and NPC (0.277 vs 0.344), equal at NPA (0 vs 0), above at LM7, NPB, NPD.

## Results: self-learning (Dream-RSI layer), online on the deployed site

Fresh sites, date 2027-06-01. Each situation: up to 4 rounds; gate = 128 paired real episodes; before / after measured on 256 separate score starts. These are online-adaptation scores on the site where the rover drives, not a held-out generalisation score. 40-minute cap, 6 situations at a time, fixed order.

| site | world | rounds accepted (of 4) | accepted change | returned before -> after | died before -> after | mean return before -> after | gain (SE) | label |
|---|---|---|---|---|---|---|---|---|
| DM1 | blocked route | 1 | map: added the 850 obstacle cells it had seen | 0.254 -> 0.668 | 0.086 -> 0.000 | +84.5 -> +187.1 | +102.5 (8.3) | IMPROVED |
| LM7 | blocked route | 1 | map: added the 850 obstacle cells it had seen | 0.254 -> 0.457 | 0.090 -> 0.000 | +55.6 -> +89.3 | +33.7 (4.1) | IMPROVED |
| NPA | blocked route | 1 | map: added the 850 obstacle cells it had seen | 0.000 -> 0.008 | 0.008 -> 0.000 | -2.8 -> +4.8 | +7.6 (2.0) | IMPROVED |
| NPB | blocked route | 1 | map: added the 850 obstacle cells it had seen | 0.590 -> 0.703 | 0.047 -> 0.000 | +131.2 -> +153.6 | +22.4 (4.6) | IMPROVED |
| NPC | blocked route | 1 | map: added the 850 obstacle cells it had seen | 0.254 -> 0.457 | 0.109 -> 0.000 | +44.6 -> +82.7 | +38.0 (4.8) | IMPROVED |
| NPD | blocked route | 1 | map: added the 850 obstacle cells it had seen | 0.398 -> 0.711 | 0.070 -> 0.000 | +96.1 -> +147.9 | +51.8 (5.4) | IMPROVED |
| DM1 | unchanged | 0 | nothing accepted | 0.664 -> 0.664 | 0.000 -> 0.000 | +173.6 -> +173.6 | +0.0 (0.0) | NO CHANGE |
| LM7 | unchanged | 0 | nothing accepted | 0.496 -> 0.496 | 0.000 -> 0.000 | +91.9 -> +91.9 | +0.0 (0.0) | NO CHANGE |
| NPA | unchanged | 0 | nothing accepted | 0.008 -> 0.008 | 0.000 -> 0.000 | +4.9 -> +4.9 | +0.0 (0.0) | NO CHANGE |
| NPB | unchanged | 0 | nothing accepted | 0.680 -> 0.680 | 0.000 -> 0.000 | +156.4 -> +156.4 | +0.0 (0.0) | NO CHANGE |
| NPC | unchanged | 0 | nothing accepted | 0.434 -> 0.434 | 0.000 -> 0.000 | +80.1 -> +80.1 | +0.0 (0.0) | NO CHANGE |
| NPD | unchanged | 0 | nothing accepted | 0.684 -> 0.684 | 0.000 -> 0.000 | +144.4 -> +144.4 | +0.0 (0.0) | NO CHANGE |
| DM1 | dusty panels | 1 | solar output x 0.600 (measured) | 0.598 -> 0.613 | 0.137 -> 0.000 | +163.6 -> +168.1 | +4.5 (6.3) | NO CHANGE |
| LM7 | dusty panels | 1 | energy margin 0 -> 0.10 | 0.375 -> 0.426 | 0.172 -> 0.051 | +76.6 -> +91.5 | +14.9 (3.9) | IMPROVED |
| NPA | dusty panels | 0 | nothing accepted | 0.000 -> 0.000 | 0.059 -> 0.059 | +2.4 -> +2.4 | +0.0 (0.0) | NO CHANGE |
| NPB | dusty panels | 0 | nothing accepted | 0.551 -> 0.551 | 0.121 -> 0.121 | +132.0 -> +132.0 | +0.0 (0.0) | NO CHANGE |
| NPC | dusty panels | 0 | nothing accepted | 0.449 -> 0.449 | 0.031 -> 0.031 | +83.7 -> +83.7 | +0.0 (0.0) | NO CHANGE |
| NPD | dusty panels | 1 | energy margin 0 -> 0.10 | 0.559 -> 0.688 | 0.230 -> 0.039 | +109.7 -> +149.3 | +39.6 (6.2) | IMPROVED |

All 18 situations ran to the end (31 of the 40 minutes). Of 72 rounds, 57 sent a proposal to the real-episode gate; 9 were accepted, 48 rejected. No accepted change broke the gate rule (checked independently).

- **Blocked route (S3: MET, 6 of 6 IMPROVED, 0 WORSE).** At every site the first round accepted one change: the rover added the obstacle it had seen to its map. After that, deaths on the score starts fell to 0 at all six sites and returned rose (for example DM1 0.254 -> 0.668, NPD 0.398 -> 0.711). Limits: the obstacle is a synthetic ring that lies wholly inside the rover's sensed window round the lander, so one recorded batch saw all of it (850 cells at every site); at NPA the effect is 2 rovers of 256 (almost no trip is survivable there in June).
- **Unchanged world (S4: MET).** Nothing was accepted in 24 rounds (16 proposals reached the gate, all rejected), so the brain is unchanged and before = after by construction. The gate did no harm; self-learning also found nothing to improve, as expected when the model is already exact.
- **Dusty panels (not part of S3): 2 IMPROVED, 4 NO CHANGE, 0 WORSE.** LM7 and NPD accepted a larger energy margin (0 -> 0.10): deaths 0.172 -> 0.051 and 0.230 -> 0.039, returned 0.375 -> 0.426 and 0.559 -> 0.688; they still lose 5.1% and 3.9%. DM1 accepted the correct solar measurement (x 0.600): deaths 0.137 -> 0.000, but the mean-return gain on the score starts is inside the noise, so its label is NO CHANGE. At NPA, NPB and NPC nothing passed the gate and rovers keep dying (5.9%, 12.1%, 3.1%). The gate was strict here: 15 rounds rejected a change although deaths fell in the gate episodes (the mean-return gain was under 2 SE).
- **How recursive is it?** Every accepted change came in round 1; rounds 2-4 accepted 0 of 54. The loop keeps running and keeps testing proposals, but in these tests it found one useful change per situation and then nothing more.
- **Cost.** Self-learning practises on the real site: up to 1,152 real rover episodes per situation (recording and gate batches, besides the score runs), and rovers were lost in them (DM1 blocked route, first gate batch: 14 of 128 under the old brain).
- What entered the belief from the real world: the true slope inside the 320 m windows round cells the recorded rovers visited (recording and gate batches, including the candidate's), and the solar factor fitted from battery telemetry. The 256 score starts were never used to choose anything.

## What went wrong / what to know

- Amendment 2 changed the self-learning schedule (stage D after C, 6 at a time instead of 3) for compute time, after validation and before any held-out number.
- The check of the harness against earlier Site11 planner numbers was not an exact reproduction (cautious 0.312 = 0.312, wait-light 0.240 vs 0.234, naive 0.364 vs 0.407): the earlier Site11 numbers came from environments reused across candidates. Stage B is the exact-protocol check (see above).
- Development smoke on Site01 and validation on six sites ran before the final launch (allowed by the plan; numbers in the lead log). The validation check passed: prediction = outcome 100%, v3 never below a planner in any of 24 worlds.
- Nothing crashed in the final evaluation and nothing was rerun. Two tests written with wrong expectations were corrected before any evaluation (lead log). A development smoke run of the blocked-route loop was stopped after its first round to free the machine.
- Dusty panels: self-learning helped at only 2 of 6 sites by the pre-registered label; at 3 sites it changed nothing while rovers kept dying.
- No second validation of the self-learning loop exists beyond one development site (Site01) smoke run.

## Caveats

- Simulation only: kinematic, one decision per hour. A "target" is a PSR edge (dark and cold), not confirmed ice.
- Unchanged-world numbers assume a perfect map and a perfect light forecast. No test of a wrong light forecast was run. Stages C and D cover a wrong map and wrong solar output only.
- The fresh sites are flatter than Site23/Site42 (median slope 2.5-5.9 deg against 9.5-11.6 deg); the two test sets are not equally hard.
- v3's zero losses rest on the lander-immunity rule of this simulator.
- The executor senses the true slope one move ahead (read from the simulator; the rover's map crop shows the same values).
- Six sites: the intervals are approximate. Ten fresh sites had clean footprints; a recon agent fetched six before any terrain statistic existed; why the other four were left out was not recorded.
- The blocked route is a synthetic obstacle. In self-learning, what a batch of rovers has seen is merged into one belief map.
- Nothing in v3 is a trained network. If the brief needs a learned driver, v3 does not provide one.

## Claims for the presentation

Safe to say:
- "Our rover brain has three layers: a physics planner, a scripted executor and a gated self-learning layer. In simulation, on six lunar south-pole sites it had never been run on, it brought back 59.6% of rovers from a PSR-edge target against 32.9% for the best of our three earlier planners, and lost none; it kept 37% of rovers at the lander because no survivable trip existed in their window."
- "The gain comes from planning the whole trip against the light forecast, mostly from waiting at the lander for light."
- "Rovers under the old planners died of energy: they set out without checking the whole trip; about half of those trips could not have been completed by any of our plans."
- "When the world differs from the map (blocked route, dusty panels) v3 still beat planners that were also not told, but it lost 7-14% of rovers."
- "After deployment the rover keeps testing changes to its own map and margins with no training data, no human in the loop and no LLM: it records its own drives, practises in imagination and keeps a change only if it wins on real paired episodes. Faced with a synthetic blocked route it had not been told about, it added the obstacle to its map at 6 of 6 sites and deaths fell from 1-11% to 0. In unchanged worlds it changed nothing."
- "With dusty panels it helped at 2 of 6 sites; the gate is conservative."

Not safe to say:
- "v3 learned to drive" or "the network generalises to unseen terrain": nothing is learned; the planner is recomputed from each site's own map.
- "v3 never loses a rover" or "0% deaths" without the lander rule and the perfect-forecast condition next to it.
- "v3 is never worse than the planner" without limits: true for returned and died in all 56 unchanged worlds tested; false for reached-target (below naive in 6 of 24 fresh worlds) and against a planner that is told about a change (2 of 6 sites).
- "v3 finds ice": targets are PSR edges, not confirmed ice.
- "Works on the real Moon" or any claim about wheel dynamics, thermal limits or a wrong light forecast.
- "Recursive self-improvement" as an open-ended claim: each situation got one accepted change, in the first round, and none in the 54 rounds after. The list of possible changes and the gate are hand-written.
- "Self-learning is free or safe": it spends real episodes and loses rovers while it learns.
- "This is Google's DREAM-RSI": it is a small key-free loop in that spirit, with no LLM.

## Files

MacBook, under `~/Documents/Ojas/Hackathon/`:
- `records/rover_v3/ROVER_V3_PLAN.md` (22 KB; pre-registered plan + 2 amendments), `ROVER_V3_REPORT.md` (this file), `eval.json` (170 KB; every number above, per world and per self-learning round), `tables.md` (7 KB; the script's own tables), `STATUS.md`
- `records/agents/rover-v3-lead.md` (attempt log), `rover3-recon.md`, `rover3-plan-reviewer.md`, `rover3-numbers-checker.md` (parts 1 and 2)
- code: `work/adinfinitium-rover3` (git branch `rover-v3`, from rover-v2 968a4e2; not pushed): `worldmodel/rover3/brain.py` (planner + executor), `run.py` (stages, self-learning loop, gate), `report.py`, `tests/test_rover3.py` (4 tests), `ops/rover3/run_rover3_mini.sh`, `ops/rover3/package_rover3.sh`

Mac mini:
- `~/adinfinitium_addendum3_rover_v3.zip` (about 0.2 MB, about 120 files): plan, report, `eval.json`, `tables.md`, `rover_brain_v3_config.json` (the deployed brain: there are no network weights), `final/` (every raw stage file and log), `validation/`, `development_smoke/`, `code/` (the rover3 package, launcher, tests, `rover_v3.diff` against rover-v2)
- working folder `~/adinfinitium-rover3/` (173 MB): `code/`, `data/` (6 fresh tiles + overlay of links to existing tiles and caches), `runs/final/`, `runs/val/`, `runs/dev/`
- Freeze check `python -m ops.freeze --check` in `~/adinfinitium-run/adinfinitium/`: ALL UNCHANGED at the start (09:55 IST) and at the end (11:18 IST). The final evaluation ran on code commit 8392e49; the report script was relabelled afterwards (718a56f, labels only).

How to rerun one world (mini): `cd ~/adinfinitium-rover3/code && export ADI_DATA=$HOME/adinfinitium-rover3/data PYTHONPATH=$PWD && ~/adinfinitium-run/adinfinitium/.venv/bin/python -m worldmodel.rover3.run eval --sites DM1 --dates 2027-06-01 --n 512 --seed 1234 --out /tmp/x.json` (about 15 s on CPU).
