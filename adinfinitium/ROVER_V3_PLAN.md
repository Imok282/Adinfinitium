# Rover v3 plan (pre-registered)

Written Sun 15 Nov 2026, before any rover v3 code was run and before any v3 number exists. Rover v1 (PPO) and rover v2 (DAgger imitation) failed and stay reported as failed. v3 is a separate, after-freeze experiment; it changes nothing in the frozen main run.

User request (verbatim): "We can add a planner and executor layer of brain...with inputs ->planner -> executor -> dream rsi" and "after the model is complete, and the training data is cut off, it should still recursively self-learn."

## 0. What was already known when this plan was written (contamination statement)

- Site23 / Site42 held-out numbers for v1, v2 and the three physics planners have been seen twice. This design is informed by them. They are therefore NOT the primary test here.
- From the code (`worldmodel/rover/env.py:215`): the only way a rover dies is battery <= 0 while farther than 20 m from the lander. Hazard never kills (reward penalty only). Battery 100 Wh, driving 12 W, idle 2 W, panel 50 W x sun fraction: in the dark a rover drives 8.3 h (300 m) or idles 50 h. Targets are 200-1600 m away and many landers are dark for most of the month (rover-v2 diagnosis). So planner rovers die of energy: they set out, or sit away from the lander, without enough light. The baseline planners look at light now, +1 h and +3 h only; none checks whether the whole trip can be survived.
- No v3 number of any kind exists. The six fresh sites (section 3) were fetched and tiled by a recon agent that computed terrain statistics only (PSR share, slope share, relief); no rover episode has run on them.

## 1. Architecture

```
INPUTS  terrain slope, hazard map, light forecast (sun fraction per cell per hour), ice (PSR-edge) target, lander position,
        rover state (position, battery, hour, mission phase)
   |
PLANNER (scripted physics, no neural network; recomputed on every world, so it works on unseen terrain)
   route:    Dijkstra cost-to-go fields over the rover's BELIEF map (same cost as the existing planner)
   schedule: for each rover, simulate a fixed list of candidate plans forward to the end of the episode inside the rover's
             own model (belief map + belief physics) and pick one: go now / wait w hours at the lander then go / charge on
             the way or not / behave as one of the three baseline planners / SAFE HARBOUR (go to the lander and stay)
   output:   one plan per rover (parameters + the predicted battery and position at every hour)
   |
EXECUTOR (scripted controller, no neural network)
   each hour: follow the plan's cost-to-go field to the next cell; never step onto ground it senses as too steep
   (true slope at the 8 cells one move ahead); go round an unexpected obstacle (wall following); stop to charge when
   the plan says so; every 6 hours compare real battery and position with the plan's prediction and ask the planner
   for a new plan when they differ
   |
DREAM-RSI (self-learning after deployment; no training data, no human, no LLM, no key)
   records its own episodes -> corrects the belief (obstacles it has seen; solar output it has measured) and proposes
   changes to planner parameters -> practises every proposal in imagination (episodes inside the corrected model)
   -> the best proposal must pass a REAL-episode gate against the current brain on the same starts -> keep or discard
```

### What is learned and what is scripted

- Planner: scripted. Executor: scripted. No neural network is trained in v3.
- Why no learned executor: the simulator is deterministic and kinematic, and the planner supplies a cost-to-go field on any terrain; following it leaves nothing to learn. Rover v2 showed a learned follower agrees with the planner on only 15-26% of steps on unseen terrain. A network would add error, not skill. If the team wants a learned component, this plan does not provide one, and the report will say so.
- What self-learns (Dream-RSI layer only): the belief map (obstacles), the belief solar output, and planner parameters (energy margin, charge thresholds). Every change must pass the real-episode gate.
- Maps the planner uses are the project's existing model outputs (hazard U-Net map, modelled light). The planner's model of the world is the same physics the simulator uses. In an UNCHANGED world this model is exact; see caveats (section 8). The CHANGED-world tests exist to measure what happens when it is wrong.

### Fixed parameters (set here, a priori; none is tuned on any site)

- Candidate plans per rover (36): wait w in {0, 4, 8, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168, 192, 216, 240} hours x charge-on-the-way {off, on: stop when battery < 35% and the panel gives more than idle + 1 W, until 95%} = 32; the three baseline planners (naive, cautious reserve 0.1, cautious + wait-for-light 1.5) exactly as coded in `RoverEnv.planner_action`, driven from the belief; safe harbour.
- Choice per rover: discard any candidate the model predicts to die or whose predicted battery away from the lander drops below margin m x capacity (safe harbour is never discarded). Among the rest take, in order: predicted returned; predicted reached ice; higher predicted episode return; the currently active plan; the lower candidate index.
- Margin m = 0.0 at deployment (trust the model; the gate may raise it).
- Executor: check every 6 steps; replan a rover when real battery differs from predicted by more than 3 Wh or position by more than 20 m. Wall-following gives up after 40 steps.
- Dream-RSI: 4 rounds per situation; gate = 128 paired real episodes (same starts for current and candidate brain, a new seed per round); imagination = 128 episodes inside the corrected belief; proposals per round, in this fixed order: (1) belief map <- ground seen by the rover in recorded episodes, (2) belief solar output x measured factor (only if it differs from 1 by more than 5%), (3) (1)+(2) together, (4) margin m - 0.05, m + 0.05, m + 0.10 (kept inside 0..0.30), (5) charge thresholds (20%, 80%) and (50%, 100%). A proposal reaches the gate only if its imagined mean return beats the current brain's imagined mean return; only the best imagined proposal of a round goes to the gate. ACCEPT only if, on the gate episodes: paired mean-return gain > 2 standard errors AND returned count not lower AND died count not higher. Otherwise the current brain is kept unchanged.

### Never worse than the plain planner

- By construction (unchanged world, model exact): the three baseline planners are among the candidates and the model predicts each exactly, so for every rover v3's choice returns whenever any baseline returns, and v3 never follows a plan predicted to die. Checked per world on validation before the test (section 4).
- By gate (after deployment): a self-learning change is kept only if it passes the real-episode gate above; the starting brain is the fallback.
- Not guaranteed, and reported as measured: v3 against the baselines in a CHANGED world before any learning.

## 2. Code and machines

- Branch `rover-v3` (from rover-v2 968a4e2), new package `worldmodel/rover3/`, launcher `ops/rover3/run_rover3_mini.sh`. No file under `contracts/`, `worldmodel/rover/`, `worldmodel/light/`, `worldmodel/hazard/`, `config/mission.yaml` is edited: the simulator is byte-identical to the frozen run (checked by hash against the frozen folder before the run).
- Runs on the Mac mini in `~/adinfinitium-rover3/` with `PYTHONPATH` set to that folder (module path printed and checked). The frozen folder `~/adinfinitium-run/adinfinitium/` is read-only; `python -m ops.freeze --check` must print ALL UNCHANGED at the start and at the end.

## 3. Sites (exact splits)

- DEVELOPMENT (debugging while the code is written; numbers go to the log, decide nothing): Site01, Site06, Haworth.
- VALIDATION (go / no-go check, section 4): Site11 plus five carved from the training pool: Site07, Site20, Shoemaker, DM2, Site04. Dates 2027-03-01, 06-01, 09-01, 12-01; 256 rovers per world; seed 4321.
- FRESH HELD-OUT (PRIMARY TEST): DM1, LM7, NPA, NPB, NPC, NPD: NASA PGDA LOLA 5 m south-pole site products that are not in the project's catalogue, never used by the hazard, light or terrain models or by any rover experiment; footprints farther than 5 km from every used site. Centre window of each (same convention as every existing rover world); all six that were fetched are used, none is dropped. Tiles: `mini:~/adinfinitium-rover3/data/tiles/lola5m_{dm1,lm7,npa,npb,npc,npd}.npz`. Dates 2027-03-01, 06-01, 09-01, 12-01 = 24 worlds; 512 rovers per world; seed 1234.
- PREVIOUSLY SEEN HELD-OUT (reported once, labelled as such, weaker evidence): Site23, Site42, the same 8 worlds, 512 rovers and seed 1234 as the earlier evaluations (the baseline rows must reproduce the earlier numbers: naive 0.196 / 0.093 / 0.868).
- Held-out sites (fresh or seen) are never used to write, debug, select or tune anything.
- Known before the test (terrain statistics only): the fresh sites are flatter than Site23/Site42 (median slope 2.5-5.9 deg against 9.5-11.6 deg), so the two test sets are not equally hard. Worlds with no reachable PSR edge get plain science targets (existing behaviour); they are counted and listed.

## 4. Selection procedure

There is nothing to select: no network, no checkpoints, and every parameter is fixed in section 1. Validation is a go / no-go check of the construction, run once the code passes its unit tests:

- G1 no crash on the 24 validation worlds;
- G2 in every validation world, predicted outcome = real outcome for at least 99% of v3 rovers (the model is exact there);
- G3 in every validation world, v3 returned >= every baseline's returned and v3 died <= every baseline's died.

If G1-G3 hold: go. If one fails it is a defect against this plan: fix the code (not the parameters), log it, rerun validation. If it cannot be fixed inside the time budget: the final evaluation still runs once and the report states that the construction check failed. Validation numbers are reported as validation, never as a score. A changed-world smoke test (blocked_route and dusty_panels on development site Site01 only) is run for crashes and timing.

## 5. The single final evaluation (one launch, no reruns for results)

All methods in one run, same worlds, same starts (same environment seed), same episode count.

- A. PRIMARY: fresh held-out, unchanged worlds (24 worlds x 512): v3 (as deployed, no self-learning), naive, cautious, cautious_wait_light.
- B. Previously seen held-out, unchanged (8 worlds x 512): same four methods.
- C. Changed worlds before any learning: fresh sites, date 2027-06-01, changes `blocked_route` and `dusty_panels` (from `worldmodel/dream/changes.py`, unchanged), 256 rovers, seed 1234. The rover's belief is the ORIGINAL world (it is not told). Methods: v3; the three baselines driven from the same stale belief (fair comparison); the three baselines told about the change (reference only).
- D. Self-learning (Dream-RSI), online on the deployed site, date 2027-06-01, on fresh sites: situations in this fixed order: DM1 blocked_route, DM1 none, LM7 blocked_route, LM7 none, NPA blocked_route, NPA none, NPB blocked_route, NPB none, NPC blocked_route, NPC none, NPD blocked_route, NPD none, then dusty_panels on DM1, LM7, NPA, NPB, NPC, NPD. The stage has a wall-clock cap (40 minutes); situations run in this order (up to 3 at a time) and none starts after the cap; every situation that started is reported, finished or not. Before / after are measured on 256 separate score starts (seed different from every gate seed) with the starting brain and the final brain.
- Reported for every method: reached ice %, returned % (primary), died %, mean return; per world and averaged over worlds; episode counts; 95% intervals: world-cluster bootstrap (10,000 resamples of worlds, fixed seed) and the paired v3-minus-baseline difference with the same bootstrap. Also: why rovers died (phase at death, light at the place of death, whether any candidate plan could have returned), and how many rovers v3 kept at the lander.
- Time rule (decided from timing on development sites, before any test number exists): if A + B would take more than 30 minutes, the rover count is halved for all methods alike; recorded in the log. If stage D cannot finish all situations the cap rule above applies. No other change after the launch.

## 6. Success criteria (decided now)

Primary (stage A, fresh held-out, unchanged). "Best baseline" = the baseline with the highest returned averaged over the 24 fresh worlds in this run.

- PASS: v3 returned > best baseline returned with the 95% bootstrap interval of the paired difference above 0, AND v3 died <= the lowest baseline died.
- PARTIAL: v3 returned >= best baseline returned but the interval includes 0, and v3 died is lower than every baseline's.
- FAIL: anything else.

There is no absolute threshold; the report states the absolute returned rate plainly, and says so if it is low.

Secondary, each reported MET / NOT MET, none can turn a primary FAIL into a PASS:

- S1 (stage B): the same PASS rule on Site23 + Site42, labelled previously seen.
- S2 (stage C): per change, v3 returned >= best stale-belief baseline returned and v3 died <= its died.
- S3 (stage D, changed world): among finished `blocked_route` situations, more than half are IMPROVED and none is WORSE. IMPROVED = at least one round accepted and mean return on the score starts after minus before > 2 standard errors (paired). WORSE = after below before by more than 2 standard errors. Otherwise NO CHANGE.
- S4 (stage D, unchanged world): no `none` situation is WORSE (the gate does no harm). An improvement there is not expected and not required.
- dusty_panels situations in D are reported the same way if any ran; they are not part of S3.

## 7. On failure

Report every number as measured, with the verdict from section 6. No v4, no second evaluation, no parameter change after the final launch. Only genuine crashes may be fixed (logged; a crash fix reruns only the crashed stage with the same seeds).

## 8. Caveats that will be printed in the report whatever the result

- Simulation only (kinematic, one decision per hour). PSR = dark and cold, not confirmed ice.
- In unchanged worlds the planner's internal model is the simulator's own physics and maps, so its predictions are exact; the baselines are short-sighted rules. A stage A win shows the value of planning the whole trip with the light forecast, not robustness to model error. Stages C and D measure model error.
- v3 keeps rovers at the lander when no candidate plan is predicted to return. That lowers deaths without reaching more ice; v3's reached-ice rate can be below the naive planner's, which reaches ice with rovers that then die.
- The executor's local sensing reads the true slope one move ahead from the simulator arrays (the rover's map crop shows the same values).
- Self-learning scores are online adaptation on the deployed site (the rover learns where it drives), not a held-out generalisation score. The blocked route is a synthetic obstacle.
- Each episode batch is many independent rovers; in Dream-RSI what one recorded batch has seen is merged into one belief map.

## Amendment 1 (after the independent plan review, before any v3 code was run and before any v3 number)

The reviewer (records/agents/rover3-plan-reviewer.md) found no leakage path and listed 5 blockers and 9 should-fix items. All are accepted. Where this amendment and the text above differ, this amendment wins.

1. Intervals. The 24 fresh worlds are 6 sites x 4 dates that share terrain, lander and targets. Intervals therefore resample the 6 SITES with replacement (each drawn site brings its 4 dates), 10,000 resamples, seed 20261115, percentile interval; PASS needs the lower bound of the paired difference > 0. Also printed: the per-site paired difference and "v3 > best baseline in k of 6 sites". With 6 clusters the interval is approximate and the report says so. Stage B has 2 sites: per-site and per-world numbers only, no interval, and S1 is judged on the point values with that weakness stated.
2. The by-construction claim is for the deployed brain (m = 0) only. After any accepted change to m or the charge thresholds the claim is by gate only; the report says which brain each number comes from.
3. A baseline candidate is executed with the executor's overrides OFF (actions = `planner_action` bit for bit, read from the belief); the planner's model is `RoverEnv.step` itself. G2 is 100%, not 99%, in unchanged validation worlds; any mismatch is a defect. If G2 or G3 fails at the final launch, the words "by construction" are removed from the report.
4. What PASS means. Per rover, v3 takes the best of three exactly-predicted baselines plus 33 more plans, so in an unchanged world PARTIAL is the floor if the code is right. PASS means: whole-trip planning with a perfect forecast beats three short-sighted rules. It is not evidence of robustness or of learning. The report and any slide state this next to the number, with the absolute returned rate, the share of rovers kept at the lander, and reached-ice beside the naive planner's.
5. Lander immunity. A rover with an empty battery within 20 m of the lander does not die in this simulator (`env.py:215`), and most rovers start outside 20 m. v3 uses this (safe harbour, waiting); the baselines wait where they landed. Reported in addition: a reference row "safe harbour only" (drive to the lander and never leave: no planning at all) on the same starts in stages A and B, so the part of the death reduction that comes from this rule alone is visible; and v3's returned rovers split into "left at once" and "waited at the lander first".
6. Metric definition. returned = the simulator's own `returned` flag (reached ice, then back within 20 m). Safe harbour is not returned. Denominator = all rovers started. Rates are means of per-world rates.
7. Stage C: S2 is also reported against the TOLD baselines (S2b: v3 returned >= best told baseline returned). The report says the stale-belief baselines have no obstacle avoidance.
8. Stage D: concurrency is fixed at 3. S3's denominator is all 6 blocked_route situations; an unfinished situation counts as NO CHANGE; a capped situation is scored with its last accepted brain. Fewer than 4 finished = S3 NOT EVALUABLE.
9. Stage isolation. A, B, C use only the deployed brain (default parameters, belief = original world); nothing learned in D is used in A-C. Each stage writes its own file.
10. Crash rule. A crash fix may not change any action of a rover that did not crash; first-launch outputs are kept untouched; the rerun stage is named in the log.
11. Seeds and starts. v3 and the baselines use the same construct-then-reset sequence as `worldmodel/rover/train.py:evaluate_policy` (the environment is constructed with the seed and then reset once more); start positions, hours and targets are asserted equal across methods per world. Gate seed = 5000 + 100 x situation index + round; score seed = 9000 + situation index; recording seed = 7000 + situation index; imagination seed = 8000 + 100 x situation index + round; blocked_ring seed = 0; build_world seed = 0; the hazard map source is written to the output.
12. Site23/Site42 reproduction. If the baseline rows do not reproduce the earlier numbers, stage B is reported as NOT COMPARABLE, with no second run. The harness is first checked against the earlier Site11 baseline numbers on validation (naive 0.407 / cautious 0.312 / wait-light 0.234 returned at 256 rovers, seed 4321).
13. Fresh-site pick. Ten PGDA sites had clean footprints (DM1, NPA, NPB, NPC, NPD, LM2, LM3, LM4, LM5, LM7). The recon agent fetched six of them (its brief asked for 2 to 4, "prefer sites with PSR area") before any terrain statistic of them was computed, and reports no selection by PSR. Why LM2-LM5 were left out was not recorded; the lead did not choose among them. All six fetched sites are used.
14. Time rule. Halving is applied at most once, to A and B alike; if still over 30 minutes the run proceeds at that count.
15. Not tested anywhere: an error in the light forecast in an otherwise unchanged world. Stage C covers map error and solar-output error only. Stated in the report.
16. Parameters in section 1 were fixed after the lead had seen the fresh sites' terrain statistics (PSR share, slope share, relief), not any rover result.

## Amendment 2 (timing only; after validation and the development-site smoke, before any held-out number)

Reason: on development site Site01 one self-learning round in the blocked-route world took 5.5 minutes (loaded machine), so with 3 situations at a time and stage C running beside it fewer than 4 blocked_route situations could finish inside the 40-minute cap and S3 would be "not evaluable" for a compute reason. Changes, all fixed here before the launch: stage D starts after stage C has finished (not beside it); stage D concurrency is fixed at 6 (was 3); the 40-minute cap, the situation order, the 4 rounds, the gate and every size are unchanged. Nothing else changes. No held-out rover number existed when this was written (fresh-site worlds had been built: terrain, hazard map and light only).

Recorded at the same time (validation, not a score): G1, G2 (prediction = outcome for 100% of v3 rovers in all 24 validation worlds) and G3 held. The harness check against the earlier Site11 baseline numbers is not an exact reproduction: cautious 0.312 (earlier 0.312), wait-light 0.240 (0.234), naive 0.364 (0.407); the earlier Site11 numbers came from environments reused across checkpoint candidates, so their starts were a later random draw. The exact reproduction test remains stage B (Site23 + Site42, same protocol as `evaluate_policy`).
