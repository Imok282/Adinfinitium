# Adinfinitium: addendum after the freeze (rover v2 and the self-learning run)

Written Sun 15 Nov 2026, about 07:00 IST. For the students and mentors.
The main overnight run is frozen and is not changed by anything here. Its results are in `RESULTS_FOR_TEAM.md` (inside `adinfinitium_results_macmini.zip`).
The main run froze at 02:01 IST on 15 Nov. The v2 plan and code were written before that (00:11 to 01:02), on the MacBook. All v2 training, the v2 evaluation and the self-learning run happened AFTER the freeze, in a separate folder on the Mac mini (`~/adinfinitium-rover2/`). Simulation only.
Words used: "reached ice" = got to a dark-and-cold shadow-edge target (not confirmed ice). "Returned" = reached ice AND came back to the lander.

## 1. Summary (failures first)

1. **Rover v2 FAILED its pre-registered test.** On the 8 held-out worlds (Site23 + Site42, 4 dates, 512 rovers each) the v2 network returned 1.2% of rovers and lost 95.3%. The best planner returned 9.3% and lost 86.8%. (`rover_v2/eval.json`)
2. v2 is better than v1 (v1 returned 0.0%, reached ice 1.7%; v2 returned 1.2%, reached ice 7.4%). It is still far below every planner. It is not a working driver.
3. Why: v2 learned the routes of its training worlds by heart. On the return leg it agrees with the planner on 86-93% of steps on terrain it trained on (88-93% for the selected driver), and on 15-26% on terrain it has not seen. (`rover-v2-diagnosis.md`)
4. **The self-learning loop ran by itself** for 135 minutes (budget 140), starting from the v2 driver, with no training data and no human. It ended by itself when its time budget was used up, not by the hard stop and not because it had stopped improving.
5. It made small accepted gains in mean return: 2 of 23 rounds accepted on Site23, 3 of 15 on Site42. The other 33 rounds were rejected.
6. **In neither situation did any rover reach ice and return.** Returned stayed 0.0% in every round. Reached ice went 0.0% -> 1.6% (Site23) and 2.3% -> 2.0% (Site42, slightly down).
7. The report prints "Verdict: MET" for both situations. That rule is by mean return only. The plain reading is in section 3.
8. **The working rover driver to present is the planner.** Naive planner: returned 9.3%, lost 86.8% of rovers. Cautious planner: returned 8.8%, lost 54.3%. The cautious one brings home almost as many and loses far fewer.
9. Even the planners lose most rovers. This task is hard in this simulator.

## 2. Rover v2

### 2.1 Method

- v1 (main run) learned by trial and error (PPO) and never found the ice. It collapsed to "stop".
- v2 is the same network (11,377,354 parameters). It learns by copying the physics planners. This is called imitation with DAgger.
- Teacher: six fixed planner variants. For each training rover, all six are simulated, and the one with the best episode return is that rover's teacher.
- After the first 8 episodes, half of the rovers are driven by the network itself. The teacher labels the states they reach. This teaches recovery from its own mistakes.
- Same simulator, reward, planners, evaluator, train worlds (8 sites x 4 dates = 32 worlds) and hazard map as v1. The launcher checks these files against the main run.
- The reported "model" is the network alone. No planner fallback.
- Training: 105 episodes, 14,605 training steps, 110.4 minutes of a 110-minute budget (`train.log`).

### 2.2 Pre-registration (the rules were written before any v2 number existed)

| what | commit time (IST, 15 Nov) | commit | evidence it used |
|---|---|---|---|
| Plan: method, metric, selection rule, verdict rule | 00:11 | d02c090 | none from v2; the v1 held-out numbers had been seen and the plan says so |
| Clarifications (nothing changed, details fixed) | 00:35 | c580188 | code review; a planner-only probe on Site11; small MacBook smoke runs that printed Site11 candidate rows |
| Amendment 1: which training states are stored | 00:52 | bc4037d | a small MacBook check on train site Site01 and validation site Site11 |
| Amendment 2: narrows Amendment 1 (it dropped too many "stay at the lander" states) | 01:02 | 968a4e2 | a code re-review that measured the dropped states in a dark flat test world (not a site) |
| Training starts on the mini | 02:17 | 968a4e2 | |

- The plan's own headings give 00:36 and 00:53 for the clarifications and Amendment 1. The commits are one minute earlier.
- Amendment 1 states that no v2 code loaded Site23 or Site42 before or for the change. Amendment 2's text does not repeat that sentence; its evidence is the test world named above. The plan says the training program refuses to load the test sites (we did not re-check that code for this addendum).
- Both amendments are dated before training started and before any v2 held-out number existed.
- Note: the generated report says "one amendment". The plan file has two. This table is the correct count.
- Primary metric: returned, averaged over the 8 held-out worlds.
- Verdict rule: SUCCESS, MATCH, PARTIAL (v2 returned at least half of the best planner's) or FAILURE.

### 2.3 Checkpoint selection (validation site Site11 only; NOT a held-out score)

4 Site11 worlds, 256 rovers each, seed 4321. Rule: highest returned. (`selection.json`)

| candidate | round | minute | returned | reached ice | died | best so far |
|---|---|---|---|---|---|---|
| 0 | 11 | 14 | 0.123 | 0.216 | 0.762 | yes |
| 1 | 24 | 27 | 0.220 | 0.288 | 0.732 | yes |
| 2 | 38 | 40 | 0.264 | 0.387 | 0.693 | yes |
| 3 | 51 | 53 | 0.216 | 0.353 | 0.696 | no |
| 4 | 62 | 66 | 0.217 | 0.335 | 0.729 | no |
| 5 | 76 | 80 | 0.230 | 0.316 | 0.730 | no |
| 6 (selected) | 87 | 93 | 0.268 | 0.339 | 0.708 | yes |
| 7 | 101 | 106 | 0.023 | 0.321 | 0.871 | no |
| 8 (final weights) | 105 | 110 | 0.026 | 0.285 | 0.822 | no |
| naive planner, same starts | | | 0.407 | 0.561 | 0.581 | |
| cautious planner, same starts | | | 0.312 | 0.338 | 0.506 | |
| cautious + wait for light, same starts | | | 0.234 | 0.250 | 0.577 | |

Plain reading: even on its own validation site the best v2 candidate (0.268) is below the naive planner (0.407). The last two candidates almost stopped returning.

### 2.4 The one held-out evaluation (run once, 04:07-04:16 IST)

8 of 8 worlds, 512 rovers per world = 4,096 rovers per driver. (`eval.json`; v1 row from the main run)

| driver | reached ice | returned (primary) | died | mean return |
|---|---|---|---|---|
| rover v1 network (main run, failed) | 0.017 | 0.000 | 0.955 | -55.2 |
| rover v2 network (alone) | 0.074 | 0.012 | 0.953 | -41.8 |
| planner: naive | 0.196 | 0.093 | 0.868 | +4.7 |
| planner: cautious | 0.105 | 0.088 | 0.543 | -18.4 |
| planner: cautious + wait for light | 0.069 | 0.060 | 0.739 | -19.2 |

v2 returned 48 of 4,096 rovers. The naive planner returned 380.

Per world (v2 network, and the planner that returned most in that world). In site42 March naive and cautious tie on returned (0.043 each); naive is listed by the plan's tie rule, and cautious died less there (0.598 against 0.896):

| world | v2 ice | v2 returned | v2 died | best planner | ice | returned | died |
|---|---|---|---|---|---|---|---|
| site23, 2027-03-01 | 0.016 | 0.000 | 1.000 | naive | 0.182 | 0.049 | 0.951 |
| site23, 2027-06-01 | 0.002 | 0.000 | 1.000 | naive | 0.133 | 0.004 | 0.996 |
| site23, 2027-09-01 | 0.041 | 0.000 | 0.990 | naive | 0.275 | 0.135 | 0.865 |
| site23, 2027-12-01 | 0.047 | 0.002 | 0.994 | naive | 0.309 | 0.162 | 0.838 |
| site42, 2027-03-01 | 0.061 | 0.008 | 0.963 | naive | 0.133 | 0.043 | 0.896 |
| site42, 2027-06-01 | 0.053 | 0.000 | 0.992 | cautious | 0.082 | 0.076 | 0.557 |
| site42, 2027-09-01 | 0.160 | 0.025 | 0.852 | cautious | 0.232 | 0.193 | 0.449 |
| site42, 2027-12-01 | 0.215 | 0.059 | 0.834 | cautious | 0.260 | 0.230 | 0.465 |

### 2.5 Verdict

**Pre-registered verdict: FAILURE.** v2 returned 0.012. PARTIAL needed at least half of the best planner's 0.093, which is 0.046. v2 is at about one eighth of the planner.
v2 never beats the planner on returned in any of the 8 worlds.

### 2.6 What happened (from `rover-v2-diagnosis.md`, a read-only diagnosis)

- **It memorised routes.** On worlds it trained on, the return leg agrees with the planner on 86-93% of steps (selected driver: 88% and 93%; the final-weights candidate 8: 86%). On unseen Site11 it agrees on 23-26%. On held-out Site42 (December) it agrees on 15%.
- Why it could memorise: in every training world the lander and the 16 targets sit in fixed places. The network can learn "this terrain picture means go that way" without learning to navigate.
- On unseen terrain the return leg turns into a back-and-forth loop. On Site11 42-74% of return steps undo the previous step across the checks (selected driver in September: 47%; candidate 8: 74%). The selected driver's rovers die a median 70 m from the lander there.
- **Site11 was an easy site.** The diagnosis calls its targets the closest of all 11 sites (median 542 m; Site23 has a nearer single target, 198 m against 225 m), and its lander is lit and hazard-free. The naive planner returns 0.407 there and 0.093 on the test sites (4.4 times less). v2 drops from 0.268 to 0.012 (22 times less). So an easy validation site explains part of the gap. The rest is v2's own.
- **A defect in the training set (Site04).** Site04's centre world is 99.5% over the slope limit. 0 of 4 targets are reachable. Its 15 training episodes stored 3 states in total. So 4 of the 32 training worlds (12.5%) taught nothing. Site20's teacher returned 0.017. Useful return examples came from 6 sites.
- Site23 in March and June is a case no training world has: the lander gets no usable light for all 709 hours.
- No bug was found in the train-versus-evaluate pipeline. The inputs and the hazard map are the same in both.
- The report's "agreed with the teacher on 62.9% of steps" includes Site04 (about 0%). Without Site04 it is 76.5%.
- Not determined: why exactly the last 50 m fails; whether Site42's raw data differs in quality; how much picking the best of 9 noisy candidates flattered the Site11 score.

### 2.7 What a real fix needs (future work, none of it done)

- Randomise the lander and target positions in training, so routes cannot be memorised.
- More training sites, and drop or fix Site04 (and check Site20).
- More than one validation site, and not the easiest one.
- Training worlds with a dark lander, like Site23 in March and June.
- A new pre-registered plan and fresh held-out sites. Site23 and Site42 have now been looked at closely and should not be reused as the only test.

## 3. Self-learning addendum (run `selflearn_v2`)

### 3.1 What was asked, and what was built

- The request (the first words are as the team lead passed them on; `DESIGN.md` records it from "after the model is complete"): "Only thing I want is it should after the model is complete, and the training data is cut off, it should still recursively self-learn."
- The project named: Dream-RSI (github google-research/dream-rsi). In Dream-RSI an LLM rewrites a small strategy program. The strategy decides how a budget of tries is spent. Each round is recorded, replayed and scored, and the best strategy is kept.
- **The original needs paid LLM API keys. We used none.** So the original was not run.
- Built instead: a key-free version of its record / replay / score / keep-best loop. Faithful: a recorded tree of tries per round, replay on saved rounds, the budget sweep, the area score, "the current strategy stays unless beaten".
- Not faithful: strategy variants come from a fixed grid of 54 settings that a person wrote. No LLM writes code. The world model and the "imagination" practice under it are our addition; Dream-RSI has neither.
- Design: `DESIGN.md`. Code: branch rover-v2 at 968a4e2, folder `worldmodel/dream/`.

### 3.2 How the loop works

1. Drive: 128 rovers x 100 steps with the current driver. Every step is recorded. This is the only new information.
2. Learn a small "what happens if I do this" model from those recorded steps.
3. Practise inside that model ("imagination"). This gives up to 6 candidate drivers per round (careful, medium, bold recipes).
4. Test each candidate on 64 real simulator episodes against the current driver, from the same starts.
5. Accept the best candidate only if its gain is above 2 standard errors. Otherwise keep the old driver.
6. Score the kept driver on 256 separate score starts. Then the next round starts from it. No human in the loop.

The starting driver is rover v2 (which failed, section 2). Its file was not changed by the run (same sha256 before and after).

### 3.3 How it ended

It ended by itself, because its time was used up. Both situations stopped with "time for this situation used up", not because learning had finished. `report.md` says "Status: finished". The ledger has an "end" line at 06:34:48 IST. It started at 04:19:33, used 135.25 of 140 minutes, and was 0.0 minutes over budget. The hard stop was at 150 minutes (about 06:49) and was not reached.

### 3.4 Situation 1: Site23, 2027-06-01, world unchanged. Online self-learning on that site, NOT a held-out score

256 score rovers. Report's printed verdict: **"MET: self-learning improved on the starting driver"** (2 rounds accepted, 21 rejected, 415,532 own steps).

| round | accepted | ice | home | died | mean return | gain vs start (se) | tried |
|---|---|---|---|---|---|---|---|
| 0 (starting driver) | n/a | 0.0% | 0.0% | 100.0% | -48.5 | 0 | 0 |
| 1 | YES: bold #0 | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 6 of 6 |
| 2 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 6 of 6 |
| 3 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 6 of 6 |
| 4 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 2 of 6 |
| 5 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 6 of 6 |
| 6 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 3 of 6 |
| 7 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 2 of 6 |
| 8 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 2 of 6 |
| 9 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 4 of 6 |
| 10 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 2 of 6 |
| 11 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 3 of 6 |
| 12 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 4 of 6 |
| 13 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 2 of 6 |
| 14 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 3 of 6 |
| 15 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 3 of 6 |
| 16 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 2 of 6 |
| 17 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 2 of 6 |
| 18 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 2 of 6 |
| 19 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 4 of 6 |
| 20 | no | 0.8% | 0.0% | 99.2% | -40.4 | +8.0 (0.7) | 4 of 6 |
| 21 | YES: careful #1 | 1.6% | 0.0% | 98.8% | -39.9 | +8.6 (0.8) | 3 of 6 |
| 22 | no | 1.6% | 0.0% | 98.8% | -39.9 | +8.6 (0.8) | 4 of 6 |
| 23 | no | 1.6% | 0.0% | 98.8% | -39.9 | +8.6 (0.8) | 5 of 6 |

References on the same score starts: naive planner ice 12.5%, home 0.0%, died 100.0%, return -26.7. Cautious planner ice 0.0%, home 0.0%, died 13.7%, return -8.7.

Plain reading: one accepted step in round 1, then 19 rejected rounds, then one tiny step. 4 of 256 rovers reached ice at the end. None returned. 253 of 256 still died. In this world the lander is dark all month, and no planner returns a rover either.

### 3.5 Situation 2: Site42, 2027-06-01, with a SYNTHETIC blocked route. Online self-learning on that site, NOT a held-out score

The obstacle is made up (a ring of steep ground around the lander with 3 gaps). It is not real terrain. 256 score rovers.
Report's printed verdict: **"MET: self-learning improved on the starting driver"** (3 rounds accepted, 12 rejected, 833,926 own steps).

| round | accepted | ice | home | died | blocked moves / episode | mean return | gain vs start (se) | tried |
|---|---|---|---|---|---|---|---|---|
| 0 (starting driver) | n/a | 2.3% | 0.0% | 90.2% | 70.0 | -84.5 | 0 | 0 |
| 1 | no | 2.3% | 0.0% | 90.2% | 70.0 | -84.5 | +0.0 (0.0) | 4 of 6 |
| 2 | YES: bold #1 | 2.7% | 0.0% | 85.9% | 61.0 | -66.2 | +18.3 (2.7) | 6 of 6 |
| 3 | no | 2.7% | 0.0% | 85.9% | 61.0 | -66.2 | +18.3 (2.7) | 3 of 6 |
| 4 | no | 2.7% | 0.0% | 85.9% | 61.0 | -66.2 | +18.3 (2.7) | 6 of 6 |
| 5 | YES: bold #0 | 1.6% | 0.0% | 84.8% | 59.8 | -68.1 | +16.4 (2.9) | 6 of 6 |
| 6 | no | 1.6% | 0.0% | 84.8% | 59.8 | -68.1 | +16.4 (2.9) | 6 of 6 |
| 7 | YES: medium #1 | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 6 of 6 |
| 8 | no | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 3 of 6 |
| 9 | no | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 6 of 6 |
| 10 | no | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 4 of 6 |
| 11 | no | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 3 of 6 |
| 12 | no | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 6 of 6 |
| 13 | no | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 5 of 6 |
| 14 | no | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 3 of 6 |
| 15 | no | 2.0% | 0.0% | 81.6% | 65.1 | -67.2 | +17.4 (3.0) | 6 of 6 |

References on the same score starts: naive planner ice 5.1%, home 3.1%, return -9.2. Cautious planner ice 7.8%, home 7.0%, died 55.5%, return -52.0.

Plain reading: one real step in round 2 (fewer rovers died, fewer blocked moves). After that the score moved sideways. The final driver is slightly worse on mean return than after round 2 (-67.2 against -66.2). Reached ice went down (6 rovers -> 5 of 256). None returned. Here the planners do return rovers (3.1% to 7.0%), and the self-learning driver does not.

### 3.6 The printed verdict and the plain reading

- **The verdict rule is by mean return only.** MET means: at least one round accepted, and the final mean return beats the start by more than 2 standard errors. That did happen (+8.6, se 0.8; +17.4, se 3.0).
- **Home (returned) stayed 0.0% in all 38 rounds.** The mission was never completed once on the score starts.
- **Reached ice barely moved:** 0.0% -> 1.6% (0 -> 4 rovers) and 2.3% -> 2.0% (6 -> 5 rovers). Mean return can rise without the mission getting done. The report says this itself: "return can rise from fewer blocked moves alone".
- **Acceptance is best-of-6, so it is optimistic.** Up to 6 candidates are tried on the same 64 test starts and the best is kept. The honest number is the score-start column.
- **One round was accepted while slightly worse on the score starts.** Site42 round 5 passed its test (+6.38, se 2.53 on the 64 test starts). On the 256 score starts mean return went from -66.2 to -68.1 and ice from 2.7% to 1.6%.
- **"Training cut short by the time budget":** 5 calls in Site23 round 23 ran 713 to 786 of their 800 steps. This is small and affects only the last, rejected round.
- **The strategy layer is a mechanism demo.** It changed strategy 2 times over 38 saved rounds (replay score 0.366 -> 0.450, then 0.438 -> 0.445). That is far too little data to say it helps. It may be chasing noise.
- **These are "online self-learning on that site" scores, not held-out scores.** The rover drove 415,532 and 833,926 steps on these exact worlds.
- **Order of events:** the single held-out evaluations of v1 and v2 were finished (v2 at 04:16:35 IST) before this run touched Site23 and Site42 (start 04:19:33). The held-out tables in section 2 are not affected.
- The report prints "dream code commit NOT RECORDED". We checked separately: the code folder on the mini matches commit 968a4e2 file for file (section 5).

### 3.7 Does it meet the request? An honest answer

Demonstrated: after training ended and the training data was cut off, the rover kept adjusting itself, round after round, from its own driving only, with no human and no new data. Each round started from the last accepted driver. The change is measurable (mean return up by 8.6 and 17.4 points; on Site42 deaths fell from 90.2% to 81.6%), and it is small. Most rounds (33 of 38) changed nothing.
Not demonstrated: learning to complete the mission. No rover reached ice and returned, before or after. Also not demonstrated: open-ended or truly recursive self-improvement. The rover does not change how it learns. It picks among settings a person wrote and learns a small correction on top of a driver that had already failed. Nothing here is about a real rover.

## 4. What to say in the presentation

Safe to say (each with its number and file):

- "Our learned rover driver failed, twice, and we report both." v1 returned 0.0%; v2 returned 1.2% against the planner's 9.3%. (`RESULTS_FOR_TEAM.md` section 3.4; `rover_v2/eval.json`)
- "The driver we present is the physics planner." Naive: returned 9.3%, lost 86.8%. Cautious: returned 8.8%, lost 54.3%. (`rover_v2/eval.json`, same as the main run's)
- "We wrote the pass/fail rule for v2 before we had any number, and it came out FAILURE." Commit d02c090, 00:11 IST. (`ROVER_V2_PLAN.md`, `ROVER_V2_REPORT.md`)
- "Copying the planner got the network much further than trial and error": reached ice 1.7% -> 7.4%, returned 0.0% -> 1.2%. Still a failure. (`rover_v2/eval.json`)
- "We found why v2 fails: it memorised routes." 86-93% agreement on trained terrain, 15-26% on unseen terrain. (`rover-v2-diagnosis.md`)
- "We found a defect in our own training set": Site04 is 99.5% too steep and taught nothing. (`rover-v2-diagnosis.md`)
- "After training ended, a loop let the rover keep adjusting itself with no new data and no human, for 135 minutes." 5 of 38 rounds accepted; mean return +8.6 and +17.4; nobody returned. (`selflearn_v2/report.md`)
- "It is a key-free version of the Dream-RSI loop, not the original. No LLM, no API key." (`DESIGN.md`)

Do NOT say:

- "The self-learning rover learned the mission" or "it improves itself recursively". Returned stayed 0.0%.
- "The self-learning verdict was MET" without adding "by mean return only; no rover returned".
- "Rover v2 works" or "v2 nearly matches the planner". It is at one eighth of the planner on returned.
- "We ran Dream-RSI." We ran a key-free loop built in its style.
- Any self-learning number as a "held-out" or "test" score, or any claim about real ice or a real rover.

## 5. Files and how to reproduce

Zip: `~/adinfinitium_addendum_after_freeze.zip` on the Mac mini (about 57 MB; outside the frozen folder).

| in the zip | what |
|---|---|
| `ADDENDUM_AFTER_FREEZE.md` | this file |
| `rover_v2/` | `ROVER_V2_REPORT.md`, `eval.json`, `selection.json`, `train.log`, `train_log.jsonl`, `launcher.log`, `launcher.out`, `eval.log`, `TRAINING_CLOSED`, `ROVER_V2_PLAN.md`, `CHANGES_ROVER_V2.md`, `rover_driver_v2.pt` (selected candidate 6, sha256 `cecdf244...9375`) |
| `selflearn_v2/` | `report.md`, `report.json`, `ledger.jsonl`, 2 `curve_*.png`, `selflearn_v2.log`, `dream_launcher.out`, `trace_pool/` (38 saved rounds), `models/` with the final file of each situation (Site23 round 23, Site42 round 15) |
| `docs/` | `DESIGN.md`, `rover-v2-diagnosis.md`, `rover-diagnosis.md` (v1) |
| `source/` | `adinfinitium_rover-v2_968a4e2.bundle` (git bundle of branch rover-v2; includes the dream code), `ROVER2_COMMIT.txt` |
| `scratch_evidence/` | the diagnosis scripts and raw probe outputs that were removed from the mini (156 KB) |

Left out of the zip: the other 36 per-round self-learning model files (about 6.7 MB each, 242 MB in all) and `rover_driver_v2_last.pt` (135 MB; candidate 8 plus optimiser state). They stay on the mini.
The self-learning model files are small corrections on top of the v2 driver. They need `rover_driver_v2.pt` to be used.

Reproduce (on the mini; needs the main run folder for its data and Python environment):

    git clone adinfinitium_rover-v2_968a4e2.bundle adinfinitium-rover2 && cd adinfinitium-rover2 && git checkout rover-v2   # HEAD 968a4e2
    nohup bash ops/rover2/run_rover2_mini.sh 110 > runs/_rover2/launcher.out 2>&1 &          # rover v2: train 110 min, select on Site11, one held-out evaluation
    START_DRIVER_FILE=$HOME/adinfinitium-rover2/data/models/rover_driver_v2.pt nohup bash ops/dream/run_dream_mini.sh 140 selflearn_v2 > runs/dream_launcher.out 2>&1 &
    cd ~/adinfinitium-run/adinfinitium && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ops.freeze --check      # must print: FROZEN check: ALL UNCHANGED

A re-run will not give the same numbers digit for digit (training is timed by the clock). The launcher refuses to repeat a finished held-out evaluation in the same folder.
Code check done for this addendum: a checksum comparison of the MacBook checkout at 968a4e2 against `~/adinfinitium-rover2/` on the mini listed no differing file.

Left on the Mac mini by these two tracks:

| where | size | what |
|---|---|---|
| `~/adinfinitium-rover2/` | 427 MB in all | code at 968a4e2 plus everything below |
| `~/adinfinitium-rover2/data/models/rover_driver_v2.pt` | 43 MB | the selected v2 driver (in the zip) |
| `~/adinfinitium-rover2/data/models/rover_driver_v2_last.pt` | 129 MB on disk | candidate 8 plus optimiser state (not in the zip) |
| `~/adinfinitium-rover2/data/rover_cache/` | 5.4 MB | Site11 caches built by v2; the rest of `data/` is links into the main run's data |
| `~/adinfinitium-rover2/runs/_rover2/` | about 200 KB | v2 report, evaluation, selection, logs |
| `~/adinfinitium-rover2/runs/_dream/selflearn_v2/` | 245 MB | report, ledger, curves, 38 saved rounds, 38 model files (244 MB) |
| `~/adinfinitium-rover2/runs/_dream/selflearn_v2.log`, `runs/dream_launcher.out` | 36 KB | logs |
| `~/adinfinitium_addendum_after_freeze.zip` | about 57 MB | this package |

Removed after packaging (small scratch from tonight's diagnosis agents; a copy of each is in the zip under `scratch_evidence/`): `~/adinfinitium-run/diag_scratch/` (12 KB), `~/adinfinitium-rover2-diag/` (120 KB, the raw probe outputs behind `rover-v2-diagnosis.md`), `~/adinfinitium-rover2/runs/_rover2/_cpuprobe/` (24 KB).
Not from these tracks and not touched: the frozen main run folder and its two zips, and the setup leftovers next to it in `~/adinfinitium-run/` (`probe*`, `rehearsal_outputs`, `prefetch_tiles*.out`, `setup*.log`, `smoke.out`, about 9.5 MB).

Copied to the MacBook (`~/Documents/Ojas/Hackathon/records/`): `rover_v2/` (report, `eval.json`, `selection.json`) and `dream_rsi/selflearn_v2/` (report, json, ledger, curves, log). No checkpoints.
