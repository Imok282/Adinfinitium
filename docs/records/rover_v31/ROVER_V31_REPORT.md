# Rover v3.1 report: stay-alive routes, strict-first planning, a self-learning layer that can fix a shifted forecast

Sun 15 Nov 2026. After-freeze experiment; nothing in the frozen main run was changed; no simulator physics change. Plan `ROVER_V31_PLAN.md` (pre-registered 13:40 IST, four amendments, all before any test number was read). One final evaluation; nothing was changed or rerun after a test number was seen. Every number below was re-derived from the raw files by an independent checker (`records/agents/rover31-numbers-checker.md`): all confirmed.

Two scoring rules are used everywhere. **Strict**: a rover whose battery is empty at any hour, anywhere, is lost; "returned" needs a trip with no empty battery. **Lenient** (the simulator's own): an empty battery within 20 m of the lander is forgiven. A strict number is never comparable with a lenient one.

## Verdict against the pre-registered criteria

| criterion | result |
|---|---|
| PRIMARY: four untouched sites (LM2, LM3, LM4, LM5), unchanged worlds, strict rule: lost falls by 2 points or more, at 3+ of 4 sites, interval below 0, strict returned not lower, lenient lost not higher | **PASS**: strict lost 61.0% -> 53.9% (-7.1 points [-13.8, -3.5], lower at 4 of 4 sites); strict returned 31.0% -> 33.7% (+2.7 [+1.9, +3.4]); lenient lost 0.0% both |
| S1: same rule on previously seen sites | PASS on the six earlier fresh sites (strict lost 56.7% -> 54.2%, -2.5 points, 6 of 6 sites; strict returned 42.0% -> 43.9%); PASS on Site23 + Site42 (38.6% -> 30.0%; 32.4% -> 38.3%; 2 sites, no interval) |
| S2: lenient returned (reported, not a criterion) | untouched 46.7% -> 48.1%; six fresh 59.6% -> 59.9%; Site23/42 37.5% -> 41.4%. It did not fall, although the plan allowed for that |
| S3: changed worlds before learning, strict rule, v3.1 vs v3 | MET for all five changes (strict lost 5.8 to 8.9 points lower, strict returned 0.6 to 2.1 points higher). **But under the lenient rule v3.1 is worse with a late forecast: lost 32.2% against 19.9%, returned 13.5% against 17.3%** |
| S4: self-learning, more than half of the sites IMPROVED and none WORSE | light 12 h late: **MET** (3 of 4 IMPROVED, 0 WORSE); blocked route: **MET** (4 of 4); dusty panels: **NOT MET** (1 of 4 IMPROVED, 3 NO CHANGE); unchanged: MET (nothing accepted, nothing worse); light 8 h early: **NOT EVALUABLE** (all 4 cut by the time cap after one round, my launcher error, see "What went wrong") |

**Read this before quoting the PASS.**
- v3.1 cannot score below v3 under the strict rule in unchanged worlds if the code is right: v3's plans are among v3.1's candidates and the model is exact there. Stage A only shows how much lit ground within reach exists under a perfect forecast; stages C and D are the only tests of a wrong model.
- The gain is modest and most of it is not science. Of the 583 rovers (of 8,192) that v3 lost and v3.1 did not, **164 are new completed trips and 419 are rovers that stayed alive on lit ground without completing a trip** (alive only until hour 300 of the episode, mean battery left 80 Wh). v3.1 still loses 53.9%.
- The size varies by site: LM4 -17.1 points, the other three -3.4 to -4.1. On the six earlier fresh sites it is -2.5 points.
- The baseline is frozen v3. The weak-flags "strict-aware reserve" variant of v3 was about 1 point better than frozen v3 on the six fresh sites (55.5% strict lost against 56.7%); v3.1 is at 54.2% there.
- LM2-LM5 are neighbours (10-30 km apart; LM5's footprint is 4 km from seen site LM7's): four sites, not independent samples; the intervals are rough.

## Diagnosis: where v3's strict losses come from (development + validation sites, 4,608 rovers, frozen v3)

Done before the design was tested, on sites that are not test sites (`diagnosis.md`, `diagnosis.json`; scripts in `scripts_diagnosis/`).

- v3 strict returned 37.0%, strict lost 56.1% (2,586 rovers) on these nine sites.
- **71% of the losses (1,846) are rovers that sat within 20 m of the lander until the battery was flat and never completed a trip**; 1,198 of them never went further than 60 m. **15% (396) were counted "returned" by the simulator but had a flat battery earlier** (they sat flat at the lander, then drove when the light came back; median first flat hour 45, median return hour 217). 13% (344) are simulator deaths, all at Site04, where the ground round the lander is too steep to reach it.
- Cause: the lander is dark for more than 50 h in a row in 78% of the 300 h windows (a rover idles 50 h on a full battery); 98% of the strict losses are such starts. In 96% of those dark runs some ground within 300 m of the lander is lit at some hour. v3 has one place to wait, the lander, and never looks for lit ground.
- **How much is unavoidable?** A range, not one number:
  - Proven unavoidable (even a rover that collects the best light of everything it could have reached, at idle power only, goes flat before the earliest possible return): **8.1% of the losses** (209). This bound is very loose.
  - An energy-only estimate (drive to a lit light-grid cell within 450 m and sit; ignores slopes): about 66% of the losses have no stay-alive plan, **about 34% (882) do**. It is an estimate, not a bound.
  - What v3.1 actually saved on the six validation sites: 348 of 1,770 strict losses (19.7%); 88 became strict returns, 260 stayed alive without a completed trip.
  - So the unavoidable share of v3's strict losses lies **between 8% (proven) and about 80% (v3.1 saved 20% on validation, so at most 80% are unavoidable); the energy estimate puts it near two thirds**. In this simulator, with this rover (100 Wh battery, 2 W idle) and random landing hours, most strict losses probably cannot be planned away. The avoidable part is real and is what v3.1 goes after.
- Self-learning (from v3's saved rounds): in rounds 2-4, 41 proposals reached the real gate and none was accepted; 19 of the 41 were re-sends of a proposal the gate had already rejected (the layer had no memory); in 13 rounds nothing beat the current brain in imagination. Dusty panels: the correct solar measurement removed all deaths in the gate episodes 7 times of 7 (47 deaths to 0) but failed the "gain above 2 standard errors" test 6 times of 7, and in 7 rounds an energy-margin proposal took its place at the gate. No proposal type existed for a time-shifted light forecast.

## What changed in the brain (v3 -> v3.1)

Still scripted: no neural network, no simulator change. Frozen v3 is `Theta()`; v3.1 is `Theta(v31=True)`.

1. **Stay-alive route search (new planner part).** For each rover, a search over a grid of places around it (36 m apart, up to 432 m away) and every remaining hour finds the highest battery level it can have at each place and hour without ever being flat, from the light forecast. Read backwards it gives a route: drive to lit ground, sit, move on when the light moves.
2. **31 new candidate plans** next to v3's 36: "shelter only" (follow that route to the end) and "shelter, then go" (follow it until hour w, then drive to the target and back; 15 waits, with or without charging stops). Every candidate is rolled forward in the rover's own model, as in v3.
3. **Strict-first choice.** A plan that returns and is never flat beats a plan that only stays alive, which beats v3's rule (anything not dead under the simulator's lenient rule). v3 could not tell these apart.
4. **Executor**: one new mode, following waypoints by hour.
5. **Self-learning layer**: two new proposal types (light forecast shifted in time, fitted from the rover's own solar readings on a grid of -36 to +36 h; battery capacity as measured), memory of rejected proposals (a rejected proposal is not sent again; the next best goes), and the gate now scores under the strict rule (same thresholds; all of v3's count conditions kept and two strict ones added).

## Results (the report script's own tables; brackets = 95% interval from resampling sites)

### A. Untouched sites LM2, LM3, LM4, LM5 (PRIMARY): 16 worlds, 8192 episodes per method

| method | returned, STRICT | lost, STRICT | returned, lenient | lost, lenient | reached target |
|---|---|---|---|---|---|
| **v3.1** | 33.7% [26.0, 42.2] | 53.9% [46.8, 59.5] | 48.1% [32.8, 63.4] | 0.0% [0.0, 0.0] | 50.4% |
| v3 (frozen) | 31.0% [24.3, 38.9] | 61.0% [50.4, 72.8] | 46.7% [30.8, 62.5] | 0.0% [0.0, 0.0] | 49.0% |
| planner: naive | 17.9% [15.4, 20.6] | 81.8% [79.1, 84.4] | 18.2% [15.7, 21.0] | 81.4% [78.8, 84.0] | 29.3% |
| planner: cautious | 12.2% [6.6, 17.9] | 83.7% [78.6, 90.3] | 19.1% [9.2, 29.2] | 44.6% [39.8, 49.4] | 22.0% |
| planner: cautious + wait for light | 14.0% [5.6, 22.3] | 78.5% [72.2, 88.5] | 14.9% [5.7, 24.1] | 72.8% [66.3, 83.2] | 16.7% |
| reference: safe harbour only | 0.0% [0.0, 0.0] | 79.0% [61.4, 96.4] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% |

- v3.1 minus v3, strict lost: -7.1 points [-13.8, -3.5]; per site: LM2 -4.1, LM3 -3.9, LM4 -17.1, LM5 -3.4

- v3.1 minus v3, strict returned: +2.7 points [+1.9, +3.4]; per site: LM2 +2.6, LM3 +3.2, LM4 +1.4, LM5 +3.6

- v3.1 minus v3, lenient returned: +1.5 points [+0.9, +2.6]; per site: LM2 +0.9, LM3 +0.8, LM4 +0.9, LM5 +3.2

- Rovers v3 lost (strict) that v3.1 did not: 583 = 164 now strict returns + 419 alive without a completed trip. Lost by v3.1 but not by v3: 0. Strict returns of v3 that v3.1 did not make: 0. Lenient returns of v3 that v3.1 did not make: 4.
- v3.1 alive at hour 300 without a completed trip: 1020 (mean battery left 80 Wh). First plan: rank 1 / 2 / 3 = [4413, 1020, 2759]; shelter only 461, shelter then go 1723.
- v3.1 minus best planner by strict returned (naive): +15.8 points, higher at 4 of 4 sites.
- Prediction = outcome (strict): v3.1 at least 100.0% in every world. Rule result: **PASS** {'strict_lost_falls_2_points': True, 'falls_at_enough_sites': True, 'interval_below_0': True, 'strict_returned_not_lower': True, 'lenient_died_not_higher': True}

### B1. Six earlier fresh sites (previously seen): 24 worlds, 12288 episodes per method

| method | returned, STRICT | lost, STRICT | returned, lenient | lost, lenient | reached target |
|---|---|---|---|---|---|
| **v3.1** | 43.9% [32.6, 53.4] | 54.2% [45.4, 65.2] | 59.9% [46.8, 71.5] | 0.0% [0.0, 0.0] | 62.9% |
| v3 (frozen) | 42.0% [31.2, 51.2] | 56.7% [48.0, 67.2] | 59.6% [46.4, 71.4] | 0.0% [0.0, 0.0] | 62.7% |
| planner: naive | 29.8% [19.3, 38.0] | 69.4% [61.1, 79.8] | 29.9% [19.4, 38.1] | 69.4% [61.0, 79.8] | 39.6% |
| planner: cautious | 23.7% [13.8, 32.5] | 75.3% [66.1, 85.4] | 32.9% [18.0, 46.2] | 38.8% [26.8, 50.8] | 36.7% |
| planner: cautious + wait for light | 25.7% [15.8, 34.0] | 70.3% [62.3, 79.4] | 26.4% [16.5, 34.8] | 65.0% [58.3, 72.6] | 28.9% |
| reference: safe harbour only | 0.0% [0.0, 0.0] | 83.0% [76.7, 89.3] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% |

- v3.1 minus v3, strict lost: -2.5 points [-3.7, -1.3]; per site: DM1 -3.6, LM7 -2.7, NPA -1.5, NPB -1.1, NPC -0.8, NPD -5.0

- v3.1 minus v3, strict returned: +1.9 points [+1.1, +3.0]; per site: DM1 +2.4, LM7 +1.1, NPA +1.6, NPB +1.1, NPC +0.8, NPD +4.4

- v3.1 minus v3, lenient returned: +0.3 points [-0.0, +0.6]; per site: DM1 +0.9, LM7 -0.0, NPA +0.6, NPB +0.5, NPC +0.2, NPD -0.3

- Rovers v3 lost (strict) that v3.1 did not: 302 = 219 now strict returns + 83 alive without a completed trip. Lost by v3.1 but not by v3: 0. Strict returns of v3 that v3.1 did not make: 0. Lenient returns of v3 that v3.1 did not make: 17.
- v3.1 alive at hour 300 without a completed trip: 228 (mean battery left 64 Wh). First plan: rank 1 / 2 / 3 = [6663, 228, 5397]; shelter only 83, shelter then go 3081.
- v3.1 minus best planner by strict returned (naive): +14.1 points, higher at 6 of 6 sites.
- Prediction = outcome (strict): v3.1 at least 100.0% in every world. Rule result: **PASS** {'strict_lost_falls_2_points': True, 'falls_at_enough_sites': True, 'interval_below_0': True, 'strict_returned_not_lower': True, 'lenient_died_not_higher': True}

### B2. Site23 + Site42 (previously seen): 8 worlds, 4096 episodes per method

| method | returned, STRICT | lost, STRICT | returned, lenient | lost, lenient | reached target |
|---|---|---|---|---|---|
| **v3.1** | 38.3% | 30.0% | 41.4% | 0.0% | 48.2% |
| v3 (frozen) | 32.4% | 38.6% | 37.5% | 0.0% | 41.4% |
| planner: naive | 9.2% | 86.9% | 9.3% | 86.8% | 19.6% |
| planner: cautious | 7.0% | 78.3% | 8.8% | 54.3% | 10.4% |
| planner: cautious + wait for light | 6.0% | 77.6% | 6.0% | 73.8% | 6.9% |
| reference: safe harbour only | 0.0% | 50.2% | 0.0% | 0.0% | 0.0% |

- v3.1 minus v3, strict lost: -8.6 points; per site: Site23 -16.5, Site42 -0.8

- v3.1 minus v3, strict returned: +5.9 points; per site: Site23 +9.0, Site42 +2.7

- v3.1 minus v3, lenient returned: +3.9 points; per site: Site23 +5.2, Site42 +2.5

- Rovers v3 lost (strict) that v3.1 did not: 354 = 182 now strict returns + 172 alive without a completed trip. Lost by v3.1 but not by v3: 0. Strict returns of v3 that v3.1 did not make: 0. Lenient returns of v3 that v3.1 did not make: 8.
- v3.1 alive at hour 300 without a completed trip: 1301 (mean battery left 91 Wh). First plan: rank 1 / 2 / 3 = [1227, 1301, 1568]; shelter only 181, shelter then go 1035.
- v3.1 minus best planner by strict returned (naive): +29.1 points, higher at 2 of 2 sites.
- Prediction = outcome (strict): v3.1 at least 100.0% in every world. Rule result: **PASS** {'strict_lost_falls_2_points': True, 'falls_at_enough_sites': True, 'interval_below_0': None, 'strict_returned_not_lower': True, 'lenient_died_not_higher': True}

### Changed worlds before any learning (untouched sites, 2027-06-01, 256 rovers per world; the rover is not told)

| change | method | returned, STRICT | lost, STRICT | returned, lenient | lost, lenient |
|---|---|---|---|---|---|
| blocked route | **v3.1** | 12.0% | 77.3% | 21.6% | 5.8% |
| blocked route | v3 (frozen) | 10.9% | 85.7% | 20.5% | 5.0% |
| blocked route | best planner not told: naive | 2.1% | 93.5% | 2.1% | 93.5% |
| blocked route | best planner told: naive | 8.0% | 91.1% | 8.1% | 91.0% |
| dusty panels | **v3.1** | 11.8% | 80.0% | 23.2% | 11.8% |
| dusty panels | v3 (frozen) | 11.0% | 87.3% | 23.1% | 10.4% |
| dusty panels | best planner not told: naive | 5.0% | 94.4% | 5.2% | 94.2% |
| dusty panels | best planner told: naive | 5.0% | 94.4% | 5.2% | 94.2% |
| light 12 h late | **v3.1** | 12.2% | 81.3% | 13.5% | 32.2% |
| light 12 h late | v3 (frozen) | 11.2% | 87.1% | 17.3% | 19.9% |
| light 12 h late | best planner not told: naive | 7.2% | 92.1% | 7.5% | 91.8% |
| light 12 h late | best planner told: naive | 7.2% | 92.1% | 7.5% | 91.8% |
| light 8 h early | **v3.1** | 17.0% | 75.4% | 32.1% | 4.3% |
| light 8 h early | v3 (frozen) | 14.9% | 83.7% | 29.8% | 3.9% |
| light 8 h early | best planner not told: naive | 6.8% | 92.7% | 7.1% | 92.3% |
| light 8 h early | best planner told: naive | 6.8% | 92.7% | 7.1% | 92.3% |
| weak battery | **v3.1** | 9.0% | 80.0% | 20.0% | 9.0% |
| weak battery | v3 (frozen) | 8.4% | 88.9% | 18.7% | 6.1% |
| weak battery | best planner not told: cautious_wait_light | 4.6% | 93.4% | 4.7% | 88.4% |
| weak battery | best planner told: cautious_wait_light | 4.0% | 94.3% | 4.1% | 89.6% |

- blocked route: v3.1 minus v3 strict lost -8.4, strict returned +1.1, lenient returned +1.1, lenient lost +0.8 points; S3 MET
- dusty panels: v3.1 minus v3 strict lost -7.3, strict returned +0.8, lenient returned +0.1, lenient lost +1.4 points; S3 MET
- light 12 h late: v3.1 minus v3 strict lost -5.8, strict returned +1.0, lenient returned -3.8, lenient lost +12.3 points; S3 MET
- light 8 h early: v3.1 minus v3 strict lost -8.3, strict returned +2.1, lenient returned +2.3, lenient lost +0.4 points; S3 MET
- weak battery: v3.1 minus v3 strict lost -8.9, strict returned +0.6, lenient returned +1.4, lenient lost +2.9 points; S3 MET

### Self-learning (v3.1 layer on the v3.1 brain; untouched sites, 2027-06-01; 2 rounds; 256 score starts)

| site | world | accepted (rounds) | accepted change | v3 no learning: strict ret / lost | v3.1 before -> after: strict returned | strict lost | lenient returned | lenient lost | gain strict score (SE) | label |
|---|---|---|---|---|---|---|---|---|---|---|
| LM2 | light 12 h late | 1 ([1]) | light forecast shifted by +12 h (fitted from solar readings) | 9.8% / 85.9% | 10.2% -> 16.4% | 81.6% -> 72.3% | 12.5% -> 34.4% | 53.9% -> 0.0% | +36.6 (4.5) | IMPROVED |
| LM2 | dusty panels | 1 ([2]) | energy margin 0.00 -> 0.10 | 12.5% / 76.6% | 13.3% -> 14.8% | 70.3% -> 66.8% | 27.7% -> 34.8% | 18.8% -> 5.5% | +2.5 (2.2) | NO CHANGE |
| LM2 | blocked route | 1 ([1]) | belief map <- ground seen (850 cells corrected) | 7.4% / 82.8% | 10.5% -> 18.8% | 73.8% -> 71.9% | 18.4% -> 36.7% | 7.4% -> 0.0% | +28.8 (3.7) | IMPROVED |
| LM2 | unchanged | 0 ([]) | nothing | 16.0% / 73.0% | 19.5% -> 19.5% | 68.4% -> 68.4% | 39.5% -> 39.5% | 0.0% -> 0.0% | +0.0 (0.0) | NO CHANGE |
| LM3 | light 12 h late | 1 ([1]) | light forecast shifted by +12 h (fitted from solar readings) | 25.8% / 74.2% | 28.1% -> 38.3% | 71.5% -> 60.5% | 31.6% -> 69.5% | 56.2% -> 0.0% | +66.0 (5.0) | IMPROVED |
| LM3 | dusty panels | 0 ([]) | nothing | 24.2% / 75.8% | 26.6% -> 26.6% | 72.3% -> 72.3% | 58.6% -> 58.6% | 19.5% -> 19.5% | +0.0 (0.0) | NO CHANGE |
| LM3 | blocked route | 1 ([1]) | belief map <- ground seen (850 cells corrected) | 20.3% / 79.3% | 26.2% -> 33.6% | 71.9% -> 64.8% | 56.2% -> 67.2% | 6.6% -> 0.0% | +13.8 (2.9) | IMPROVED |
| LM3 | unchanged | 0 ([]) | nothing | 32.4% / 67.6% | 36.7% -> 36.7% | 62.9% -> 62.9% | 70.7% -> 70.7% | 0.0% -> 0.0% | +0.0 (0.0) | NO CHANGE |
| LM4 | light 12 h late | 1 ([1]) | light forecast shifted by +12 h (fitted from solar readings) | 15.6% / 84.4% | 16.0% -> 23.0% | 65.6% -> 53.1% | 18.4% -> 25.0% | 11.7% -> 0.0% | +19.1 (3.3) | IMPROVED |
| LM4 | dusty panels | 1 ([1]) | belief solar output x 0.600 (measured from battery telemetry) | 6.6% / 93.4% | 5.9% -> 10.9% | 67.2% -> 58.6% | 5.9% -> 12.1% | 10.9% -> 0.0% | +12.9 (2.4) | IMPROVED |
| LM4 | blocked route | 1 ([1]) | belief map <- ground seen (850 cells corrected) | 9.0% / 91.0% | 9.4% -> 17.6% | 64.8% -> 56.6% | 9.4% -> 18.4% | 6.2% -> 0.0% | +18.6 (3.3) | IMPROVED |
| LM4 | unchanged | 0 ([]) | nothing | 12.9% / 87.1% | 13.7% -> 13.7% | 53.5% -> 53.5% | 14.5% -> 14.5% | 0.0% -> 0.0% | +0.0 (0.0) | NO CHANGE |
| LM5 | light 12 h late | 0 ([]) | nothing | 4.3% / 95.7% | 4.7% -> 4.7% | 95.3% -> 95.3% | 4.7% -> 4.7% | 0.4% -> 0.4% | +0.0 (0.0) | NO CHANGE |
| LM5 | dusty panels | 0 ([]) | nothing | 1.6% / 98.4% | 1.6% -> 1.6% | 98.4% -> 98.4% | 2.0% -> 2.0% | 0.4% -> 0.4% | +0.0 (0.0) | NO CHANGE |
| LM5 | blocked route | 1 ([1]) | belief map <- ground seen (850 cells corrected) | 3.5% / 96.5% | 3.5% -> 5.1% | 96.5% -> 94.9% | 3.9% -> 5.5% | 2.0% -> 0.0% | +4.0 (1.6) | IMPROVED |
| LM5 | unchanged | 0 ([]) | nothing | 5.1% / 94.9% | 5.1% -> 5.1% | 94.9% -> 94.9% | 5.1% -> 5.1% | 0.0% -> 0.0% | +0.0 (0.0) | NO CHANGE |
| LM2 | light 8 h early | 0 ([]) | nothing | 15.6% / 79.3% | 16.4% -> 16.4% | 75.8% -> 75.8% | 45.3% -> 45.3% | 3.1% -> 3.1% | +0.0 (0.0) | NOT FINISHED (counts as NO CHANGE) |
| LM3 | light 8 h early | 0 ([]) | nothing | 32.4% / 67.6% | 36.7% -> 36.7% | 60.2% -> 60.2% | 72.3% -> 72.3% | 3.9% -> 3.9% | +0.0 (0.0) | NOT FINISHED (counts as NO CHANGE) |
| LM4 | light 8 h early | 0 ([]) | nothing | 8.2% / 91.8% | 9.4% -> 9.4% | 69.5% -> 69.5% | 9.8% -> 9.8% | 13.3% -> 13.3% | +0.0 (0.0) | NOT FINISHED (counts as NO CHANGE) |
| LM5 | light 8 h early | 0 ([]) | nothing | 4.3% / 95.7% | 4.3% -> 4.3% | 95.7% -> 95.7% | 4.3% -> 4.3% | 0.0% -> 0.0% | +0.0 (0.0) | NOT FINISHED (counts as NO CHANGE) |

- light 12 h late: 3 IMPROVED, 1 NO CHANGE, 0 WORSE of 4 (4 finished); rounds accepted 3 of 8 (in round 2: 0); S4 MET
- dusty panels: 1 IMPROVED, 3 NO CHANGE, 0 WORSE of 4 (4 finished); rounds accepted 2 of 8 (in round 2: 1); S4 NOT MET
- blocked route: 4 IMPROVED, 0 NO CHANGE, 0 WORSE of 4 (4 finished); rounds accepted 4 of 8 (in round 2: 0); S4 MET
- unchanged: 0 IMPROVED, 4 NO CHANGE, 0 WORSE of 4 (4 finished); rounds accepted 0 of 8 (in round 2: 0); S4 MET
- light 8 h early: 0 IMPROVED, 4 NO CHANGE, 0 WORSE of 4 (0 finished); rounds accepted 0 of 4 (in round 2: 0); S4 NOT EVALUABLE

Notes on the tables.
- All methods ran on the same worlds, starts (seed 1234) and episode counts. The v3 row is the frozen v3 logic (`v31=False`): on the six fresh sites it reproduces the frozen counts exactly (7,327 returned, 0 died; strict 5,163 returned, 6,965 lost), and on validation it equals the frozen code's output in all 24 worlds.
- v3.1 never lost a rover that v3 kept, and never missed a strict return that v3 made (0 and 0 on all three test sets). It gave up 4 (untouched), 17 (six fresh) and 8 (Site23/42) of v3's lenient returns, the ones that needed a flat battery at the lander, and made more new ones.
- Prediction = outcome for 100% of v3.1 rovers in every unchanged world.
- Against the best simple planner by strict returned (naive) on the untouched sites: v3.1 +15.8 points, v3 +13.1 points, both at 4 of 4 sites.
- LM5 in June: 18 of 512 strict returns and 494 lost for both brains; nothing nearby is lit long enough.

## Self-learning: what happened, round by round

Stage D ran 36 rounds in 20 situations; 23 rounds sent a proposal to the real-episode gate; **9 were accepted, 8 in round 1 and 1 in round 2** (LM2 dusty panels: energy margin 0 -> 0.10). Every accepted change satisfied the gate rule and would also have been accepted by v3's original gate. No rejected proposal was sent twice (the memory fix works).
- **Light 12 h late (the case v3 could not learn):** at all four sites the rover fitted a shift of exactly +12 h from its own solar readings. At LM2, LM3 and LM4 the shifted forecast was accepted in round 1: lenient lost 53.9% -> 0%, 56.2% -> 0%, 11.7% -> 0%; strict lost 81.6% -> 72.3%, 71.5% -> 60.5%, 65.6% -> 53.1%. At LM5 the gate rejected it (almost no trip is survivable there in June, so the gain was nil): NO CHANGE.
- **Blocked route:** 4 of 4 sites added the obstacle they had seen to the map in round 1 (as v3 did); lenient lost 2-7% -> 0%.
- **Dusty panels:** still weak. LM4 accepted the correct solar measurement (x 0.600) in round 1 and improved; LM2 accepted a larger margin in round 2 but the gain on the score starts is inside the noise (NO CHANGE); LM3 and LM5 accepted nothing and LM3 keeps losing 19.5% under the lenient rule. The gate's "gain above 2 standard errors on 128 paired episodes" test remains the limit.
- **Unchanged world:** nothing accepted in 8 rounds, nothing changed.
- **Light 8 h early (off the example the layer was built on):** the rover fitted exactly -8 h at all four sites, so the detection generalises to another shift. But only one round ran per site and none was accepted: at LM4 the shift went to the gate and every count improved (strict lost 87 -> 81, lenient lost 9 -> 0 of 128) but the gain (+4.2) was below 2 standard errors (5.5); at LM2 and LM3 a charge-threshold proposal scored higher in imagination, went to the gate instead and was rejected; at LM5 nothing beat the current brain. The second round, where the shift would have been the next proposal, never ran because of the cap. Result: not evaluable; no improvement shown.
- Honest summary of "recursive": one useful change per situation, almost always in the first round; the layer then stops. It is a gated belief-correction loop, not open-ended self-improvement. It spends real episodes and loses rovers while it learns.

## What went wrong / what to know

- **Stage D ran with a 30-minute cap, not the 45 minutes of Amendment 2.** I amended the plan but left `--cap-minutes 30` in the launcher. All 20 situations started; the cap stopped the four early-light situations after their first round. They are reported as not finished (NO CHANGE) and were not rerun.
- **Launcher fault at the single launch (14:13):** the job runner refused the stage A/B/C command lines as too long, so no A/B/C job started and stage D ran first. A/B/C were started once at 14:32 with the same command lines (Amendments 3 and 4; the CPU budget had been raised to 14 cores). No stage ran twice; no test number had been read. A stray marker file `ABC_DONE` (14:13) is left from the failed step.
- **With a wrong belief v3.1 loses more rovers under the lenient rule than v3** (late forecast 32.2% against 19.9%; weak battery 9.0% against 6.1%; dusty panels 11.8% against 10.4%): it leaves the lander to shelter on lit ground, and when the forecast is wrong the battery runs out away from the lander. Under the strict rule those rovers were lost either way. Known from the development smoke before launch and deliberately not tuned.
- Timestamps in the plan headings are 1-2 minutes ahead of the machine clock in three places (noted in the amendments).
- One unit-test fixture was wrong at first (the target lay on the lit side) and was corrected; the code was not.
- The diagnosis ran in parallel with the design, not before it; the design did not change after its numbers came in.

## Caveats

- Simulation only: kinematic, one decision per hour; a "target" is a PSR edge, not confirmed ice. Unchanged-world numbers assume a perfect map and forecast.
- "Alive" means the battery never reached zero within the 300-hour episode. A real rover on lit ground would still face thermal limits not modelled here.
- The strict rule is our reading of "a flat battery is a lost rover"; both rules are shown.
- Changed worlds are synthetic and each has a matching proposal type (map, solar output, forecast shift, battery capacity); no change without a matching proposal type was tested, so this is not evidence of handling arbitrary new situations. Weak battery was tested before learning only.
- Stage C and D use one date (2027-06-01), four neighbouring sites, 256 rovers; self-learning ran 2 rounds, not 4.
- The gate's gain is measured on the strict score (not comparable in strictness with v3's gate on that one test); all of v3's count conditions were kept and two strict ones added.
- The mini's code folder is a copy, not a git checkout; the commit (48b459c) is what the launch log printed.

## Claims for the website

Safe to say:
- "Rover brain v3.1 adds a survival mode: when the lander will be dark for too long, the rover plans a route to nearby ground that stays lit and waits there. On four lunar south-pole sites no rover experiment had touched, counting every flat battery as a lost rover, it lost 53.9% of rovers against 61.0% for v3 and completed 33.7% of trips against 31.0%; the simple planners lost 78-84%."
- "Most of the rovers it saved stayed alive on lit ground without completing the trip (419 of 583); 164 were extra completed trips."
- "Most losses under the strict rule are probably unavoidable for this rover (100 Wh battery, 50 hours of idling) with random landing times: v3.1 still loses more than half."
- "The self-learning layer can now notice from its own solar readings that the light forecast is shifted in time. With a forecast 12 hours late it found the 12 hours at 4 of 4 sites and, after its real-episode gate accepted the fix at 3 of them, losses under the simulator's rule fell from 12-56% to 0."
- "With a blocked route it corrected its map at 4 of 4 sites. In unchanged worlds it changed nothing."

Not safe to say:
- "v3.1 solves the survival problem" or any "0% lost": 53.9% are still lost under the strict rule.
- "v3.1 is more robust than v3 to a wrong forecast": before learning it loses more rovers under the simulator's rule (32.2% against 19.9%).
- "Self-learning handles new, unrelated situations": every tested change has a hand-written proposal type; dusty panels improved at 1 of 4 sites; the 8-hour-early forecast was detected but no fix was accepted in the single round that ran.
- "Recursive self-improvement": 9 accepted changes in 36 rounds, 8 of them in round 1.
- "The network generalises": there is no network; the planner is recomputed on each site's map.
- Any comparison of a strict number with a lenient one, or "finds ice".

## Recommendation

Present **v3.1** as the rover brain, quoted under the strict rule with the caveats above: on every test set it is at least as good as v3 on strict and lenient returned and on strict lost, with zero rovers made worse in unchanged worlds, and its self-learning layer fixes the shifted-forecast case v3 could not. Keep v3's numbers beside it, and state the one place v3.1 is worse (lenient losses under a wrong forecast before learning).

## Files

MacBook, `~/Documents/Ojas/Hackathon/`:
- `records/rover_v31/`: `ROVER_V31_PLAN.md` (plan + 4 amendments), `ROVER_V31_REPORT.md` (this file), `eval.json` (every number, per world and per self-learning situation), `tables.md`, `diagnosis.md`, `diagnosis.json`, `scripts_diagnosis/`, `STATUS.md`
- `records/agents/rover-v31-lead.md` (attempt log), `rover31-diagnosis.md`, `rover31-plan-reviewer.md`, `rover31-numbers-checker.md`
- code: `work/adinfinitium-rover31` (git branch `rover-v3.1`, from `rover-v3` 6742c7b; not pushed): `worldmodel/rover3/brain.py` (stay-alive search, shelter plans, strict-first choice), `run.py` (stages, v3.1 self-learning layer and gate), `report31.py`, `worldmodel/dream/changes.py` (late / early light, weak battery), `tests/test_rover3.py` (8 tests: 4 old, 4 new), `ops/rover31/`

Mac mini:
- `~/adinfinitium_addendum5_rover_v31.zip`: plan, report, `eval.json`, `tables.md`, diagnosis, `rover_brain_v31_config.json`, `final/` (every raw stage file and log), `validation/`, `development_smoke/`, `code/` (package, launchers, tests, `rover_v31.diff` against rover-v3)
- working folder `~/adinfinitium-rover31/`: `code/` (evaluated copy), `code-v3/` (frozen v3 copy), `data/` (4 new tiles LM2-LM5 + links), `runs/final/`, `runs/val/`, `runs/val_frozen/`, `runs/dev/`, `diag/`
- Freeze check `python -m ops.freeze --check` in `~/adinfinitium-run/adinfinitium/`: ALL UNCHANGED at the start (13:35), before the launch (14:13) and at the end (15:05).

How to run one world (mini): `. ~/adinfinitium-rover31/env.sh && cd ~/adinfinitium-rover31/code && $PY -m worldmodel.rover3.run eval --sites LM4 --dates 2027-03-01 --n 512 --seed 1234 --threads 1 --out /tmp/x.json` (about 1.5 minutes on one core).
