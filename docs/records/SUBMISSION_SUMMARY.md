# Adinfinitium: submission summary

NASA Space Apps 2026, Lunar Landing Site Navigator, plus a lunar lander pilot.
Final version, Sun 15 Nov 2026. Written about 12:30 IST and restructured during the afternoon as the improvement work
(addendum 4), the lander job, rover v3.1 (addendum 5) and its early-light completion run finished. No result slot is left open.

Every number in this document is copied from a named file (section 9). Paths starting `records/` are under
`~/Documents/Ojas/Hackathon/` on the MacBook. Paths starting `runs/` are under `~/adinfinitium-run/adinfinitium/` on the Mac mini.

Team rules, verbatim: "Only fix genuine crashes, never tune models to look better, record every failure honestly, and do not
change anything after the freeze step."

Words used: "held-out" = a site never used for training or for choosing a checkpoint. "Reached ice" = reached a dark-and-cold
shadow-edge target, not confirmed ice. All rover and lander results are simulation only.

## 1. Summary

We built a chain of models for choosing and reaching a landing site at the lunar south pole from NASA LOLA 5 m terrain. One
unattended overnight run trained four networks and scored each against a physics backup on two sites it never saw (Site23,
Site42); the run was then frozen and nothing in it was changed. Afterwards we attacked our own weak points in pre-registered
addenda, and several headline results became more modest. What stands: the hazard network and the light network beat their
physics backups on both held-out sites and on six further fresh sites, although the hazard network is much less accurate on
flat fresh ground (F1 0.757 pooled against 0.965 and 0.917). What changed: for terrain, a classical kriging fill has 40% to
60% of the error of our 49.9 M parameter diffusion network on the same hidden laser pixels, so kriging is the terrain estimator
we present and the network is reported as an experiment that a classical method beats. The two learned rover drivers (v1, v2)
FAILED and are reported as failed. The rover brain we present is v3.1: a scripted planner and executor with a gated
self-learning layer and no neural network. Counting every empty battery as a lost rover (the strict rule), on four sites no
rover experiment had touched it returned 33.7% of rovers and lost 53.9%, against 31.0% and 61.0% for the earlier brain v3 and
17.9% and 81.8% for the best simple planner. It still loses more than half its rovers. An earlier "lost none" for v3 came
from a simulator rule that forgives an empty battery next to the lander. The lander pilot (a learned
correction on classic guidance) lands 96% to 98% of missions pooled on maps never used in its training (94.3% to 99.1% per
map and condition). It is slightly better than
classic guidance on the three unseen maps pooled under normal failure rates (0.9839 against 0.9766, p = 0.0186) and on
the Site23 final exam it ties classic under normal rates (0.9795 both) and is not better under harsh rates (0.9629 against
0.9648, not significant). Today's self-improvement job changed nothing: its gate rejected all 7 candidates.

## 2. What was built

```
NASA data ............ LOLA 5 m terrain + slope + error map, NASA 20 m shadow (PSR) map, SPICE Sun/Earth directions
   | (1)
TILES ................ 1024 x 1024 px windows at 5 m; 85 of 89 planned windows on disk; shadow mask on 85 tiles
   | (2)                      | (3)                           | (4)
HAZARD U-Net          LIGHT skyline transformer        TERRAIN fill between laser points
60,361,561 params     38,279,809 params                presented: classical kriging (no learned parameters)
unsafe-pixel map      horizon height, 360 directions   experiment: diffusion U-Net, 49,943,681 params
   | (5)                      : (6) not wired                 : (7) not wired
ROVER BRAIN v3.1 ..... scripted planner -> scripted executor -> gated self-learning; 0 learned parameters
   : (8)
LANDER PILOT ......... learned residual on classic ZEM/ZEV guidance (separate package)
```

What each arrow really is (solid `|` = wired and run; dotted `:` = not wired, or not verified):

| arrow | status | evidence |
|---|---|---|
| (1) NASA data -> tiles | Wired, run in the frozen run. 4 training windows are missing. | `records/RESULTS_FOR_TEAM.md` section 2 and section 5 item 3 |
| (2) tiles -> hazard U-Net | Wired. The network's inputs include the backup's gap-filled slope and NASA's error map. | `work/adinfinitium/worldmodel/hazard/features.py:20`, `:59`; RESULTS section 8 |
| (3) tiles -> light network | Wired for its own training and evaluation. Input is the same grid averaged 4x. | RESULTS sections 3.2 and 8 |
| (4) tiles -> terrain diffusion | Wired for its own training and hide-and-check evaluation. | RESULTS section 3.3 |
| (5) hazard U-Net -> rover | **Wired.** The rover world loads the hazard checkpoint and uses its probability map. It sets route cost and a score penalty; it cannot kill a rover. | `work/adinfinitium-rover3/worldmodel/rover/world.py:52-76` and `:135`; `eval.json` field `hazard_sources`; `records/benchmarks/BENCHMARKS.md` section 5 |
| (6) light network -> rover | **Not wired.** The rover's light forecast is the exact physics ray-cast (`raycast_skyline_torch` from `worldmodel.light.teacher`), not the light network. | `work/adinfinitium-rover3/worldmodel/rover/world.py:105-124`; BENCHMARKS section 5: "The light forecast is a physics ray-cast, not our light network." |
| (7) terrain diffusion -> anything | **Not wired.** A search of the hazard, light, rover and rover3 code found no use of the terrain model. Its output is labelled "GENERATED - NOT OBSERVED". | search of `work/adinfinitium/worldmodel/{hazard,light,rover}` and `work/adinfinitium-rover3/worldmodel/rover3` for `worldmodel.terrain`; RESULTS section 3.3 |
| (8) rover <-> lander | **Not wired (separate package).** The rover's lander position is the centre of its world, not a landing point from the lander pilot. The lander pilot's camera sees the slope-rule backup hazard map, not the hazard U-Net. | `work/adinfinitium-rover3/worldmodel/rover/world.py:145`; `adinfinitium_lander/ops/lander_mini/START_HERE.md` section 2 |

Hand-offs run in one end-to-end pass on an unseen site, with timings and a figure: section 2a.

Splits. Train: Site01, 04, 06, 07, 20, Haworth, Shoemaker, DM2. Validation: Site11. Held-out: Site23 (13 windows), Site42
(6 windows). Six fresh sites, used in no training or checkpoint choice: DM1, LM7, NPA, NPB, NPC, NPD. Four more sites, LM2, LM3, LM4 and LM5, were
used only for the final test of rover v3.1 ("untouched sites"). Rover v3 was first run on them;
addendum 4 then scored hazard, light and terrain there, and the end-to-end test ran on DM1.

Order of work. The overnight run `macmini_night` (14 stages, all `ok` on the first attempt, "87 checks, 0 FAIL, 0 WARN") is
FROZEN. Everything after it is an addendum and changes nothing in it: addendum 1 (rover v2 + self-learning), addendum 2
(terrain band calibration), addendum 3 (rover v3), benchmarks, addendum 4 (the improvement work on our weak points), the
end-to-end test, the lander job (finished 15:02 IST), and addendum 5 (rover v3.1, finished 15:05 IST).

## 2a. End-to-end test on an unseen site

One script ran the chain in the order the code wires it, on a site no model was trained or tuned on, with the three frozen
checkpoints (their SHA-256 equals `runs/FROZEN.json`). It is a does-it-work-together test, not a benchmark: nothing was tuned
or chosen from it. Site DM1 is the first of the six fresh sites by name and the default site of the test script; the date 2027-06-01 is the
default date of the rover v3 code.

- **Verdict: PASS** (baseline run, rover v3 code, while the machine was shared). 56 checks, 0 failed; 1,293.9 s on the MacBook CPU with 2 threads; peak memory 7,440 MB.
- **What ran, in order:** LOLA tile (1024 x 1024 px, 5 m) -> hazard U-Net -> skyline network and the exact ray-cast on the
  rover's 64 x 64 light grid -> SPICE Sun directions and light geometry -> rover world -> rover brain v3 and the three
  planners (512 rovers, 4 dates) -> one recorded traverse -> terrain diffusion on a 128 px crop (side branch).
- **Hand-offs checked** (shape, unit, NaN, value range at each): the hazard map is a 1024 x 1024 probability in [0, 1] and
  equals the map the rover v3 evaluation used on the Mac mini (largest difference 2.98e-08, 0 pixels change side at 0.5). The
  rover world's hazard is that map's centre crop, bit for bit. The rover's light equals the exact ray-cast light, bit for bit.
- **What is really wired.** The rover reads the hazard U-Net map (route cost and score penalty). The rover does NOT read the
  light network: it reads the physics ray-cast. The light network ran on the same grid and gives the same kind of array, so it
  could be connected, but no code does it. The terrain diffusion output is read by no later stage. The lander pilot is a
  separate package. No hand-off was found broken and no stage is wired to a stub; no integration defect needed a fix.
- **Rover result on this world** (DM1, 2027-06-01, simulation, the simulator's own rule): v3 returned 338 of 512 rovers; 132
  stayed at the lander. The simulator counted no rover as lost, but its rule forgives an empty battery within 20 m of the
  lander; under the strict rule of section 3 many of these rovers count as lost. The reached / returned / died rates of all five methods on all four DM1 dates equal `records/rover_v3/eval.json`
  exactly.
- **Sanity numbers (one site, not a benchmark):** the network skyline differs from the exact ray-cast by 0.266 deg on average
  on the rover grid; the coarse ray-cast backup differs by 0.352 deg.

Time per stage, MacBook CPU, 2 threads, machine shared with other jobs (so times vary between runs):

| stage | seconds | note |
|---|---|---|
| checkpoints: SHA-256 | 1.5 | |
| terrain tile load | 0.4 | |
| hazard U-Net, whole tile | 235.7 | slowest stage on the rover's path |
| light network, 4,096 cells | 610.9 | not on the rover's path; its speed-up is the improvement team's item |
| exact ray-cast skyline, 4,096 cells | 5.1 | the light the rover uses |
| SPICE directions + light geometry | 0.9 | |
| rover world | 4.7 | |
| rover v3 + 3 planners, 512 rovers, per date | 47.0 to 56.0 | 4 dates |
| recorded traverse | 38.7 | |
| terrain diffusion, 128 px crop, 2 versions | 182.6 | side branch |

Speed-ups (results unchanged; made in a separate copy of the code, `work/final/code`; the frozen run and the rover v3
package are untouched):

| stage | before | after | proof that the result is unchanged |
|---|---|---|---|
| Rover evaluation, 4 DM1 worlds: the light sampler read a whole 64 x 64 slice per rover to get one value; it now reads the 4 neighbouring cells | 168.0 s | 69.8 s | sampled values differ by at most 1.19e-07; every rover has the same outcome and the same return (difference 0.0) in all 4 worlds; rates equal `records/rover_v3/eval.json` |
| Hazard map for the rover: the U-Net ran on all 16 windows of the tile; the rover needs the central 4 | 130.3 s | 61.9 s | the rover-world crop is identical bit for bit (run back to back in one process) |

- A full re-run of the rover's path with both speed-ups passed 55 checks, 0 failed (266.97 s without the light-network and
  terrain stages; peak memory 5,366 MB).
- Not done: running the hazard U-Net on the Apple GPU instead of the CPU (the GPU was in use by another job).
- **Final re-run with rover brain v3.1** (Mac mini, own folder `~/adinfinitium-final/`, a copy of the evaluated rover-v3.1
  code, fresh cache, CPU with 6 threads): PASS, 65 checks, 0 failed, 432.41 s, peak memory 14,622 MB. Stage seconds: hazard
  U-Net 20.7; light network 209.7; rover v3.1, v3 and three planners 31.5 to 32.6 per date; terrain diffusion crop 48.5. On
  all four DM1 dates the reached, returned, died, strict-returned and strict-lost rates of every method equal
  `records/rover_v31/eval.json` exactly. The same run on the MacBook a few minutes earlier (338.42 s, 6 threads) gave the same
  65 checks and identical rover rates (`records/final/macbook_run_v31/`).
- **Rover result on DM1, 2027-06-01, strict rule:** v3.1 returned 232 of 512 rovers, kept 19 alive without a completed trip and
  lost 261 (51.0%); v3 lost 58.6% and the naive planner 70.3%. (The baseline and optimised runs above used rover v3.)
- Figure for the presentation: `records/final/e2e_demo_DM1_v31_strict.png` (terrain, hazard map, both illumination maps, v3.1
  routes, strict-rule outcome for v3.1, v3 and the planners; the caption states the rule and what is model output and what is
  physics). The earlier figure `e2e_demo_DM1_v3_lenient_superseded.png` showed v3 under the simulator's lenient rule and is
  superseded.

## 3. Headline results

One row per model: the result on sites the model never used, against its baseline. Detail rows follow in section 3.1.

| model | parameters | metric | result on unseen sites vs baseline | interval or significance | caveat |
|---|---|---|---|---|---|
| **Hazard U-Net** | 60,361,561 | F1 for unsafe pixels (higher is better) | Site23 0.965 vs 0.899; Site42 0.917 vs 0.567; six fresh sites pooled 0.757 vs 0.292 (slope-rule backup) | none computed; ahead in 19 of 19 test windows and on 6 of 6 fresh sites | much weaker on flat fresh ground (per site 0.239 to 0.823); most of the truth is NASA's interpolation |
| **Light skyline transformer** | 38,279,809 | skyline error, degrees (lower is better) | Site23 0.534 vs 0.675; Site42 0.270 vs 0.342; six fresh sites pooled 0.2426 vs 0.3205 (coarse ray-cast backup) | none computed; ahead in 19 of 19 test windows and on 6 of 6 fresh sites | about 200 times slower than the backup; not connected to the rover |
| **Terrain fill: classical kriging** (the estimator we present) | 0 learned | RMSE on hidden laser heights, metres (lower is better) | Site23 0.94; Site42 1.11; six fresh sites 0.97. Our diffusion network (49,943,681 parameters, an experiment): 1.83; 2.80; 1.62 | none computed; kriging ahead in 19 of 19 test windows and on 6 of 6 fresh sites | the classical method beats our network; the test favours interpolation (hidden pixels sit next to shown ones) |
| **Rover brain v3.1** (scripted, v3 beside it) | 0 | returned / lost under the strict rule, four untouched sites, simulation | v3.1 33.7% / 53.9%; v3 31.0% / 61.0%; best simple planner (naive) 17.9% / 81.8% | v3.1 minus v3, lost: -7.1 points [-13.8, -3.5], lower at 4 of 4 sites; returned +2.7 points [+1.9, +3.4] | still loses more than half; most rovers saved only stay alive on lit ground; the four sites are neighbours, so the interval is rough |
| **Lander pilot** (learned residual on classic guidance) | about 15 million | landing success, simulation | three maps unseen by the pilot, normal rates: 0.9839 vs 0.9766 (classic guidance); Site23 final exam: 0.9795 vs 0.9795 | exact McNemar p = 0.0186 (unseen maps, normal rates); p = 0.1843 under harsh rates; p = 1.000 on Site23 | the gain comes from one map; today's self-improvement job accepted 0 of 7 candidates |

Failed and reported as failed: rover v1 (PPO network) and rover v2 (imitation network), 11,377,354 parameters each (section 3.1).

### 3.1 Detail rows

Unseen sites, ours against the baseline. "Fresh sites" are DM1, LM7, NPA, NPB, NPC, NPD (one centre tile each), never used to
train or select these models. Intervals are shown where the source has them; the hazard, light and terrain sources give none.

| model | parameters | metric | ours vs baseline on unseen sites | verdict | source |
|---|---|---|---|---|---|
| Hazard U-Net | 60,361,561 | F1, all pixels (higher is better) | Site23 0.965 vs 0.899; Site42 0.917 vs 0.567 | beats backup | `records/RESULTS_FOR_TEAM.md` 3.1 |
| Hazard U-Net | same | F1, laser-measured pixels only | Site23 0.949 vs 0.832; Site42 0.926 vs 0.677 | beats backup | RESULTS 3.1 |
| Hazard U-Net, six fresh sites (addendum 4) | same | F1, all pixels, pooled | 0.757 vs 0.292; ahead on 6 of 6 sites; per site 0.239 to 0.823 | beats backup, but much weaker than on Site23 / Site42; poor on the four flat sites | `records/weak_flags/REPORT_hazard.md` |
| Light transformer (frozen checkpoint) | 38,279,809 | skyline MAE, mean of windows (lower is better) | Site23 0.534 deg vs 0.675 deg; Site42 0.270 deg vs 0.342 deg | beats backup; about 200 times slower than the backup | RESULTS 3.2; `records/weak_flags/REPORT_light.md` |
| Light transformer, six fresh sites (addendum 4) | same | skyline MAE, pooled | 0.2426 deg vs 0.3205 deg (24.3% lower); ahead on 6 of 6 sites | beats backup | REPORT_light, "Fresh sites" |
| **Terrain: classical kriging (the estimator we present)** | 0 learned | hide-and-check RMSE (lower is better) | Site23 0.94 m; Site42 1.11 m; six fresh sites 0.97 m. Diffusion network on the same pixels: 1.83 m; 2.80 m; 1.62 m | kriging beats the network on both held-out sites and on 6 of 6 fresh sites | `records/weak_flags/REPORT_terrain.md` |
| Terrain: kriging "90%" band | 0 learned | coverage / mean width | Site23 85.9% / 2.31 m; Site42 88.2% / 2.76 m; fresh sites 92.8% / 2.82 m. Network calibrated band: 84.9% / 7.06 m; 96.9% / 15.51 m; 96.6% / 10.48 m | similar coverage, much narrower; below 90% on Site23 and Site42 | REPORT_terrain |
| Terrain diffusion network (experiment) | 49,943,681 | hide-and-check RMSE | frozen run: Site23 1.84 m, Site42 2.82 m, against the old gap-fill backup 17.92 m and 24.82 m | beats only a weak backup; a classical method beats it; combining it with kriging did not help | RESULTS 3.3; REPORT_terrain |
| Terrain network raw "90%" band | same | coverage (about 90% is the goal) | Site23 21.9%; Site42 37.3% | **FAILS coverage**; calibrated in addendum 2 (scale 8.232 fitted on Site11) | RESULTS 3.3; `records/terrain_calibration/TERRAIN_CALIBRATION_REPORT.md` |
| Rover v1 (PPO network) | 11,377,354 | returned, 8 held-out worlds, simulation | 0.0% vs 9.3% (naive planner); reached ice 1.7% vs 19.6%; died 95.5% | **FAILED** | RESULTS 3.4 |
| Rover v2 (imitation network) | 11,377,354 | returned, same 8 worlds, simulation | 0.012 vs 0.093 (naive planner); died 0.953 | **FAILED** (pre-registered verdict FAILURE) | `records/ADDENDUM_AFTER_FREEZE.md` 2.4, 2.5 |
| Rover v3 (the earlier brain; scripted, no network), STRICT rule | 0 | returned / lost, six fresh sites, 24 worlds, simulation | returned 42.0% [31.2, 51.1] vs 29.8% [19.3, 38.0] (naive, the best planner under this rule); lost 56.7% [48.1, 67.3] vs 69.4% [61.1, 79.8]; lead in returned +12.2 points [10.3, 13.6], 6 of 6 sites | PASS by the pre-registered rule; not evidence of learning (section 4) | `records/weak_flags/REPORT_rover.md` section (a) |
| Rover v3, simulator's lenient rule (secondary) | 0 | returned / lost, same worlds | returned 59.6% [46.4, 71.4] vs 32.9% [18.4, 46.2] (cautious planner); lead +26.7 points [19.1, 33.7], 6 of 6 sites; the simulator counts 0 of 12,288 v3 rovers as lost because it forgives an empty battery within 20 m of the lander | PASS (pre-registered primary of addendum 3); the "none lost" does not survive the strict rule | `records/rover_v3/ROVER_V3_REPORT.md` section A; REPORT_rover |
| Rover v3, Site23 + Site42 (previously seen), strict rule | 0 | returned / lost, 8 worlds | 32.4% / 38.6% vs 9.2% / 86.9% (naive planner); no interval (2 sites) | ahead at 2 of 2 sites (weaker evidence) | REPORT_rover |
| Rover v3, wrong light forecast (12 h late), fresh sites | 0 | returned / lost, 1,536 rovers | lenient rule: 30.9% [16.5, 41.3] returned, 27.3% [15.0, 39.4] lost; strict rule: 24.8% [12.6, 36.3] / 74.4% [63.1, 86.7]. Best planner not told (naive): 20.4% / 79.6% | ahead of the planners on the 6-site mean; loses rovers; self-learning helped at 0 of 6 sites | REPORT_rover section (b) |
| **Rover v3.1 (the brain we present), STRICT rule, four untouched sites LM2, LM3, LM4, LM5** | 0 | returned / lost, 16 worlds, 8,192 rovers per method, simulation | v3.1 33.7% [26.0, 42.2] / 53.9% [46.8, 59.5]; v3 31.0% [24.3, 38.9] / 61.0% [50.4, 72.8]; naive planner 17.9% [15.4, 20.6] / 81.8% [79.1, 84.4]. v3.1 minus v3: lost -7.1 points [-13.8, -3.5], 4 of 4 sites; returned +2.7 points [+1.9, +3.4]. v3.1 minus naive, returned: +15.8 points, 4 of 4 sites | PASS (pre-registered primary of addendum 5); v3.1 cannot score below v3 here if the code is right, so this shows how much lit ground is within reach, not robustness | `records/rover_v31/ROVER_V31_REPORT.md` section A |
| Rover v3.1, lenient rule, same worlds (secondary) | 0 | returned | 48.1% [32.8, 63.4] vs v3 46.7% [30.8, 62.5]; the simulator counts no rover of either brain as lost | reported, not a criterion | ROVER_V31_REPORT section A |
| Rover v3.1, six earlier fresh sites and Site23 + Site42 (previously seen), strict rule | 0 | returned / lost | six fresh: 43.9% / 54.2% vs v3 42.0% / 56.7% (lost -2.5 points [-3.7, -1.3], 6 of 6 sites). Site23 + Site42: 38.3% / 30.0% vs v3 32.4% / 38.6% (2 sites, no interval) | PASS on both (weaker evidence: sites seen before) | ROVER_V31_REPORT sections B1, B2 |
| Rover v3.1, light forecast 12 hours late, before learning, untouched sites | 0 | returned / lost, 256 rovers per world | strict: 12.2% / 81.3% vs v3 11.2% / 87.1%. Lenient: 13.5% / 32.2% vs v3 17.3% / 19.9% | better than v3 under the strict rule; **worse than v3 under the lenient rule** | ROVER_V31_REPORT, changed worlds |
| Lander pilot (learned residual on classic guidance; shipped pilot, unchanged today) | about 15 million (as stated in the package; no exact count on record) | landing success, 3 maps unseen by the pilot (Site42, Shoemaker, DM2) pooled, 2,304 paired missions per condition, simulation | normal rates: 0.9839 [0.978, 0.988] vs classic 0.9766 [0.970, 0.982]; difference +0.0074 [0.002, 0.013]; only pilot landed 32, only classic 15; exact McNemar p = 0.0186. Harsh rates: 0.9692 [0.961, 0.975] vs 0.9631 [0.955, 0.970]; +0.0061 [-0.002, 0.014]; 55 / 41; p = 0.1843 | small significant gain under normal rates; not significant under harsh rates; the gain comes from Site42 | `records/lander/RESULTS_SUMMARY.md` section 5 |
| Lander pilot, Site23 final exam (flown once) | same | landing success, 1,024 paired missions per condition | normal: 0.9795 [0.969, 0.987] vs 0.9795 [0.969, 0.987]; 11 / 11; p = 1.000. Harsh: 0.9629 vs 0.9648; difference -0.0020 [-0.014, 0.010]; 19 / 21; p = 0.875 | tie under normal rates; not better under harsh rates | RESULTS_SUMMARY section 6 |
| Lander self-improvement job (today) | same | candidates accepted by the Site11 gate (1,024 paired missions; current pilot 990 landed, classic 965) | 0 of 7 accepted (4 authority rules, 3 training rounds); curve flat at 0.9668 | **nothing improved**; final pilot = shipped pilot (same SHA-256 before and after) | RESULTS_SUMMARY section 3 |

Strict rule for the rover: a rover is LOST when its battery is empty at the end of any step, anywhere. It RETURNED when it
returned under the simulator's rule and was never empty up to and including the arrival step. The simulator's own (lenient) rule
does not count an empty battery within 20 m of the lander as a loss: the rover sits there and wakes when sunlight returns.

What the improvement work (addendum 4) found, item by item. Each was pre-registered before its held-out number, chosen on
training sites and validation Site11 only, run once per pre-registered stage, and re-derived by an independent checker (`records/weak_flags/CHECK.md`:
no wrong headline number).

| item | result |
|---|---|
| Light: finish training to 6,000 steps | Done: 6,000 of 6,000 steps. Held-out error moved from 0.4503 deg to 0.4489 deg (backup 0.5700 deg): -0.0014 deg, below the 0.005 deg fixed beforehand as a real change. On the fresh sites the new checkpoint is 0.2439 deg against the frozen 0.2426 deg. The frozen 3,258-step checkpoint stays the one presented. The added steps ran at 1.6% of the peak learning rate or less, so this does not say what an uninterrupted run would reach. |
| Light: inference speed-up | None found that keeps the output. Bigger batches and one transfer: 0.99x. Half precision was 1.3 to 1.4 times faster but changed the output by up to 0.327 deg and was rejected. The network stays 199 to 266 times slower than the backup on the MacBook (7.41 s against 0.033 s per window, 227 times, on the Mac mini). |
| Terrain: strong classical competitors on the same hidden pixels | Seven of nine classical methods beat the network on both held-out sites, including plain linear interpolation. Kriging: 0.94 m (Site23) and 1.11 m (Site42) against the network's 1.83 m and 2.80 m. A plain average of network and kriging is worse than kriging (1.22 m and 1.69 m); the fitted weight on the network is 0.0064. |
| Hazard: six fresh sites | F1 0.757 against 0.292 for the backup, pooled (0.782 against 0.352 on laser-measured pixels); ahead on 6 of 6 sites. On the four flat sites (NPA, NPB, NPC, NPD) it misses 38.4% to 74.9% of the unsafe pixels. No gain without retraining was adopted: the best threshold on Site11 is the frozen 0.5, and test-time augmentation gained 0.0011 on Site11, below the 0.005 needed. |
| Rover v3: strict-loss scoring | v3 returned 42.0% and lost 56.7% on the fresh sites (lenient: 59.6% and 0.0%). Still ahead of the best planner by +12.2 points at 6 of 6 sites. A permitted fix (refuse plans whose battery reaches empty) barely helped: 42.8% returned, 55.5% lost. |
| Rover v3: wrong light forecast | With true sunlight 12 hours later than forecast, v3 returned 30.9% and lost 27.3% under the lenient rule. Self-learning: 0 of 6 sites IMPROVED, 6 NO CHANGE. |
| Rover v3.1 (addendum 5, after addendum 4) | Adds a stay-alive route search (drive to ground that stays lit and wait there), a strict-first choice of plan, and two new self-learning proposal types (light forecast shifted in time; battery capacity). Results in the rows above and in section 5. |

Lander, in full (`records/lander/RESULTS_SUMMARY.md`, `records/lander/LANDER_DIAGNOSIS_AND_RECOMMENDATIONS.md`):

- **The gate rejected all 7 candidates.** Four authority rules (pilot at 80% authority; classic flies after an engine failure
  with hand-over below 5-15 m; hand-over to classic below 5-15 m; pilot at 60% authority) landed 989 to 991 of 1,024 gate
  missions against the current pilot's 990, and each lost 12 to 14 missions that classic or the current pilot lands. Three
  training rounds landed 977 to 981 and lost 21 to 25. The gate accepts only a candidate that loses no such mission.
- **The training rounds were too short to matter:** 4 to 5 updates each (12 minutes per round). All three came back worse than
  the pilot they started from.
- **Per map, unseen benchmark (768 missions each):** Site42 normal 0.9701 vs 0.9505 (22 / 7, p = 0.008), harsh 0.9427 vs 0.9297
  (p = 0.220); Shoemaker normal 0.9909 vs 0.9896 (6 / 5, p = 1.000), harsh 0.9831 vs 0.9805 (p = 0.832); DM2 normal 0.9909 vs
  0.9896 (4 / 3, p = 1.000), harsh 0.9818 vs 0.9792 (p = 0.824). The whole pooled gain under normal rates comes from Site42;
  Shoemaker and DM2 are ties.
- **The unseen-map numbers are the shipped ones.** The chain skipped the benchmark step because the pilot did not change. One
  extra replication on the Mac mini after the final exam reproduced the shipped table byte for byte; it is a confirmation, not
  new evidence.
- **Diagnosis, in three lines.** Classic guidance already lands 94% to 99% depending on the map, and the pilot is a residual
  capped at 0.3 of full thrust, so it can only move the last few percent. The failures are borderline cases: of the pilot's 77
  crashes in 2,048 Site11 missions, 37 are sideways-speed touchdowns and 27 of those touch down at 0.51 to 0.78 m/s against a limit of
  0.5 m/s. The gate ("no lost mission out of 1,024") rejects any candidate whose marginal missions reshuffle, even a better one.
- **What a longer run needs:** training with the touchdown-speed penalty on from the start, for hours (the original run was
  sized for 600 updates; today's rounds had 5); a residual decided more often near the ground; a paired acceptance test on
  more missions, agreed before the run; longer rounds that keep the optimiser state.
- **Recorded deviations.** Step 3 replayed 200 of the 1,701 pooled missions (a fixed random sample) under 15 strategies instead
  of 8, and sent 4 candidates to the gate instead of 3. Step 4 ran 3 of 5 rounds. Two launches of step 3 failed (one too large
  to finish, one out of GPU memory) and wrote nothing. Nothing was changed in the simulator, the mission config, the failure
  rates, the seeds or the gate.
- **Disclosed.** The lander lead opened the shipped unseen-map benchmark file (`runs/lander/benchmark.json`) while designing
  the engine-failure rule. That rule was ranked on training-map missions, judged by the Site11 gate only, and rejected.
- "Unseen" for the lander means unseen by the pilot. Shoemaker and DM2 are training sites of the navigator networks (section 2,
  splits); that does not affect the lander pilot, which is a separate model.

Details that belong next to the table:

- Hazard: missed-hazard rate Site23 2.9% vs 13.7%, Site42 5.3% vs 56.9% (all pixels). The U-Net has the higher F1 in all 19
  test windows. On Site23 measured pixels the backup has the higher precision (0.961 vs 0.940). On the fresh sites the network
  misses 18.7% of unsafe pixels pooled, and 70.8% of its alarms are right (96.0% and 88.8% on Site23 and Site42).
- Hazard and the rover: the map the rover read on the fresh sites is the map scored here. On NPA to NPD it missed 38% to 75% of
  the unsafe pixels. In the simulator a hazard costs points and cannot lose a rover, so this affects route cost, not losses.
- Light: the network has the lower MAE in all 19 test windows, about 21% lower on both sites. The median error is lower in only
  9 of 13 Site23 windows and 1 of 6 Site42 windows. On the fresh sites the frozen checkpoint is better than the backup on 61% to
  76% of the cells of a window.
- Terrain: the test favours interpolation. Hidden pixels lie on the same laser tracks as shown pixels, usually a few pixels
  away. Filling wide gaps with no measurement nearby was never tested, for the network or for kriging.
- Terrain: the smoothed thin-plate spline is slightly ahead of kriging on the fresh sites (0.960 m against 0.972 m pooled).
  Kriging was fixed as the final estimator by the Site11 rule and was not changed.
- Rover v3.1, what the gain is made of (untouched sites): of the 583 rovers (of 8,192) that v3 lost and v3.1 did not, 164 are
  new completed trips and 419 stayed alive on lit ground without completing a trip (alive until hour 300 of the episode).
  In all, 1,020 v3.1 rovers were alive at hour 300 without a completed trip, with 80 Wh left on average. v3.1 lost no rover that v3 kept. The gain varies by site: LM4 -17.1 points, the other three -3.4
  to -4.1.
- Rover v3.1 and the unavoidable share: the diagnosis puts the unavoidable share of v3's strict losses between 8% (proven) and
  about 80%; an energy-only estimate puts it near two thirds (`records/rover_v31/diagnosis.md`).
- Rover v1: the scored file is the network after update 1 of 144. It never returned home in any world.
- Rover v2: v2 returned 48 of 4,096 rovers; the naive planner returned 380.
- Rover v3, where the strict losses come from (fresh sites): 4,801 rovers sat with an empty battery at the lander and never
  made the trip; 2,164 were counted "returned" by the simulator although their battery had been empty earlier while they
  waited at the lander. The planners are scored under the same rule on the same starts.

## 4. Generalisation evidence

- **Splits are by site.** No held-out site contributed a training or validation window (splits in section 2).
- **Spatial check.** 60 train + 6 validation tiles were checked against 19 test tiles with a 1,280 m margin: 0 excluded. The
  closest training tile to a test tile is 28.8 km away (Haworth r0c3 to Site23 r3c0); the closest validation tile 90.1 km.
- **Held-out vs validation** (Site11 is the validation site; it picked the hazard, light and terrain checkpoints):

| model | validation Site11 (ours vs backup) | held-out Site23 | held-out Site42 |
|---|---|---|---|
| Hazard F1, all pixels | 0.946 vs 0.862 | 0.965 vs 0.899 | 0.917 vs 0.567 |
| Light skyline MAE | 0.381 deg vs 0.491 deg | 0.534 deg vs 0.675 deg | 0.270 deg vs 0.342 deg |
| Terrain RMSE | 1.90 m vs 17.92 m | 1.84 m vs 17.92 m | 2.82 m vs 24.82 m |
| Rover v2 returned | 0.268 vs 0.407 (naive) | 0.012 vs 0.093, both sites together | |

- **Rover v2 did not generalise.** It agrees with the planner on 86-93% of return-leg steps on trained terrain and on 15-26% on
  unseen terrain. Site11 was an easy validation site: the naive planner returns 0.407 there and 0.093 on the test sites.
- **What "unseen" means for each number.**
  - Hazard, light, terrain: Site23 and Site42, scored once in the frozen run. Unseen terrain of the same region and product.
  - Terrain calibrated band: the scale was fitted on Site11 only and applied once to Site23 and Site42.
  - Rover v1, v2: 8 worlds = centre window of Site23 and Site42 on 4 dates, 512 rovers each. One evaluation each.
  - Rover v3 primary: six fresh sites, 24 worlds, 12,288 episodes per method. Site23 and Site42 had been seen twice before v3
    was designed, so they are reported as "previously seen" and are weaker evidence.
  - Self-learning scores (addendum 1 and v3): online scores on the site where the rover drives. They are not held-out scores.
  - Six fresh sites (addendum 4): hazard, light and terrain were each scored once on DM1, LM7, NPA, NPB, NPC and NPD, one centre
    tile per site. No model or setting had used these sites. Rover v3 had already driven on them.
- **Why the rover v3 PASS is not evidence of learning** (from the v3 report): in an unchanged world the v3 planner tries 36
  plans per rover inside an exact copy of the simulator (same map, same light forecast, same physics) and picks the best. Three
  of the 36 are the baseline planners themselves. So v3 could not lose to them there if the code is right. PASS means: planning
  the whole trip with a perfect forecast beats three short-sighted rules. It is not evidence of learning, and not evidence of
  robustness. The changed-world rows are the only test of a wrong model.
- **Fresh sites against the two original test sites.** Hazard: F1 0.757 pooled on the fresh sites against 0.965 and 0.917; the
  lead over the backup holds on 6 of 6 sites but the accuracy does not carry over to flat ground. Light: 24.3% below the backup
  on the fresh sites against about 21% on Site23 and Site42; ahead on 6 of 6. Terrain: kriging beats the network on 6 of 6
  fresh sites (0.97 m against 1.62 m), as on Site23 and Site42. The fresh sites are flatter and have fewer unsafe pixels, so
  "new place" and "flatter terrain" cannot be separated.
- **Changed worlds, rover not told** (fresh sites, 1,536 episodes per method, simulator's lenient rule): blocked route, v3
  returned 0.299 and died 0.074 against the best not-told planner 0.066 and 0.255; dusty panels, v3 0.413 and 0.139 against
  0.202 and 0.797. Against a planner that is told about the blocked route (0.237), v3 is below at 2 of 6 sites. Under the
  strict rule v3 returned 18.8% on the blocked route, not above the told planner (20.6%), and lost 78.8%; with dusty panels
  28.1% returned and 71.2% lost against 20.2% and 79.7%.
- **Four untouched sites for rover v3.1** (LM2, LM3, LM4, LM5): no rover experiment had used them. They are neighbours (10-30
  km apart; LM5's footprint is 4 km from the seen site LM7's), so they are four sites, not independent samples, and the
  intervals are rough. On the six earlier fresh sites the gain of v3.1 over v3 is smaller (strict lost -2.5 points).
- **Wrong light forecast** (true sunlight 12 hours late, rover not told): v3 returned 30.9% and lost 27.3% under the lenient
  rule, ahead of the best planner that is told (cautious, 23.4% returned) on the 6-site mean but below it at NPD.
- **The two rover test sets are not equally hard.** The fresh sites are flatter (median slope 2.5-5.9 deg against 9.5-11.6 deg).
- End-to-end pass on an unseen site (DM1): the frozen checkpoints ran together without error; section 2a. It is one site and
  one integration test, not a generalisation score.

## 5. Self-learning: what it does and what it does not

**Addendum 1: Dream-RSI-style loop on the failed rover v2 driver.**

- What it is: a key-free version of the record / replay / score / keep-best loop of Dream-RSI. The original needs paid LLM
  keys; none was used, so the original was not run. Strategy variants come from a fixed grid of 54 settings written by a person.
- What it does each round: drive 128 rovers x 100 steps; learn a small model from those steps; practise inside it; test up to 6
  candidates on 64 real episodes; accept the best only if its gain is above 2 standard errors; score on 256 separate starts.
- It ran by itself for 135 minutes (budget 140), with no training data and no human.
- Accepted: 2 of 23 rounds on Site23, 3 of 15 on Site42. The other 33 rounds were rejected.
- Mean return rose by 8.6 (se 0.8) and 17.4 (se 3.0). On Site42 deaths fell from 90.2% to 81.6%.
- **It did not learn the mission.** Returned stayed 0.0% in all 38 rounds. Reached ice went 0.0% -> 1.6% and 2.3% -> 2.0%.
- The report prints "MET" for both situations. That rule is by mean return only.
- One round was accepted while slightly worse on the score starts (Site42 round 5: -66.2 to -68.1).

**Rover v3: gated self-learning layer.**

- What it adapts: the belief map, the belief about solar output, an energy margin, and charge thresholds. No network weights.
- Gate: a proposal must beat the current brain on 128 paired real episodes (gain > 2 SE, no fewer returned, no more lost).
- Rounds: of 72 rounds, 57 sent a proposal to the gate; 9 were accepted, 48 rejected. All 9 came in round 1. Rounds 2-4
  accepted 0 of 54.

| world (6 fresh sites each) | rounds accepted | outcome by the pre-registered label |
|---|---|---|
| blocked route (synthetic) | 6 of 24 | 6 of 6 IMPROVED, 0 WORSE; losses on the score starts fell to 0 at all six sites under the simulator's lenient rule (not re-scored under the strict rule) |
| dusty panels (solar output x 0.6) | 3 of 24 | 2 IMPROVED, 4 NO CHANGE, 0 WORSE; rovers keep dying at NPA, NPB, NPC (5.9%, 12.1%, 3.1%) |
| unchanged | 0 of 24 | nothing accepted; NO CHANGE at every site; before = after by construction |

- Examples, returned before -> after: DM1 blocked route 0.254 -> 0.668; NPD blocked route 0.398 -> 0.711; NPD dusty panels
  0.559 -> 0.688. At NPA blocked route the effect is 2 rovers of 256.
- Limits. The blocked route is a synthetic ring that lies wholly inside the rover's sensed window, so one recorded batch saw all
  of it. What a batch of rovers has seen is merged into one belief map. The list of possible changes and the gate are
  hand-written. Each situation got at most one accepted change.
- Cost. Up to 1,152 real rover episodes per situation, and rovers were lost in them (DM1 blocked route, first gate batch: 14
  of 128 under the old brain).
- The loop was validated only by one smoke run on one development site (Site01).
- **Wrong light forecast (addendum 4): self-learning did not help.** 0 of 6 sites IMPROVED, 6 NO CHANGE, 0 WORSE; 2 rounds per
  site; 8 proposals reached the real gate and all were rejected. Pooled over the 1,536 score starts: returned 31.8% -> 31.8%,
  lost 26.2% -> 26.2%. The layer can correct the map, the solar output, the energy margin and the charge thresholds. It has
  no way to notice or correct a forecast that is shifted in time.
- The self-learning scores above use the simulator's lenient rule. The blocked-route, dusty-panel and unchanged-world
  situations were not re-scored under the strict rule.

**Rover v3.1: self-learning layer (addendum 5; four untouched sites, 2027-06-01, 2 rounds, 256 score starts, strict gate).**

- What is new: two proposal types (light forecast shifted in time, fitted from the rover's own solar readings; battery
  capacity as measured), a memory of rejected proposals, and a gate that scores under the strict rule.
- Rounds: 36 rounds in 20 situations; 23 sent a proposal to the real-episode gate; 9 were accepted, 8 in round 1 and 1 in
  round 2.

| world (4 untouched sites each) | sites IMPROVED | what happened |
|---|---|---|
| light forecast 12 hours late | 3 of 4 (LM5: NO CHANGE) | the rover fitted a shift of +12 h at all four sites; the gate accepted it at LM2, LM3 and LM4: lenient lost 53.9% -> 0%, 56.2% -> 0%, 11.7% -> 0%; strict lost 81.6% -> 72.3%, 71.5% -> 60.5%, 65.6% -> 53.1% |
| blocked route (synthetic) | 4 of 4 | each site added the obstacle it had seen to its map in round 1 |
| dusty panels | 1 of 4 (3 NO CHANGE) | LM4 accepted the correct solar measurement; LM3 keeps losing 19.5% under the lenient rule |
| unchanged | 0 of 4 (nothing accepted) | nothing changed, nothing worse |
| light forecast 8 hours early, as run in the final evaluation (30-minute cap by the lead's launcher mistake instead of the registered 45) | not evaluable (0 of 4 finished; 1 of 2 rounds each) | the shift (-8 h) was fitted at all four sites; nothing was accepted in the one round that ran |
| light forecast 8 hours early, completion run with the registered 45-minute cap (run after the final numbers had been seen; nothing changed, round 1 reproduced exactly) | 0 of 4 (4 NO CHANGE, 0 WORSE): **NOT MET** | the shift (-8 h) was fitted at all four sites in both rounds, and no fix was accepted at any site in either round (0 of 8 rounds). The shift reached the real gate once (LM4, round 1): every count improved (strict lost 87 -> 81, lenient lost 9 -> 0 of 128) but the gain +4.25 was below the 5.47 needed |

- **What works:** a map correction for a blocked route, and a time shift of the light forecast when the forecast is 12 hours
  late. **What does not:** a forecast 8 hours early (0 of 4), dusty panels (1 of 4), and anything open-ended: each situation gets one useful change, almost
  always in the first round, and then the layer stops.
- Every tested change has a hand-written proposal type (map, solar output, forecast shift, battery capacity). No change
  without a matching proposal type was tested, so "handles arbitrary new situations" is not supported.
- It spends real episodes and loses rovers while it learns.

**Lander: self-improvement loop (today's job).** The loop mines failures, proposes authority rules and short training rounds,
and passes a candidate only through a Site11 gate that allows no lost mission. It accepted 0 of 7 candidates, so the pilot
did not change. What works: the gate never let a worse pilot through. What does not: no improvement was found; the training
rounds had 4 to 5 updates each, too short to matter (section 3, "Lander, in full").

## 6. Known limitations

Frozen run (substance of `records/RESULTS_FOR_TEAM.md` sections 8 and 5):

- **Hazard truth is mostly interpolated.** Only 3.7% to 14.2% of the pixels in a test window hold a laser return. The all-pixel
  score mostly measures agreement with NASA's interpolation. The measured pixels are the network's own input, so that score
  does not measure gap recovery either.
- **The hazard comparison leans towards the network.** The backup's slope and NASA's error map are network inputs.
- Haworth is 19 of the 60 training tiles (32%). One crater dominates training.
- One validation site. Site11 (6 windows) picked the hazard, light and terrain checkpoints.
- Light: Sun and Earth directions are modelled (SPICE) and exist for centre tiles only. The 17 non-centre test windows
  borrowed the DM2 January directions for the sun-fraction numbers. Skyline MAE does not depend on this.
- Light: truth and model both ignore terrain beyond 1,280 m. The worst single error stays large for both (61.5 vs 63.8 deg on
  Site23).
- Terrain: hidden pixels sit next to shown pixels on the same laser track, which flatters both model and backup. The test says
  nothing about wide gaps.
- Shadow (PSR) mask is NASA's 20 m map upsampled 4x. Position may be off by up to 10 m.
- "Reached ice" means reaching a dark-and-cold shadow-edge target. It is not confirmed ice.
- The rover is simulation only. Rover v1's "best" label is misleading: the scored network is the one after update 1.
- Less training than requested fitted the budgets: light 3,258 of 6,000 steps, hazard 37 of 40 epochs, rover 144 of 800
  updates, terrain 7,671 of 8,000 steps.
- 4 of 89 planned tile windows are missing, all training windows. For three of them the root cause is unknown.
- The terrain network's raw band fails (coverage 16.2% to 41.5% per window where about 90% is claimed), and its win over the
  backup was a win over a weak opponent (addendum 4).
- Record-keeping items not resolved: a pytest count discrepancy (167 vs about 30); an unproven claim of "5 jax collection
  errors" on the mini; 11 corrected statements in `records/RUN_LOG.md`.

Addendum 1 (rover v2, self-learning):

- v2 memorised routes: the lander and the 16 targets sit in fixed places in every training world.
- A defect in the training set: Site04's centre world is 99.5% over the slope limit and taught nothing (4 of the 32 training
  worlds).
- Self-learning acceptance is best-of-6 on the same 64 test starts, so it is optimistic. The strategy layer is a mechanism demo.

Addendum 2 (terrain calibration):

- The calibrated band is wide: 7.1 m and 15.5 m, which is 3.9 and 5.5 times the RMSE.
- It is uneven: 6 of 19 windows are individually outside 80-97%. In the narrowest third of pixels coverage is 66.7% on Site23.
- Site42 passed by a thin margin (96.9%, 0.12 points under the upper limit). Leaving out one Site11 window moves the scale
  between 6.97 and 8.56.
- No formal guarantee. The scale is tied to 6 versions, 25 sampler steps and the hide-and-check task. The sampler is not seeded.

Addendum 3 (rover v3):

- Unchanged-world numbers assume a perfect map and a perfect light forecast. One wrong forecast was tested in addendum 4 (a
  uniform 12-hour delay, one date, six sites); the other direction and other sizes were not.
- The simulator's rule that an empty battery within 20 m of the lander is not a loss hid every v3 loss on the fresh sites (0 of 12,288 under that rule): under the strict rule v3
  loses 56.7% on the fresh sites. On validation Site04, where the lander cannot be reached, v3 and every planner lose the same
  55-80% even under the lenient rule.
- Most strict losses are rovers with no survivable option in this simulator (2 W idle draw, 100 Wh battery, a random landing
  hour in a month in which the lander is often dark for more than 50 hours). They say more about the energy budget and the
  landing time than about the planner. The strict rule is our reading of "a flat battery is a lost rover"; both rules are shown.
- The executor senses the true slope one move ahead, read from the simulator.
- Six sites: the intervals are approximate. Why 4 of 10 candidate fresh sites were left out was not recorded.
- Nothing in v3 is a trained network.

Addendum 4 (improvement work):

- Hazard on fresh sites: six tiles, one per site. Only 3.9% of the fresh pixels hold a laser return, so the label is mostly
  NASA's interpolation. NPC has only 6,300 unsafe pixels, so its numbers are noisy. Whether retraining with flat sites would
  fix the flat-terrain weakness was not tested.
- Light: the planned learning-rate schedule could not be recreated, so "finishing the schedule" means more steps at a very
  small rate. No confidence intervals on the fresh sites (one window per site).
- Terrain: one validation site chose every kriging setting. The kriging band has no formal guarantee; it covers 85.9% and
  88.2% on Site23 and Site42, not 90%, and two Site23 windows are below 80%. The band reflects distance from shown pixels, not
  roughness, and under-covers on rough ground. The network's numbers are one sampler run. The network still produces several
  plausible terrain versions with texture; that property was not scored.
- Rover: the wrong forecast is synthetic. The fixed brain was run on fresh sites that had been used once before. No
  per-rover outcomes were saved; the checker re-derived the numbers from per-world counts. Rover intervals come from
  resampling six sites and are approximate.

Addendum 5 (rover v3.1):

- v3.1 still loses 53.9% of rovers under the strict rule on the untouched sites. In this simulator, with this rover (100 Wh
  battery, 2 W idle) and random landing hours, most strict losses probably cannot be planned away.
- "Alive" means the battery never reached zero within the 300-hour episode. Thermal limits on lit ground are not modelled.
- With a wrong belief v3.1 loses more rovers under the lenient rule than v3 (late forecast 32.2% against 19.9%; weak battery
  9.0% against 6.1%; dusty panels 11.8% against 10.4%): it leaves the lander to shelter on lit ground, and when the forecast is
  wrong the battery runs out away from the lander.
- Changed-world and self-learning stages use one date, four neighbouring sites and 256 rovers; self-learning ran 2 rounds.
- The early-light self-learning stage is not evaluable as run (30-minute cap by mistake). A completion run with the right cap,
  made after the numbers had been seen, gave 0 of 4 sites improved (NOT MET); it is weaker evidence for that reason only. A launcher fault delayed stages A, B and C;
  no stage of the final evaluation ran twice and no test number had been read.

Benchmarks (`records/benchmarks/BENCHMARKS.md`):

- Only its "measured in this project, same data" tables compare two scores. Published work is listed there as "not directly
  comparable": different data, resolution and targets. Their scores must not be read against ours.
- Parameter counts of most published models are not stated in their papers; none was estimated.
- These are local models, tested on unseen sites of the same region and data product only.

## 7. How to reproduce

Frozen run (Mac mini, `~/adinfinitium-run/adinfinitium/`, code commit `07eb0f0`; `work/adinfinitium/START_HERE.md`):

    python -m ops.verify before
    bash ops/macmini/start_night.sh macmini_night
    python -m ops.verify after --run-id macmini_night
    .venv/bin/python -m ops.freeze --check          # must print: FROZEN check: ALL UNCHANGED

Do not re-run any `evaluate` command by hand: it would overwrite a frozen `eval.json`.

Addendum 1 (mini; needs the main run folder for data and Python environment):

    git clone adinfinitium_rover-v2_968a4e2.bundle adinfinitium-rover2 && cd adinfinitium-rover2 && git checkout rover-v2
    nohup bash ops/rover2/run_rover2_mini.sh 110 > runs/_rover2/launcher.out 2>&1 &
    START_DRIVER_FILE=$HOME/adinfinitium-rover2/data/models/rover_driver_v2.pt nohup bash ops/dream/run_dream_mini.sh 140 selflearn_v2 > runs/dream_launcher.out 2>&1 &

A re-run will not give the same numbers digit for digit (training is timed by the clock).

Addendum 2: script `records/terrain_calibration/terrain_calib.py`, plan `CALIBRATION_PLAN.md`. It refuses a second held-out run.

Addendum 3, one world (mini, about 15 s on CPU):

    cd ~/adinfinitium-rover3/code && export ADI_DATA=$HOME/adinfinitium-rover3/data PYTHONPATH=$PWD && ~/adinfinitium-run/adinfinitium/.venv/bin/python -m worldmodel.rover3.run eval --sites DM1 --dates 2027-06-01 --n 512 --seed 1234 --out /tmp/x.json

Addendum 4 (MacBook; scripts in `records/weak_flags/scripts/{terrain,hazard,light,rover,checker}/`, plan in
`records/weak_flags/PLAN.md`). Each held-out script refuses a second run, so a re-run needs a fresh copy:

    records/weak_flags/scripts/terrain/run.sh        # classical methods on the frozen hide-and-check pixels
    records/weak_flags/scripts/terrain/run_fresh.sh  # the same on the six fresh sites
    records/weak_flags/scripts/hazard/run.sh         # hazard on Site23, Site42 and the six fresh sites
    records/weak_flags/scripts/rover/run_a.sh        # rover v3 re-scored under the strict rule

Lander (Mac mini, `~/adinfinitium_lander`; runbook `ops/lander_mini/START_HERE.md`): the job is the chain
`ops/lander_mini/run_lander_job.sh` (steps 1 smoke, 2 mine, 3 solve, 4 targeted training rounds, 5 benchmark, 6 Site23 final
exam). Do not re-run step 6: the Site23 exam is flown once. Today's code changes against the pristine package:
`records/lander/lander_changes_vs_pristine.diff`.

Rover v3.1, one world (mini, about 1.5 minutes on one core):

    . ~/adinfinitium-rover31/env.sh && cd ~/adinfinitium-rover31/code && $PY -m worldmodel.rover3.run eval --sites LM4 --dates 2027-03-01 --n 512 --seed 1234 --threads 1 --out /tmp/x.json

End-to-end test (MacBook, `work/final/`, about 22 minutes on
CPU):

    cd ~/Documents/Ojas/Hackathon/work/final && export ADI_DATA=$PWD/data PYTHONPATH=$PWD/code OMP_NUM_THREADS=2
    ../venv-local/bin/python ../../records/final/scripts/e2e_test.py --site DM1 --date 2027-06-01 --all-dates --out e2e.json --arrays e2e.npz --frozen <FROZEN.json> --ref-eval ../../records/rover_v3/eval.json
    ../venv-local/bin/python ../../records/final/scripts/make_figure.py --result e2e.json --arrays e2e.npz --out e2e_demo_DM1.png

Final end-to-end run with rover brain v3.1 (Mac mini, own folder `~/adinfinitium-final/`, CPU, 6 threads; `code/` is a copy of
the evaluated v3.1 code, `data/` links the tiles, kernels and frozen checkpoints read-only):

    cd ~/adinfinitium-final && export ADI_DATA=$PWD/data PYTHONPATH=$PWD/code OMP_NUM_THREADS=6
    ~/adinfinitium-run/adinfinitium/.venv/bin/python scripts/e2e_test_v31.py --site DM1 --date 2027-06-01 --all-dates --threads 6 --out out/e2e_result_final_v31.json --arrays out/e2e_arrays_final_v31.npz --frozen ~/adinfinitium-run/adinfinitium/runs/FROZEN.json --ref-eval ref/rover_v31_eval.json
    ~/adinfinitium-run/adinfinitium/.venv/bin/python scripts/make_figure_v31.py --result out/e2e_result_final_v31.json --arrays out/e2e_arrays_final_v31.npz --out out/e2e_demo_DM1_v31_strict.png

File map:

| package | machine and path | size as stated | contents |
|---|---|---|---|
| Frozen run results | mini `~/adinfinitium-run/adinfinitium/adinfinitium_results_macmini.zip` | 637,147,760 bytes before the report was added | ledger, logs, 4 `eval.json`, 4 evaluated checkpoints, `FROZEN.json`, `RESULTS_FOR_TEAM.md` |
| Frozen run, resumable checkpoints (optional) | mini `~/adinfinitium-run/adinfinitium/adinfinitium_results_macmini_last_checkpoints_optional.zip` | 1,777,693,553 bytes | the 4 `*_last.pt` files |
| Addendum 1 | mini `~/adinfinitium_addendum_after_freeze.zip` | about 57 MB | rover v2, self-learning run, git bundle of branch rover-v2 |
| Addendum 2 | mini `~/adinfinitium_addendum2_terrain_calibration.zip` | 29,165 bytes, 12 files | plan, report, numbers |
| Addendum 3 | mini `~/adinfinitium_addendum3_rover_v3.zip` | about 0.2 MB, about 120 files | plan, report, `eval.json`, brain config, code |
| Lander results | mini `~/adinfinitium_lander/adinfinitium_lander_results_macmini.zip` | 164,182,171 bytes; sha256 `de5da8f7a5bb4ad24707e97d7d854967871a58437cbc3823fecb7c64e1cd05b6` | benchmark tables, gate changelog, logs, final pilot (unchanged) |
| Lander package as received | MacBook `~/Documents/Ojas/Hackathon/adinfinitium_lander_macmini_package.zip` | see the final package list below | pristine code, data, shipped pilot |
| Improvement work (addendum 4) | MacBook `records/weak_flags/adinfinitium_addendum4_weak_flags.zip` | 405,786 bytes (checksum: see the final package list below) | plan, four item reports, `WEAK_FLAGS_REPORT.md`, `CHECK.md`, json files, scripts |
| Rover v3.1 (addendum 5) | mini `~/adinfinitium_addendum5_rover_v31.zip` | 439,403 bytes | plan, report, `eval.json`, `tables.md`, diagnosis, brain config, raw stage files, code |
| End-to-end test and figure | MacBook `records/final/`; copied into the final submission documents zip | folder, not a zip | `e2e_result_baseline.json`, `e2e_result_optimised.json`, `e2e_result_final_v31.json`, `opt_rover.json`, `opt_hazard.json`, `e2e_demo_DM1_v31_strict.png`, `scripts/`, `NUMBER_CHECK*.md`, `PACKAGE_LIST.md` |
| Write-ups | MacBook `records/` (this file, `RESULTS_FOR_TEAM.md`, `ADDENDUM_AFTER_FREEZE.md`, `terrain_calibration/`, `rover_v3/`, `benchmarks/`) | folder; the reports are copied into the final submission documents zip | |
| Code | MacBook `work/adinfinitium` (`07eb0f0`), `work/adinfinitium-rover3` (branch `rover-v3`), `work/adinfinitium-rover31` (branch `rover-v3.1`); none pushed | folders; the addendum zips carry their own code | |

### Final package list

Sizes and SHA-256 were read from the files on Sun 15 Nov 2026 (`records/final/PACKAGE_LIST.md` is the same list).

| deliverable | machine and full path | bytes | sha256 |
|---|---|---|---|
| Frozen run results | mini `~/adinfinitium-run/adinfinitium/adinfinitium_results_macmini.zip` | 637,158,836 | `d27df50ad7fe55a49f2fb5cb06d0140c5c43ad5b2835c14916b92006f6e940bd` |
| Frozen run, resumable checkpoints (optional) | mini `~/adinfinitium-run/adinfinitium/adinfinitium_results_macmini_last_checkpoints_optional.zip` | 1,777,693,553 | `7dc0f4f8eb38d1cfe7bf0c5330f4a07b40f843283ef83d070c303c8ba0d1399f` |
| Addendum 1: rover v2 + self-learning | mini `~/adinfinitium_addendum_after_freeze.zip` | 57,222,310 | `2f67e03fff2ee4fb1a1ea884d42020a8b5a204133aad5962cce49d4941e9031a` |
| Addendum 2: terrain band calibration | mini `~/adinfinitium_addendum2_terrain_calibration.zip` | 29,165 | `28e94febcdcad0707218eec428ccc14e43690d7f78eb2aa72afc64365e04f271` |
| Addendum 3: rover v3 | mini `~/adinfinitium_addendum3_rover_v3.zip` | 224,461 | `947669df539fb4c4b3972b147526113045424676ddbc1d078bbf5f30fa9a567b` |
| Addendum 4: improvement work | MacBook `~/Documents/Ojas/Hackathon/records/weak_flags/adinfinitium_addendum4_weak_flags.zip`; copy on the mini `~/adinfinitium_addendum4_weak_flags.zip` | 405,786 | `5291ada95eed5784b73674b997a5d94425852de8cfc73335351a1523f49101ec` |
| Addendum 5: rover v3.1 | mini `~/adinfinitium_addendum5_rover_v31.zip` | 439,403 | `f17c5a4224435a42bd49386047fd94cf11cda88fe92e29763406840dc4b60bfa` |
| Lander results | mini `~/adinfinitium_lander/adinfinitium_lander_results_macmini.zip` | 164,182,171 | `de5da8f7a5bb4ad24707e97d7d854967871a58437cbc3823fecb7c64e1cd05b6` |
| Final submission documents (this file, benchmarks, figures, package list, per-item reports) | mini `~/adinfinitium_final_submission_docs.zip` | recorded after building | in `~/adinfinitium_final_submission_docs.zip.sha256` beside the zip and in `records/final/PACKAGE_LIST.md` on the MacBook (a zip cannot hold its own checksum) |
| Lander package as received (input, not a result) | MacBook `~/Documents/Ojas/Hackathon/adinfinitium_lander_macmini_package.zip` | 163,698,063 | `0503b89a57968a262f210efae9fb52c6d05e41d980228e092cdb18789cd06e87` |

The frozen run's own SHA-256 values (4 checkpoints, 4 `eval.json`, code) are in `runs/FROZEN.json`; check with
`.venv/bin/python -m ops.freeze --check`, which must print `FROZEN check: ALL UNCHANGED`.

## 8. Honest claims list

You can say:

- "On two lunar south-pole sites it never saw, our hazard network scored F1 0.965 and 0.917 against 0.899 and 0.567 for the
  slope-rule backup. On six fresh sites it still clearly beats the backup (F1 0.76 against 0.29), but it is much less
  accurate there, and on the four flat sites it misses 38% to 75% of the unsafe pixels." Add that most of the truth is NASA's
  interpolation.
- "Our light network's skyline error is about 21% lower than a coarse ray-cast on both held-out sites, and 24% lower on six
  fresh sites (0.243 deg against 0.321 deg), where it wins on all six." Add that it is about 200 times slower than the backup,
  and that running the planned 6,000 training steps (the added steps at a very small learning rate) did not move the result.
- "When we hide half of the real laser heights, a classical kriging fill predicts them to 0.9 m (Site23) and 1.1 m (Site42)
  RMSE; our diffusion network reaches 1.8 m and 2.8 m. On this test the classical method is better, and combining the network
  with it did not help. On six further sites kriging won again on every site (0.97 m against 1.62 m), with a 90% band that
  held 93% of the time at 2.8 m width (86% and 88% on Site23 and Site42)."
- "We tested our own headline and it did not survive: the terrain network had only been compared with a weak backup."
- "Our learned rover driver failed, twice, and we report both."
- "Rover brain v3.1 adds a survival mode: when the lander will be dark for too long, the rover plans a route to nearby ground
  that stays lit and waits there. In simulation, on four lunar south-pole sites no rover experiment had touched, counting
  every flat battery as a lost rover, it lost 53.9% of rovers against 61.0% for v3 and completed 33.7% of trips against 31.0%;
  the simple planners lost 78-84%. It still loses more than half its rovers."
- "Most of the rovers it saved stayed alive on lit ground without completing the trip (419 of 583); 164 were extra completed
  trips."
- "Most losses under the strict rule are probably unavoidable for this rover (100 Wh battery, 50 hours of idling) with random
  landing times: an energy-only estimate puts the unavoidable share near two thirds; only 8% of the losses are proven
  unavoidable, and the upper limit is about 80%. v3.1 still loses more than half."
- "The self-learning layer can now notice from its own solar readings that the light forecast is shifted in time. With a
  forecast 12 hours late it found the 12 hours at 4 of 4 sites and, after its real-episode gate accepted the fix at 3 of them,
  losses under the simulator's lenient rule fell from 12-56% to none at those three sites. Under the strict rule they fell
  from 82%, 71% and 66% to 72%, 61% and 53%; at the fourth site nothing changed."
- "With a blocked route it corrected its map at 4 of 4 sites. In unchanged worlds it changed nothing."
- "When the light forecast was 8 hours early, the rover measured the shift exactly at all four untouched sites, but its
  self-learning did not turn that into an accepted fix at any of them: in two full rounds, 0 of 4 sites improved and none got
  worse. We first cut this test short by a launcher mistake and completed it afterwards, unchanged, with the same result."
- About the earlier brain v3: "On six fresh sites v3 returned 59.6% of rovers under the simulator's rule, but that rule forgives
  a flat battery next to the lander; counting every flat battery as a lost rover, it returned 42.0% and lost 56.7%, still
  ahead of the best simple planner (29.8% returned, 69.4% lost) at all six sites. With a light forecast 12 hours off its
  self-learning layer did not recover anything."
- "We ran the frozen hazard, light and terrain checkpoints and the rover brain together, in order, on a site none of them was
  trained on; all 65 integration checks passed." Add that only the hazard network feeds the rover; the rover's light is physics.
- "The run was frozen with checksums, and every later model experiment is a separate addendum with a rule written before its
  held-out numbers." (The benchmarks table and the end-to-end test are not experiments of that kind.)
- "Our lander pilot lands about 96% to 98% of simulated missions on maps never used in its training (94.3% to 99.1% per map
  and condition). It is about as good as classic guidance or better on every map measured, and slightly better on three unseen maps pooled under normal failure rates (0.9839
  against 0.9766, p = 0.0186)." Add that the gain comes from one map (Site42), that the Site23 final exam is a tie under normal rates, and that under harsh
  rates on Site23 the pilot landed 2 missions of 1,024 fewer than classic (not significant).
- "Our self-improvement loop for the lander never let a worse pilot through: it tried 7 candidates today and rejected all 7 (each lost at least 12 gate missions that classic or the current pilot lands)."

Do not say:

- "The terrain network beats interpolation", "10 times better than the backup", or any terrain-network win without the kriging
  result next to it. Do not show the network's raw band as an uncertainty.
- "The kriging band is a 90% band" without its coverage (85.9%, 88.2%, 92.8%).
- "The hazard network is 92 to 97% accurate" as a general claim. On fresh ground it is F1 0.757 pooled and 0.24 to 0.59 on
  three of the four flat sites.
- "Finishing the training improved the light network." It changed nothing real.
- "v3 lost no rovers", "0% lost" or "0% deaths", in any form, for v3 or v3.1. That number came from one simulator rule; v3.1
  still loses 53.9% under the strict rule. Do not say "v3.1 solves the survival problem".
- "v3.1 is more robust than v3 to a wrong forecast." Before learning it loses more rovers under the simulator's rule (32.2%
  against 19.9%).
- "Self-learning handles new, unrelated situations." Every tested change has a hand-written proposal type; dusty panels
  improved at 1 of 4 sites; the 8-hour-early forecast was detected at 4 of 4 sites but no fix was accepted, in the single round of the final evaluation or
  in the two full rounds of the completion run (0 of 4 sites improved). Do not say the shift fix "would have been accepted
  with more rounds" (untested).
- Any comparison of a strict number with a lenient one.
- "Self-learning adapts to a changed world" without "only for changes that have a hand-written proposal type (map, solar
  output, a time shift of the light forecast, battery capacity)".
- "Rover v2 works" or "the self-learning rover learned the mission". Returned stayed 0.0% in addendum 1.
- "v3 learned to drive" or "the network generalises to unseen terrain". Nothing in v3 or v3.1 is a trained network.
- "Recursive self-improvement" as an open-ended claim (v3.1: 9 accepted changes in 36 rounds, 8 of them in round 1), or
  "self-learning is free or safe".
- "We ran Dream-RSI." We ran a small key-free loop in its style.
- "The rover uses our light network." It uses the physics ray-cast (section 2, arrow 6).
- "The models form one connected pipeline." The end-to-end test shows they run together on an unseen site; only the hazard
  network feeds the rover. The light network and the terrain estimators feed nothing downstream (section 2a).
- "The pilot beats classic guidance on the final exam." On Site23 it ties under normal rates (0.9795 both) and is not better
  under harsh rates (0.9629 against 0.9648).
- "The lander self-improved today" or "the lander learned from its failures". 0 of 7 candidates were accepted; the final pilot
  is the shipped pilot.
- "The pilot is better under harsh conditions." That difference is not significant (p = 0.1843).
- "Finds ice", "works on the real Moon", or any score read against a published model's score.

## 8a. Website-ready results

Sentences that can be published as they stand. Each carries its number and its source file.

1. On two lunar south-pole sites it never saw, our hazard network (60,361,561 parameters) scored F1 0.965 and 0.917 against
   0.899 and 0.567 for a slope-rule backup; most of the truth is NASA's interpolation between laser tracks.
   (`records/RESULTS_FOR_TEAM.md` section 3.1)
2. On six further sites the hazard network still beat the backup on every site (F1 0.757 against 0.292 pooled), but it was
   much less accurate than on the first two sites and missed 38% to 75% of the unsafe pixels on the four flat sites.
   (`records/weak_flags/REPORT_hazard.md`)
3. Our light network (38,279,809 parameters) predicts the horizon line with an error about 21% below a coarse ray-cast on the
   two held-out sites and 24% below it on six fresh sites (0.243 deg against 0.321 deg), winning on all six; it is about 200
   times slower than the ray-cast. (`records/RESULTS_FOR_TEAM.md` section 3.2; `records/weak_flags/REPORT_light.md`)
4. For filling terrain between laser tracks we tested our 49.9 M parameter diffusion network against classical methods on the
   same hidden laser heights, and the classical kriging fill won: 0.94 m and 1.11 m RMSE against 1.83 m and 2.80 m on the two
   held-out sites, and 0.97 m against 1.62 m on six fresh sites. We present kriging. (`records/weak_flags/REPORT_terrain.md`)
5. In simulation, our two learned rover drivers failed on unseen terrain (0.0% and 1.2% of rovers returned, against 9.3% for a simple
   planner), and we report both failures. (`records/RESULTS_FOR_TEAM.md` section 3.4; `records/ADDENDUM_AFTER_FREEZE.md` section 1)
6. In simulation on four sites no rover experiment had touched, counting every flat battery as a lost rover, our scripted rover
   brain v3.1 completed 33.7% of trips and lost 53.9% of rovers, against 31.0% and 61.0% for our earlier brain and 17.9% and
   81.8% for the best simple planner; it still loses more than half, and most of the rovers it saved stayed alive on lit ground
   without completing a trip (419 of 583). (`records/rover_v31/ROVER_V31_REPORT.md`)
7. In simulation, our lander pilot (a learned correction on classic guidance) landed 98.4% of missions on three maps it never
   trained on, against 97.7% for classic guidance alone under normal failure rates (2,304 paired missions, exact McNemar p =
   0.0186); under harsh rates the difference is not significant (96.9% against 96.3%, p = 0.1843). On the final exam map
   Site23 the two tie at 97.95% under normal rates and the pilot is not better under harsh rates (96.29% against 96.48%).
   The whole gain comes from one of the three maps (Site42); the other two are ties. Today's self-improvement job accepted
   none of its 7 candidates.
   (`records/lander/RESULTS_SUMMARY.md` sections 3, 5, 6)
8. We ran the frozen models and the rover brain together, in order, on a site none of them was trained on, and all 65
   integration checks passed; only the hazard network feeds the rover, whose light forecast is a physics ray-cast.
   (`records/final/e2e_result_final_v31.json`)

## 9. Sources appendix

| numbers | copied from |
|---|---|
| Parameter counts 60,361,561; 38,279,809; 49,943,681; 11,377,354; 0 | `records/benchmarks/BENCHMARKS.md` section 1; `records/benchmarks/params.json`; RESULTS section 7 |
| Run facts: 14 stages `ok`, "87 checks, 0 FAIL, 0 WARN", training amounts fitted, 85 of 89 windows | `records/RESULTS_FOR_TEAM.md` sections 1, 2 |
| Hazard F1, missed rates, precision 0.961 vs 0.940, 19 windows, Site11 0.946 vs 0.862, 3.7% to 14.2% | RESULTS section 3.1 and section 8 |
| Light MAE, 9 of 13 and 1 of 6, 61.5 vs 63.8 deg, Site11 0.381 vs 0.491, "about 21%" | RESULTS sections 1, 3.2 |
| Light speed 7.41 s vs 0.033 s; "54% of the plan" | BENCHMARKS section 3A caveats |
| Terrain RMSE, coverage 21.9% / 37.3%, backup coverage 17.8% / 4.9%, 16.2% to 41.5%, Site11 1.90 vs 17.92 | RESULTS sections 3.3, 5 |
| Calibration: 8.232, 22.0% -> 84.9%, 37.1% -> 96.9%, widths, 3.9 and 5.5 times, 66.7%, 6 of 19, 0.12 points, 6.97 to 8.56, claim wording 85% / 97% / 7 m / 16 m | `records/terrain_calibration/TERRAIN_CALIBRATION_REPORT.md` (Verdict, Held-out result, What to be careful about, What the team can say) |
| Rover v1: 0.0%, 1.7%, 95.5%, planners 9.3% / 19.6%, update 1 of 144 | RESULTS sections 3.4, 4 |
| Rover v2: 0.012, 0.093, 0.953, 48 and 380 of 4,096, Site11 0.268 vs 0.407, 86-93% and 15-26%, Site04 99.5%, 4 of 32 | `records/ADDENDUM_AFTER_FREEZE.md` sections 2.3 to 2.6 |
| Self-learning addendum 1: 135 of 140 minutes, 2 of 23, 3 of 15, 33 rejected, +8.6 (0.8), +17.4 (3.0), 90.2% -> 81.6%, 38 rounds, 54 settings, loop sizes, round 5 | ADDENDUM sections 1, 3.1 to 3.7 |
| Rover v3 fresh sites: 0.596, 0.329, intervals, +0.267 [0.191, 0.337], died 0.000 / 0.388, 37.3%, 12,288, 36 plans | `records/rover_v3/ROVER_V3_REPORT.md` (Verdict, section A) |
| Rover v3 Site23 + Site42: 0.375, 0.093, 0.868 | ROVER_V3_REPORT section B |
| Changed worlds: 0.299 / 0.074, 0.066 / 0.255, 0.237, 0.413 / 0.139, 0.202 / 0.797, 2 of 6 sites | ROVER_V3_REPORT "changed worlds before any learning" |
| v3 self-learning: 72, 57, 9, 48, 0 of 54, per-world rounds and labels, 1,152, 14 of 128, site examples, slopes 2.5-5.9 / 9.5-11.6, 55-80%, 4 of 10 | ROVER_V3_REPORT (self-learning, Caveats); per-world round counts from BENCHMARKS table 5A-4 notes |
| Spatial check: 60 + 6 vs 19, 1,280 m, 28.8 km, 90.1 km; Haworth 19 of 60 (32%); 17 windows; 10 m | RESULTS section 8 |
| Zip sizes | RESULTS section 9; ADDENDUM section 5; `records/agents/terrain-calibration.md` (09:37 entry); ROVER_V3_REPORT "Files" |
| Commands | `work/adinfinitium/START_HERE.md` steps 1, 2, 4; RESULTS section 6; ADDENDUM section 5; ROVER_V3_REPORT "Files" |
| End-to-end test (section 2a): checks, seconds, memory, 338 / 132 / 0 of 512, 2.98e-08, 0.266 and 0.352 deg | `records/final/e2e_result_baseline.json` (`stages`, `checks_total`, `peak_rss_mb`, `hazard`, `light.sanity_only`, `rover.traverse`); equality with eval.json: `records/final/e2e_result_optimised.json` (`rover.worlds[].equals_reported_eval_json`) and `records/final/opt_rover.json` |
| Speed-ups: 168.0 -> 69.8 s, 1.19e-07; 130.3 -> 61.9 s; 55 checks, 266.97 s, 5,366 MB | `records/final/opt_rover.json`; `records/final/opt_hazard.json`; `records/final/e2e_result_optimised.json` |
| Addendum 4, hazard: 0.757 / 0.292, 0.782 / 0.352, 0.239 to 0.823, 38.4% to 74.9%, 18.7%, 70.8%, 96.0% / 88.8%, 3.9%, 6,300, threshold 0.5, 0.0011 / 0.005, 0.24 to 0.59 | `records/weak_flags/REPORT_hazard.md` (Result, What the table says, Verdict, Caveats, not done); `records/weak_flags/hazard.json` |
| Addendum 4, light: 0.4503 / 0.4489 / 0.5700, -0.0014, 0.005, 0.2426 / 0.2439 / 0.3205, 24.3%, 6 of 6, 61% to 76%, 0.99x, 1.3 to 1.4 times, 0.327 deg, 199 to 266 times, 7.41 s / 0.033 s / 227, 1.6%, 0.243 / 0.321 | `records/weak_flags/REPORT_light.md` (Result tables, speed table, Verdict, Caveats, Fresh sites); `records/weak_flags/light.json` |
| Addendum 4, terrain: 0.94 / 1.11 / 0.97, 1.83 / 2.80 / 1.62, 17.92 / 24.82, seven of nine, 1.22 / 1.69, 0.0064, bands 85.9% / 2.31, 88.2% / 2.76, 92.8% / 2.82, 84.9% / 7.06, 96.9% / 15.51, 96.6% / 10.48, 0.960 / 0.972, 6 of 6, 93% / 2.8 m | `records/weak_flags/REPORT_terrain.md` (main tables, band table, Fresh sites); `records/weak_flags/terrain.json`, `terrain_fresh.json` |
| Addendum 4, rover: 42.0% / 56.7% and intervals, 29.8% / 69.4% and intervals, +12.2 [10.3, 13.6], 59.6% [46.4, 71.4], 32.9% [18.4, 46.2], 32.4% / 38.6%, 9.2% / 86.9%, 42.8% / 55.5%, 4,801, 2,164, 18.8% / 78.8%, 20.6%, 28.1% / 71.2%, wrong light 30.9% / 27.3% and intervals, 24.8% / 74.4%, 20.4% / 79.6%, 23.4%, 0 of 6, 8 proposals, 31.8%, 26.2%, 2 W / 100 Wh / 50 hours | `records/weak_flags/REPORT_rover.md` (Read this first, Results (a) and (b), Caveats); `records/weak_flags/rover.json` |
| Addendum 5, rover v3.1: 33.7% / 53.9%, 31.0% / 61.0%, 17.9% / 81.8% and intervals, -7.1 [-13.8, -3.5], +2.7 [+1.9, +3.4], +15.8, 48.1% / 46.7%, 43.9% / 54.2%, -2.5 [-3.7, -1.3], 38.3% / 30.0%, late light 12.2% / 81.3%, 11.2% / 87.1%, 13.5% / 32.2%, 17.3% / 19.9%, 583 / 164 / 419, 80 Wh, LM4 -17.1, -3.4 to -4.1, self-learning table, 36 / 23 / 9 / 8 / 1, 30-minute cap, 10-30 km, 4 km, weak battery 9.0% / 6.1%, dusty 11.8% / 10.4%, 78-84%, 12-56% | `records/rover_v31/ROVER_V31_REPORT.md` (Verdict, sections A, B1, B2, changed worlds, self-learning, What went wrong, Caveats, Claims); `records/rover_v31/eval.json`, `tables.md` |
| Rover v3.1 early-light completion run: 0 of 4, 2 rounds, 45 / 30 minutes, 87 -> 81, 9 -> 0 of 128, +4.25 / 5.47 | `records/rover_v31/COMPLETION_EARLY_LIGHT.md`; `records/rover_v31/completion_early_light.json` |
| Rover v3.1 diagnosis: 8% proven, about 80%, near two thirds | `records/rover_v31/ROVER_V31_REPORT.md` (Diagnosis); `records/rover_v31/diagnosis.md` |
| Addendum 4, independent check | `records/weak_flags/CHECK.md` |
| Wiring statements | code lines cited in section 2 (read on the MacBook checkouts) |
| Lander: 0.9839 / 0.9766 and intervals, +0.0074, 32 / 15, 0.0186; 0.9692 / 0.9631, +0.0061, 55 / 41, 0.1843; per-map rows; Site23 0.9795 / 0.9795, 11 / 11, 1.000; 0.9629 / 0.9648, -0.0020, 19 / 21, 0.875; gate 990 / 965 of 1,024, 0.9668, 0 of 7, 989 to 991, 12 to 14, 977 to 981, 21 to 25, 4 to 5 updates, 200 of 1,701, 15 strategies, 3 of 5 rounds | `records/lander/RESULTS_SUMMARY.md` sections 3 to 8 |
| Lander diagnosis: 94 to 99%, cap 0.3, 77 / 37 / 27, 0.51 to 0.78 m/s, 0.5 m/s, 600 updates; "15 million" | `records/lander/LANDER_DIAGNOSIS_AND_RECOMMENDATIONS.md` sections 2, 5; `adinfinitium_lander/ops/lander_mini/START_HERE.md` section 2 |
| Lander zip size and checksum | `records/lander/STATUS.md` |

## Discrepancies noticed

- **Planner rows differ slightly between runs on the same 8 worlds.** Earlier runs (RESULTS 3.4, ADDENDUM 2.4): cautious
  0.105 / 0.088 / 0.543, mean return -18.4; wait for light 0.069 / 0.060 / 0.739, -19.2. The v3 run: cautious 0.104 / 0.088 /
  0.543, -18.5; wait for light 0.069 / 0.060 / 0.738, -19.0. The naive planner is identical. Cause not established.
- **Terrain, frozen run vs calibration re-run.** Site23 coverage 21.9% vs 22.0%, RMSE 1.838 m vs 1.834 m; Site42 37.3% vs
  37.1%, 2.815 m vs 2.795 m. The sampler is not seeded. RESULTS prints the frozen RMSE as 1.84 m and 2.82 m.
- **Light sun-fraction error.** RESULTS 3.2 gives centre windows (Site23 0.022 vs 0.029); BENCHMARKS 3A gives means over
  windows (0.013 vs 0.017). Different aggregation, not a conflict.
- **Light seconds per window.** BENCHMARKS: 7.41 s (model time). RESULTS section 7: 7.7 s per window (whole evaluation stage).
- **Tile counts in RESULTS.** Section 2 says 35 windows were built during the run; section 7 says 36 windows were fetched.
- **Rover v2 report says "one amendment"; the plan has two** (ADDENDUM 2.2).
- **Results zip size.** 637,147,760 bytes = 637 MB, while `ls -lh` prints 608M (RESULTS section 9).
- **Site11 planner check in the v3 harness** was not an exact reproduction (naive 0.364 vs 0.407; ROVER_V3_REPORT).
- **Terrain backup.** RESULTS says a stronger classical competitor was not tested; addendum 4 has since tested it and the
  network loses (section 3). RESULTS is frozen and was not edited.
- **Rover v3 "best planner" depends on the rule.** Under the lenient rule the best planner by returned is cautious (32.9%);
  under the strict rule it is naive (29.8%).
- **Rover v3 lenient intervals** are quoted from REPORT_rover in percent; the v3 report gives the same values as fractions.
