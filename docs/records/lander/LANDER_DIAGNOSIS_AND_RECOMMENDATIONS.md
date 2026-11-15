# Lander pilot: why the scores are where they are, what was tried, what to do next

Sunday 15 November 2026, Mac mini run. Written for teammates; every number is from a file named next to it.
Paths are inside the package (`~/adinfinitium_lander` on the mini) unless stated.

## 1. Headline

- The lander pilot was **not improved today**. Seven candidates went to the Site11 gate (four authority rules, three training
  rounds) and the gate rejected every one. The production pilot is byte-for-byte the shipped one
  (sha256 `2501e02b...`).
- The shipped pilot is already better than classic guidance, by a small margin: on the validation map Site11 it lands 990 of 1024
  gate missions (0.9668) against 965 for classic (0.9424) (`runs/lander/rsi_changelog.md`). On the three unseen maps it lands 0.9839
  against 0.9766 under normal failure rates (p = 0.0186) and 0.9692 against 0.9631 under harsh rates (p = 0.184, not significant)
  (`runs/lander/benchmark.md`: the shipped table; the chain skipped the re-run because the pilot did not change, and an extra
  replication after the exam reproduced the table byte for byte on the mini, `runs/lander/benchmark_replication_macmini.md`).
- The Site23 final exam was flown once with that pilot (`runs/lander/benchmark_site23_final.md`): classic 0.9795, pilot 0.9795 under
  normal rates (11 missions only the pilot landed, 11 only classic, p = 1.0); classic 0.9648, pilot 0.9629 under harsh rates (19 / 21,
  p = 0.875). **On the exam map the pilot is not better than classic guidance.** Together with the unseen maps (the whole pooled gain
  comes from Site42; Shoemaker and DM2 are ties) the honest claim is: the pilot is about as safe as classic or safer on every map measured,
  clearly better on Site42 and on the validation map Site11, and a tie on Shoemaker, DM2 and Site23.

## 2. Why the scores are low (the diagnosis)

1. **Classic guidance is the floor and it is already high.** Classic lands 94 to 99 % depending on the map. The pilot is a residual
   capped at 0.3 of full thrust (`worldmodel/lander/pilot.py:23`), so it can only move the last few percent.
2. **The biggest single failure is touching down a few centimetres per second too fast sideways.** On Site11 (2048 paired missions,
   `runs/lander/failure_analysis.json`) the pilot crashes 77 missions. 37 are `hard_horizontal`, and 27 of those touch down at
   0.51 to 0.78 m/s against a limit of 0.5 m/s (`config/mission.yaml:31`). Classic landed 28 of the 37.
   The cause: a navigation drift is a 0.3 m/s velocity bias (`config/mission.yaml:71`, `worldmodel/lander/env.py:200`). Classic nulls
   the *measured* velocity, so the true sideways speed at touchdown is about 0.3 m/s, leaving a 0.2 m/s margin. The pilot decides
   every 0.5 s and holds its residual in between (`worldmodel/lander/analysis.py:59`), and velocity noise adds to it. That this is
   what eats the margin is my reading of the numbers, not something I measured directly.
3. **These failures are coin flips, not a systematic habit.** When the same pooled failures are replayed, the current pilot lands
   153 of 200 missions it had mostly failed or nearly failed (`runs/lander/solve_report.md`). Every rule tried (hand over to classic
   below 5-15 m, 20-40 m, ...; 60 % or 80 % authority; classic flies after an engine failure) solves 9 to 16 of the pilot's 47
   crashes and loses 1 to 11 of its landings. On the gate the same thing: each rule lost 4 to 9 missions the current pilot lands and
   won about as many others, net -1 to +1 of 1024 (`runs/lander/rsi_changelog.jsonl`).
4. **The gate cannot be passed by a candidate of this kind.** The gate (`worldmodel/lander/rsi_targeted.py:82-98`) accepts only if
   the candidate loses no mission that classic OR the current pilot lands. The shipped pilot itself loses 37 of 2048 Site11
   missions that classic lands (`runs/lander/failure_analysis.md`), so a candidate must repair every such mission and flip none of
   the marginal ones. None of the rules did: each still lost 10 to 12 gate missions that classic lands
   (`runs/lander/rsi_changelog.jsonl`, `regressions_detail.vs_classic`).
5. **The training rounds are too short and they damage the pilot.** One round is 12 minutes; on this machine that is 5 updates
   (`runs/lander/logs/4_rsi_targeted.log`: "time budget 12 min: 5 of 100000 requested units fit"), two to three minutes of simulated
   flight per lander. All three rounds came back worse than they started: 0.9570, 0.9541 and 0.9580 against 0.9668, with 21 to 25
   missions lost. A fresh optimiser on a
   15-million-parameter network with exploration noise moves the policy before it has seen enough finished landings to learn from.
6. **Compute is the hard limit.** The pilot network costs about 0.5 ms per lander per decision on the mini's GPU (measured: 0.29 s
   for a batch of 512, 2.16 s for 4096), and a mission needs up to 480 decisions. 1000 missions cost about 4 minutes; one gate costs
   4.3 minutes; replaying 1701 pooled missions under 17 strategies would have taken about 2 hours.

## 3. What was changed today (all inside the five hard rules; diff in `lander_changes_vs_pristine.diff`)

| change | file | why | what it bought |
|---|---|---|---|
| Seven more authority rules in the solve ladder (hand-over below 2-6 m and 10-25 m, 80 % authority, 80 % + 5-15 m, three "classic flies after an engine failure" rules) | `worldmodel/lander/solve.py` | the failure analysis pointed at touchdown speed and engine failures | two of them ranked in the top four and went to the gate; both rejected |
| New optional rule key `engine_out_scale` (uses the measured engine-health reading, never the hidden failure) | `worldmodel/lander/pilot.py`, one assertion in `tests/test_mission_brain.py` | about half of the pilot's losses against classic on Site11 involve an engine failure | on the training pool it solves 11 crashes and loses 11 landings: no net gain |
| Ladder flights in chunks of at most 4096 landers | `worldmodel/lander/solve.py` | the known memory risk in START_HERE section 5 | not needed in the end (sample of 200) |
| `--max-pool` (fixed random sample of the pool), `SOLVE_MAX_POOL`, `SOLVE_TRIES` | `worldmodel/lander/solve.py`, `ops/lander_mini/run_lander_job.sh` | time: see section 2 point 6 | step 3 finished in 34 minutes; four rules gated instead of three |

Two mistakes of mine cost about 27 minutes of GPU time and are recorded in `RESULTS_SUMMARY.md` section 7: the first step-3 launch was too
big to finish (killed after 11 minutes), and a speed-up that ran the network only for landers still flying ran the GPU out of memory
(a changing batch size makes MPS keep one compiled graph per shape). The second was reverted.

Nothing was changed in the simulator, the mission config, the failure rates, the seeds or the gate. Site23 was flown once. The unseen
maps were not used for any choice.

## 4. What it bought

Nothing accepted. Honest accuracy before and after is the same pilot:

| where | classic | pilot (shipped = final) |
|---|---|---|
| Site11 gate, 1024 missions, normal + harsh | 0.9424 | 0.9668 |
| Unseen maps pooled, normal, 2304 missions | 0.9766 [0.970, 0.982] | 0.9839 [0.978, 0.988], p = 0.0186 |
| Unseen maps pooled, harsh, 2304 missions | 0.9631 [0.955, 0.970] | 0.9692 [0.961, 0.975], p = 0.1843 (not significant) |
| Site23 final exam, normal, 1024 missions (flown once today) | 0.9795 [0.969, 0.987] | 0.9795 [0.969, 0.987], p = 1.0 |
| Site23 final exam, harsh, 1024 missions (flown once today) | 0.9648 [0.952, 0.974] | 0.9629 [0.949, 0.973], p = 0.875 |

Self-improvement curve on the Site11 gate (1024 missions; current pilot 990 landed, classic 965):

| round | what was tried | candidate landed | missions lost | decision | pilot after the round |
|---|---|---|---|---|---|
| solve 1 | 80 % authority | 990 | 13 | rejected | 0.9668 |
| solve 2 | classic flies after an engine failure + hand-over below 5-15 m | 989 | 13 | rejected | 0.9668 |
| solve 3 | hand-over to classic below 5-15 m | 990 | 12 | rejected | 0.9668 |
| solve 4 | 60 % authority | 991 | 14 | rejected | 0.9668 |
| training 1 | 5 updates, lr 5e-5 | 980 | 22 | rejected | 0.9668 |
| training 2 | 4 updates, lr 3e-5, touchdown penalty 0.3 | 977 | 25 | rejected | 0.9668 |
| training 3 | 5 updates, lr 2e-5, touchdown penalty 0.5 | 981 | 21 | rejected | 0.9668 |

The loop itself works as designed (it mines, trains, gates, records, and never lets a worse pilot through); it did not find an
improvement in the time it had.

## 5. What a longer run needs (recommendations, in order of expected value)

1. **Fix the touchdown margin at the source, in training.** Train with the touchdown-speed penalty on from the start
   (`--td-penalty`, `worldmodel/lander/train.py:64`) and with failure rates boosted (`--fail-boost`), for hours, not minutes: the
   original run was sized for 600 updates (`train.py` docstring); today's rounds had 5.
2. **Let the residual be decided every 0.1 s near the ground**, or fade it by measured sideways speed. Holding a 0.5 s old residual
   through the last metres is what turns 0.3 m/s into 0.52 m/s. This is a pilot-side change (not the simulator) and needs retraining.
3. **Change how a candidate is accepted, as a team decision, before the next run.** "Zero lost missions out of 1024" rejects any
   candidate whose marginal missions reshuffle, including one that is truly better. A paired test on more missions (for example:
   candidate-only wins against current-only wins, exact McNemar, on 4096 fresh Site11 missions, plus "not worse than classic") measures
   the same thing without that flaw. I did not touch the gate: this is your call, and it must be fixed before looking at results.
4. **Make each training round long enough to matter.** At 2.2 minutes per update, 8 updates is 18 minutes and is still tiny. Either
   give a round 60 to 90 minutes, or cut the cost per update (smaller network for the residual, or `--horizon 32` with more updates).
   Keep the optimiser state between rounds instead of starting Adam fresh each time (`rsi_targeted.py:140` saves the warm start without it).
5. **Make solve affordable.** Replay only missions the pilot failed and classic landed (384 `pilot_fail` of 1701), fly each strategy
   with the same sensor noise (today strategies saw different noise on the same mission, which blurs a ranking whose differences are
   1 to 10 missions), and never change the batch size between network calls on MPS.
6. **Mining finds plenty** (1701 failed or near-miss missions from 10240 flown), so the pool is not the limit.

## 6. Not verified

- The cause named in section 2 point 2 (the 0.5 s hold of the residual) is inferred from the touchdown numbers, not measured.
- The pool ranking in step 3 used 200 of 1701 pooled missions and unpaired sensor noise, so differences of a few missions mean nothing.
- The MacBook was not used for compute (it was overloaded by other work), so there is no second-machine cross-check of the gate.
