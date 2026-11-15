# Adinfinitium — final benchmarks and model findings (15 Nov 2026)

All numbers are measured in simulation or against NASA LRO LOLA 5 m terrain truth, unless marked **Projected** or **Scenario**.
Paired fixed-seed design throughout (every model and its baseline see identical missions / windows / worlds).
Full tables, sources and per-row provenance: `website/assets/data/benchmarks.json`, `mission_bench.json`, `dream_rsi.json`,
`projections.json`; detailed write-up `data/handoff_macmini_1115/BENCHMARKS.md` (local, not in git).
Note: every real test site has now been used at least once; the procedural Moon maps are the only never-seen terrain.

## 1. Lander pilot (14,842,567 params, PPO residual on classic ZEM/ZEV guidance, checkpoint d53a162df9)
Touchdown success = vertical ≤ 2 m/s, horizontal ≤ 0.5 m/s, slope ≤ 12°. Simplified 3-DOF terminal descent.

| Test | n / condition | Classic | Pilot | Pilot − classic [95 % CI] | Discordant (P/C) | Exact p |
|---|---|---|---|---|---|---|
| 3 unseen maps, normal | 2,304 | 97.66 % | **98.39 %** | +0.74 [+0.16, +1.32] | 32 / 15 | 0.019 (Holm 0.074) |
| 3 unseen maps, harsh (3× failures) | 2,304 | 96.31 % | 96.92 % | +0.61 [−0.23, +1.44] | 55 / 41 | 0.18 |
| Site 23 final exam (flown once), normal | 1,024 | 97.95 % | 97.95 % | 0.00 | 11 / 11 | 1.00 |
| Site 23 final exam, harsh | 1,024 | 96.48 % | 96.29 % | −0.20 | 19 / 21 | 0.88 |
| Mission rehearsal at planned cell (Site 23, 2027-03-17) normal | 512 | 99.02 % | **99.61 %** | +0.59 [−0.4, +1.6] | 5 / 2 | 0.45 |
| Mission rehearsal, harsh | 512 | 99.22 % | 99.02 % | −0.20 [−1.3, +1.0] | 4 / 5 | 1.00 |
| Procedural Moon, 20 new maps, normal | 640 | 97.50 % | **98.12 %** | +0.63 [−0.3, +1.6] | 7 / 3 | 0.34 |
| Procedural Moon, harsh | 640 | 96.72 % | **97.50 %** | +0.78 [−0.6, +2.1] | 12 / 7 | 0.36 |

- Fuel: ≈ 9 % less in every test (e.g. 65.0 vs 71.5 kg unseen maps; 64.0 vs 71.4 kg procedural).
- Precision: same mean miss (≈ 4 m normal); 95th-percentile miss at the planned cell 29.0 vs 32.7 m.
- Weak spots: engine-out on real maps 95.9 % vs classic 98.4 % (regression); on Site 23 worse in every injected-failure class;
  crashes dominated by sideways touchdowns at 0.51–0.78 m/s; touches down harder (1.12 vs 1.00 m/s); unseen-map gain comes from Site 42 only.
- Verdict: at least as safe as classic guidance in every test, ahead in most point estimates, ~9 % fuel saving; only one comparison is
  individually significant and none survives multiple-comparison correction.

## 2. Hazard U-Net (60,361,561 params) vs slope-rule backup
| Test | U-Net F1 | Slope rule F1 | Always-unsafe F1 |
|---|---|---|---|
| Held-out Site 23 + 42 (19 windows) | **0.951** [0.932, 0.967] | 0.817 | 0.843 |
| Six fresh sites | **0.757** [0.501, 0.805] | 0.292 | 0.233 |
Wins every window. Weak on flat ground (F1 0.24–0.70 at flat sites; 38–75 % of unsafe pixels missed); ~90 % of label pixels are interpolation;
Site 42 whole tile 0.803 (flags flat zones). The team's 1.94 M model scores 0.950 on the Site 23 centre window where this 60 M model scores 0.926.

## 3. Light / skyline transformer (38,279,809 params) vs coarse ray-cast
Skyline MAE 0.450° vs 0.570° (held-out), 0.243° vs 0.321° (fresh); sunlit-hours error 7.6 h vs 10.5 h; better in 19/19 windows.
Weak spots: ~227× slower; median error worse in 9/19 windows (gain is in the tail); worst cell ≈ 61°; not wired into the rover.

## 4. Terrain
Kriging (0 params) 0.97 m RMSE beats the diffusion U-Net (49.9 M) at 2.01 m in 19/19 held-out windows → kriging ships; the network is an
experiment (raw 90 % band covered only 22–37 % of truth).

## 5. Planner (Site 23, 7 dates × 96 candidates)
Chosen-candidate simulated success: logistic 15.2 %, Navigator ranking 12.8 % (ships), PlannerSetNet 12.2 %; none significant; all ~30 pts
below the best candidate (42.4 %). PlannerSetNet Brier 0.0145 is worse than a constant (0.0125). Limit: 23 labelled worlds, noisy labels.

## 6. Rover (LM2–LM5 × 4 dates, 8,192 runs per method, strict rule)
Scripted v3.1: returned **33.7 %**, lost 53.9 % vs naive planner 17.9 % / 81.8 % (≈ 1.9×). Learned drivers failed: PPO 0.0 % returned
(scored checkpoint was update 1 of 144), imitation 1.2 % (memorised routes). Main loss: rovers idling at a dark lander until the battery dies.

## 7. Dream-RSI (gated offline self-improvement)
| Loop | Accepted | Effect |
|---|---|---|
| Planner | 1 of 5 | validation Brier 0.01543 → 0.01353 (−12 %) |
| Lander pilot (zero-regression gate) | 0 of 7 | unchanged; gate held |
| Rover v3.1 | 9 of 36 | fixes planted faults (blocked route 4/4 sites, late light 3/4) |
| Rover v3 | 9 of 72 | blocked route 6/6 |
| Rover v2 | 5 of 38 | returned stayed 0 % |
Finding: Dream-RSI reliably repairs specific failures and never ships a regression; in unchanged conditions it rarely finds a safe gain.
Projected (planner, own log): Brier 0.0145 → 0.0094 after 20 generations [0.0043, 0.0145]. Scenarios for other models are on the website, labelled.

## 8. Not benchmarked
Swarm / NEAT: code exists, never run. Light and terrain are not wired into the rover, so no end-to-end benefit can be measured.

## Reproduce
```
python -m worldmodel.lander.benchmark --n 256 --seeds 3 --harsh 3
python -m worldmodel.lander.mission_bench rehearsal --n 512 --batch 256
python -m worldmodel.lander.mission_bench procedural --maps 20 --n 32 --batch 32
```
