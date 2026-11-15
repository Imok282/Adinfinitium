---
license: apache-2.0
library_name: pytorch
tags:
- lunar
- moon
- lunar-south-pole
- nasa-lola
- space-apps
- adinfinitium
- simulation
---

<p align="center"><img src="logo.png" alt="Adinfinitium" width="320"></p>

# Adinfinitium Light / Skyline Transformer

**Status:** Final · **License:** Apache-2.0 · **Domain:** lunar south pole (NASA LRO LOLA 5 m)

Predicts the 360° horizon elevation (skyline) at a lunar surface cell. Physics then converts the skyline into Sun fraction and Earth visibility.

## Model details

| | |
|---|---|
| Task | Skyline regression: horizon height in 360 compass directions per cell |
| Parameters | **38,279,809** |
| Architecture | `SkylineFormer`: circular conv stem along each compass ray → transformer across 360 directions → zero-initialised head. Prediction = physics baseline + learned correction (an untrained model equals the physics backup). `width 64, n_down 3, d 512, layers 12, heads 8, k_steps 64, n_az 360` |
| Training | 3,258 of 6,000 planned steps (54%); best checkpoint by validation skyline error on Site11 |
| Checkpoint | `light_skyline_big.pt` · 153,175,919 bytes · sha256 `673b6715…e11d3` |
| Wiring | Not wired to the rover (the rover uses exact physics ray-casting) |

## Benchmarks

Skyline mean absolute error in degrees (lower is better), against the coarse ray-cast baseline.

![Light transformer vs coarse ray-cast baseline](assets/benchmarks.png)

| Test set | Ours | Coarse ray-cast baseline |
|---|---|---|
| Skyline MAE, Site23 + Site42 | **0.450°** | 0.570° |
| Skyline MAE, fresh sites | **0.243°** | 0.321° |
| Sunlit-hours error | **7.6 h** | 10.5 h |

Lower error on every one of the 19 test windows. Per site: Site23 0.534° vs 0.675°, Site42 0.270° vs 0.342°.

## Limitations

- About **227× slower** than the baseline (about 7.4 s vs 0.033 s per window); half precision changed the output by up to 0.327° and was rejected.
- Truth and model both ignore terrain beyond 1,280 m.
- Finishing the planned 6,000 steps changed nothing real (−0.0014°), so the frozen 3,258-step checkpoint is the one released.

## Usage

```python
# project code: https://github.com/oasisk/adinfinitium-handoff
from worldmodel.light.model import SkylineStudent, load_checkpoint, pick_device
```

## Dream-RSI: gated offline self-improvement

Dream-RSI lets a model keep improving after the main training ends, offline and safely:

1. **Find failures**: fly or replay many missions and record where the current model fails.
2. **Dream harder**: generate and replay those hard cases (failure-boosted scenarios) and retrain a candidate, warm-started from the current model.
3. **Gate**: the candidate is scored on a frozen validation exam (fixed seeds, paired with the current model and the physics backup).
   - **Lander:** accept only if the candidate loses no mission that the current model or the physics backup lands (zero regressions) AND improves total success.
   - **Planner and rover:** accept only if the frozen validation metric improves by a margin.
4. **Repeat** from whichever model won. Rejected candidates are discarded, and the shipped model never gets worse.

Test maps and the final-exam site are never used in Dream-RSI: training uses training maps, and gating uses a separate validation map.

### Measured results (all logged rounds)

![Dream-RSI rounds accepted per loop](assets/dream_rsi.png)

| Loop | Rounds accepted | Effect on its gate metric |
|---|---|---|
| Planner (PlannerSetNet) | 1 of 5 | Validation Brier 0.01543 → 0.01353 (−12.3%) |
| Lander pilot (Site11 gate, 1,024 paired missions) | 0 of 7 | Pilot unchanged; every candidate lost 12–25 missions, so the gate held |
| Rover brain v3.1 (self-learning on stressed worlds) | 9 of 36 | Blocked route improved at 4/4 sites, late light 3/4, dusty panels 1/4, early light 0/4 |
| Rover brain v3 | 9 of 72 | Blocked route improved at 6/6 sites, dusty panels 2/6 |
| Rover learned driver v2 | 5 of 38 | Returned rate stayed 0.0% |

Across all loops, 18 of 42 first rounds on a new problem were accepted, against 6 of 120 later rounds.
**What this shows:** Dream-RSI reliably repairs specific failures (blocked routes, late light, model faults) without ever shipping a regression. In unchanged conditions it rarely finds a safe gain (0 of 32 rover rounds accepted).

### Projections after more Dream-RSI (Projected, not measured)

Median after 20 more generations, from the planner's own Dream-RSI log:

- **Planner Brier:** 0.0145 → 0.0094 [0.0043, 0.0145]

Scenario only, for models where Dream-RSI has not been run yet (it borrows the improvement rate seen in the planner and rover loops):

- **Hazard fresh F1:** 0.757 → 0.779
- **Light error:** 0.450° → 0.410°
- **Terrain error:** 2.01 → 1.84 m

The lander and rover v3.1 are projected to stay unchanged, because their gates have not accepted anything in unchanged conditions.

### Limitations and next steps

- The zero-regression gate is strict: it also rejected lander candidates that were slightly better overall (best: 991 vs 990 landings).
- Training rounds were short (4–5 updates) because of the compute budget.
- **Next:** harder self-generated scenarios, longer rounds, and a fresh unseen-map exam after each accepted generation.

## Intended use and scope

Research, education and benchmarking of lunar landing-site navigation methods. **Not for flight, safety-critical or
operational landing decisions.** All results are simulation or held-out-lunar-site results, not real-mission performance.
Models were frozen (SHA-256) before held-out evaluation; each test site was evaluated once.

## About Adinfinitium

Adinfinitium is an open-source effort to make lunar landing and surface navigation easier to study. It began as our
NASA Space Apps 2026 project, *Lunar Landing Site Navigator*, and trains on NASA LRO LOLA 5 m terrain of the lunar south
pole. The model chain is: hazard network → light network → terrain model → rover brain → lander pilot.
See the whole family at https://huggingface.co/Imok282.

Source code, full records and the hand-off package: https://github.com/oasisk/adinfinitium-handoff (checkpoint SHA-256 values: `CHECKPOINTS.sha256`).

## Citation

```bibtex
@misc{adinfinitium2026lunar,
  title  = {Adinfinitium open models for lunar landing-site navigation},
  author = {Adinfinitium},
  year   = {2026},
  howpublished = {Hugging Face, https://huggingface.co/Imok282},
  note   = {NASA Space Apps 2026, Lunar Landing Site Navigator}
}
```
