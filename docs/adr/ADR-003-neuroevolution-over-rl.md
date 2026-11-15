# ADR-003: Neuroevolution (TensorNEAT) for the pilot, not reinforcement learning

**Status:** Accepted
**Date:** 2026-11-14
**Deciders:** Imok, team

## Context
The genome must hold both a **brain** (thrust control) and **mission genes** (site, landing
time, battery size, heater power, rover route preference). Fitness is gated (an unsafe
landing scores 0), sparse, and only known at the end of a mission. Judges need to *see* the
AI improve, and the plan already includes a swarm map and a live network drawing.

## Decision
Evolve the pilots with TensorNEAT (GPU NEAT on JAX), in this build order: ① NEAT (topology
grows) → ② fixed-topology network with evolved weights (robust fallback) → ③ Dream-RSI-style
meta layer → ④ recurrent memory (stretch). Classic ZEM/ZEV guidance is the baseline and the backup.

## Options Considered

### Option A: NEAT via TensorNEAT (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Cost | Large sample counts, paid for by the 10k-parallel sim |
| Scalability | Population-parallel on the GPU |
| Team familiarity | Low–Medium |

**Pros:** gradient-free, so gated and discontinuous fitness is fine; evolves discrete mission genes with the same machinery; small networks that can be drawn; the "brain grows" story for the video.
**Cons:** sample-hungry; genome padding limits network size; library risk.

### Option B: PPO / SAC (RL)
**Pros:** sample-efficient for continuous control.
**Cons:** credit assignment for a sparse, gated reward needs shaping; mission genes don't fit naturally; harder to explain in 30 s.

### Option C: Classic guidance only
**Cons:** no AI story; it can't choose the site and time or handle surprise hazards on its own.

## Trade-off Analysis
The simulator makes samples cheap, which removes NEAT's main weakness. NEAT's strengths
(mixed genomes, visible structure, no reward shaping) match what we need for judging.

## Consequences
- Easier: one fitness function for brain + mission; the swarm visual comes for free.
- Harder: the exam must test cases classic guidance can't handle, or the AI shows no gain (see SYSTEM_DESIGN §4.2).
- Revisit: hybrid — evolve the guidance gains and gate altitude instead of raw thrust if NEAT plateaus.

## Action Items
1. [ ] TensorNEAT problem wrapper over `sim_jax.rollout` (KAN-105)
2. [ ] Fixed-network fallback + AI vs classic plot (KAN-106)
3. [ ] Exam sets that stress hazards near the target, surprise rocks and engine-out (KAN-100)
