# ADR-004: Two simulation fidelities — JAX swarm for search, Basilisk for checking

**Status:** Accepted
**Date:** 2026-11-14
**Deciders:** Imok (JAX), Tanush + Arnav (Basilisk)

## Context
The search needs millions of cheap missions. Credibility needs a recognised astrodynamics
framework with orbit → release → deorbit and engine models. No single tool does both in 48 hours.

## Decision
- **Search** in the JAX 3-DOF simulator (`sim_jax/`), starting at the terminal phase.
- **Check** the top champions in Basilisk, using our Python modules (terrain, powered descent, rover, power/comms) by replaying `champion_XXX.json`.
- Basilisk writes `telemetry_XXX.csv` for mission control, plus Vizard binaries for 3D.

## Options Considered

### Option A: JAX search + Basilisk check (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Medium — two sims must agree |
| Cost | Basilisk runs only a handful of missions |
| Scalability | Search scales on the GPU; checking stays small |
| Team familiarity | Basilisk work goes to the vibe-coders, who have clear inputs and outputs |

**Pros:** both speed and credibility; a natural parallel split of work.
**Cons:** a model gap (3-DOF vs 6-DOF, uniform vs point-mass gravity) must be reported honestly.

### Option B: Basilisk only
**Cons:** around 10–100 missions per minute at most, so no swarm.

### Option C: JAX only
**Cons:** "we wrote our own sim" carries weaker validity with judges.

## Trade-off Analysis
Showing the agreement between the two sims becomes a validity exhibit: "champion success
87% in the swarm and 85% in Basilisk on the same missions."

## Consequences
- Easier: credibility, a 3D replay, a clear split of work.
- Harder: an agreement metric is needed, and initial states and frames must match (contracts/runs.py `Champion.initial_state`).
- Revisit: move point-mass gravity and orbit phases into JAX.

## Action Items
1. [ ] Champion export with the full control log (KAN-110)
2. [ ] Basilisk re-fly + agreement table (KAN-43)
