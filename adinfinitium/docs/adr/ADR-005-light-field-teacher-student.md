# ADR-005: Light field = exact physics teacher + neural student

**Status:** Accepted
**Date:** 2026-11-14
**Deciders:** Imok

## Context
The brief asks for Sun and Earth visibility against the local horizon (terrain + curvature,
no refraction) for any site and date. The swarm needs these values at thousands of
(x, y, t) points per generation. An exact ray-cast over LOLA is correct but slow.

## Decision
- **Teacher:** horizon by ray-casting LOLA (with lunar curvature) + Sun/Earth vectors from SPICE (`MOON_ME_DE440_ME421`). It outputs the finite-disk Sun fraction, Sun elevation and Earth elevation.
- **Student:** a neural field f(x, y, t) trained on the teacher's samples and used by the swarm for speed.
- The Navigator and validation use the **teacher**. The student is shown with its error map.

## Options Considered
| Option | Speed | Correctness | Risk |
|---|---|---|---|
| A. Teacher + student (chosen) | fast in the swarm | teacher is exact; student error is measured | training time |
| B. Teacher only, precomputed grids | fast lookups, but memory-heavy for fine t | exact | storage; coarse time resolution |
| C. Neural only | fast | unverifiable | fails validity |

## Trade-off Analysis
Option B is the fallback and also the training data for A, so A strictly dominates once the
student's error map is acceptable (target: Sun elevation error < 0.5°).

## Consequences
- Easier: instant queries for the swarm and the Navigator's countdowns.
- Harder: needs a validation of the teacher against JPL Horizons at ≥ 3 sites and dates (KAN-22).
- Revisit: horizon precompute per cell (azimuth bins) to speed up the teacher.

## Action Items
1. [ ] SPICE + Horizons check (KAN-59/60)
2. [ ] Teacher maps for the demo region (KAN-61/62)
3. [ ] Student training + error map + fallback switch (KAN-95)
