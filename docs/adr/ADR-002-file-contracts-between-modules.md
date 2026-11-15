# ADR-002: Modules communicate only through versioned files (`contracts/`)

**Status:** Accepted
**Date:** 2026-11-14
**Deciders:** whole team

## Context
Four people work in parallel for 48 hours. Two of them vibe-code with AI agents and can't
debug cross-module imports. The 4090 has no Claude and gets big data by pen drive. The
producers (data prep, swarm, Basilisk) are batch jobs, and the consumers (screens) only need
to refresh every few seconds.

## Decision
Every hand-off is a file with a typed loader and validator in `contracts/`: tiles, light
fields, sites, mission.yaml, and generation / champion / telemetry files. Folders never import
each other's internals. Each file carries a `Meta` block (provenance, source, frame, units,
schema_version).

## Options Considered

### Option A: File contracts (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | None |
| Scalability | Fine for batch producers and second-level refresh |
| Team familiarity | High — everyone understands "write a file, read a file" |

**Pros:** no one waits on anyone; works offline and over a pen drive; a demo tile unblocks everyone on day 1; easy to test (round-trip tests); agents get a precise spec.
**Cons:** consumers must poll; no back-pressure; schema drift is possible without discipline.

### Option B: Python services / REST between modules
**Cons:** process management on 5 machines; network setup in a hackathon venue; agents add boilerplate.

### Option C: Message queue (ZeroMQ / Redis)
**Cons:** live streaming isn't needed for judging; adds an install on the 4090.

### Option D: Shared database (SQLite / Postgres)
**Cons:** large arrays fit `.npz` better; locking across a pen drive copy is fragile.

## Trade-off Analysis
Coordination cost is the scarcest resource in 48 hours, and files minimise it. We give up
live latency, which the judges never see; the screens refresh every 1–2 s.

## Consequences
- Easier: parallel work, validation in CI, swapping AI ↔ backup (same file shape).
- Harder: a schema change needs a contracts PR plus a heads-up to the team.
- Revisit: a streaming bus if the screens need under 1 s of latency.

## Action Items
1. [x] `contracts/` loaders + validators + tests
2. [x] `data_tools.make_demo_tile` (GENERATED) so consumers can start now
3. [ ] Kick-off: everyone reads §3 of SYSTEM_DESIGN.md
