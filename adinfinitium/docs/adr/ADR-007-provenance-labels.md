# ADR-007: Every output carries a provenance label

**Status:** Accepted
**Date:** 2026-11-14
**Deciders:** whole team

## Context
About 90% of 5 m LOLA pixels are interpolated, and we generate terrain with diffusion and
craters synthetically. Judges score Validity. A known failure in similar projects is presenting
synthetic-terrain results as real.

## Decision
`contracts.provenance.Provenance` ∈ {observed, modelled, generated, unknown} is mandatory in
every file's `Meta`. The UI shows `Provenance.banner`, and generated data is always labelled
**"GENERATED — NOT OBSERVED"**. Generated data is never used alone for a safety claim.

## Options Considered
| Option | Validity | Effort |
|---|---|---|
| **Mandatory label in every file + UI banner (chosen)** | high | low — one field |
| Note in README only | low | none |
| Separate folders for real vs generated | medium | brittle |

## Consequences
- Easier: honest claims; a validation slide almost writes itself.
- Harder: every producer must set it (enforced by pydantic: `provenance` has no default).
- Revisit: per-pixel provenance (`real_mask`) is already optional in tiles; make it required once KAN-21 has spot-density data.

## Action Items
1. [x] `Meta` + `Provenance.banner` + tests
2. [ ] UI banner component shared by the 3 screens
