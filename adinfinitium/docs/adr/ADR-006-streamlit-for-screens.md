# ADR-006: Streamlit for the three screens

**Status:** Accepted
**Date:** 2026-11-14
**Deciders:** Nimansh (Navigator), Arnav (mission control), Imok (brain window)

## Context
There are three screens: Navigator (:8501), mission control (:8502) and brain window (:8503).
Two builders vibe-code. Everything else is Python. The screens read contract files and
refresh every second or so.

## Decision
Use Streamlit + Plotly for all three screens, each one a standalone app reading files via `contracts/`.

## Options Considered
| Option | Build speed | Look | Fits Python data | Vibe-code friendly |
|---|---|---|---|---|
| **Streamlit (chosen)** | very fast | good enough | yes | yes |
| Dash | fast | good | yes | medium |
| React + FastAPI | slow | best | needs an API layer | no |
| Jupyter | fast | poor for demo | yes | yes |

## Consequences
- Easier: a working screen in about an hour; reuses the contract loaders directly.
- Harder: the custom 3D sky dome must be done in Plotly; less control over the visual style.
- Revisit: a React front-end only after the hackathon.

## Action Items
1. [ ] Navigator skeleton (KAN-63)
2. [ ] Mission-control window reading `telemetry_XXX.csv` (KAN-89)
3. [ ] Brain window reading `generation_XXX.json` (KAN-108)
