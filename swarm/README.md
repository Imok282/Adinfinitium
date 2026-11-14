# swarm/

**Owner:** Imok · **Jira:** KAN-30 to KAN-32, KAN-34

TensorNEAT evolution + Dream-RSI-style meta layer + champion export.

| | |
|---|---|
| Reads | sim_jax env |
| Writes | `runs/<run>/generation_XXX.json`, `champion_XXX.json` (contracts/runs.py) |

Fallback: fixed-topology network with evolved weights (KAN-106).

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.
