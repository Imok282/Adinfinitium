# sim_jax/

**Owner:** Imok · **Jira:** KAN-28, KAN-29

GPU simulator: 10,000 landers at once (JAX, vmap + scan).

| | |
|---|---|
| Reads | `config/mission.yaml`, tiles, light fields |
| Writes | Per-mission metrics + failure reasons |

`lander_env.py` (3-DOF env), `guidance.py` (classic ZEM/ZEV baseline), `rollout.py` (vmap), `terrain.py`. Next: surface/power/rover/night (KAN-103), failure injection extras (KAN-104).

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.
