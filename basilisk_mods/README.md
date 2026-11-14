# basilisk_mods/

**Owner:** Tanush (orbit/descent), Arnav (terrain/rover/power) · **Jira:** KAN-41 to KAN-46

Basilisk Python modules: orbit + release, powered descent, terrain, rover, power/comms.

| | |
|---|---|
| Reads | `champion_XXX.json`, tiles, mission.yaml |
| Writes | `runs/<run>/telemetry_XXX.csv` + Vizard binaries |

Re-flies swarm champions at higher fidelity to check the JAX sim.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.
