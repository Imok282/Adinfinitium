# Brief for Tanush (from the orchestrator, 2026-11-14)

Your tasks and prompts are in `docs/team/SIMULATION_TEAM_HANDBOOK.md` (T1..T7). Priority order for today:

1. **T3 / KAN-38 `sites/sites.json` first.** Nimansh's Navigator and the light-field tiles depend on it. 15+ sites, official sources only,
   `provenance: "observed"` (region centres: "modelled"), validate with `python -m data_tools.validate_formats sites/sites.json`.
   Each site's `region` field must equal the tile/light region name (one tile per site; e.g. `chandrayaan3_shiv_shakti` at about -69.373, 32.319).
2. **T2 / KAN-42 descent on a terrain tile** with Arnav's `Terrain` (agree the function names). Engine-out demos at 5, 20, 60 s. Don't change touchdown limits.
   Stretch (braking from PDI): cut at **Sun 12:00** and keep the 2 km start.
3. **T1 / KAN-41** small items (plot check, move `PROPULSION_MODULE_MASS_KG` into `config/mission.yaml` + `contracts/mission.py` with a test).
4. **T5 / KAN-40 Chandrayaan-3 replay** needs Imok's light field for 20 Aug - 10 Oct 2023: format is light contract v2 (window start date, 1 h cadence).
5. **T6 / KAN-43 re-fly champions Sunday morning** from `runs/<run>/champion_XXX.json` (Imok's overnight swarm run). Report JAX-vs-Basilisk differences as they are.
6. **T7 / KAN-50 submission page** Sunday evening. The residual pilot must be described as "evolved residual on classic guidance".

FYI: the shared sim has an opt-in look-ahead change (`imok/KAN-100-lookahead`, default off): nothing for you to change.
Write questions or blockers to `docs/outbox/tanush.md`.
