# Brief for Arnav (from the orchestrator, 2026-11-14)

Your tasks and prompts are in `docs/team/SIMULATION_TEAM_HANDBOOK.md` (A1..A5). Priority order:

1. **A1 / KAN-48 mission-control window** (port 8502) from `tests/fixtures/telemetry_basilisk_engine_out.csv`. Needed for Checkpoint 1 (Sat 21:00).
2. **A2 / KAN-44 terrain module + OBJ.** Imok owes you **3 known lat/lon -> x,y projection test points** from the LOLA label; use them for the
   `ps_from_latlon` test. Agree `Terrain.ground_height` / `Terrain.slope` names with Tanush.
3. **A3 / KAN-47 Vizard**: terrain-in-Vizard is untested; fall back to Blender after 1 hour. Any vertical exaggeration must be labelled.
4. **A4 / KAN-46 power + comms** reads the light data: use light contract v2 (`contracts/light.py`: Earth link = `earth_elev_deg > 0`; point series for
   per-site runs; date = window start). Until Imok's real light field exists use a clearly labelled **GENERATED** synthetic day/night cycle.
5. **A5 / KAN-45 rover** after A2 and A4. Add `ROVER_COLUMNS` to `contracts/runs.py` with a test first.
6. Team plan also lists a **JPL Horizons check** of Sun/Earth angles at 3 sites x 3 dates; confirm with Imok whether you or Imok (KAN-22) own it.
7. Confirm the Vizard download link in the handbook (1 minute).

FYI: the shared sim has an opt-in look-ahead change (`imok/KAN-100-lookahead`, default off): nothing for you to change.
Write questions or blockers to `docs/outbox/arnav.md`.
