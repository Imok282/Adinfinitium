# basilisk_mods/

**Owners:** Tanush (orbit, descent, re-fly) · Arnav (terrain, rover, power/comms, Vizard)
**Handbook (read first):** [`docs/team/SIMULATION_TEAM_HANDBOOK.md`](../docs/team/SIMULATION_TEAM_HANDBOOK.md)

Basilisk (`pip install bsk`) scenarios + **Python** modules (no C++). They re-fly the swarm's
champions at higher fidelity and make the Vizard 3D footage.

## Starters that already run (tested with bsk 2.12.2)
| File | What it does | Run |
|---|---|---|
| `bsk_common.py` | sim/task setup, Moon gravity, frames, `runs/basilisk/` output dir | — |
| `py_module_template.py` | copy-me template for a Basilisk Python module | — |
| `scenario_orbit_release.py` | 100 km polar orbit → release → deorbit → coast to 15 km PDI; writes `pdi_state.json`, plot, Vizard file | `python -m basilisk_mods.scenario_orbit_release` |
| `lander_descent.py` | lander: 5 × 800 N thrusters, fuel tank, attitude chain, ZEM/ZEV guidance module, engine-out balancing, touchdown check | — |
| `scenario_descent.py` | terminal descent → touchdown; writes contract telemetry CSV + Vizard file | `python -m basilisk_mods.scenario_descent [--engine-out-at 20]` |
| `lunar_terrain.py` | tile height/slope lookup, polar-stereographic ↔ Moon-fixed, OBJ export for Vizard | `python -m basilisk_mods.lunar_terrain data/tiles/demo_craters.npz` |

Measured: nominal touchdown 1.2 m/s vertical, 0.02 m/s horizontal, 0.4° tilt, 0.2 m from target;
engine-out at 5 s / 20 s / 60 s all land within limits (opposite engine shut down for balance).

| | |
|---|---|
| Reads | `config/mission.yaml`, `data/tiles/<region>.npz`, `data/light/*.npz`, `runs/<run>/champion_XXX.json` |
| Writes | `runs/basilisk/pdi_state.json`, `telemetry_<id>.csv`, `descent_<id>.json`, `_VizFiles/*.bin` |

Rules: talk to other folders only through `contracts/`; label outputs observed/modelled/generated/unknown;
physics backup before AI; no C++.
