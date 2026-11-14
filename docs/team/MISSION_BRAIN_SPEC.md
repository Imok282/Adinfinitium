# Mission brain + autonomous rover: spec (v1, written Sat 14 Nov 2026 for the Sun 12:00 cut rule)

Decided with Imok in the planning session. **PyTorch only** (the 4090 PC is native Windows: no JAX GPU there; the JAX swarm moves to Colab). Hermes on the 4090 is remote hands for training runs only and is **not part of the project**.
Everything below keeps the repo rules: files between folders only (`contracts/`), provenance labels, a **physics backup for every AI part**, no magic numbers (config), honest limits.

## 1. What we are building, in one picture
```
 inputs from everywhere                         PLANNER (learned, L2)                 EXECUTORS (learned, L1)             PHYSICS (backup, L0)
 ─────────────────────                          ─────────────────────                 ───────────────────────             ───────────────────
 tiles: height/slope/hazard(U-Net)/PSR  ─┐      picks site + landing date             Lander pilot  (evolved, Imok,        ZEM/ZEV classic guidance
 light: Sun/Earth over time (SPICE+skyline)─┼──▶ + landing cell + go/wait   ──plan──▶  JAX, Colab) -> touchdown cell       Dijkstra route planner (time-aware)
 terrain uncertainty (generated versions)  ─┤      + rover goals + return deadline     Rover driver  (this work, PyTorch)   Navigator table ranking
 mission needs (config) + vehicle state    ─┘            ▲        │                    -> drive commands                     (exact, always works)
                                                         └── ORCHESTRATOR (state machine, L3): orbit → descent → surface → return → night, writes the decision log
```
Basilisk (other team: Tanush, Arnav) provides the 6-DOF physics and Vizard 3D; it **replays** what the models decide. Our models never depend on Basilisk to train.

## 2. Rover driver model ("Tesla-style", the new ML part)
**Honest framing:** same mechanism as an autonomous-driving stack (sense → plan → control, learned), but no real lunar driving data exists: it is trained **in simulation on real LOLA terrain** and shown as a simulated autonomy result.
- **Input (per decision, hourly):** egocentric top-down crop 64×64 px (5 m/px = 320 m) around the rover with channels: relative height, slope / limit, hazard probability (our U-Net), permanent-shadow mask (NASA PSR),
  sunlight fraction, Earth margin; plus a vector: battery fraction, hours to the next sunset at the rover and at the lander, sun elevation, relative direction + distance to the ice target and to the lander, link to the lander (bool).
- **Output:** 9 actions (8 compass headings, or stop and charge). One step = one hour of mission time; distance per step = `rover.speed_m_s` × 3600 s (36 m for the Pragyan-class 1 cm/s) reduced on slopes.
- **Rules of the world (config, design assumptions):** slope limit 20° (`rover.max_slope_deg`), unsafe cells are blocked; battery 100 Wh, panel 50 W × sunlight fraction; drive / idle loads and the link range go in a new `rover_sim` config block; the rover talks only **via the lander relay** (needs line of sight to the lander).
- **Reward = the mission score already in config:** +50 reach an ice target (PSR edge), +25 return to the lander, minus fuel-like penalties (energy), death if the battery empties away from the lander (`night_death`) or the link is lost for too long (`no_comms`).
- **Training:** (1) behaviour cloning from the physics route planner (labels for free), then (2) policy-gradient fine-tuning (PPO) with thousands of parallel rovers on the GPU. Held-out terrain for the exam: **Site 23 and Site 42** (never trained on).
- **Physics backup:** time-aware Dijkstra on a cost map (slope, hazard, sunlight) with a hard "turn back by T" rule. **Proof it works:** success rate, energy used and ice reached vs the planner and vs a naive straight-line driver on the held-out sites, per-seed variance reported.
- **Provenance:** terrain OBSERVED/MODELLED, light MODELLED (SPICE + exact skyline), ice from NASA PSR (OBSERVED) but "ice present" is UNKNOWN: PSR only means dark and cold, not confirmed ice.

## 3. Mission planner (learned selector, L2)
- **Job:** over candidate (site, date, landing cell) choose the one that maximises probability of mission success, and decide go / wait (later date).
- **Input per candidate:** Navigator-style metrics (Sun hours, longest darkness, Earth hours, blackout), hazard fraction near the cell, landing scatter radius (from the lander pilot's touchdown spread), distance to nearest ice target, rover feasibility (from the rover driver's simulated success).
- **Model:** a small MLP trained on **simulated outcomes** (rover env + lander scatter) as a success-probability predictor, with calibration shown; output = ranked candidates + confidence. **Backup:** the exact ranking of Navigator metrics.
- **Output file (new contract `contracts/plan.py`, to add with tests):** `runs/<run>/plan_XXX.json`: site, landing_time_utc, target cell, scatter radius, rover goals (cells), return deadline, battery/heater genes, rationale (top factors), provenance.

## 4. Lander pilot (existing, Imok): interface only
`swarm/` evolves the pilot (JAX, now on Colab). The planner needs only its **touchdown scatter** around a target and its success rate; the orchestrator hands it the plan's target cell. Basilisk re-flies champions (KAN-43) for the 3D story. Orbit-phase decisions (when to descend, abort and retry) are **out of scope for v1** (cut line).

## 5. Orchestrator (L3, state machine)
orbit → descent → touchdown → surface (rover out, power) → return before sunset → night → sunrise. Each transition writes one row into the telemetry `decision` column (the "no commentary, just decisions" log the mission-control window shows).

## 6. Compute and the Sun 12:00 cut rule
| Part | Where | Notes |
|---|---|---|
| Rover env + policy, planner | 4090 PC, native Windows, PyTorch | scripts cross-platform, `--resume`, small CPU smoke test on the Mac |
| JAX swarm (lander pilot) | Colab (user drives Chrome) | `swarm/README.md` |
| Basilisk rover + Vizard | other team, laptops | consumes our policy outputs |
Order: spec (this) → rover environment + PSR → physics planner (backup, labels) → behaviour cloning → PPO → exam on held-out sites → planner model → orchestrator log. **Anything not working by Sun 12:00 ships as its physics backup** and the slides say so.

## 7. Honest limits to state out loud
Simulation only; hourly steps and a kinematic rover (Basilisk adds the wheel physics); light from our skyline model out to ~1.3 km inside a ~5 km LOLA window; LOLA heights mostly interpolated; PSR means dark and cold, not proven ice; no real lunar driving data; one lunar region family (south pole); small models (millions of parameters) trained from scratch.

## 8. Who builds what
This session (Mac): spec, rover environment, PSR ice map, physics planner, driver training scripts, planner model. 4090 (via SSH, remote hands): the training runs. Other team: Basilisk rover module (KAN-45), power/comms (KAN-46), Vizard (KAN-47), mission control (KAN-48). Ojas's session: swarm on Colab, Navigator, Dream-RSI with the pilot.
