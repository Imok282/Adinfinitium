# KAN-48 · Professional lunar landing simulator (plan)

Branch `ojas/KAN-48-live-mission` · 15 Nov 2026 · decided with Imok in the planning chat.

## What it is

A live, operator-run simulator of the whole landing: **Basilisk** flies the physics (orbit → separation → deorbit →
braking → AI-piloted terminal descent → touchdown → rover), **three screens** show it in the style of the ISRO mission
control room, and an **operator console** runs it like a real sim: time warp, failures, GO/NO-GO polls, aborts.
Not a video: every number on screen comes from the running simulation.

## Three screens

| Screen | Content |
|---|---|
| 1 · Wall (left) | ISRO-style front wall. Status strip on top (UTC from the SPICE epoch, mission elapsed time, phase, time warp, AI or classic, GO/NO-GO lights). Panels: south-pole map with ground track and landing site; trajectory (orbit → PDI → braking → descent, altitude vs downrange); gauges (altitude, speeds, fuel, thrust, throttle); vehicle schematic (5 engines, 8 attitude thrusters, tank, 4 footpads lighting up live); event ticker. |
| 2 · 3D + Vizard (centre) | Our three.js render and Basilisk's Vizard, side by side, both live from the same run (`/showcase/duo.html`). |
| 3 · Console (right) | Operator controls (below) + Mission Control desks + the AI view (pilot brain). |

Colours: midnight black, electric blue, cantaloupe orange, plus **green / amber / red only for GO / CAUTION / NO-GO**.

## Basilisk physics upgrades

1. **Full mission in one run.** Lander docked on the propulsion module in a 100 km polar orbit whose plane passes over
   the site at the landing time; real push-off separation (two spacecraft); deorbit burn on the main engines (finite
   burn, not impulsive) timed so that the braking ends above the site; coast to the 15 km periapsis (PDI).
2. **Braking leg 15 km → 2 km** (new, written for this task): classic guidance from the PDI state (~1.7 km/s) to the
   terminal gate the AI pilot was trained for (2 km, ±600 m, 30–60 m/s down, ≤25 m/s sideways). Labelled classic.
3. **Terminal descent 2 km → touchdown**: the learned residual pilot (classic backup on any error) as today.
4. **SPICE (DE440)** at the planner's landing slot (demo 2027-03-02T00:00Z): Sun, Earth and Moon from the NAIF kernels
   in `data/kernels`; Moon at the origin; Earth and Sun as third bodies; the site turns with the Moon (MOON_PA frame).
   Correct lighting in Vizard and in our 3D view.
5. **8 × 58 N attitude thrusters** at the model's `rcs0..7` nozzles replace the ideal torque
   (control torque → thruster force mapping → on/off firing). Puffs in 3D, plumes in Vizard.
6. **Footpad terrain check** on real LOLA heights and slopes: which pad touches first, tilt, slope limit, leg stroke.

## Operator console

- Time warp and pause (1×, 10×, 60×, 600×; descent is never warped past 10× unless asked).
- Failure injection: engine out (which, when), stuck attitude thruster, navigation drift, blind camera.
- Site and start state: Site01 / Site23 (and any tile in `data/tiles`), start altitude and offset; restart.
- AI on / off: switch the AI pilot and classic guidance live.
- GO / NO-GO polls before separation, deorbit and powered descent: the sim holds; each check is computed (fuel margin,
  attitude error, Sun angle, Earth link, hazard at the target), and the flight director confirms.
- Abort and retarget: abort to hover, divert to the next safe cell from the hazard map, abort to orbit while braking.
- Record, replay, ghosts: every run is recorded; any run can be replayed; failed runs show as ghosts next to the live one.
- Comms: 1.3 s Earth–Moon light delay on commands, loss of signal when the Earth is below the horizon, and a procedure
  checklist per phase.

## Professional features (after the core)

- **Scenario library**: one-click presets (nominal; engine out at 30 s; navigation drift; blind camera; stuck attitude
  thruster; steep site; low Sun), each a saved start state + failure schedule + seed, so every run is repeatable.
- **Limit checking and alarms**: every telemetry value has caution / warning limits from `config/mission.yaml`
  (amber / red); alarms are logged, must be acknowledged, and show which rule fired.
- **Save, restore, rewind**: snapshot the full Basilisk state at any time (and automatically at each GO/NO-GO), restore
  it and fly again ("rewind to PDI").
- **Monte Carlo batch**: N runs with dispersed start states and failures (no graphics, as fast as possible); the
  landing ellipse, success rate and failure reasons per scenario, AI vs classic.
- **Console roles**: FLIGHT (director: GO/NO-GO, aborts), GNC (guidance, attitude, AI pilot), PROP (engines, attitude
  thrusters, fuel), SURFACE (terrain, hazards, rover); each role sees its own console page.
- **Data export**: every run's telemetry, events and commands as CSV/JSONL with provenance, for the debrief.

## Order of work

1. SPICE + full-mission Basilisk run (orbit, separation, deorbit, braking, descent) with the time-warp/pause loop and a
   command channel from the console to Basilisk (server forwards console commands to the producer).
2. Attitude thrusters, footpad check, Vizard look (terrain patch, lighting, plumes, cameras, our coloured models).
3. Our 3D view follows the whole run (orbit scene from Basilisk states, attitude-thruster puffs, SPICE Sun).
4. Wall screen (ISRO style) and console screen with the operator controls.
5. GO/NO-GO, aborts/retarget, record/replay/ghosts, comms delay and checklists.

## Stated simplifications and open items

- Point-mass Moon gravity (no mascons); no terrain occlusion of the Sun inside Basilisk (the 3D view and light model
  have it).
- RCS specific impulse is not in `config/mission.yaml`; the main-engine value (310 s) is used until one is added.
- The pilot was trained in a 3-DOF twin with ideal attitude; with attitude thrusters it flies a 6-DOF vehicle for the
  first time. Measured, not assumed: AI vs classic success on the same seeds is reported per run.
- Waiting for: `data/models/lander_pilot.pt` (trained pilot), the Site23 tile, Dream-RSI histories.
