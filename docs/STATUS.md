# Project status — read this at the start of every coding session

_Last updated: Sat 14 Nov 2026, 10:45 IST. Update this file whenever something important changes._

## Where the code is
| Branch | PR | Contains | Merge order |
|---|---|---|---|
| `imok/KAN-19-skeleton` | #1 → `main` | contracts, mission.yaml, JAX lander sim + classic guidance, system design, ADRs, 12 diagrams, CI | **1st** |
| `imok/KAN-41-basilisk-starter` | #2 → `imok/KAN-19-skeleton` | tested Basilisk starters (orbit, descent, terrain), mission-control fixture CSV, simulation team handbook | **2nd** (after #1, retarget to `main` if GitHub doesn't do it) |

After both are merged, everyone works from `main` + their own task branch.

## What works (tested)
- `python -m contracts.mission` · `python -m data_tools.validate_formats --selftest`
- `python -m scripts.bench_swarm --n 1000`: classic guidance 99.7% success on the demo crater tile with failures on; ~1.8 M lander-steps/s on CPU
- `python -m basilisk_mods.scenario_orbit_release`: 15 km PDI at 1,692 m/s
- `python -m basilisk_mods.scenario_descent [--engine-out-at 20]`: lands within limits
- `pytest`: 27 pass with Basilisk installed; 24 + 3 skipped without

## Next tasks for Imok (in order)
1. **KAN-101:** on the 4090, run `make bench` (10,000 landers) and record the time in the PR / `docs/VALIDATION.md`
2. **KAN-21/22:** LOLA tiles + SPICE kernels → `data/tiles/*.npz` (pen drive); 3 projection test points for Arnav
3. **KAN-23:** light-field teacher → `data/light/*.npz` (Arnav's power module and Tanush's CH-3 replay wait on this)
4. **KAN-105:** TensorNEAT wrapper over `sim_jax.rollout` + generation logs + overnight script (start before sleeping)
5. **KAN-103/104:** surface phase (rover, power, night) + surprise-hazard failure
6. **KAN-100:** exam sets that stress what classic guidance can't do (hazards near target, surprise rocks, engine-out)

Team tasks: `docs/team/SIMULATION_TEAM_HANDBOOK.md` (Tanush, Arnav). Nimansh: Navigator (KAN-35), tests (KAN-36), README (KAN-37).

## Hazard detector (KAN-25, Nimansh) — branch `nimansh/KAN-25-hazard`
Trained on real LOLA 5 m tiles (Sites 01/04/06/07/20; val Site 11; **test Site 23, never seen**). Same sparse input for both:
slope-rule backup F1 0.814 (misses 20.9% of hazards) vs U-Net F1 0.950 (misses 4.7%). Label = NASA slope map (computed from
interpolated heights), so it "recovers the official hazard map from sparse laser data", not proof of safety. Rocks and PSR = UNKNOWN.
Shared files touched: `config/mission.yaml` + `contracts/mission.py` (`hazard_detector` block), `contracts/paths.py` (`model_path`).
Details + how to run: `worldmodel/hazard/README.md`.

## Shared-sim change (KAN-100, branch `imok/KAN-100-lookahead`) — read if you touch `sim_jax/lander_env.py`
- New opt-in `EnvParams.lookahead` (default **off**): appends 8 `ring_unsafe_*` values AFTER the base observation.
  Observation indices 0..12, `OBS_NAMES`, `OBS_DIM` (=13) and `classic_policy` are unchanged; with the flag off nothing
  behaves differently. `EnvParams.obs_dim` gives the real length. `make_params(..., lookahead=True, lookahead_m=30.0)`.
- Basilisk / Navigator / mission control do not read the sim's observation vector, so no action is needed from you;
  tell Imok if you see anything change. Exams (`worldmodel/exams.py`, `python -m worldmodel.exams`) and the swarm
  (`--exams`, `--lookahead`, `--resume`) are documented in `swarm/README.md`.

## Navigator (KAN-35, Nimansh) — branch `nimansh/KAN-35-navigator` (based on `imok/light-contract-v2`)
Streamlit app on port 8501: compare 4 site/date combos, lunar-day timeline, sunset/Earthset countdowns, sky panorama + dome over the terrain
horizon, terrain/hazard map. Runs today on a **GENERATED** stand-in (`python -m data_tools.make_demo_light`: analytic Sun/Earth, arbitrary
phases, NOT SPICE; horizon ray-cast from the real LOLA windows). Switches to Imok's real `*_points.npz` with no code change.
Ice/PSR = UNKNOWN until KAN-39. Details: `navigator/README.md`.

## World models (Nimansh, 2026-11-14 evening) — branches `nimansh/KAN-24-light-model`, `nimansh/KAN-26-terrain` (stacked on `imok/integration`)
- **Light model (KAN-24), trained:** predicts the terrain skyline; physics does the rest. Held-out Site 23: skyline error 0.63° -> 0.47°, sunlit-hours error 21 h -> 14 h vs the physics backup. `worldmodel/light/README.md`.
- **Real ephemeris:** NASA SPICE DE440 Sun/Earth directions (`worldmodel/light/ephemeris.py`), checked against JPL Horizons to 0.00003°. `worldmodel.light.make_points` writes REAL point series (MODELLED) for 12 months of 2027 per site; the analytic stand-in is retired for those sites.
- **Terrain generator (KAN-26):** code + tests + physics backup done; model not trained (planned on the 4090). Outputs always GENERATED.
- **Data:** the fetcher can now pull a window grid per site and 4 more sites (Site42, Haworth, Shoemaker, DM2): `python -I -m worldmodel.hazard.lola all --grid --sites ...`. Not run locally (disk nearly full); plan: Colab/4090. Test sites: Site23 + Site42.
- **Viewer:** `streamlit run model_viewer/app.py --server.port 8504` (3D, trial).
- Shared files touched: `config/mission.yaml` `light_model` block, `contracts/mission.py` `LightModel`.

## Decisions already made
- Submission: **Final project = all three** (hosted web app + GitHub repo + Google Drive). **Repo goes public Sunday** before submitting. **Demo = 30 s video with English subtitles.** **Imok owns the Drive** and tests every link in incognito.
- AI-generated images/video need a visible "AI-generated" watermark, and no NASA logos may appear in generated content. AI-written code is disclosed in the "Use of AI" field.
- Don't use Atlassian/Jira from the agent unless asked. Task text lives in the repo docs.

## Known gaps / risks
- The JAX sim starts at the terminal phase (2 km), not at PDI. The Basilisk braking phase from PDI is a stretch goal (cut rule Sun 12:00).
- Terrain-in-Vizard via `vizSupport.createCustomModel` is untested.
- The polar-stereographic convention in `lunar_terrain.py` must be verified against the LOLA label.
- Classic guidance already succeeds at 99%+ on clear targets. The exams must test harder cases, or the AI shows no gain.

## Deadlines (IST)
Checkpoint 1 Sat 21:00 · cut rule Sun 12:00 · code freeze Sun 17:00 · submit Sun 22:00 (official close is 23:59 local; "Submit for Judging" is final)
