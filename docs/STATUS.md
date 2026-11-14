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
