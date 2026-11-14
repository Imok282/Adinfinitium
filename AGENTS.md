# Instructions for coding agents (Claude Code, Codex, Codebuff, …) and humans

Adinfinitium — NASA Space Apps 2026, *Lunar Landing Site Navigator*. An autonomous lander + rover
that land where and when a mission survives, tested by an evolving 10,000-mission swarm.

Read first: `docs/SYSTEM_DESIGN.md` (how the parts fit) and the folder `README.md` you are working in.
Your task text is in Jira (project KAN); the key is in your branch name.

## Hard rules
1. **Branches.** Never commit to `main`. Branch as `<name>/KAN-<n>-<short>` (e.g. `tanush/KAN-41-orbit`).
   Merge only when `pytest` passes. `main` must always run.
2. **Contracts.** Folders talk ONLY through the files defined in `contracts/` (tiles, light, runs,
   sites, mission.yaml). Never import another folder's internals. Need a new field? Add it in
   `contracts/` + a test in `tests/test_contracts.py`, and tell the team.
3. **Numbers live in `config/mission.yaml`.** No magic numbers for lander/rover/scoring in code.
   Load with `contracts.mission.load_mission()`.
4. **Provenance.** Every file you write has a `Meta` with `provenance` = observed / modelled /
   generated / unknown, plus source, frame and units. Anything generated is shown as
   **"GENERATED — NOT OBSERVED"** and never used alone for a safety claim.
5. **Physics backup first, AI second.** Every AI part has a non-AI fallback that runs first.
6. **Big files never go in git** (`data/`, `runs/`, kernels, checkpoints). They move by pen drive.
7. **Tests.** Add a test for anything physics-related. `pytest` must stay green and fast (< 60 s on CPU).

## Commands
```bash
pip install -e ".[dev]"          # laptop (CPU JAX)
python -m contracts.mission       # validate mission.yaml
python -m data_tools.make_demo_tile   # synthetic tile so you can work before real data
pytest
python -m scripts.bench_swarm --n 10000   # 4090 check
```

## 4090 PC (no Claude on it)
Everything there is pre-planned in `docs/plan/02_4090_SETUP_AND_PENDRIVE.md`. Code reaches it via
GitHub; data via pen drive. Set `ADI_DATA=~/adi-data` and `ADI_RUNS=~/adi-runs`.

## Deadlines (IST)
Checkpoint 1 Sat 21:00 · cut rule Sun 12:00 · code freeze Sun 17:00 · submit Sun 22:00.
