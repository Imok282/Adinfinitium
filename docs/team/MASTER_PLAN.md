# MASTER PLAN + HANDOVER: Adinfinitium (read this first, in a fresh session)

Written Sat 14 Nov 2026 (~15:30 IST) by the Claude session that worked with **Imok** (with Nimansh beside) on a MacBook. The user asked for the whole plan, in depth, so another
Claude session can take over. Everything here is either verified (tested/measured, numbers given) or explicitly marked **TODO / UNTESTED / OPEN**. Nothing is invented.
Companion docs: `docs/team/HANDOVER_WORLD_MODELS.md` (state after the world models), `docs/team/MISSION_BRAIN_SPEC.md` (rover + planner + orchestrator spec),
`docs/STATUS.md`, `AGENTS.md`, `docs/inbox/` + `docs/outbox/` (how the orchestrator's Claude talks to us: repo files only).

**User preference (from Imok, keep it):** build REAL working, validated solutions for the hackathon, not prototypes/demos; keep everything honest (physics baselines, held-out tests, provenance labels).

---------------------------------------------------------------------------------------------------------------------------------------------------------
## 0. UPDATE Sat 14 Nov, evening (read this first; sections below are older where they disagree)
- **Overnight kit EXISTS** (`ops/`): `run_all` runner (ledger with failures, status.json, per-stage time budgets, resume), `report` (RUN_REPORT.md + curves),
  `verify before|after`, `freeze` (SHA-256 of checkpoints/evals/code; nothing changes after), `make_bundle --target macmini|4090` (code + seed data zip + brief).
  Smoke-tested on the MacBook from an unpacked package (bugs it found and fixed: hazard attention window 37 GiB -> 256 px windows; seed missing `data/raw`;
  runner import path). Training scripts size themselves to `--max-minutes` (worldmodel/budget.py). Rover plan corrected: 1024 x 32 per update, not 4096 x 64.
- **Mac mini M5 run is IN PROGRESS** (sent 14 Nov ~16:00, Opus session there, ONE access): light skyline transformer ~38 M, hazard attention U-Net ~60 M,
  rover driver mini ~9 M (equal must-have), terrain diffusion mini ~50 M; ~11.7 h; brief = `ops/macmini/START_HERE.md`; results come back as
  `adinfinitium_results_macmini.zip` + `RESULTS_FOR_TEAM.md`. Known: its smoke-stage terrain uses depth 2 (fixed in the repo afterwards; harmless on 64 GB).
- **Pilot brain (NEW, `worldmodel/lander/`)**: PyTorch twin of the JAX lander (parity-tested step for step), classic ZEM/ZEV backup, `PilotFormer` (~15 M vision
  transformer residual on ZEM/ZEV, zero-init so untrained == backup), PPO, best checkpoint by validation Site11, exam on Site23 (Site42 when its tiles exist).
  Honest baseline: classic already lands 97.7 % on Site23 / 94.9 % on Site11 with failures injected.
- **Meta layer (NEW, `worldmodel/planner/`, `contracts/plan.py`)**: candidate features (SPICE+skyline light, Earth link, hazard, PSR ice edges), labels =
  simulated full missions (lander to that cell x rover drive at that hour; label kept STRICT: land AND ice AND return, decided with Imok), set-transformer planner
  + logistic baseline vs Navigator ranking backup (Brier, ECE, rank correlation, regret), `plan.py` writes `plan_XXX.json` + `decisions_XXX.csv` (orchestrator).
  Note: the physics rover rarely completes ice-and-return (mean p ~3 % in the first worlds), so the planner's signal is weak; reported as is.
- **4090: 2-hour run** (`ops/windows/run_brain.ps1`, profile `brain4090`): pilot on GPU in parallel with 3 label simulators, then exams, planner, demo plan,
  verify, freeze, zip. Windows scripts UNTESTED (written on a Mac). Package: `python -m ops.make_bundle --target 4090`.
- **3D vehicles** (branch `nimansh/KAN-47-vehicle-models`, pushed): procedural lander/rover OBJ+GLB from mission.yaml; Tripo prompts for cinematic versions in
  `docs/team/TRIPO_PROMPTS.md`.

## 1. The hackathon, the product, the rules
- **Event:** NASA Space Apps 2026, challenge **Lunar Landing Site Navigator** (brief: an intuitive tool to compare lunar south-pole landing sites and dates on Sun and Earth vs the horizon,
  power, direct-to-Earth windows, countdowns to sunset/Earthset). Pune, online team of 4 + orchestrator Ojas.
- **Deadlines (IST):** Sat 14 Nov 21:00 checkpoint 1 · **Sun 15 Nov 12:00 cut rule** (anything not working ships as its physics backup) · 17:00 code freeze · 21:00 team read-through · **22:00 submit** (official close 23:59; "Submit for Judging" is final).
- **Judging (1-5 each):** Impact, Creativity, Validity, Relevance, Presentation. Global judges mostly read the submission page, video and repo; they will not run our apps. **Judges cannot open localhost**: we need a hosted/public Navigator link.
- **Submission rules:** demo = video **<= 30 s with English subtitles** (YouTube/Drive, public); Final Project = one hub linking hosted web app + GitHub repo + Google Drive (Imok owns Drive, set "Anyone with the link: Viewer", test in incognito);
  repo is **private until Sunday**, then public before submitting; disclose that Claude and Codex wrote code; any AI-generated image/video needs a visible "AI-generated" watermark and **no NASA logos**; credit NASA data (LOLA/PGDA, SPICE/NAIF, LRO Diviner, LEND, M3) and tools (Basilisk, Vizard, TensorNEAT, JAX, Streamlit).
  Product name: undecided (decide Sunday). Do not claim Chandrayaan-3 was "the only soft landing" (false; it was the first near the south pole). Say "to our knowledge, the first" for the survive-after-landing framing.
- **What we are building (the chat-settled story):** "Adinfinitium: an AI that learns the Moon and evolves landers to discover where and when a mission can land AND survive." The Navigator is the challenge tool judges see;
  the swarm/AI models are the novelty; survival (Sun + Earth + battery + rover) is the score, not the goal.
- **Framing decided with the user: "physics-informed ML"**: neural networks trained on real NASA data, always shown next to an exact-physics baseline with held-out tests. Not "like an LLM"; our models are small (33 k to 2 M parameters) and narrow.

## 2. Team, repo, branches
| Person | Owns |
|---|---|
| Imok | 4090 PC, world models, swarm (TensorNEAT/JAX), data pipeline, Google Drive |
| Nimansh | Navigator (KAN-35), tests (KAN-36), README (KAN-37), hazard (KAN-25), light model (KAN-24) |
| Tanush | Basilisk orbit/descent (KAN-41/42), site library (KAN-38), ice layers (KAN-39), CH-3 replay (KAN-40), re-fly champions (KAN-43), submission page (KAN-50) |
| Arnav (Jira: Amit Kade) | Basilisk terrain (KAN-44), rover (KAN-45), power/comms (KAN-46), Vizard (KAN-47), mission control (KAN-48) |
| Ojas | Orchestrator; his Claude session is online and reads/writes `docs/inbox/<name>.md`, `docs/outbox/<name>.md` |
Tanush and Arnav mostly vibe-code. Repo `github.com/Imok282/Adinfinitium` (private). **Never commit to `main`; never merge (Imok merges). No `gh` CLI: give GitHub compare links.**
Branches: `main` empty; `imok/integration` (everything of Imok's + my KAN-25/KAN-35, tests green); mine: `nimansh/KAN-25-hazard`, `nimansh/KAN-35-navigator` (has a newer commit "grid azimuth" that integration lacks),
`nimansh/KAN-24-light-model`, `nimansh/KAN-26-terrain` (light + terrain + real ephemeris + viewer + handover), and **`nimansh/KAN-45-rover-brain`** (this handover: rover world/env/planner/PPO, rover_sim config, multi-window light, specs; created with this document).
Working rules (AGENTS.md): folders talk only via files in `contracts/`; numbers live in `config/mission.yaml`; every file carries provenance (observed/modelled/generated/unknown), generated terrain is shown as "GENERATED — NOT OBSERVED";
physics backup first, AI second; big files never in git (`data/`, `runs/`, `kernels/`, checkpoints); pytest green and fast. Shared files I touched (Imok must review): `config/mission.yaml` (`hazard_detector`, `light_model`, `rover_sim` blocks), `contracts/mission.py`, `contracts/paths.py` (`model_path`).

## 3. System architecture and what exists
```
NASA data → tiles (height, slope, hazard, uncertainty, PSR, laser mask) ──┬─► Hazard U-Net ─► hazard maps ─┬─► Rover world ─► Rover driver (PPO)        ─► Basilisk replay (other team)
SPICE DE440 (Sun/Earth) + exact skyline ray-cast ─► light point series ──┴─► Neural skyline model ────────┤   (+ ice targets from NASA PSR)
real laser pixels ─► Terrain diffusion (generated versions, uncertainty) ─────────────────────────────────┘
Evolved lander pilot (TensorNEAT, JAX → Colab) ─► touchdown scatter ─► Mission planner (learned selector) ─► Orchestrator (decision log) ─► Navigator / mission control / Vizard
```
### 3.1 Status table (verified unless marked)
| Part | State | Numbers |
|---|---|---|
| **Hazard U-Net** (1.94 M params), `worldmodel/hazard/` | **Trained**, merged into `imok/integration` | Held-out Site 23: F1 **0.950** vs 0.814 for the slope-rule backup (missed hazards 4.7% vs 20.9%); val Site 11: 0.962 vs 0.870. Same sparse laser-pixel input for both; label = NASA slope >= 12 deg. Checkpoint b0ac044446 |
| **Neural skyline / light model** (32.7 k params), `worldmodel/light/` | **Trained** | Held-out Site 23: skyline error 0.63 -> **0.47 deg** (p95 2.65 -> 1.99); sun up/down wrong 3.0% -> 2.2%; Earth visible wrong 2.3% -> 1.6%; sunlit-hours error 21.2 h -> **14.1 h**. Gain modest because input is a coarse 20 m terrain. Checkpoint b1fac75ad7 |
| **Real ephemeris**, `worldmodel/light/ephemeris.py` | **Done** | NASA SPICE DE440 Sun/Earth directions vs **JPL Horizons**: 0.00003 deg (elevation and azimuth, Sun and Earth, two 30-day windows at Site 23); pinned by `tests/test_ephemeris.py` |
| **Real light files**, `worldmodel/light/make_points.py` | **Done** (Mac) | 84 point-series files (7 sites x 12 months of 2027, MODELLED) in `data/light/`; replaces the analytic stand-in |
| **NASA PSR (permanent shadow)**, `worldmodel/rover/psr.py` | **Done** (7 tiles) | LOLA LPSR 20 m read as a small window over HTTP; Site 04 (Shackleton rim) 32% PSR. PSR = dark and cold, NOT confirmed ice |
| **Terrain generator** (diffusion, 2.04 M), `worldmodel/terrain/` | Code + tests + physics backup (gap-fill + LOLA-error noise) done. **NOT TRAINED** | Evaluation is hide-and-check vs the backup; if it cannot beat the backup, ship the backup |
| **Rover world + env + planner**, `worldmodel/rover/` | **Done, 11 tests** | Baselines on held-out Site 23 (Sept 2027, 512 rovers): naive 28% ice / 12% home / 88% die; cautious 6% / 4% / 77%; cautious+wait-for-light 4% / 3% / 85% |
| **Rover driving model** (PPO, 506 k params), `worldmodel/rover/train.py` | Code done and smoke-tested; **the Mac validation run (80 updates, 6 worlds x 512 envs) STALLED and was killed: the Mac had ~58 MB free RAM and a 98%-full disk and was thrashing on swap, so NO rover model has been trained yet**. Run it on the 4090/Colab, or on the Mac only with a tiny config (1 world, `--envs 128`) | Two bugs fixed on the way (MPS out-of-memory, MPS `randperm` garbage indices). No held-out evaluation script yet (TODO) |
| **Mission planner + orchestrator** | Spec only (`MISSION_BRAIN_SPEC.md`) | `contracts/plan.py` not written |
| **Evolved lander pilot** (TensorNEAT/JAX) | Imok's `swarm/` + `worldmodel/exams.py` on `imok/integration`; never run end to end here | classic guidance 99.7% safe on easy target; weak on tight fuel (25.7%, tuned classic 81%) |
| **Dream-RSI meta layer** (KAN-106) | **Not started** | needs several swarm runs with different strategies; user wants it with the pilot |
| **Navigator** (KAN-35), `navigator/` | Runs; 4 sites x dates, timeline, countdowns, sky panorama/dome, terrain tab. Now reads REAL light files | TODO: launch-slip control, hosted public link + small data bundle, use real PSR in the ice panel, README |
| **Basilisk** (`basilisk_mods/`, Tanush/Arnav) | orbit 100 km -> deorbit to 15 km works; powered descent with 5 x 800 N engines and engine-out balancing lands within limits (1.2 m/s down, 0.2 m from target) | rover/power/comms/Vizard/mission control = README stubs only |
| **3D model viewer**, `model_viewer/` (port 8504) | Works, tested | trial harness; not a deliverable |

### 3.2 Data (what is real)
- **Terrain:** NASA LRO LOLA 5 m south-pole products (PGDA, Barker et al. 2021): surface height, official slope, laser count (`real_mask`), uncertainty. ~90% of pixels are interpolated by LOLA (about 10% have laser spots).
  On the Mac: centre 1024x1024 windows of Sites 01/04/06/07/11/20/23 (`data/tiles/lola5m_site*.npz`, 74 MB, zipped in `data/handoff/lola5m_tiles_for_imok.zip` 76 MB).
- **Ephemeris:** SPICE kernels in `data/kernels/` (46 MB): `naif0012.tls, de440s.bsp, pck00011.tpc, moon_pa_de440_200625.bpc, moon_de440_250416.tf` from `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/` (lsk/, spk/planets/, pck/, fk/satellites/).
- **Splits (whole tiles; single source of truth `worldmodel/hazard/lola.py`):** train Sites 01, 04, 06, 07, 20, Haworth, Shoemaker, DM2 · validation Site 11 · **test Site 23 and Site 42 (never trained on)**. `Site20v2` extends Site20: do NOT use (leakage).
- **Fetcher:** `python -I -m worldmodel.hazard.lola all --grid [--sites ...]` pulls non-overlapping 1024 px windows (HTTP range reads of the GeoTIFFs; resumes; skips bad windows). Windows are named `lola5m_<site>_r<row>c<col>`; helper `regions_for_site(site)`.
  **Never run on the Mac: its disk is ~96% full (free space swung 1 to 8 GB).** NASA's server throttles with HTTP 503: use at most ~4 parallel site groups. Full-file download speed is ~170 KB/s, so range reads (2-3 min per window) are used.
- **Not NASA / not real:** the demo crater tile (tests only), `data_tools/make_demo_light.py` stand-in (tests only), ice presence (UNKNOWN: PSR only), rocks (UNKNOWN), far mountains beyond the ~5 km LOLA window (skyline ray-cast only to 1,280 m).
- **Known unverified:** grid azimuth = true azimuth + east longitude assumes the LOLA x = r sin(lon), y = r cos(lon) convention (KAN-21 to verify). Hazard labels are NASA's slope map computed from interpolated heights, so the U-Net "recovers the official hazard map from sparse laser data", not "proves landing safety".

## 4. Compute plan (decided with Imok)
| Machine | Role | Constraints |
|---|---|---|
| **MacBook (Apple GPU/MPS)** | development, validation, short runs, monitoring over SSH | disk ~96% full: no big downloads; MPS bugs (OOM at ~9 GB; `randperm` on MPS returns garbage: use CPU permutations); `python -I` for anything reading downloaded files; **zsh does not word-split unquoted variables** |
| **4090 PC (i7 + RTX 4090, FRESH Windows, native, no WSL)** | PyTorch training: rover PPO, terrain diffusion, scaled hazard and light | nothing installed; no JAX GPU on native Windows; user chose **no WSL**; code arrives by **zip (pen drive/Drive)**, no GitHub login; target = **working within 2 hours, then continuous training with nothing else on the PC, and RECORD everything (failures too) for the showcase**; SSH from the Mac wanted (OpenSSH Server, user must enable) |
| **Colab free (T4, may disconnect)** | resumable lighter jobs (light model scale-up, NASA data fetch from Colab's fast network); Ojas's JAX swarm (Linux + GPU) | session limits; keep checkpoints on Drive; every training script supports `--resume` |
| Hermes agent (Nous, MIT) on the 4090 | optional "remote hands" for training runs only; **not part of the project** (do not ship it in the submission) | native Windows support is experimental; control would be SSH from the Mac |
The JAX swarm cannot use the 4090 under these choices: **it moves to Colab** (Ojas's decision to confirm).
**Scale-up plan (user's wish: bigger, more accurate models):** more data first (about 16x more real terrain: grid windows + 4 more sites), then grow models: hazard `--base 48` (17.5 M params; base 32 = 7.8 M, 64 = 31 M), light `--width 128` (907 k; 64 = 228 k, 192 = 2.04 M),
terrain `--base 64` (8.1 M; 96 = 18.3 M, 128 = 32.6 M), rover `--width 64 --hidden 512`. Honest note told to the user: parameter count alone does not help; with 7 windows a huge model overfits; data + held-out tests are what make it accurate.

## 5. Overnight run kit: SPEC (now BUILT, see section 0)
Goal: one command per machine that runs every model stage, resumably, and records everything.
1. **`ops/overnight/run_all.py`** (`python -m ops.overnight.run_all --profile smoke|real4090|colab [--only stage,...] [--resume] [--dry-run]`). Stages (each a subprocess `sys.executable -I -m ...`, own log file, wall-clock budget; timeout = kill, checkpoints are resumable):
   `env_check` (python/torch/CUDA/GPU/VRAM/disk/RAM versions) -> `kernels` (download the 5 NAIF files with stdlib urllib; TODO new module `worldmodel/light/fetch_kernels.py`) -> `tiles` (`lola all --grid`, up to 3 parallel site groups, 503 backoff) -> `psr` (`worldmodel.rover.psr`, default regions must be ALL `lola5m_*` tiles except `_gen`) ->
   `points` (`worldmodel.light.make_points` 12 months; default regions = centre windows of ALL_SITES) -> `hazard` -> `light` -> `terrain` -> `rover` -> `report`. Failure of one stage must NOT stop the rest; it is recorded and the run continues.
   Suggested real-profile settings (UNTESTED estimates): hazard `train --base 48 --epochs 40 --steps 200 --name hazard_unet_big` then `evaluate --name hazard_unet_big`; light `train --width 128 --steps 6000 --cells 6000 --name light_skyline_big` then `evaluate --name light_skyline_big`;
   terrain `train --base 64 --steps 12000 --name terrain_diffusion_big` then `evaluate --name terrain_diffusion_big`; rover `train --envs 4096 --updates 400 --width 64 --hidden 512 --name rover_driver_big` then (TODO) `evaluate`. Colab profile: tiles + light + smaller hazard only.
2. **Ledger + report ("record failures and workings"):** `runs/overnight/<run_id>/ledger.jsonl` (stage, command, start/end, duration, exit code, last 30 log lines on failure, exception text), `status.json` refreshed every minute (current stage, last log line, `nvidia-smi` query),
   `ops/overnight/report.py` -> `RUN_REPORT.md` + PNG curves from every `runs/*/train_log.jsonl` and `runs/*/eval.json`, including a **Failures and retries** table and the baseline-vs-model results. Optional `ops/dashboard.py` (Streamlit, port 8505) showing live curves + the ledger (screen-record this for the video).
3. **Bundles for the zip transfer:** `ops/make_bundle.py` -> `dist/adinfinitium_code.zip` (`git ls-files --cached --others --exclude-standard`) and `dist/adinfinitium_data_seed.zip` (tiles + kernels + checkpoints; ~130 MB), so the PC need not wait on NASA downloads.
4. **Windows setup (TODO, UNTESTED, to write as `ops/windows/setup_4090.ps1`):** `nvidia-smi`; `winget install -e --id Python.Python.3.12`; venv; `pip install torch --index-url https://download.pytorch.org/whl/cu124` (match the driver); `pip install numpy scipy pydantic pyyaml rasterio spiceypy matplotlib pandas plotly streamlit pytest`; unzip code; `pip install -e .`;
   `python -c "import torch; print(torch.cuda.is_available())"`; `pytest -q`; run the smoke profile. For unattended work: `powercfg /change standby-timeout-ac 0`; optional SSH: `Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0; Start-Service sshd; Set-Service -Name sshd -StartupType Automatic` (user sets keys).
   Python 3.11+ is required; **never use same-quote nested f-strings** (3.12-only syntax broke 3.11 once).
5. **Colab notebook (TODO):** cells: GPU check -> upload code zip (or Drive) -> unzip -> `pip install rasterio spiceypy` -> set `ADI_DATA` to a Drive folder (survives disconnects) -> `run_all --profile colab --resume` -> download checkpoints and the run report.
6. **Code gaps to close first (multi-window data):** done for hazard train/evaluate and light train/evaluate (`regions_for_site`, shared site lists); **TODO** terrain (`worldmodel/terrain/data.py` lists the old 5 sites and the old test set; `train.py`/`evaluate.py` load one tile per site), rover (`train.py world_list` uses only the centre window; add `--windows-per-site`),
   `psr.default_regions`, `make_points.default_regions`, tests for all of it.

## 6. Ordered task list (owner, time, acceptance)
| # | Task | Owner/where | Time | Done when |
|---|---|---|---|---|
| 1 | Run the rover validation on the 4090/Colab (not the Mac: it stalled there); write `worldmodel/rover/evaluate.py` (held-out Site 23 + 42, 4 dates, vs the 3 baselines, JSON report) | any session | 45 min | model vs baselines table with per-seed spread, honest if it does not win |
| 2 | Multi-window code gaps (section 5.6) + tests | any session | 1 h | `pytest` green, scripts skip missing sites |
| 3 | Overnight kit (section 5): run_all, ledger, report, dashboard, bundles, Colab notebook, PowerShell setup, `--dry-run` + smoke profile tested on the Mac | any session | 2-3 h | smoke profile passes end to end on the Mac in < 15 min |
| 4 | **4090 bring-up** (user + SSH): unzip, setup script, smoke profile, then real profile; time-box 2 h | user + session | 2 h | `torch.cuda.is_available()`, smoke green, first real stage logging |
| 5 | **Overnight continuous training** (about 21:00 Sat to 08:00 Sun) on the 4090; Colab runs light/data in parallel | 4090 + Colab | ~11 h | ledger + curves recorded; every stage has a result or a recorded failure |
| 6 | Sunday morning: evaluate on held-out Site 23 + 42, fill the results table, update READMEs/STATUS | session | 2 h before 12:00 | numbers in READMEs match `eval.json` |
| 7 | Navigator: launch-slip, real PSR ice panel, hosted public link + small data bundle (84 point files = 6 MB; tiles 74 MB or downsampled), README (KAN-37) | session (Nimansh) | 3 h | public URL opens without login; banners show MODELLED not GENERATED |
| 8 | Mission planner model + `contracts/plan.py` + orchestrator decision log | session | 3 h | writes `runs/<run>/plan_XXX.json`; falls back to Navigator ranking |
| 9 | Basilisk handoff: export the rover policy and a replay format for Arnav's module (KAN-45) | session + Arnav | 1 h | Arnav can replay a route |
| 10 | Dream-RSI (with the pilot, JAX on Colab) | Ojas's session / Imok | open | A/B plot with vs without, or shipped as "not done" |
| 11 | Submission: 30 s video + subtitles (record training/failures), project README, hub page, Drive, repo public Sunday, AI disclosure, NASA credits | team | Sun | checklist in section 1 complete |
**Cut line Sunday 12:00:** anything not working ships as its physics backup and the slides say so. Priority if time is short: (1) validated hazard + light results and the hosted Navigator, (2) rover baselines + trained driver if it beats them, (3) terrain generator only if hide-and-check beats the backup.

## 7. Commands cheat sheet (run from the repo root; `-I` for isolation)
```bash
pip install -e ".[hazard,apps,dev]" spiceypy                       # Python >= 3.11; torch/scipy/rasterio/matplotlib in the `hazard` extra
python -I -m worldmodel.hazard.lola all --grid --sites Site23 Site42        # NASA data (not on the Mac)
python -I -m worldmodel.rover.psr                                             # NASA permanent shadow into the tiles
python -I -m worldmodel.light.make_points --dates 2027-01-01 2027-02-01 ... 2027-12-01
python -I -m worldmodel.hazard.train --base 48 --epochs 40 --steps 200 --name hazard_unet_big [--resume]; python -I -m worldmodel.hazard.evaluate --name hazard_unet_big --figure
python -I -m worldmodel.light.train --width 128 --steps 6000 [--resume]; python -I -m worldmodel.light.evaluate
python -I -m worldmodel.terrain.train --base 64 --steps 12000 [--resume]; python -I -m worldmodel.terrain.evaluate
python -I -m worldmodel.rover.train --envs 4096 --updates 400 --width 64 --hidden 512 [--resume]
streamlit run navigator/app.py --server.port 8501      # Navigator      (8502 mission control, 8503 brain window, 8504 model viewer)
pytest -q                                              # ~100 tests; 3 skipped (Basilisk not installed)
```
Checkpoints go to `<ADI_DATA>/models/` (default `./data/models/`, never git). Set `ADI_DATA` to relocate data. Training scripts print the device (cuda / mps / cpu) and all support `--resume`.

## 8. Lessons and failures so far (use these in the video: honest, they ARE the story)
1. **The slope rule was already perfect at full resolution** (NASA's slope map correlates 0.9999 with the finite-difference slope of its own heights), so a hazard U-Net would only tie it. We redesigned the task to sparse-input recovery (only real laser pixels): rule F1 0.81 vs U-Net 0.95.
2. **Light model gain is modest (25-33%)**: it only sees a coarse terrain; capacity cannot recover detail averaged away.
3. **Stand-in light data hid reality.** Our analytic Sun/Earth stand-in was arbitrary; real SPICE shows Site 23 cell 0 with 0% Sun in Feb-Jul and ~30% in Nov (neighbouring cells reach 78-91%). We replaced it and validated against JPL Horizons (0.00003 deg).
4. **Navigator bugs found by testing:** horizon subtracted twice for Earth visibility after the contract defined elevations relative to the skyline; picker lost a site when two display names collided.
5. **Rover environment bugs:** "reached" radius (20 m) smaller than one step (36 m), so rovers stepped over targets (fixed by checking the path's closest approach); the turn-back rule ignored solar income and sent every rover home (fixed; naive/cautious/wait-for-light baselines all kept: they die 77-88% in the dark).
6. **Platform bugs:** Apple MPS out-of-memory at ~9 GB; MPS `randperm` returned garbage indices (CPU permutation); CPU jobs during GPU training doubled training time (55 min vs ~25); nested same-quote f-string broke Python 3.11; zsh unquoted variables not word-split; NASA 503 throttling; Mac disk nearly full.
7. **The Mac cannot train the rover**: a 6-world PPO run stalled the whole machine (swap thrash, 98% disk). Heavy training belongs on the 4090/Colab; this is why the overnight kit matters.
8. **Tooling dead ends:** browsers/Chrome tools were unavailable several times; the shared planning chat needed the Claude-in-Chrome extension to be read.

## 9. Open questions / decisions still needed
- Ojas/Imok: where the JAX swarm runs (Colab free T4 is the only Linux GPU under the "no WSL" choice) and whether Dream-RSI is realistic by 12:00.
- Imok: SSH reachability of the 4090 PC from the Mac (host/IP, user, key; the user sets it up; no passwords through the chat), and whether the PC stays awake/powered all night.
- Imok: hosting for the public Navigator link (needs the repo public or a bundle host) and who builds the final hub page (Tanush, KAN-50).
- Tanush: `sites/sites.json` and ice layers (KAN-38/39); our regions use names `lola5m_site*` until then; `sites[].region` must equal the tile/light region name.
- Everyone: product name; video storyboard (30 s: 60% lander / 20% rover / 20% Navigator was the earlier plan).

## 10. How to continue in a new session (paste this)
> Read `docs/team/MASTER_PLAN.md`, then `docs/STATUS.md`, `AGENTS.md`, `docs/team/MISSION_BRAIN_SPEC.md`, `docs/team/HANDOVER_WORLD_MODELS.md`. Check out branch `nimansh/KAN-45-rover-brain`. Run `pytest -q`. Then do section 6 in order, starting with task 1 (rover validation + evaluation) and task 3 (overnight kit). Real, validated work only; label provenance; keep every physics backup; report in `docs/outbox/nimansh.md`; never commit to `main`.
