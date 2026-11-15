<p align="center"><img src="assets/adinfinitium-banner.png" alt="Adinfinitium" width="100%"></p>

# Adinfinitium

<p align="center">
  <a href="https://adinfinitium.vercel.app"><img alt="Website" src="https://img.shields.io/badge/website-adinfinitium.vercel.app-d97757"></a>
  <a href="https://huggingface.co/Imok282"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20models-7%20public-ffcc4d"></a>
  <img alt="NASA Space Apps 2026" src="https://img.shields.io/badge/NASA%20Space%20Apps-2026-0b3d91">
  <img alt="Code license" src="https://img.shields.io/badge/code-MIT-informational">
  <img alt="Weights license" src="https://img.shields.io/badge/weights-Apache--2.0-informational">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776ab">
</p>

**Contents:** [For judges](#for-judges-the-60-second-version) · [How it works](#how-it-works) · [Models](#models) · [Headline results](#headline-results-held-out-simulation) · [Quick start](#quick-start) · [Repository layout](#repository-layout) · [Start here](#start-here) · [License](#license)

<p align="center">
  <a href="https://adinfinitium.vercel.app"><img src="assets/readme/website_home.jpg" alt="adinfinitium.vercel.app: autonomy that lands where missions survive" width="100%"></a>
  <br><sub>Live site: <a href="https://adinfinitium.vercel.app">adinfinitium.vercel.app</a> · watch a full mission from lunar orbit to touchdown and replay real recorded landings.</sub>
</p>

**Open models for lunar landing-site navigation**, built from NASA LRO LOLA 5 m terrain of the lunar south pole.
Adinfinitium began as our NASA Space Apps 2026 project, *Lunar Landing Site Navigator*, plus a lunar-lander pilot.

Models were frozen before testing, held-out sites were evaluated once, and failures are reported next to successes.
All results are **simulation** results on held-out lunar sites. This is research code, **not for flight or safety-critical use**.

<p align="center">
  <a href="https://adinfinitium.vercel.app"><b>Website</b></a> ·
  <a href="https://adinfinitium.vercel.app/landing.html"><b>Watch a landing</b></a> ·
  <a href="https://adinfinitium.vercel.app/research/benchmarks.html"><b>Benchmarks</b></a> ·
  <a href="https://huggingface.co/Imok282"><b>Models on Hugging Face</b></a>
</p>

## For judges: the 60-second version

**Challenge:** NASA Space Apps 2026, *Lunar Landing Site Navigator*. At the lunar south pole, *where* and *when* you land decides
whether a mission survives: slope and hazards, sunlight that barely clears the horizon, and line of sight to Earth.

**What we built**
- **Navigator** chooses the landing cell and date from real NASA LRO LOLA 5 m terrain and NASA/JPL SPICE Sun/Earth geometry,
  then tests each candidate in simulated missions.
- **Pilot** flies the descent: a learned 14.8 M-parameter pilot that only *corrects* classic ZEM/ZEV guidance (the physics backup
  always flies), plus a hazard U-Net, a horizon/light model and a rover brain that drives to shadowed ice edges and back.
- **Dream-RSI**: gated offline self-improvement. A model may replace itself only if it loses no mission that the current model
  or the physics backup lands.

**Lander results** (paired fixed-seed simulation; touchdown limits vz ≤ 2 m/s, vh ≤ 0.5 m/s, slope ≤ 12°)

| Test | n per condition | Classic guidance | Learned pilot | Exact p |
|---|---|---|---|---|
| 3 unseen maps, normal | 2,304 | 97.66 % | **98.39 %** | 0.019 (Holm 0.074) |
| Site 23 final exam (flown once) | 1,024 | 97.95 % | 97.95 % | 1.00 |
| Our planned landing cell (rehearsal) | 512 | 99.02 % | **99.61 %** | 0.45 |
| 20 procedurally generated Moon maps | 640 | 97.50 % | **98.12 %** | 0.34 |

At least as safe as classic guidance in every test, ahead in most, with **≈ 9 % less fuel** in every test. Weak spot: after an
engine failure it lands less often than classic (95.9 % vs 98.4 %). Full tables, harsh conditions and every caveat:
[`docs/records/SUBMISSION_SUMMARY.md`](docs/records/SUBMISSION_SUMMARY.md) and the
[benchmarks page](https://adinfinitium.vercel.app/research/benchmarks.html).

**Dream-RSI record:** planner 1 of 5 generations accepted (validation Brier −12 %); rover brain repaired planted faults
(blocked route fixed at 4/4 sites); lander 0 of 7 accepted: the zero-regression gate held.

**Data and AI use.** Terrain: NASA LRO LOLA (PGDA); imagery: NASA/GSFC/Arizona State University (LROC); ephemeris: NASA/JPL
SPICE (DE440). No NASA endorsement implied. The team built the software together with AI coding agents; cinematic video and
imagery on the website are AI-generated (Google Gemini).

**Team:** Ojas · Nimansh · Tanush · Arnav

## How it works

<table>
<tr>
<td width="50%" valign="top"><img src="assets/readme/model_pipeline.png" alt="Pipeline: real NASA data feeds the image, planner and execution models; landing results feed back to the planner"><br><sub><b>Pipeline.</b> Real NASA data (LOLA terrain, SPICE ephemerides, descent physics) feeds the surface, planning and execution models; landing results feed back to the planner.</sub></td>
<td width="50%" valign="top"><img src="assets/readme/mission_timeline.png" alt="Mission timeline from lunar orbit to sunrise"><br><sub><b>Mission timeline (design).</b> Fully autonomous from lunar orbit: hazard scan, choose when to descend, powered descent, hover and re-target, touchdown, then the rover's surface mission.</sub></td>
</tr>
<tr>
<td width="50%" valign="top"><img src="assets/readme/hazard_site23.webp" alt="Hazard U-Net vs slope rule vs truth on held-out Site 23"><br><sub><b>Hazard model, held-out Site 23.</b> U-Net prediction against the slope-rule backup and the LOLA truth.</sub></td>
<td width="50%" valign="top"><img src="assets/readme/rover_dm1.webp" alt="Rover brain v3.1 routes on site DM1"><br><sub><b>Rover brain v3.1 on site DM1.</b> End-to-end test: the hazard map feeds the rover's route to a shadowed ice edge and back.</sub></td>
</tr>
</table>

## Models

All weights and model cards are on Hugging Face: **https://huggingface.co/Imok282** (weights: Apache-2.0).

| Model | What it does | Params | Status |
|---|---|---|---|
| [lunar-hazard-unet](https://huggingface.co/Imok282/lunar-hazard-unet) | per-pixel slope hazard map | 60.4 M | final |
| [lunar-light-skyline](https://huggingface.co/Imok282/lunar-light-skyline) | skyline / illumination prediction | 38.3 M | final |
| [lunar-terrain-diffusion](https://huggingface.co/Imok282/lunar-terrain-diffusion) | terrain gap-filling | 49.9 M | experimental (kriging beats it) |
| [lunar-rover-brain](https://huggingface.co/Imok282/lunar-rover-brain) | scripted rover planner v3 / v3.1 | 0 learned | final |
| [lunar-lander-pilot](https://huggingface.co/Imok282/lunar-lander-pilot) | residual-on-classic landing guidance | 14.8 M | final |
| [lunar-rover-driver-v1](https://huggingface.co/Imok282/lunar-rover-driver-v1) / [v2](https://huggingface.co/Imok282/lunar-rover-driver-v2) | learned rover policies (PPO / imitation) | 11.4 M | **deprecated** (failed; research record) |

The chain: hazard network → light network → terrain model → rover brain → lander pilot. The hazard U-Net feeds the rover brain; the lander pilot is a separate package.

## Headline results (held-out, simulation)

| Model | Metric | Ours | Baseline |
|---|---|---|---|
| Hazard U-Net | F1, held-out Site23 + Site42 | **0.951** | 0.817 (slope rule) |
| Hazard U-Net | F1, six fresh sites | **0.757** | 0.292 |
| Light transformer | skyline MAE, Site23 + Site42 | **0.450°** | 0.570° |
| Lander pilot | landing success, 3 unseen maps | **98.39%** | 97.66% (classic) |
| Rover brain v3.1 | returned, four untouched sites | **33.7%** | 17.9% (naive planner) |
| Terrain diffusion | RMSE, held-out | 2.01 m | **0.97 m** (kriging wins) |

Known weaknesses are listed on every model card. In short: the hazard model is weak on flat fresh ground, the light model is ~227× slower than ray-casting, the lander's gain comes from one site and is not significant after correction, and the rover brain still loses over half its rovers.

## Quick start

```bash
git clone https://github.com/Imok282/Adinfinitium.git && cd Adinfinitium/lander
pip install -e ".[dev]"                      # Python 3.11+, CPU is fine
python -m contracts.mission                  # validate the mission config (lander, rover, failures, limits)
python -m data_tools.make_demo_tile          # synthetic crater tile to try things without NASA data (GENERATED, NOT OBSERVED)
python -m scripts.bench_swarm --n 1000       # fly 1,000 landers with classic guidance
pytest                                       # the test suite
```

Trained weights: `huggingface-cli download Imok282/lunar-lander-pilot` (and the other models under [Imok282](https://huggingface.co/Imok282)).
Real NASA terrain: LOLA 5 m south-pole tiles from [PGDA](https://pgda.gsfc.nasa.gov/products/90). Per-project setup is in [`lander/README.md`](lander/README.md) and [`adinfinitium/README.md`](adinfinitium/README.md).

## Repository layout

| Path | Contents |
|---|---|
| [`adinfinitium/`](adinfinitium/) | Main codebase: hazard, light and terrain models, rover simulator and brains (v1–v3.1), Dream-RSI self-improvement, tests, ops scripts |
| [`lander/`](lander/) | Lander pilot: 3-DOF terminal-descent simulator, classic ZEM/ZEV guidance, `PilotFormer`, benchmark and gate code |
| [`models/`](models/) | Model cards and benchmark charts (mirrors Hugging Face), `CHECKPOINTS.sha256` |
| [`docs/records/`](docs/records/) | Final results summaries, rover v3.1 and lander reports, Dream-RSI design |
| [`assets/`](assets/) | Logo and banner |

Large files are not in this repo: checkpoints are on Hugging Face, and the raw LOLA tiles, light samples and training logs are not published.
Each sub-project has its own `README.md` / `START_HERE.md` with setup instructions.

## Start here

1. [`docs/records/SUBMISSION_SUMMARY.md`](docs/records/SUBMISSION_SUMMARY.md): the full results, with numbers and caveats.
2. [`docs/records/RESULTS_FOR_TEAM.md`](docs/records/RESULTS_FOR_TEAM.md): per-model results and training notes.
3. [`adinfinitium/README.md`](adinfinitium/README.md) and [`lander/README.md`](lander/README.md): how to run the code.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Model weights on Hugging Face: Apache-2.0. Terrain data: NASA LRO LOLA (public domain).
