# Handover: world models + real NASA data (to Ojas's Claude session)

Written Sat 14 Nov 2026 (IST, afternoon) by the Claude session working with Imok and Nimansh. **Imok asked to stop and hand over.** Nothing is running; all servers are stopped.
Read this, then `docs/STATUS.md`, `AGENTS.md`, and the folder READMEs listed below. Everything you need is on branch **`nimansh/KAN-26-terrain`**
(stacked on `imok/integration`; it contains the light model, the terrain generator, the fetcher upgrade, the viewer, the real ephemeris). Nothing is merged to `main`.

```bash
git fetch origin && git checkout nimansh/KAN-26-terrain
pip install -e ".[hazard,apps,dev]" spiceypy        # torch, scipy, rasterio, matplotlib, streamlit, plotly
pytest -q                                           # 70+ pass; 3 skipped (Basilisk). Always run python with  -I  when it reads downloaded files.
```

## 1. State of the five AI parts

| # | Part | State | Where | Honest note |
|---|---|---|---|---|
| 1 | Hazard detector (U-Net, 1.94 M params) | **Trained.** Merged into `imok/integration`. | `worldmodel/hazard/` | Held-out Site 23: F1 **0.950** vs 0.814 for the slope-rule backup (missed hazards 4.7% vs 20.9%) |
| 2 | Neural light field (32.7 k params) | **Trained** (this session) | `worldmodel/light/` | Held-out Site 23: skyline error 0.63° → **0.47°**; sunlit-hours error 21.2 h → **14.1 h**; Earth-visible wrong 2.3% → 1.6% |
| 3 | Terrain generator (diffusion, 2.04 M params) | **Code + tests + physics backup done. NOT trained.** | `worldmodel/terrain/` | Plan: train on the 4090 (user decision). `evaluate.py` = hide-and-check vs the backup |
| 4 | Evolved pilot (TensorNEAT) | Imok's, `swarm/` | — | Not touched by this session |
| 5 | Dream-RSI meta layer (KAN-106) | **Not started** | — | User: build it together with the pilot on the 4090 |

Also done this session: **real NASA ephemeris** (below), real light point series, the 3D model viewer, the LOLA window-grid fetcher, hazard scripts that load every window of a site.

## 2. Real NASA data: what is real, what is not
- **Terrain** (heights, official slope, laser counts, uncertainty): NASA LRO LOLA 5 m from PGDA (`pgda.gsfc.nasa.gov/data/LOLA_5mpp/`), fetched by `worldmodel/hazard/lola.py` as 1024×1024 windows.
  On the Mac: centre windows of Sites 01/04/06/07/11/20/23 only (`data/tiles/lola5m_site*.npz`).
- **Sun/Earth ephemeris: now real** (user insisted on NASA data). `worldmodel/light/ephemeris.py` uses NASA/JPL **SPICE DE440** (kernels from `naif.jpl.nasa.gov`, 46 MB in `data/kernels/`,
  names in the module docstring). Checked against **JPL Horizons** at Site 23 over two 30-day windows: Sun and Earth, elevation and azimuth, agree to **0.00003°**.
  Pinned in `tests/test_ephemeris.py` (Sun values from Horizons).
- `python -I -m worldmodel.light.make_points --dates 2027-01-01 ... 2027-12-01` wrote **84 real point-series files** (7 sites × 12 months, provenance **MODELLED**, 6 MB) into `data/light/`. They replace the analytic stand-in for those sites.
  Real seasons matter: Site 23 cell 0 has Sun up 0% in Feb–Jul and ~30% in Nov; its 9 cells together reach 78–91% in good months.
- **Still not NASA**: the demo crater tile (tests only, GENERATED); PSR / ice layers (UNKNOWN, KAN-39 not delivered); `make_demo_light.py` stand-in generator is kept only for tests.
- Limits to say out loud: LOLA heights are mostly interpolated; the skyline is ray-cast only to 1,280 m inside a ~5 km window (biased low); grid azimuth = true azimuth + east longitude assumes the LOLA
  x = r·sin(lon), y = r·cos(lon) convention, **still unverified (KAN-21)**; hazard labels are NASA's slope map (made from interpolated heights).

## 3. Decisions the user (Imok) already made. Do not re-ask.
1. **Framing: "physics-informed ML"**: networks trained on real NASA data, always shown next to their exact-physics baseline with held-out tests. Not "like an LLM".
2. **Make models more accurate: more data first, then scale up.** Models today are small (33 k to 2 M params). User wants bigger; the honest plan: ~16× more real terrain, 3 more sites, then
   hazard ~20 M (`--base 48` = 17.5 M), light 0.2–2 M (`--width 64/128/192` = 228 k / 907 k / 2.04 M), terrain 8–33 M (`--base 64/96/128` = 8.1 / 18.3 / 32.6 M). Parameter count alone does not help: more data and held-out testing do.
3. **Splits (whole tiles):** train Sites 01, 04, 06, 07, 20, Haworth, Shoemaker, DM2 · validation Site 11 · **test Site 23 and Site 42** (never trained on). `Site20v2` is an extension of Site20: do NOT use (leakage).
4. **Compute:** hazard on the Mac is fine; light on Colab or the second laptop; **terrain generator + pilot + Dream-RSI on the 4090**. The Mac's disk is ~96% full (free space swung 1 to 8 GB): **do the big data work elsewhere** (the user said to shift it to you).
5. **Colab will be driven through the user's Chrome** (Claude in Chrome extension; the user said they will reconnect it). Colab should fetch the NASA windows itself (fast network).
6. **Next priority for the Mac session was: Navigator on real data + launch-slip + a public link** (judges cannot open localhost).
7. **Parked, "talk later": how JAX/TensorNEAT runs on the 4090 PC** (Imok's note says native Windows, no WSL, no JAX GPU). Options asked: WSL2 Ubuntu, Colab for the swarm, or check first.
8. Keep publishing details to the orchestrator via `docs/outbox/nimansh.md`.

## 4. Work for you, in order
**A. Data scale-up (not on the Mac).** `python -I -m worldmodel.hazard.lola all --grid` (all 11 sites; non-overlapping 1024 px windows that skip the centre window; resumes; skips bad windows). Roughly 1–2 GB.
NASA's server returned **503** when hit hard: run at most ~4 groups in parallel (e.g. `--sites Site23 Site42 Site11`, `--sites Site01 Site04`, ...). Each window is ~2–3 min (HTTP range reads of the GeoTIFF; the full files download at ~170 KB/s).
Window tiles are named `lola5m_<site>_r<row>c<col>`; helper `regions_for_site(site)` lists them. **Hazard training/evaluation already uses it** (skips sites not on disk). Still TODO to use multi-window data:
`worldmodel/light/data.py` (`build_site` loads one tile via `region_name`; its `TRAIN_SITES` is still the original 5), `worldmodel/terrain/train.py` + `data.py` (`SitePatches(site)` / `load_site(site)` load one tile; site lists still the original 5).
**B. Colab notebook + scale-up training** (not written yet). Suggested: `git archive` the branch → upload zip (the repo is private; never paste tokens) → `pip install rasterio spiceypy` (torch is preinstalled) → run the fetcher →
`python -I -m worldmodel.hazard.train --base 48`, `python -I -m worldmodel.light.train --width 128 --steps 6000` → download `data/models/*.pt` back. Training scripts are cross-platform and support `--resume`.
**C. Terrain generator on the 4090:** `python -I -m worldmodel.terrain.train --base 64 --steps 12000 --resume`, then `python -I -m worldmodel.terrain.evaluate`. If it cannot beat the physics backup on held-out hide-and-check (RMSE, 90% coverage), ship the backup (cut rule Sun 12:00) and say so.
**D. Dream-RSI (KAN-106):** nothing exists. The swarm's strategy knobs are fixed per run (`--weight-mutate-power`, `--survival-threshold`, `--species-size`, `--compatibility-threshold`, missions per individual, topk); `GenerationLog.strategy` records them.
A meta layer needs several runs with different strategies (or per-generation changes) to learn from; decide with Imok. Not started, and it needs JAX/TensorNEAT, which are not installed on the Mac.
**E. Navigator (KAN-35) on real data + launch-slip + hosting + KAN-37 README.** `data/light/*_points.npz` are real now (MODELLED): the red GENERATED banner disappears automatically. Ideas: launch-slip control (shift the window start; needs more dates or on-demand generation via `make_points`);
a small public data bundle (84 point files = 6 MB; tiles 74 MB or downsampled); update `navigator/README.md` (still says stand-in). Note: `nimansh/KAN-35-navigator` has commit a433ed3 ("grid azimuth" label) that `imok/integration` does not.
**F. After new data:** re-run `python -I -m worldmodel.hazard.evaluate` and `python -I -m worldmodel.light.evaluate` including Site 42, report both test sites.

## 5. Files NOT in git and how to get them back
| Path (on the Mac) | Size | How |
|---|---|---|
| `data/tiles/lola5m_site{01,04,06,07,11,20,23}.npz` | 74 MB | `lola.py all`; also zipped: `data/handoff/lola5m_tiles_for_imok.zip` (76 MB) |
| `data/models/hazard_unet.pt` (best), `light_skyline.pt` (best) | 7.8 MB, 0.14 MB | retrain, or copy by hand; `*_last.pt` hold optimizer state for `--resume` |
| `data/kernels/*` (SPICE) | 44 MB | `naif.jpl.nasa.gov/pub/naif/generic_kernels/` (names in `ephemeris.py`) |
| `data/light/*_points.npz` (84 files) | 6 MB | `python -I -m worldmodel.light.make_points --dates ...` (needs tiles + kernels) |
| `runs/hazard/`, `runs/light/` (eval.json, logs, figures) | small | rerun the evaluate scripts |

Set `ADI_DATA` (default `<repo>/data`) to put data on another drive.

## 6. Gotchas
- **zsh does not word-split unquoted variables** (`--dates $DATES` passed one string). Pass lists explicitly.
- Run Python with **`-I`** when it reads downloaded files. Heavy CPU jobs slow GPU training that runs at the same time (training took 55 min instead of ~25 because of this).
- I changed shared files: `config/mission.yaml` (`hazard_detector`, `light_model` blocks), `contracts/mission.py` (`HazardDetector`, `LightModel`), `contracts/paths.py` (`model_path`). Imok reviews.
- Wording from the user to keep: the hazard model "recovers the official hazard map from sparse laser data", not "proves landing safety".

## 7. Where each thing is documented
`worldmodel/hazard/README.md` · `worldmodel/light/README.md` (results, ephemeris, limits) · `worldmodel/terrain/README.md` · `model_viewer/README.md` (3D trial viewer, port 8504) · `navigator/README.md` · `docs/STATUS.md` (World models section).
