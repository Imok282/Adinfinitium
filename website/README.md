# Adinfinitium — company website

Static site for **Adinfinitium**, which builds safety-first landing and surface autonomy:
**Adinfinitium Pilot** (landing pilot, hazard, light and rover-driver models, each with a physics backup that runs first) and
**Adinfinitium Navigator** (landing site + date planning, a simulation swarm and mission control). No build step.

## Site map

| Page | What |
|---|---|
| `index.html` | Company home: 3D hero (`js/hero3d.js`), product cards, real-number credibility band, latest research, waitlist |
| `landing.html` | "Watch a landing": the cinematic scroll story (orbit → descent → touchdown → rover), `js/main.js` + `js/scene.js` |
| `pilot.html`, `navigator.html` | Product pages |
| `mission-control.html` | Replay of one real simulated mission (lander then rover) |
| `research.html`, `research/*.html` | Research index and posts (`<base href="../">` in research pages) |
| `company.html` | About, team (TODO), careers, news, contact (TODO), Method & data, how we used AI, credits |

The shared top nav, waitlist band and footer live in `tools/shell.html`; after editing it run `python3 tools/sync_shell.py`.
The waitlist form is **visual only** (`js/waitlist.js` sends and stores nothing; TODO: wire a form service).
Real-scroll QA: `cd tools/qa && npm i && node qa_scroll.mjs 1440 900` (set `BASE=http://localhost:8780/landing.html` for the story),
`node pages_shots.mjs [pages…]` for every page at 1440 and 390 px.

## Run it

```bash
cd website
python3 -m http.server 8780        # then open http://localhost:8780/
```

Needs a browser with ES modules; three.js is loaded from jsDelivr. Without WebGL or with
`prefers-reduced-motion` the site shows static posters and a flat drawing of the network instead.
Screenshot / deep-link helpers: `?scroll=0.42` (fraction of the page) or `?scroll=results&off=0.3`
(section id + viewport heights); `?static` forces the no-WebGL path.

## File map

| Path | What |
|---|---|
| `index.html` | Home: the scroll story (orbit → pilot → build → teasers → ghosts → touchdown → rover) |
| `mission.html`, `models.html`, `results.html`, `benchmarks.html`, `team.html` | Subpages (same nav, palette and footer; no 3D scene behind them) |
| `assets/data/benchmarks.json` | Benchmarks page data. `"placeholder": true` shows "Sample figures — final benchmarks coming"; paste real rows and set it to false |
| `assets/video/manifest.json` | Lists the cinematic clips that exist (written by `tools/process_video.sh`); no 404 probing |
| `tools/qa/qa_scroll.mjs` | Real-scroll QA (puppeteer-core, wheel events): `cd tools/qa && npm i && node qa_scroll.mjs 1440 900` → `out_*/failures.txt`, `contact.jpg` |
| `css/style.css` | Theme tokens (`:root`), top bar, sections, results, media slots, footer |
| `css/buildlog.css`, `css/gauge.css`, `css/models.css` | Built-by-agents log, descent gauge, models views |
| `js/main.js` | Scroll orchestration, lazy module loading, nav highlight, fallbacks |
| `js/nav.js` | Pill dropdowns and the round "all sections" menu |
| `js/scene.js`, `js/gauge.js` | Fixed 3D scene (Moon, lander, rover) and the descent gauge |
| `js/net3d.js`, `js/net-style.js`, `js/net2d.js` | Pilot network fly-through (real sampled weights) + static fallback |
| `js/stage.js`, `css/stage.css` | Shared scroll-story helpers (word reveal, fly-in-and-settle, deck stack, dim) and the sticky-stage pattern |
| `js/pilot.js`, `css/pilot.css` | Pilot beat: fly into the lander, real camera patches stream into the network, thrust corrections |
| `js/cursors.js`, `css/cursors.css`, `js/buildlog.js` | "How it was built": cursors assemble the software from real code/images; compact real git-history ticker |
| `js/modelcards.js`, `css/modelcards.css` | Clickable model cards (#agents) with a pop-up per model: diagram, training data, real metrics |
| `js/ghosts.js`, `js/ghostdata.js`, `js/models3d.js` | Simulated-descent ghosts (touchdown beat + 3D tab) and the 3D model views |
| `js/rover.js`, `css/rover.css` | Rover beat: real planned route, overlays, dashboard |
| `assets/data/models.json` | Every model-card number with its source; edit and reload |
| `assets/data/ghosts_*.{bin,json}` | Compact ghost tracks (uint16 x,y,z,t per point + header with counts) from `tools/export_ghosts_web.py` |
| `assets/data/rover_route_site23.json` | Real rover route + per-step battery/light from `tools/export_rover_route.py` |
| `assets/img/pilot_patches.*` | Real 32×32 lander-camera crops from LOLA tiles (`tools/export_patches.py`) |
| `js/results.js` | Renders the Results section from `assets/data/results.json` |
| `js/media.js` | Optional cinematic video slots (see below) |
| `assets/data/results.json` | **Every number shown in the Results section** |
| `assets/data/numbers.json` | Full extracted metrics (lander, hazard, light, planner) for reference |
| `assets/net.json`, `assets/build_log.json` | Network sample and git history (made by `tools/export_*.py`) |
| `assets/img/`, `assets/models/` | Terrain / model images, 3D lander and rover |
| `tools/` | Asset exporters, screenshot script, `process_video.sh` |

## Model diagrams (drop-in)

Each model pop-up first looks for `assets/diagrams/<key>.svg`, then `<key>.png`, and shows it instead of the drawn SVG.
Keys: `hazard`, `light`, `terrain`, `pilot`, `planner`, `rover`, `swarm`, `rsi`. Delete the file to go back to the drawn one.
`?model=<key>` opens that pop-up on load (screenshots). The user's originals are kept outside the published folder in `../website_dev/diagrams_incoming/`.

## Rebuilding data

```bash
~/venvs/adi/bin/python -m worldmodel.ghosts --kind lander --region lola5m_site06 --n 160 --pilot lander_pilot   # 960 runs, ~23 min on an M1
python website/tools/export_ghosts_web.py runs/ghosts_web/lander_lola5m_site06.json runs/ghosts_web/rover_lola5m_site23_2027-03-01.json
~/venvs/adi/bin/python website/tools/export_rover_route.py
~/venvs/adi/bin/python website/tools/export_patches.py
```

Debug helpers: `?force=q,f,rover` sets the 3D timeline directly; `?debug` prints scene state.

## Updating results

Edit `assets/data/results.json` and reload; no code changes. Each card has a `type`:

- `paired` — classic vs pilot bars: `baseline` / `candidate` (`label`, `success_pct`), `diff_points`,
  `ci95_points`, `test`, `p_value`, `significant`, `only_candidate`, `only_baseline`, `note`, `source`.
- `kpis` — big numbers: `kpis: [{ "value": "≈4 m", "label": "mean miss, normal" }, …]`, `note`, `source`.
- `pending` — a slot waiting for a result. For the Site 23 final exam set `"value"` (e.g. `"97.9 %"`) and
  `"detail"` (what it measures); while `value` is `null` the card shows `pending_text` ("Result pending").

`intro` (section text), `bar_axis` (bar scale, default 94–100 %) and `footnote` (sources line) are also in
the file. Text fields accept `**bold**`. Keep `meta.provenance` = `modelled`: all results are simulation.

## Dropping in the cinematic videos

Three slots stay hidden until their files exist (the 3D scene is the fallback):

| Slot | Where | Files |
|---|---|---|
| `hero` | full-screen backdrop behind the title, fades out on scroll | `assets/video/hero.mp4` (+ `.webm`), `hero.jpg` |
| `descent` | card inside the Mission section | `assets/video/descent.mp4` (+ `.webm`), `descent.jpg` |
| `landing` | card in the Touchdown section, above the ghost tracks | `assets/video/landing.mp4` (+ `.webm`), `landing.jpg` |

One command per clip (needs `ffmpeg`), run from the repo root:

```bash
website/tools/process_video.sh path/to/clip.mp4 hero       # or: descent | landing
```

It crops the generator's watermark corner (`CROP_PCT=7` by default, cropped symmetrically so the aspect
stays the same; `CROP_PCT=10 website/tools/process_video.sh …` for more), scales to 1280 px wide (hero) or
960 px (others), writes H.264 MP4 (≤ 8 MB, no audio, fast start), VP9 WebM and a poster JPEG from ~1 s.
Reload the page and the slot appears. Delete the files to hide it again. Videos autoplay muted and loop;
the two cards only play while on screen; with reduced motion only the poster is shown.

## Asset provenance

- **Observed:** NASA LOLA 5 m terrain tiles (PGDA), LROC Moon colour imagery, SPICE DE440 geometry.
- **Modelled:** every result, hazard / light map, network weight drawing and ghost track (our simulations
  and trained models, sources noted on each card).
- **Illustrative:** the 3D lander and rover models (not flight CAD).
- **AI-generated (decorative only):** the cinematic video slots, made with Google Gemini (Omni / Nano Banana)
  and credited in the footer. Never used as data or for any safety claim.
