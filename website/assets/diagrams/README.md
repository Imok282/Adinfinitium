# Model diagrams (optional overrides)

Drop a hand-made architecture diagram here and the model card's pop-up shows it instead of the diagram drawn
from `assets/data/models.json`. **Two steps: add the file, then add its file name to `manifest.json`** (the pop-ups only
try files listed there, so the site never requests a missing file).

| Key | Model |
|---|---|
| `hazard` | Hazard model (U-Net) |
| `light` | Light / sunlight model (skyline CNN) |
| `terrain` | Terrain generator (diffusion U-Net) |
| `pilot` | Lander pilot (transformer residual on ZEM/ZEV) |
| `planner` | Mission planner (set transformer) + Navigator ranking |
| `rover` | Rover driver (vision transformer) + rule drivers |
| `rsi` | Dream-RSI self-improvement loop |
| `swarm` | Swarm (slot card; shows `swarm.png`) |

File name: `<key>.svg` (preferred) or `<key>.png`, e.g. `pilot.svg`; then `manifest.json` becomes
`{ "files": ["swarm.png", "system_architecture.png", "pilot.svg"] }`. SVG wins if both are listed. Remove the entry
(and the file) to go back to the drawn diagram. Keep each file under ~300 KB; PNGs are shown on a white panel.
`system_architecture.png` is the overview card at the top of the models page.
