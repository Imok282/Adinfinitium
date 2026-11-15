# worldmodel/hazard — hazard detector (KAN-25)

**Owner:** Nimansh · marks each pixel of a terrain tile as safe or dangerous for landing.

Result in one line: **it recovers NASA's official hazard map (slope ≥ 12°) from sparse laser data.**
It is **not** proof of landing safety.

## Result (real NASA LOLA 5 m data, held-out region)

Both methods see the same input: only the real laser pixels (~11% of the ground) plus uncertainty.
Score = agreement with the official LOLA slope ≥ `lander.touchdown_limits.max_slope_deg` (12°).

| Site 23 (Malapert massif) — never trained or tuned on | Precision | Recall | F1 | Missed hazards |
|---|---|---|---|---|
| Physics backup: slope rule on the gap-filled surface | 0.839 | 0.791 | 0.814 | 20.9% |
| U-Net | 0.948 | 0.953 | 0.950 | 4.7% |

Validation site (Site 11, used to pick the checkpoint): backup F1 0.870, U-Net F1 0.962.
Checkpoint `b0ac044446`, trained on Sites 01, 04, 06, 07, 20. Reproduce: `python -I -m worldmodel.hazard.evaluate --figure`.

## Honest caveats (say these out loud)
- **The label is NASA's slope map, and that map was computed from interpolated heights.** So the U-Net partly learns NASA's
  gap-filling. With full-resolution input the slope rule is already ~perfect (correlation 0.9999), so the U-Net would add
  nothing; that is why the task is sparse-input recovery.
- **1 held-out site, 1 window (1024 × 1024 px = 5 km) per site.** Small sample; not a statistical claim.
- **Rocks / boulders are not covered (UNKNOWN).** No labelled rock data; skipped below the Sunday cut line.
- **PSR (shadow) masks are UNKNOWN** (all False in these tiles).
- Hazard maps are **MODELLED**. A GENERATED input tile stays GENERATED. Never use this alone for a safety claim.

## Physics backup (runs first, no AI, no torch)
`backup.py`: slope of the gap-filled surface ≥ the config limit (`features.gap_fill`, fixed gaussian widths from
`config/mission.yaml → hazard_detector`). `apply.py` falls back to it when there is no checkpoint or no torch.

## Files
| File | What |
|---|---|
| `lola.py` | Fetch a window of real LOLA 5 m GeoTIFFs over HTTP (PGDA) and write `data/tiles/lola5m_<site>.npz` |
| `features.py` | Hide interpolated pixels, gap-fill, build the 5 input channels |
| `backup.py`, `metrics.py` | Slope-rule backup, label, precision/recall |
| `model.py`, `train.py` | U-Net (torch) and training with `--resume` (device: cuda / mps / cpu) |
| `evaluate.py`, `apply.py` | Score vs backup (+ figure); write the `hazard` grid into a tile |

## Run
```bash
pip install -e ".[hazard,dev]"
python -I -m worldmodel.hazard.lola all                 # download ~16 MB x 7 sites (slow: ~15 min) and build tiles
python -I -m worldmodel.hazard.train --epochs 30        # ~15 min on a laptop GPU; add --resume to continue
python -I -m worldmodel.hazard.evaluate --figure
python -I -m worldmodel.hazard.apply --region lola5m_site23 --out lola5m_site23_hazard
```
Splits: train = Sites 01 04 06 07 20 · validation = Site 11 · **test = Site 23** (never loaded by `train.py`).
Site20v2 is an extension of Site20 and is not used (leakage). Checkpoints go to `<ADI_DATA>/models/` (not git).

Data: LOLA/PGDA 5 m south-pole products (Barker et al. 2021), NASA. Rules: see `AGENTS.md`.
