"""Export REAL lander-camera crops for the website's pilot section (website/assets/img/pilot_patches.png + .json).

Run from the repo root (read-only on data/tiles, writes only website/assets/img/pilot_patches.{png,json}):
    ~/venvs/adi/bin/python website/tools/export_patches.py

Each patch is computed exactly the way the lander environment feeds the learned pilot:
    LanderBatch.crop() (worldmodel/lander/env.py): span = clamp(altitude + 60 m, 60 m, 640 m), px = 32
    TerrainT.crop()   (worldmodel/lander/terrain.py): egocentric, north-up, bilinear grid_sample of 3 channels
        ch0 = slope / slope limit (12 deg), ch1 = hazard belief (slope-rule backup map, as in scenes.py), ch2 = (height - centre height) / 50 m
    clamped to [-4, 4].
The three channels are then turned into one display image (deterministic, stated in the JSON):
    grey   = 0.5 + ch2 / 2   (relative height: brighter = higher than the lander's ground point)
    amber  = blended in proportional to clip(ch0, 0, 1)   (slope as a fraction of the 12 deg limit)
    red    = where ch1 >= 0.5   (hazard belief)
Cells 0..N-1 are those composites. The last three cells are the raw channels of the featured patch, each as greyscale (ch0 0..1, ch1 0..1, ch2 -2..2).
Positions: around safe landing cells of each tile (safe_target_cells), altitudes 0..600 m, sorted high -> low (a descent).
Tiles: lola5m_site06 (a training map) and lola5m_site42 / lola5m_shoemaker (maps the pilot never saw; see runs/lander/benchmark.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from contracts.mission import load_mission
from contracts.paths import tile_path
from contracts.tiles import load_tile
from worldmodel.hazard.backup import backup_hazard
from worldmodel.lander.terrain import safe_target_cells, terrain_from_tile

OUT = Path(__file__).resolve().parents[1] / "assets" / "img"
PX, SPAN_MIN, SPAN_MAX = 32, 60.0, 640.0          # LanderBatch.crop defaults
TILES = [("lola5m_site06", "training map"), ("lola5m_site42", "unseen map"), ("lola5m_shoemaker", "unseen map")]
PER_TILE = 24
COLS = 10


def composite(img: np.ndarray) -> np.ndarray:
    slope, haz, rel = img
    g = np.clip(0.5 + rel / 2.0, 0, 1)
    base = np.stack([g, g, g], -1) * np.array([0.80, 0.86, 1.0])          # cool grey moon
    s = np.clip(slope, 0, 1)[..., None]
    rgb = base * (1 - 0.55 * s) + np.array([1.0, 0.71, 0.28]) * 0.55 * s * (0.35 + 0.65 * g[..., None])
    h = (haz >= 0.5)[..., None]
    rgb = np.where(h, rgb * 0.45 + np.array([1.0, 0.25, 0.3]) * 0.55, rgb)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def grey(ch: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = (np.clip((ch - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def main() -> None:
    cfg = load_mission()
    limit = float(cfg.lander.touchdown_limits.max_slope_deg)
    rng = np.random.default_rng(23)
    cells, meta = [], []
    raw0 = None
    for region, role in TILES:
        tile = load_tile(tile_path(region))
        T = terrain_from_tile(tile, belief=backup_hazard(tile, cfg))
        safe = safe_target_cells(tile)
        rc = safe[rng.integers(len(safe), size=PER_TILE)]
        xy = T.cell_to_xy(rc) + rng.normal(0, 40, size=(PER_TILE, 2))       # around a landing cell, like a descent
        alt = rng.uniform(0, 600, size=PER_TILE)
        span = np.clip(alt + SPAN_MIN, SPAN_MIN, SPAN_MAX)
        f = lambda a: torch.as_tensor(a, dtype=torch.float32)
        with torch.no_grad():
            img = T.crop(f(xy[:, 0]), f(xy[:, 1]), f(span), PX, limit).numpy()
        r, c = T.to_rc(f(xy[:, 0]), f(xy[:, 1]))
        for i in range(PER_TILE):
            cells.append((alt[i], img[i]))
            meta.append({"tile": region, "role": role, "row": int(round(float(r[i]))), "col": int(round(float(c[i]))),
                         "altitude_m": round(float(alt[i]), 1), "span_m": round(float(span[i]), 1),
                         "m_per_px": round(float(span[i]) / (PX - 1), 2), "tile_m_per_px": float(T.mpp)})
    order = np.argsort([-a for a, _ in cells])                      # high -> low
    cells = [cells[k] for k in order]
    meta = [meta[k] for k in order]
    for k, m in enumerate(meta):
        m["cell"] = k
    # featured patch (shown big, with its 3 raw channels): a low pass with some hazard and the most relief
    cand = [k for k, (a, im) in enumerate(cells) if a < 300 and 0.05 < (im[1] >= 0.5).mean() < 0.35] or [len(cells) - 1]
    feat = max(cand, key=lambda k: float(cells[k][1][2].std()))
    raw0 = cells[feat][1]
    imgs = [composite(im) for _, im in cells] + [grey(raw0[0], 0, 1), grey(raw0[1], 0, 1), grey(raw0[2], -2, 2)]
    n = len(imgs)
    rows = (n + COLS - 1) // COLS
    atlas = np.zeros((rows * PX, COLS * PX, 3), np.uint8)
    for k, im in enumerate(imgs):
        atlas[(k // COLS) * PX:(k // COLS + 1) * PX, (k % COLS) * PX:(k % COLS + 1) * PX] = im
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(OUT / "pilot_patches.png", optimize=True)
    doc = {
        "meta": {"provenance": "observed", "source": "NASA LOLA 5 m/px tiles " + ", ".join(t for t, _ in TILES) + " (data/tiles/), cropped by "
                 "worldmodel/lander/terrain.py TerrainT.crop as LanderBatch.crop does", "frame": "egocentric north-up, centred on the lander",
                 "units": "ch0 slope/12deg, ch1 hazard belief 0..1, ch2 relative height /50 m",
                 "note": "Terrain is observed (LOLA); the hazard channel is the slope-rule backup belief (modelled from observed slope). "
                         "Lander positions and altitudes are sampled (random, seed 23), not from a flown trajectory. Display colours: grey = relative "
                         "height, amber = slope fraction of the limit, red = hazard belief >= 0.5. Last 3 cells = raw channels of the featured cell."},
        "px": PX, "cols": COLS, "rows": rows, "n_patches": len(meta), "featured": int(feat), "channel_cells": [len(meta), len(meta) + 1, len(meta) + 2],
        "channel_labels": ["slope ÷ 12° limit", "hazard belief", "height vs. centre"],
        "slope_limit_deg": limit, "patches": meta,
    }
    (OUT / "pilot_patches.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote {OUT / 'pilot_patches.png'} ({(OUT / 'pilot_patches.png').stat().st_size} B), {n} cells")


if __name__ == "__main__":
    main()
