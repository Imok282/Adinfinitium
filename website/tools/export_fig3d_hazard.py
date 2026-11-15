"""Textures for the 3D hazard drape figure on research/hazard-model.html (js/fig3d/hazard.js `drape`).

Usage (read-only on the repo; writes website/assets/fig3d/hazard/):
    ~/venvs/adi/bin/python -I website/tools/export_fig3d_hazard.py

Window: held-out Site 23 centre tile, data/tiles/lola5m_site23.npz, 1024 x 1024 px at 5 m (5.12 km), the same window
scored in runs/hazard/eval.json and the same ground as assets/models3d `a` (whose 256-grid mesh the textures drape on).
Maps, all at the full 1024 x 1024 resolution (no downsampling, so the confusion counts equal eval.json):
  label  : official LOLA slope >= limit       worldmodel.hazard.backup.label_hazard   (observed/derived by NASA)
  backup : slope rule on the gap-filled input worldmodel.hazard.backup.backup_hazard  (physics, no AI)
  unet   : U-Net P(unsafe), data/viewer_cache/lola5m_site23_hazard_<ckpt mtime>.npy (model_viewer cache of
           data/models/hazard_unet.pt), unsafe at >= hazard_detector.decision_threshold
False negatives = label unsafe, map says safe. Written as separate "fn" variants of the unet / backup textures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from contracts.mission import load_mission  # noqa: E402
from contracts.paths import tile_path  # noqa: E402
from contracts.tiles import load_tile  # noqa: E402
from worldmodel.hazard.backup import backup_hazard, label_hazard, slope_limit_deg  # noqa: E402
from worldmodel.hazard.metrics import confusion  # noqa: E402

OUT = REPO / "website" / "assets" / "fig3d" / "hazard"
REGION = "lola5m_site23"

SAFE = np.array([0.36, 0.62, 0.50])     # muted green (--ok family)
UNSAFE = np.array([0.90, 0.42, 0.38])   # muted red (--bad family)
FN = np.array([1.00, 0.86, 0.10])       # vivid yellow: missed hazards


def hillshade(h, mpp, az_deg=315.0, alt_deg=35.0):
    gy, gx = np.gradient(h, mpp)
    gy = -gy
    nx, ny = -gx, -gy
    norm = np.sqrt(nx * nx + ny * ny + 1.0)
    az, alt = np.deg2rad(az_deg), np.deg2rad(alt_deg)
    sx, sy, sz = np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)
    return np.clip((nx * sx + ny * sy + sz) / norm, 0, 1)


def save(name, rgb, q=82):
    tmp = OUT / (name + ".tmp")
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), "RGB").save(tmp, format="JPEG", quality=q, optimize=True, progressive=True)
    tmp.replace(OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_mission()
    lim = slope_limit_deg(cfg)
    thr = cfg.hazard_detector.decision_threshold
    tile = load_tile(tile_path(REGION))
    mpp = float(tile.meters_per_pixel)
    label = label_hazard(tile, cfg) > 0.5
    backup = backup_hazard(tile, cfg) > 0.5
    ckpt = REPO / "data/models/hazard_unet.pt"
    cache = REPO / "data/viewer_cache" / f"{REGION}_hazard_{ckpt.stat().st_mtime_ns}.npy"
    prob = np.load(cache).astype(np.float32)
    unet = prob >= thr
    assert prob.shape == label.shape == backup.shape, (prob.shape, label.shape)

    ev = json.loads((REPO / "runs/hazard/eval.json").read_text())["regions"][REGION]
    stats = {}
    for k, pred in (("unet", unet), ("backup", backup)):
        c = confusion(pred, label)
        ref = ev["unet" if k == "unet" else "backup_slope_rule"]
        for f in ("tp", "fp", "fn", "tn"):
            assert int(c[f]) == int(ref[f]), (k, f, c[f], ref[f])
        stats[k] = {"tp": int(c["tp"]), "fp": int(c["fp"]), "fn": int(c["fn"]), "tn": int(c["tn"]),
                    "f1": round(float(c["f1"]), 4), "missed_hazard_rate": round(float(c["missed_hazard_rate"]), 6),
                    "missed_area_km2": round(int(c["fn"]) * mpp * mpp / 1e6, 3)}
        print(k, stats[k])

    shade = hillshade(tile.height.astype(np.float64), mpp)[..., None]
    lit = 0.62 + 0.62 * shade

    def binary(m):
        return np.where(m[..., None], UNSAFE, SAFE) * lit

    col_unet = (SAFE + (UNSAFE - SAFE) * np.clip(prob, 0, 1)[..., None]) * lit
    maps = {"label": binary(label), "backup": binary(backup), "unet": col_unet}
    for k, rgb in maps.items():
        save(f"drape_{k}.jpg", rgb)
    for k, pred in (("unet", unet), ("backup", backup)):
        fn = label & ~pred
        rgb = maps[k] * 0.72   # dim the rest a little so the misses stand out
        rgb[fn] = (FN * (0.75 + 0.35 * shade[fn])).clip(0, 1)
        save(f"drape_{k}_fn.jpg", rgb)

    meta = {
        "created_by": "website/tools/export_fig3d_hazard.py",
        "region": REGION, "window_px": list(label.shape), "m_per_px": mpp, "size_m": mpp * label.shape[0],
        "height": "assets/models3d/site23_height.bin (models3d.json 'a')",
        "slope_limit_deg": lim, "decision_threshold": thr,
        "prevalence": round(float(label.mean()), 4),
        "maps": {
            "label": {"texture": "drape_label.jpg", "provenance": "observed",
                      "source": f"official LOLA slope >= {lim:g} deg (worldmodel.hazard.backup.label_hazard on {REGION})"},
            "backup": {"texture": "drape_backup.jpg", "fn_texture": "drape_backup_fn.jpg", "provenance": "modelled",
                       "source": "slope rule on the gap-filled laser input (worldmodel.hazard.backup.backup_hazard)", **stats["backup"]},
            "unet": {"texture": "drape_unet.jpg", "fn_texture": "drape_unet_fn.jpg", "provenance": "modelled",
                     "source": f"U-Net P(unsafe), checkpoint data/models/hazard_unet.pt, cache data/viewer_cache/{cache.name}", **stats["unet"]},
        },
        "check": "confusion counts recomputed at full 1024 x 1024 resolution equal runs/hazard/eval.json exactly",
        "colours": {"safe": SAFE.tolist(), "unsafe": UNSAFE.tolist(), "missed": FN.tolist(), "shading": "x hillshade (az 315, alt 35)"},
    }
    tmp = OUT / "drape.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(OUT / "drape.json")


if __name__ == "__main__":
    main()
