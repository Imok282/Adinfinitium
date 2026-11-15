"""Real hazard U-Net outputs for 3 windows of held-out LOLA tiles -> website/assets/examples/hazard/.

    ~/venvs/adi/bin/python -I website/tools/examples/hazard_examples.py

Read-only use of data/models/hazard_unet.pt (no training). Runs the whole 1024 px tile through the model
exactly as worldmodel.hazard.evaluate does, then crops 256 px windows chosen by unsafe fraction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from contracts.mission import load_mission  # noqa: E402
from contracts.paths import model_path, tile_path  # noqa: E402
from contracts.tiles import load_tile  # noqa: E402
from worldmodel.hazard.backup import backup_hazard, label_hazard, slope_limit_deg  # noqa: E402
from worldmodel.hazard.features import build_features, gap_fill, measured_mask  # noqa: E402
from worldmodel.hazard.metrics import confusion  # noqa: E402
from worldmodel.hazard.model import load_checkpoint, pick_device, predict_probability  # noqa: E402

OUT = ROOT / "website/assets/examples/hazard"
W = 256  # window size px (5 m/px -> 1.28 km)


def hillshade(h, mpp, az=315.0, alt=35.0):
    gy, gx = np.gradient(h, mpp)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(az), np.radians(90 - alt)
    s = np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(s, 0, 1)


def save(arr_rgb, name):
    p = OUT / name
    Image.fromarray(arr_rgb).save(p, "WEBP", quality=82, method=6)
    return p


def gray(x):
    x = (x - x.min()) / (np.ptp(x) + 1e-9)
    g = (x * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def prob_rgb(p):
    import matplotlib
    cm = matplotlib.colormaps["magma"]
    return (cm(np.clip(p, 0, 1))[..., :3] * 255).astype(np.uint8)


def mask_rgb(m):
    out = np.zeros(m.shape + (3,), np.uint8)
    out[~m] = (34, 160, 90)   # safe green
    out[m] = (220, 50, 47)    # unsafe red
    return out


def error_rgb(pred, truth, base):
    out = (gray(base) * 0.45).astype(np.uint8)
    out[pred & truth] = (200, 200, 200)       # TP light grey
    out[~pred & truth] = (230, 30, 30)        # FN red
    out[pred & ~truth] = (40, 110, 240)       # FP blue
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_mission()
    device = pick_device(None)
    model, info = load_checkpoint(model_path("hazard_unet"), device)
    thr = cfg.hazard_detector.decision_threshold
    lim = slope_limit_deg(cfg)
    sig = cfg.hazard_detector.gap_fill_sigma_px

    cache = {}

    def run(region):
        if region in cache:
            return cache[region]
        t = load_tile(tile_path(region))
        feats, filled_slope = build_features(t, lim, sig)
        prob = predict_probability(model, feats, device)
        filled = gap_fill(t.height, measured_mask(t), sig)
        cache[region] = (t, prob, backup_hazard(t, cfg) > 0.5, label_hazard(t, cfg) > 0.5, filled, filled_slope)
        return cache[region]

    # candidate windows on the two held-out sites (never trained or tuned on)
    cands = []
    for region in ("lola5m_site23", "lola5m_site42"):
        t, prob, bk, lab, filled, _ = run(region)
        H, Wd = lab.shape
        for y in range(0, H - W + 1, 128):
            for x in range(0, Wd - W + 1, 128):
                cands.append((region, y, x, float(lab[y:y + W, x:x + W].mean()),
                              float(measured_mask(t)[y:y + W, x:x + W].mean())))
    # steep / moderate / flat from Site23 (the headline TEST site); a 4th "hard_case" moderate window from Site42
    want = {"steep": ("lola5m_site23", 0.45), "moderate": ("lola5m_site23", 0.15), "flat": ("lola5m_site23", 0.02),
            "hard_case": ("lola5m_site42", 0.15)}
    chosen = {}
    for kind, (reg, target) in want.items():
        pool = [c for c in cands if c[0] == reg]
        best = min(pool, key=lambda c: abs(c[3] - target) - 0.05 * c[4]
                   + (1.0 if any(c[0] == v[0] and abs(c[1] - v[1]) < W and abs(c[2] - v[2]) < W for v in chosen.values()) else 0))
        chosen[kind] = best

    windows = []
    for kind, (region, y, x, uf, lf) in chosen.items():
        t, prob, bk, lab, filled, fslope = run(region)
        sl = (slice(y, y + W), slice(x, x + W))
        p, b, l = prob[sl], bk[sl], lab[sl]
        u = p >= thr
        meas = measured_mask(t)[sl]
        hs = hillshade(filled[sl], t.meters_per_pixel)
        inp = gray(hs)
        # mark the real laser pixels the model sees (amber) on the input image
        inp_l = inp.copy()
        inp_l[meas] = (inp_l[meas] * 0.4 + np.array([255, 170, 40]) * 0.6).astype(np.uint8)
        files = {
            "input_hillshade": save(inp, f"{kind}_input.webp").name,
            "input_laser_pixels": save(inp_l, f"{kind}_input_laser.webp").name,
            "unet_probability": save(prob_rgb(p), f"{kind}_unet_prob.webp").name,
            "unet_mask": save(mask_rgb(u), f"{kind}_unet_mask.webp").name,
            "backup_slope_rule": save(mask_rgb(b), f"{kind}_backup.webp").name,
            "truth": save(mask_rgb(l), f"{kind}_truth.webp").name,
            "unet_error": save(error_rgb(u, l, hs), f"{kind}_unet_error.webp").name,
            "backup_error": save(error_rgb(b, l, hs), f"{kind}_backup_error.webp").name,
        }
        cu, cb = confusion(u, l), confusion(b, l)
        keep = ("precision", "recall", "f1", "missed_hazard_rate", "tp", "fp", "fn", "tn")
        windows.append({
            "kind": kind, "region": region, "site": region.replace("lola5m_", "").title(),
            "split": "TEST (held out: never trained or tuned on)",
            "window_px": [int(y), int(x), W, W], "meters_per_pixel": t.meters_per_pixel,
            "size_m": W * t.meters_per_pixel,
            "unsafe_fraction_truth": round(uf, 4), "laser_pixel_fraction": round(lf, 4),
            "mean_official_slope_deg": round(float(t.slope[sl].mean()), 2),
            "max_official_slope_deg": round(float(t.slope[sl].max()), 2),
            "unet": {k: (round(cu[k], 4) if isinstance(cu[k], float) else cu[k]) for k in keep},
            "backup_slope_rule": {k: (round(cb[k], 4) if isinstance(cb[k], float) else cb[k]) for k in keep},
            "unet_prob_mean": round(float(p.mean()), 4),
            "images": files,
        })
        print(kind, region, y, x, f"uf={uf:.3f}", "unet f1", round(cu["f1"], 3), "backup f1", round(cb["f1"], 3))

    doc = {
        "model": "hazard U-Net", "checkpoint": "data/models/hazard_unet.pt", "checkpoint_id": info["checkpoint_id"],
        "trained_on": info["trained_on"], "validated_on": info["val_regions"],
        "threshold": thr, "slope_limit_deg": lim,
        "input": "only real LOLA laser pixels (~11% of ground) + uncertainty; gaps refilled by fixed gaussian gap-fill; "
                 "the input image is a hillshade of that gap-filled surface (amber dots = real laser pixels in the _laser variant)",
        "truth": "official LOLA 5 m slope >= 12 deg (NASA product; computed from interpolated heights)",
        "error_legend": {"red": "missed hazard (FN)", "blue": "false alarm (FP)", "light grey": "hazard found (TP)", "dark": "safe, correct"},
        "mask_legend": {"red": "unsafe", "green": "safe"},
        "probability_colormap": "magma 0..1 (black = safe, yellow = unsafe)",
        "provenance": "observed LOLA input; hazard maps MODELLED; not a landing-safety claim on its own",
        "windows": windows,
        "whole_tile_1024px": {r: {"unet_f1": round(confusion(v[1] >= thr, v[3])["f1"], 4),
                                  "backup_f1": round(confusion(v[2], v[3])["f1"], 4),
                                  "unet_recall": round(confusion(v[1] >= thr, v[3])["recall"], 4),
                                  "unsafe_fraction": round(float(v[3].mean()), 4)} for r, v in cache.items()},
        "note": "windows chosen automatically by truth unsafe fraction (targets 0.45 / 0.15 / 0.02), not by model score",
    }
    (OUT / "hazard_examples.json").write_text(json.dumps(doc, indent=1))
    print("wrote", OUT / "hazard_examples.json")


if __name__ == "__main__":
    main()
