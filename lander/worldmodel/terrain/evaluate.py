"""Hide-and-check: can the terrain generator predict REAL laser measurements it was not shown, better than interpolation?

    python -I -m worldmodel.terrain.evaluate            # held-out TEST tile (Site23) + validation tile (Site11)

For random 64 x 64 patches: hide half of the measured (laser) pixels, give the rest, generate N versions, and compare with the hidden
REAL measurements. Reported against the physics backup (gap-fill of the shown pixels, plus noise from LOLA's error map):
  rmse        error of the version mean vs the real hidden heights (m), lower is better
  coverage90  share of real heights inside the versions' 5-95% band (ideal 90%): are the uncertainty bars honest?
  width90     mean width of that band (m): a band that is right only because it is huge is useless
Generated terrain is NOT observed data; this test is the evidence of how far it can be trusted.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from contracts.mission import load_mission
from contracts.paths import model_path, run_dir
from worldmodel.terrain.backup import backup_versions
from worldmodel.terrain.data import (
    PATCH,
    SD_FLOOR_M,
    TEST_SITES,
    VAL_SITES,
    load_region,
    regions_for_site,
    sample_patch_origins,
)


def band_stats(real: np.ndarray, versions: np.ndarray) -> dict:
    """real [P] hidden heights; versions [N,P] generated heights at the same pixels."""
    mean = versions.mean(0)
    lo, hi = np.percentile(versions, 5, axis=0), np.percentile(versions, 95, axis=0)
    return {"rmse_m": float(np.sqrt(np.mean((mean - real) ** 2))), "mae_m": float(np.mean(np.abs(mean - real))),
            "coverage90_pct": float(100 * np.mean((real >= lo) & (real <= hi))), "width90_m": float(np.mean(hi - lo))}


def hide_and_check(model, tile, cfg, device, n_patches: int, n_versions: int, steps: int, seed: int, p_c: float = 0.5, eta: float = 1.0) -> dict:
    import torch

    from worldmodel.terrain.model import sample

    rng = np.random.default_rng(seed)
    origins = sample_patch_origins(tile.real_mask, n_patches, rng)
    h = tile.height.astype(np.float64)
    real_all, gen_all, bak_all, interp_all = [], [], [], []
    sigmas = cfg.hazard_detector.gap_fill_sigma_px
    for r, c in origins:
        hp, mp = h[r:r + PATCH, c:c + PATCH], tile.real_mask[r:r + PATCH, c:c + PATCH]
        sig = tile.uncertainty[r:r + PATCH, c:c + PATCH].astype(np.float64)
        C = mp & (rng.random(mp.shape) < p_c)
        Hm = mp & ~C
        if C.sum() < 8 or Hm.sum() < 8:
            continue
        mu = hp[C].mean()
        sd = max(hp[C].std(), SD_FLOOR_M)
        kv = torch.from_numpy(((hp - mu) / sd * C).astype(np.float32))[None, None].repeat(n_versions, 1, 1, 1).to(device)
        km = torch.from_numpy(C.astype(np.float32))[None, None].repeat(n_versions, 1, 1, 1).to(device)
        gen = sample(model, kv, km, steps=steps, eta=eta)[:, 0].cpu().numpy() * sd + mu          # [N,H,W]
        bak = backup_versions(hp, C, sig, sigmas, n_versions, rng)
        real_all.append(hp[Hm])
        gen_all.append(gen[:, Hm])
        bak_all.append(bak[:, Hm])
        interp_all.append(bak.mean(0)[Hm])
    real, gen, bak = np.concatenate(real_all), np.concatenate(gen_all, axis=1), np.concatenate(bak_all, axis=1)
    return {"n_patches": len(real_all), "n_hidden_pixels": int(real.size), "model": band_stats(real, gen), "backup": band_stats(real, bak)}


def main(argv: list[str] | None = None) -> int:
    from worldmodel.terrain.model import load_checkpoint, pick_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="terrain_diffusion")
    ap.add_argument("--patches", type=int, default=96)
    ap.add_argument("--versions", type=int, default=8)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_mission()
    device = pick_device(a.device)
    model, info = load_checkpoint(model_path(a.name), device)
    report = {"checkpoint_id": info["checkpoint_id"], "trained_on": info["trained_on"], "label": "GENERATED — NOT OBSERVED", "regions": {}}
    print(f"checkpoint {info['checkpoint_id']} trained on {info['trained_on']}  [outputs are GENERATED — NOT OBSERVED]")
    print(f"{'region':14s} {'role':5s} {'method':24s} {'rmse m':>7s} {'MAE m':>7s} {'cover90%':>9s} {'width90 m':>10s}")
    for site in VAL_SITES + TEST_SITES:
        if not regions_for_site(site):
            print(f"note: no tiles for {site} on this machine; skipped", flush=True)
    for site, region in [(s, r) for s in VAL_SITES + TEST_SITES for r in regions_for_site(s)]:
        role = "TEST" if site in TEST_SITES else "val"
        tile = load_region(region)
        res = hide_and_check(model, tile, cfg, device, a.patches, a.versions, a.steps, seed=11, eta=a.eta)
        report["regions"][tile.region] = {"role": role, **res}
        for k, lab in (("backup", "gap-fill + LOLA-error noise"), ("model", "diffusion model")):
            m = res[k]
            print(f"{tile.region:14s} {role:5s} {lab:24s} {m['rmse_m']:7.2f} {m['mae_m']:7.2f} {m['coverage90_pct']:9.1f} {m['width90_m']:10.2f}")
    out = run_dir("terrain") / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
