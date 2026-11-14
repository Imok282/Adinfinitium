"""Train the skyline model on real LOLA tiles (cross-platform: cuda / mps / cpu, no Linux-only code).

    python -I -m worldmodel.light.train                        # laptop
    python -I -m worldmodel.light.train --steps 6000 --resume   # the 4090 (native Windows PyTorch is enough)

Splits: train Site01 04 06 07 20 · validation Site11 (picks the best checkpoint) · TEST Site23 is never loaded here.
Checkpoints: <ADI_DATA>/models/light_skyline.pt (best validation) and light_skyline_last.pt (resume). Not in git.
Prerequisite: tiles from  python -I -m worldmodel.hazard.lola all  (or Imok's real tiles, same format).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from contracts.mission import load_mission
from contracts.paths import model_path, run_dir
from worldmodel.budget import TimeBudget
from worldmodel.light.data import TEST_SITES, TRAIN_SITES, VAL_SITES, build_site_windows
from worldmodel.light.model import build_model, pick_device, ray_inputs, save_checkpoint


def to_device(site, device):
    return {"coarse": torch.from_numpy(site.coarse)[None, None].to(device), "cells": torch.from_numpy(site.cells_coarse.astype(np.float32)).to(device),
            "truth": torch.from_numpy(site.truth).to(device), "mpp": site.mpp_coarse, "region": site.region}


def make_batch(sites: list[dict], batch: int, rng: np.random.Generator, cfg, augment: bool):
    lm = cfg.light_model
    s = sites[rng.integers(len(sites))]
    idx = torch.from_numpy(rng.integers(0, s["cells"].shape[0], batch)).to(s["cells"].device)
    x, base = ray_inputs(s["coarse"], s["cells"][idx], s["mpp"], lm.range_steps, lm.n_azimuth, cfg.moon.radius_m)
    y = s["truth"][idx]
    if augment:                                   # rotating or mirroring the compass is exact physics: relabel the directions
        shift = int(rng.integers(lm.n_azimuth))
        x, base, y = torch.roll(x, shift, -1), torch.roll(base, shift, -1), torch.roll(y, shift, -1)
        if rng.random() < 0.5:
            m = (-torch.arange(lm.n_azimuth, device=x.device)) % lm.n_azimuth
            x, base, y = x[..., m], base[:, m], y[:, m]
    return x, base, y


@torch.no_grad()
def validate(model, sites: list[dict], cfg, chunk: int = 256) -> dict:
    lm = cfg.light_model
    e_net, e_base = [], []
    for s in sites:
        for a in range(0, s["cells"].shape[0], chunk):
            x, base = ray_inputs(s["coarse"], s["cells"][a:a + chunk], s["mpp"], lm.range_steps, lm.n_azimuth, cfg.moon.radius_m)
            y = s["truth"][a:a + chunk]
            e_net.append((model(x, base) - y).abs().flatten().cpu())
            e_base.append((base - y).abs().flatten().cpu())
    e_net, e_base = torch.cat(e_net), torch.cat(e_base)
    return {"mae_net_deg": float(e_net.mean()), "mae_baseline_deg": float(e_base.mean()), "p95_net_deg": float(e_net.quantile(0.95)) if len(e_net) < 16_000_000 else float("nan")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--arch", choices=["conv", "former"], default="conv", help="conv = 33 k at width 24; former = transformer across the 360 directions (~38 M at --width 64 --d 512 --layers 12)")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--cells", type=int, default=4000, help="random cells per training tile")
    ap.add_argument("--val-cells", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock budget: the run sizes its own step count to finish in time (0 = off)")
    ap.add_argument("--name", default="light_skyline")
    a = ap.parse_args(argv)

    cfg = load_mission()
    assert not set(TRAIN_SITES + VAL_SITES) & set(TEST_SITES), "test region leaked into training"
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    train = [to_device(s, device) for i, site in enumerate(TRAIN_SITES) for s in build_site_windows(site, cfg, a.cells, a.seed + 100 * i, device)]
    val = [to_device(s, device) for i, site in enumerate(VAL_SITES) for s in build_site_windows(site, cfg, a.val_cells, 1000 + 100 * i, device)]
    if not train or not val:
        raise SystemExit("no tiles for the train/validation sites: fetch them first (python -I -m worldmodel.hazard.lola all)")
    print(f"device={device} train={[s['region'] for s in train]} val={[s['region'] for s in val]} (test {TEST_SITES} never loaded) "
          f"data ready in {time.time() - t0:.0f}s", flush=True)

    lm = cfg.light_model
    kw = {"width": a.width} if a.arch == "conv" else {"width": a.width, "d": a.d, "layers": a.layers, "heads": a.heads, "k_steps": lm.range_steps, "n_az": lm.n_azimuth}
    model = build_model(a.arch, **kw).to(device)
    print(f"architecture {a.arch}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    last_p, best_p = model_path(a.name + "_last"), model_path(a.name)
    step, best = 0, float("inf")
    if a.resume and last_p.exists():
        state = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        step, best = state["info"]["step"], state["info"].get("best_val_mae_deg", float("inf"))
        print(f"resumed at step {step} (best val MAE {best:.4f} deg)", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps - step, 1))
    v0 = validate(model, val, cfg)
    print(f"start: val MAE net {v0['mae_net_deg']:.4f} deg | physics baseline (coarse ray-cast) {v0['mae_baseline_deg']:.4f} deg", flush=True)
    log = run_dir("light") / "train_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    run_loss, t1 = 0.0, time.time()
    bud = TimeBudget(a.max_minutes, a.steps)
    bud.begin(step)
    while step < a.steps:
        model.train()
        x, base, y = make_batch(train, a.batch, rng, cfg, augment=True)
        opt.zero_grad()
        loss = F.l1_loss(model(x, base), y)
        loss.backward()
        opt.step()
        sched.step()
        run_loss += float(loss.detach())
        step += 1
        if bud.update(step):
            a.steps = bud.total
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps - step, 1))
            print(f"{bud.summary()}", flush=True)
        if bud.expired():
            a.steps = step
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            v = validate(model, val, cfg)
            info = {"step": step, "train_l1_deg": run_loss / a.eval_every, "val": v, "trained_on": [s["region"] for s in train],
                    "val_regions": [s["region"] for s in val], "best_val_mae_deg": min(best, v["mae_net_deg"]),
                    "data": "real LOLA 5 m (PGDA), exact fine skyline vs coarse input", "coarse_factor": cfg.light_model.coarse_factor,
                    "max_ray_range_m": cfg.light_model.max_ray_range_m}
            if v["mae_net_deg"] < best:
                best = v["mae_net_deg"]
                save_checkpoint(best_p, model, info)
            save_checkpoint(last_p, model, info, extra={"optimizer": opt.state_dict()})
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**info, "seconds": time.time() - t1}) + "\n")
            print(f"step {step:5d}/{a.steps} train L1 {run_loss / a.eval_every:.4f} | val MAE net {v['mae_net_deg']:.4f} deg "
                  f"(baseline {v['mae_baseline_deg']:.4f}) | best {best:.4f} | {time.time() - t1:.0f}s", flush=True)
            run_loss = 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
