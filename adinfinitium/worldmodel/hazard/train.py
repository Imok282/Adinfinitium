"""Train the hazard U-Net on real LOLA tiles (sparse-input recovery).

    python -I -m worldmodel.hazard.train                       # laptop: picks cuda / mps / cpu
    python -I -m worldmodel.hazard.train --epochs 60 --resume  # 4090 overnight: continue from the last checkpoint

Splits (regions are whole tiles, never mixed): train = Site01 04 06 07 20, validation = Site11
(picks the best epoch), TEST = Site23, which is never loaded here. Checkpoints go to
<ADI_DATA>/models/ (never git): hazard_unet.pt (best validation F1) and hazard_unet_last.pt (resume).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from contracts.mission import load_mission
from contracts.paths import model_path, run_dir, tile_path
from contracts.tiles import load_tile
from worldmodel.budget import TimeBudget
from worldmodel.hazard.backup import label_hazard, slope_limit_deg
from worldmodel.hazard.features import N_CHANNELS, build_features
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, regions_for_site
from worldmodel.hazard.metrics import confusion
from worldmodel.hazard.model import build_model, pick_device, predict_probability, save_checkpoint

VAL_SITES = ["Site11"]


def load_split(sites: list[str], cfg):
    """Every built window of every site (centre + grid windows). Sites whose tiles are not on this machine are skipped with a note."""
    out = []
    for s in sites:
        regions = regions_for_site(s)
        if not regions:
            print(f"note: no tiles for {s} on this machine (python -I -m worldmodel.hazard.lola all --sites {s}); skipped", flush=True)
        for region in regions:
            t = load_tile(tile_path(region))
            feats, _ = build_features(t, slope_limit_deg(cfg), cfg.hazard_detector.gap_fill_sigma_px)
            out.append((t.region, feats, label_hazard(t, cfg)))
    if not out:
        raise SystemExit(f"no tiles found for {sites}: fetch them first (python -I -m worldmodel.hazard.lola all)")
    return out


def sample_batch(data, batch: int, crop: int, rng: np.random.Generator, device):
    xs, ys = [], []
    for _ in range(batch):
        _, f, y = data[rng.integers(len(data))]
        r = rng.integers(0, f.shape[1] - crop + 1)
        c = rng.integers(0, f.shape[2] - crop + 1)
        fx, fy = f[:, r:r + crop, c:c + crop], y[r:r + crop, c:c + crop]
        k = int(rng.integers(4))  # rotations/flips: every channel is a scalar field, so this is safe
        fx, fy = np.rot90(fx, k, axes=(1, 2)), np.rot90(fy, k)
        if rng.random() < 0.5:
            fx, fy = fx[:, :, ::-1], fy[:, ::-1]
        xs.append(np.ascontiguousarray(fx))
        ys.append(np.ascontiguousarray(fy))
    return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device)


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


def evaluate_split(model, data, thr, device) -> dict:
    ms = []
    for _, f, y in data:
        ms.append(confusion(predict_probability(model, f, device) >= thr, y > 0.5))
    return {k: float(np.mean([m[k] for m in ms])) for k in ("precision", "recall", "f1")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100, help="steps per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--arch", choices=["unet", "attn"], default="unet", help="unet = plain (1.9 M at base 16); attn = deep residual attention U-Net (~60 M at --base 72)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2, help="residual blocks per level (attn only)")
    ap.add_argument("--heads", type=int, default=8, help="attention heads (attn only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from hazard_unet_last.pt")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock budget: the run sizes its own epoch count to finish in time (0 = off)")
    ap.add_argument("--name", default="hazard_unet")
    a = ap.parse_args(argv)

    t_start = time.time()
    cfg = load_mission()
    thr = cfg.hazard_detector.decision_threshold
    train_sites = [s for s in TRAIN_SITES if s not in VAL_SITES]
    assert not set(train_sites + VAL_SITES) & set(HOLDOUT_SITES), "hold-out region leaked into training"
    train, val = load_split(train_sites, cfg), load_split(VAL_SITES, cfg)
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    kw = {"base": a.base, "depth": a.depth} | ({"blocks": a.blocks, "heads": a.heads} if a.arch == "attn" else {})
    model = build_model(a.arch, in_ch=N_CHANNELS, **kw).to(device)
    print(f"architecture {a.arch}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    start, best = 0, -1.0
    last_p, best_p = model_path(a.name + "_last"), model_path(a.name)
    if a.resume and last_p.exists():
        state = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        start, best = state["info"]["epoch"], state["info"].get("best_val_f1", -1.0)
        print(f"resumed from epoch {start} (best val F1 {best:.3f})", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs - start, 1))
    log = run_dir("hazard") / "train_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train={[r for r, _, _ in train]} val={[r for r, _, _ in val]} "
          f"(test region {HOLDOUT_SITES} is never loaded)", flush=True)

    bud = TimeBudget(a.max_minutes, a.epochs, spent_s=time.time() - t_start)
    bud.begin(start)
    epoch = start
    while epoch < a.epochs:
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(a.steps):
            x, y = sample_batch(train, a.batch, a.crop, rng, device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        sched.step()
        v = evaluate_split(model, val, thr, device)
        info = {"epoch": epoch + 1, "train_loss": tot / a.steps, "val": v, "best_val_f1": max(best, v["f1"]),
                "trained_on": [r for r, _, _ in train], "val_regions": [r for r, _, _ in val],
                "data": "real LOLA 5 m (PGDA), sparse-input recovery", "limit_deg": slope_limit_deg(cfg),
                "gap_fill_sigma_px": cfg.hazard_detector.gap_fill_sigma_px}
        if v["f1"] > best:
            best = v["f1"]
            save_checkpoint(best_p, model, info)
        save_checkpoint(last_p, model, info, extra={"optimizer": opt.state_dict()})
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**info, "seconds": time.time() - t0}) + "\n")
        print(f"epoch {epoch + 1:3d}/{a.epochs} loss {tot / a.steps:.4f} | val P {v['precision']:.3f} "
              f"R {v['recall']:.3f} F1 {v['f1']:.3f} | best {best:.3f} | {time.time() - t0:.0f}s", flush=True)
        epoch += 1
        if bud.update(epoch):
            a.epochs = bud.total
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs - epoch, 1))
            print(bud.summary(), flush=True)
        if bud.expired():
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
