"""Train the terrain diffusion model on real laser-measured pixels only (hide-and-check training).

    python -I -m worldmodel.terrain.train                         # laptop (cuda / mps / cpu)
    python -I -m worldmodel.terrain.train --steps 12000 --resume   # the 4090 (native Windows PyTorch is enough)

Each sample: a 64 x 64 patch of a real tile. Its measured (laser) pixels M are split at random into conditioning C and hidden H.
The model sees noisy terrain + C and must predict the HIDDEN real measurements: the loss is computed on H only, never on interpolated
pixels. Unmeasured pixels start from LOLA's gap-fill only to give the noisy canvas a shape; they are never supervised.
Splits: train Site01 04 06 07 20 · validation Site11 · TEST Site23 is never loaded here. Checkpoints: <ADI_DATA>/models/terrain_diffusion*.pt
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from contracts.paths import model_path, run_dir
from worldmodel.budget import TimeBudget
from worldmodel.terrain.data import (
    PATCH,
    SD_FLOOR_M,
    TEST_SITES,
    TRAIN_SITES,
    VAL_SITES,
    load_region,
    regions_for_site,
    sample_patch_origins,
)
from worldmodel.terrain.model import build_model, cosine_alpha_bar, pick_device, q_sample, save_checkpoint


class SitePatches:
    def __init__(self, region: str, device, rng: np.random.Generator, n_origins: int = 6000):
        t = load_region(region)
        self.region, self.h, self.m = t.region, t.height.astype(np.float32), t.real_mask
        self.origins = sample_patch_origins(self.m, n_origins, rng)


def make_batch(sites: list[SitePatches], batch: int, rng: np.random.Generator, device, p_c_range=(0.3, 0.9), fixed_p_c: float | None = None):
    hs, ms = [], []
    for _ in range(batch):
        s = sites[rng.integers(len(sites))]
        r, c = s.origins[rng.integers(len(s.origins))]
        hs.append(s.h[r:r + PATCH, c:c + PATCH])
        ms.append(s.m[r:r + PATCH, c:c + PATCH])
    h = torch.from_numpy(np.stack(hs))[:, None].to(device)
    m = torch.from_numpy(np.stack(ms))[:, None].to(device).float()
    p_c = torch.full((batch, 1, 1, 1), fixed_p_c, device=device) if fixed_p_c else torch.empty(batch, 1, 1, 1, device=device).uniform_(*p_c_range)
    C = m * (torch.rand_like(m) < p_c).float()
    Hm = m * (1 - C)
    n = C.sum((1, 2, 3), keepdim=True).clamp(min=1)
    mu = (h * C).sum((1, 2, 3), keepdim=True) / n
    sd = (((h - mu) ** 2 * C).sum((1, 2, 3), keepdim=True) / n).sqrt().clamp(min=SD_FLOOR_M)
    return (h - mu) / sd, C, Hm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--arch", choices=["unet", "attn"], default="unet", help="unet = 2 M at base 32; attn = attention U-Net (~112 M at --base 192)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="wall-clock budget: the run sizes its own step count to finish in time (0 = off)")
    ap.add_argument("--name", default="terrain_diffusion")
    a = ap.parse_args(argv)
    assert not set(TRAIN_SITES + VAL_SITES) & set(TEST_SITES), "test region leaked into training"
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    for s in TRAIN_SITES + VAL_SITES:
        if not regions_for_site(s):
            print(f"note: no tiles for {s} on this machine; skipped", flush=True)
    train = [SitePatches(r, device, rng) for s in TRAIN_SITES for r in regions_for_site(s)]
    val = [SitePatches(r, device, np.random.default_rng(1000 + i), 1500) for i, r in enumerate(r for s in VAL_SITES for r in regions_for_site(s))]
    if not train or not val:
        raise SystemExit("no tiles for the train/validation sites: fetch them first (python -I -m worldmodel.hazard.lola all)")
    print(f"device={device} train={[s.region for s in train]} val={[s.region for s in val]} (test {TEST_SITES} never loaded)", flush=True)

    kw = {"base": a.base, "T": a.T} | ({"depth": a.depth, "blocks": a.blocks, "heads": a.heads} if a.arch == "attn" else {})
    model = build_model(a.arch, **kw).to(device)
    print(f"architecture {a.arch}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    ab = cosine_alpha_bar(a.T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    last_p, best_p = model_path(a.name + "_last"), model_path(a.name)
    step, best = 0, float("inf")
    if a.resume and last_p.exists():
        state = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        step, best = state["info"]["step"], state["info"].get("best_val_mse", float("inf"))
        print(f"resumed at step {step}", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps - step, 1))
    log = run_dir("terrain") / "train_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def validate() -> float:
        """Mean squared error (normalised units) on HIDDEN real measurements, at fixed noise levels, fixed split."""
        model.eval()
        vrng = np.random.default_rng(7)
        tot, cnt = 0.0, 0.0
        g = torch.Generator(device="cpu").manual_seed(7)
        for _ in range(4):
            x0, C, Hm = make_batch(val, 128, vrng, device, fixed_p_c=0.5)
            for tt in (a.T // 4, a.T // 2, 3 * a.T // 4):
                t = torch.full((x0.shape[0],), tt, device=device, dtype=torch.long)
                noise = torch.randn(x0.shape, generator=g).to(device)
                pred = model(q_sample(x0, t, noise, ab), x0 * C, C, t)
                tot += float(((pred - x0) ** 2 * Hm).sum())
                cnt += float(Hm.sum())
        return tot / max(cnt, 1.0)

    t1, run = time.time(), 0.0
    bud = TimeBudget(a.max_minutes, a.steps)
    bud.begin(step)
    while step < a.steps:
        model.train()
        x0, C, Hm = make_batch(train, a.batch, rng, device)
        t = torch.randint(1, a.T + 1, (x0.shape[0],), device=device)
        pred = model(q_sample(x0, t, torch.randn_like(x0), ab), x0 * C, C, t)
        loss = ((pred - x0) ** 2 * Hm).sum() / Hm.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += float(loss.detach())
        step += 1
        if bud.update(step):
            a.steps = bud.total
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps - step, 1))
            print(f"{bud.summary()}", flush=True)
        if bud.expired():
            a.steps = step
        if step % a.eval_every == 0 or step == a.steps:
            v = validate()
            info = {"step": step, "train_mse": run / a.eval_every, "val_mse": v, "best_val_mse": min(best, v),
                    "trained_on": [s.region for s in train], "val_regions": [s.region for s in val],
                    "data": "real LOLA 5 m laser-measured pixels only (hide-and-check)", "patch": PATCH}
            if v < best:
                best = v
                save_checkpoint(best_p, model, info)
            save_checkpoint(last_p, model, info, extra={"optimizer": opt.state_dict()})
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**info, "seconds": time.time() - t1}) + "\n")
            print(f"step {step:5d}/{a.steps} train MSE {run / a.eval_every:.4f} | val MSE on hidden real pixels {v:.4f} | best {best:.4f} | {time.time() - t1:.0f}s", flush=True)
            run = 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
