"""Train the lander pilot with PPO on real LOLA terrain (cross-platform: cuda / mps / cpu). The pilot is a RESIDUAL on classic ZEM/ZEV guidance, so it
starts exactly at the physics baseline and can only be better where it learned something. Whole-tile splits: train Site01 04 06 07 20 (+ Haworth, Shoemaker,
DM2 when fetched), validation Site11, TEST Site23 + Site42 never loaded.

    python -m worldmodel.lander.train --arch former --d 384 --layers 8 --envs 512 --updates 600 --max-minutes 180      # the 4090 / big run
    python -m worldmodel.lander.train --arch mlp --envs 64 --updates 3                                                 # CPU smoke test

Checkpoints: <ADI_DATA>/models/<name>.pt (best by VALIDATION success vs the classic baseline) and <name>_last.pt (resume). Not in git.
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
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, VAL_SITES
from worldmodel.lander.env import LanderBatch, make_params
from worldmodel.lander.fly import fly, summarize
from worldmodel.lander.guidance import classic_action, guidance_params
from worldmodel.lander.pilot import build_model, pick_device, pilot_features, residual_action, save_checkpoint
from worldmodel.lander.scenes import load_scenes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96, help="control steps per rollout")
    ap.add_argument("--repeat", type=int, default=5, help="sim steps (0.1 s) per pilot decision")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--mb-size", type=int, default=0, help="samples per gradient step (overrides --minibatches; keeps transformer memory bounded)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--arch", choices=["mlp", "former"], default="mlp")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--sites", nargs="+", default=TRAIN_SITES)
    ap.add_argument("--val-sites", nargs="+", default=VAL_SITES)
    ap.add_argument("--windows-per-site", type=int, default=1)
    ap.add_argument("--belief", choices=["backup", "model"], default="backup", help="hazard map the camera image shows (the truth for touchdown is the official slope rule)")
    ap.add_argument("--val-n", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--name", default="lander_pilot")
    a = ap.parse_args(argv)

    cfg = load_mission()
    assert not set(a.sites) & set(HOLDOUT_SITES) and not set(a.val_sites) & set(HOLDOUT_SITES), "a TEST site was requested"
    device = pick_device(a.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True                      # RTX 4090: TF32 tensor cores for the transformer matmuls
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    import contextlib

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext   # half memory, faster on RTX 40/50
    t0 = time.time()
    scenes = load_scenes(a.sites, cfg, device, a.belief, a.windows_per_site)
    val_scenes = load_scenes(a.val_sites, cfg, device, a.belief, 1)
    if not scenes or not val_scenes:
        raise SystemExit("no tiles for the training / validation sites (python -m worldmodel.hazard.lola all)")
    print(f"device={device} train scenes={[s.region.split('lola5m_')[1] for s in scenes]} val={[s.region.split('lola5m_')[1] for s in val_scenes]} "
          f"test {HOLDOUT_SITES} never loaded; ready in {time.time() - t0:.0f}s", flush=True)

    params = make_params(cfg)
    gp = guidance_params(params, cfg.scenarios.terminal.hover_gate_altitude_m, cfg.scenarios.terminal.final_descent_speed_m_s)
    kw = {"hidden": a.hidden} if a.arch == "mlp" else {"d": a.d, "layers": a.layers, "heads": a.heads}
    model = build_model(a.arch, **kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    last_p, best_p = model_path(a.name + "_last"), model_path(a.name)
    update, best_val = 0, -1.0
    if a.resume and last_p.exists():
        st = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["optimizer"])
        update, best_val = st["info"]["update"], st["info"].get("best_val_success", -1.0)
        print(f"resumed at update {update}", flush=True)
    log = run_dir("lander") / "train_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_par:,}", flush=True)

    def validate() -> dict:
        res = {}
        for s in val_scenes:
            res[s.region] = {"classic": summarize(fly(s, params, gp, None, a.val_n, 777, a.repeat, device=device)),
                             "pilot": summarize(fly(s, params, gp, model.eval(), a.val_n, 777, a.repeat, device=device))}
        model.train()
        return res

    first = validate()
    base = float(np.mean([v["classic"]["success_rate"] for v in first.values()]))
    print(f"start: validation success pilot {np.mean([v['pilot']['success_rate'] for v in first.values()]):.3f} | classic baseline {base:.3f} (untrained pilot == baseline)", flush=True)

    envs = {}
    states = {}
    def get_env(scene):
        key = scene.region
        if key not in envs:
            tg = scene.sample_targets(a.envs, rng, device)
            envs[key] = LanderBatch(params, scene.terrain, a.envs, device, seed=a.seed + len(envs), targets=tg)
            states[key] = envs[key].observe()
        return envs[key]

    bud = TimeBudget(a.max_minutes, a.updates, warmup_s=90.0)
    bud.begin(update)
    model.train()
    while update < a.updates:
        scene = scenes[rng.integers(len(scenes))]
        env = get_env(scene)
        obs, H, N = states[scene.region], a.horizon, a.envs
        buf = {k: [] for k in ("state", "img", "r", "logp", "val", "rew", "done")}
        eps = ok = 0
        for _ in range(H):
            classic = classic_action(gp, obs)
            st = pilot_features(obs, classic)
            img = env.crop(32)
            with torch.no_grad(), amp():
                mu, val = model(st, img)
            mu, val = mu.float(), val.float()
            with torch.no_grad():
                dist = torch.distributions.Normal(mu, model.log_std.exp())
                r = dist.sample()
                logp = dist.log_prob(r).sum(1)
            rew = torch.zeros(N, device=device)
            newly = torch.zeros(N, dtype=torch.bool, device=device)
            for _k in range(a.repeat):
                if _k:
                    classic = classic_action(gp, obs)
                obs, rw, jd = env.step(residual_action(classic, r))
                rew += rw
                newly |= jd
            buf["state"].append(st); buf["img"].append(img.half()); buf["r"].append(r); buf["logp"].append(logp); buf["val"].append(val)
            buf["rew"].append(rew); buf["done"].append(newly.float())
            eps += int(newly.sum()); ok += int((newly & env.metrics()["success"]).sum())
            if bool(env.done.any()):
                fin = env.done
                obs = env.reset(fin, scene.sample_targets(int(fin.sum()), rng, device))
        states[scene.region] = obs
        with torch.no_grad(), amp():
            last_v = model(pilot_features(obs, classic_action(gp, obs)), env.crop(32))[1].float()
        rew, val, dn = torch.stack(buf["rew"]), torch.stack(buf["val"]), torch.stack(buf["done"])
        adv = torch.zeros_like(rew)
        g = torch.zeros(N, device=device)
        for t in reversed(range(H)):
            nv = last_v if t == H - 1 else val[t + 1]
            delta = rew[t] + a.gamma * nv * (1 - dn[t]) - val[t]
            g = delta + a.gamma * a.lam * (1 - dn[t]) * g
            adv[t] = g
        stacked = torch.stack(buf.pop("img"))
        flat = {"img": stacked.reshape(H * N, *stacked.shape[2:]), "state": torch.stack(buf["state"]).reshape(H * N, -1), "r": torch.stack(buf["r"]).reshape(H * N, 3),
                "logp": torch.stack(buf["logp"]).reshape(-1), "adv": adv.reshape(-1), "ret": (adv + val).reshape(-1)}
        del stacked
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        total = H * N
        mb = min(a.mb_size, total) if a.mb_size else total // a.minibatches
        for _ in range(a.epochs):
            perm = torch.randperm(total).to(device)
            for s in range(0, total - mb + 1, mb):
                ix = perm[s:s + mb]
                with amp():
                    mu, v = model(flat["state"][ix], flat["img"][ix].float())
                mu, v = mu.float(), v.float()
                dist = torch.distributions.Normal(mu, model.log_std.exp())
                ratio = (dist.log_prob(flat["r"][ix]).sum(1) - flat["logp"][ix]).exp()
                pg = -torch.min(ratio * flat["adv"][ix], ratio.clamp(1 - a.clip, 1 + a.clip) * flat["adv"][ix]).mean()
                loss = pg + 0.5 * F.mse_loss(v, flat["ret"][ix])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        update += 1
        if bud.update(update):
            a.updates = bud.total
            print(bud.summary(), flush=True)
        if bud.expired():
            a.updates = update
        info = {"update": update, "scene": scene.region, "episodes": eps, "train_success": ok / max(eps, 1), "mean_reward": float(rew.mean()), "best_val_success": best_val,
                "parameters": n_par, "device": str(device), "trained_on": [s.region for s in scenes], "val_regions": [s.region for s in val_scenes]}
        if update % a.eval_every == 0 or update >= a.updates:
            v = validate()
            pil = float(np.mean([x["pilot"]["success_rate"] for x in v.values()]))
            info["val"] = {"pilot_success": pil, "classic_success": base}
            if pil > best_val:
                best_val = pil
                info["best_val_success"] = pil
                save_checkpoint(best_p, model, info)
            print(f"update {update:4d}/{a.updates} val success pilot {pil:.3f} vs classic {base:.3f} | train success {info['train_success']:.2f} | {time.time() - t0:.0f}s", flush=True)
        save_checkpoint(last_p, model, info, extra={"optimizer": opt.state_dict()})
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(info) + "\n")
    if not best_p.exists():
        save_checkpoint(best_p, model, {"update": update, "note": "no validation improvement over the start; equals the classic baseline at init"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
