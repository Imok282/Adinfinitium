"""Train the rover driver with PPO on real LOLA worlds (cross-platform: cuda / mps / cpu; native Windows PyTorch is enough).

    python -I -m worldmodel.rover.train                       # laptop smoke run
    python -I -m worldmodel.rover.train --envs 4096 --updates 600 --resume   # the 4090

Worlds = (site, date): terrain from NASA LOLA, real SPICE sunlight, NASA PSR ice targets. Train sites 01 04 06 07 20 (+ any extra windows), several months of 2027.
TEST sites (Site23, Site42) are never loaded here. Reward = the mission score in config (ice +50, return +25, battery death -50, small shaping).
Checkpoints: <ADI_DATA>/models/rover_driver.pt (best by training success) and rover_driver_last.pt (resume). Not in git.
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
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, regions_for_site
from worldmodel.rover.env import RoverEnv
from worldmodel.rover.policy import build_model, pick_device, save_checkpoint
from worldmodel.rover.world import build_world

TRAIN_DATES = ("2027-03-01", "2027-06-01", "2027-09-01", "2027-12-01")


def world_list(sites: list[str], dates: list[str], windows_per_site: int = 1) -> list[tuple[str, str]]:
    """(region, date) worlds: the first `windows_per_site` windows of each site (centre window first) x every date."""
    return [(region, d) for s in sites for region in regions_for_site(s)[:windows_per_site] for d in dates]


def evaluate_policy(model, region: str, date: str, cfg, device, n: int = 512, seed: int = 1234) -> dict:
    """Success statistics of one policy (greedy) in one world, plus the three physics baselines on the SAME starts."""
    w = build_world(region, date, cfg, device=device)
    res = {}
    for name, fn in (("naive", lambda e: e.planner_action(turn_back=False)), ("cautious", lambda e: e.planner_action(turn_back=True)),
                     ("cautious_wait_light", lambda e: e.planner_action(turn_back=True, wait_for_light=True)),
                     ("model", None)):
        if fn is None and model is None:
            continue
        env = RoverEnv(w, n, device, cfg, seed=seed)
        obs = env.reset()
        ice = ret = dead = 0
        for _ in range(env.rs.max_steps):
            with torch.no_grad():
                a = fn(env) if fn else model(obs)[0].argmax(1)
            obs, _, done, info = env.step(a)
            ice += int(info["reached_ice"].sum()); ret += int(info["returned"].sum()); dead += int(info["dead"].sum())
            if bool(done.all()):
                break
        res[name] = {"reached_ice": ice / n, "returned": ret / n, "died": dead / n, "mean_return": float(env.ep_return.mean())}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=64, help="steps per rollout")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--arch", choices=["conv", "former"], default="conv", help="conv = 0.5 M; former = vision transformer (~32 M at --d 512 --layers 10)")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dates", nargs="+", default=list(TRAIN_DATES))
    ap.add_argument("--sites", nargs="+", default=TRAIN_SITES)
    ap.add_argument("--windows-per-site", type=int, default=1, help="terrain windows per site (more = more varied worlds, more GPU memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default="rover_driver")
    a = ap.parse_args(argv)

    cfg = load_mission()
    assert not set(a.sites) & set(HOLDOUT_SITES), "a TEST site was requested for training"
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    specs = world_list(a.sites, a.dates, a.windows_per_site)
    if not specs:
        raise SystemExit("no tiles found for the training sites (python -I -m worldmodel.hazard.lola all)")
    envs = [RoverEnv(build_world(r, d, cfg, device=device), a.envs, device, cfg, seed=a.seed + i) for i, (r, d) in enumerate(specs)]
    labels = [r.split("lola5m_")[1] + "@" + d[5:7] for r, d in specs][:6]
    print(f"device={device} worlds={len(envs)} ({labels}...) test {HOLDOUT_SITES} never loaded; ready in {time.time() - t0:.0f}s", flush=True)

    kw = {"width": a.width, "hidden": a.hidden} if a.arch == "conv" else {"d": a.d, "layers": a.layers, "heads": a.heads}
    model = build_model(a.arch, **kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    last_p, best_p = model_path(a.name + "_last"), model_path(a.name)
    update, best = 0, -1e9
    if a.resume and last_p.exists():
        st = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["optimizer"])
        update, best = st["info"]["update"], st["info"].get("best_score", -1e9)
        print(f"resumed at update {update}", flush=True)
    log = run_dir("rover") / "train_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    obs_store = {}
    for e in envs:
        obs_store[id(e)] = e.reset()
    while update < a.updates:
        env = envs[rng.integers(len(envs))]
        obs = obs_store[id(env)]
        H, N = a.horizon, env.n
        buf = {k: [] for k in ("map", "vec", "act", "logp", "val", "rew", "done")}
        ice = ret = dead = eps = 0
        for _ in range(H):
            with torch.no_grad():
                logits, val = model(obs)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
            buf["map"].append(obs["map"].half()); buf["vec"].append(obs["vec"]); buf["act"].append(act); buf["logp"].append(dist.log_prob(act)); buf["val"].append(val)
            obs, rew, done, info = env.step(act)
            buf["rew"].append(rew); buf["done"].append(info["newly_done"].float())
            ice += int(info["reached_ice"].sum()); ret += int(info["returned"].sum()); dead += int(info["dead"].sum()); eps += int(info["newly_done"].sum())
            if bool(done.any()):
                obs = env.reset(done)
        obs_store[id(env)] = obs
        with torch.no_grad():
            last_v = model(obs)[1]
        rew, val, dn = torch.stack(buf["rew"]), torch.stack(buf["val"]), torch.stack(buf["done"])
        adv = torch.zeros_like(rew)
        g = torch.zeros(N, device=device)
        for t in reversed(range(H)):
            nv = last_v if t == H - 1 else val[t + 1]
            delta = rew[t] + a.gamma * nv * (1 - dn[t]) - val[t]
            g = delta + a.gamma * a.lam * (1 - dn[t]) * g
            adv[t] = g
        ret_t = adv + val
        stacked = torch.stack(buf.pop("map"))                                # [H, N, C, c, c] half precision (memory)
        flat_map = stacked.reshape(H * N, *stacked.shape[2:])
        del stacked
        flat = {"map": flat_map, "vec": torch.stack(buf["vec"]).reshape(H * N, -1),
                "act": torch.stack(buf["act"]).reshape(-1), "logp": torch.stack(buf["logp"]).reshape(-1), "adv": adv.reshape(-1), "ret": ret_t.reshape(-1)}
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        total = H * N
        mb = total // a.minibatches
        for _ in range(a.epochs):
            perm = torch.randperm(total).to(device)                     # on the CPU: randperm on Apple's MPS can return garbage indices
            for s in range(0, total - mb + 1, mb):
                ix = perm[s:s + mb]
                logits, v = model({"map": flat["map"][ix].float(), "vec": flat["vec"][ix]})
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(flat["act"][ix]) - flat["logp"][ix]).exp()
                pg = -torch.min(ratio * flat["adv"][ix], ratio.clamp(1 - a.clip, 1 + a.clip) * flat["adv"][ix]).mean()
                loss = pg + 0.5 * F.mse_loss(v, flat["ret"][ix]) - a.ent * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        update += 1
        score = (ice + ret) / max(eps, 1)
        info_d = {"update": update, "world": f"{env.world.region}@{env.world.date}", "episodes": eps, "reached_ice_per_episode": ice / max(eps, 1),
                  "returned_per_episode": ret / max(eps, 1), "died_per_episode": dead / max(eps, 1), "mean_reward": float(rew.mean()),
                  "best_score": max(best, score), "device": str(device)}
        if score > best and eps >= 50:
            best = score
            save_checkpoint(best_p, model, info_d)
        save_checkpoint(last_p, model, info_d, extra={"optimizer": opt.state_dict()})
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(info_d) + "\n")
        if update % 5 == 0 or update == 1:
            print(f"update {update:4d}/{a.updates} {info_d['world']:24s} ep {eps:5d} ice {info_d['reached_ice_per_episode']:.2f} return {info_d['returned_per_episode']:.2f} "
                  f"died {info_d['died_per_episode']:.2f} mean_r {info_d['mean_reward']:+.3f} | {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
