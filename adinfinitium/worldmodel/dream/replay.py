"""Recorded rover experience for the world model: chunks of [T, N] steps kept on the CPU (maps pooled to 16 x 16, half precision)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from worldmodel.dream.model import pool_map
from worldmodel.rover.env import N_ACTIONS, RoverEnv


@dataclass
class Chunk:
    map: torch.Tensor      # [T+1,N,6,16,16] float16
    vec: torch.Tensor      # [T+1,N,18]
    act: torch.Tensor      # [T,N] long
    rew: torch.Tensor      # [T,N]
    done: torch.Tensor     # [T,N] 1.0 where the episode ended on this step (the next observation is then not its successor)
    valid: torch.Tensor    # [T,N] bool: the rover was still running when it took this step
    prior: torch.Tensor    # [T,N,9] float16: the frozen driver's action logits at this step (zeros if none was recorded)
    tag: str = ""

    @property
    def steps(self) -> int:
        return int(self.valid.sum())


def clone_env(env: RoverEnv, n: int, seed: int) -> RoverEnv:
    """Another batch of rovers in the same world, sharing its maps and cost fields (building those is the slow part)."""
    e = copy.copy(env)
    e.n = n
    reseed(e, seed)
    return e


def reseed(env: RoverEnv, seed: int) -> dict:
    env.gen = torch.Generator(device="cpu").manual_seed(seed)
    return env.reset()


def rollout(env: RoverEnv, act_fn, steps: int, reset_done: bool, record: bool = True, tag: str = "") -> tuple[Chunk | None, dict]:
    """Drive `env` from its current reset state. reset_done=True: finished rovers restart (data collection). False: one episode each (scoring)."""
    obs = env.observe()
    maps, vecs, acts, rews, dones, valids, priors = [], [], [], [], [], [], []
    n = env.n
    ice = ret = dead = eps = blocked = alive_steps = 0
    ret_sum = haz_sum = 0.0
    for _ in range(steps):
        with torch.no_grad():
            a = act_fn(env, obs)
        a, prior = a if isinstance(a, tuple) else (a, None)       # a driver may also hand back the frozen driver's logits for the record
        alive = ~env.done
        if record:
            maps.append(pool_map(obs["map"]).half().cpu()); vecs.append(obs["vec"].cpu()); acts.append(a.cpu()); valids.append(alive.cpu())
            priors.append(torch.zeros(n, N_ACTIONS, dtype=torch.float16) if prior is None else prior.half().cpu())
        obs, rew, done, info = env.step(a)
        nd = info["newly_done"]
        ice += int(info["reached_ice"].sum()); ret += int(info["returned"].sum()); dead += int(info["dead"].sum()); eps += int(nd.sum())
        blocked += int(info["blocked"].sum()); alive_steps += int(alive.sum())
        haz_sum += float((env._sample_full(env.hazard, env.pos)[:, 0] * alive).sum())
        ret_sum += float(env.ep_return[nd].sum())
        if record:
            rews.append(rew.cpu()); dones.append(nd.float().cpu())
        if reset_done:
            if bool(done.any()):
                obs = env.reset(done)
        elif bool(done.all()):
            break
    returns = env.ep_return.detach().cpu().clone()
    if reset_done:
        stats = {"episodes": eps, "reached_ice": ice / max(eps, 1), "returned": ret / max(eps, 1), "died": dead / max(eps, 1),
                 "mean_return": ret_sum / max(eps, 1)}
    else:
        stats = {"episodes": n, "reached_ice": ice / n, "returned": ret / n, "died": dead / n, "mean_return": float(returns.mean())}
    stats.update(blocked_moves_per_episode=blocked / max(stats["episodes"], 1), mean_hazard_under_rover=haz_sum / max(alive_steps, 1), returns=returns)
    chunk = None
    if record:
        maps.append(pool_map(obs["map"]).half().cpu()); vecs.append(obs["vec"].cpu())
        chunk = Chunk(torch.stack(maps), torch.stack(vecs), torch.stack(acts), torch.stack(rews), torch.stack(dones), torch.stack(valids),
                      torch.stack(priors), tag)
        stats["real_steps"] = chunk.steps
    return chunk, stats


class Replay:
    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, chunk: Chunk) -> None:
        if chunk.steps:
            self.chunks.append(chunk)

    def steps(self, tag: str | None = None) -> int:
        return sum(c.steps for c in self.chunks if tag is None or c.tag == tag)

    def sample(self, batch: int, length: int, gen: torch.Generator, prefer: str | None = None, prefer_frac: float = 0.5) -> dict:
        """`batch` sequences of `length` steps (+1 observation). With `prefer`, that share of the batch comes from chunks with that tag (the changed world)."""
        w = torch.tensor([float(c.steps) for c in self.chunks])
        if prefer is not None:
            is_p = torch.tensor([c.tag == prefer for c in self.chunks])
            if bool(is_p.any()) and not bool(is_p.all()):
                w = torch.where(is_p, prefer_frac * w / w[is_p].sum(), (1 - prefer_frac) * w / w[~is_p].sum())
        counts = torch.bincount(torch.multinomial(w / w.sum(), batch, replacement=True, generator=gen), minlength=len(self.chunks))
        out = {k: [] for k in ("map", "vec", "act", "rew", "done", "valid", "prior")}
        for c, k in zip(self.chunks, counts.tolist()):
            if not k:
                continue
            t_max, n = c.act.shape
            ln = min(length, t_max)
            flat = c.valid.reshape(-1).nonzero()[:, 0]
            pick = flat[torch.randint(0, len(flat), (k,), generator=gen)]
            t0 = torch.div(pick, n, rounding_mode="floor").clamp(max=t_max - ln)
            col = pick % n
            ts = t0[:, None] + torch.arange(ln + 1)[None]
            pad = length - ln
            for key, arr, sl in (("map", c.map, ts), ("vec", c.vec, ts), ("act", c.act, ts[:, :-1]), ("rew", c.rew, ts[:, :-1]), ("done", c.done, ts[:, :-1]),
                                 ("valid", c.valid, ts[:, :-1]), ("prior", c.prior, ts[:, :-1])):
                x = arr[sl, col[:, None]]
                if pad:           # chunk shorter than the sequence: repeat the last step, marked invalid
                    x = torch.cat([x, x[:, -1:].expand(-1, pad, *x.shape[2:])], dim=1)
                    if key == "valid":
                        x[:, ln:] = False
                out[key].append(x)
        return {k: torch.cat(v) for k, v in out.items()}
