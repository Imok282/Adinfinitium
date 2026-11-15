"""Self-learning after training has ended: (1) fit the world model to the rover's own recorded steps, (2) improve the correction to the frozen driver
inside the model (imagination), (3) the paired test on real episodes that decides whether a candidate is accepted.

Nothing here touches the simulator: train_world_model reads recorded steps, dream_improve uses no real steps at all.
"""

from __future__ import annotations

import copy
import time

import torch
import torch.nn.functional as F

from worldmodel.dream.model import DreamNet, symexp, symlog
from worldmodel.dream.replay import Chunk, Replay


def _to(batch: dict, device: torch.device) -> dict:
    return {k: (v.float() if k in ("map", "prior") else v).to(device) for k, v in batch.items()}


def wm_loss(net: DreamNet, b: dict, teacher_delta: torch.Tensor | None = None, clone_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Multi-step latent prediction loss on a batch of sequences. b: map [B,L+1,..], vec [B,L+1,18], act/rew/done/valid [B,L], prior [B,L,9].

    The clone term teaches prior(z) the frozen driver's recorded choices. `teacher_delta` [B,9] (the accepted driver's correction on the same
    observations) holds the correction steady while the encoder under it moves.
    """
    bsz, ln = b["act"].shape
    z = net.encode(b["map"][:, 0], b["vec"][:, 0])
    with torch.no_grad():
        zt = net.encode(b["map"][:, 1:].reshape(bsz * ln, *b["map"].shape[2:]), b["vec"][:, 1:].reshape(bsz * ln, -1), target=True).reshape(bsz, ln, -1)
    mask = torch.ones(bsz, device=z.device)
    tot = {k: z.new_zeros(()) for k in ("consistency", "reward", "cont", "vec", "clone")}
    v0 = b["valid"][:, 0].float()
    per = F.kl_div(F.log_softmax(net.prior(z), 1), F.softmax(b["prior"][:, 0], 1), reduction="none").sum(1)
    if teacher_delta is not None:
        per = per + ((net.delta(z) - teacher_delta) ** 2).mean(1)
    tot["clone"] = (per * v0).sum() / v0.sum().clamp(min=1)
    wsum = 0.0
    for k in range(ln):
        mask = mask * b["valid"][:, k].float()
        out = net.step(z, b["act"][:, k])
        alive = mask * (1 - b["done"][:, k])                      # the next observation belongs to the same episode
        w = 0.5 ** k
        den, den_a = mask.sum().clamp(min=1), alive.sum().clamp(min=1)
        tot["reward"] = tot["reward"] + w * ((out["reward"] - symlog(b["rew"][:, k])) ** 2 * mask).sum() / den
        tot["cont"] = tot["cont"] + w * (F.binary_cross_entropy_with_logits(out["cont"], 1 - b["done"][:, k], reduction="none") * mask).sum() / den
        tot["consistency"] = tot["consistency"] + w * (((out["z"] - zt[:, k]) ** 2).mean(1) * alive).sum() / den_a
        tot["vec"] = tot["vec"] + w * (((out["vec"] - b["vec"][:, k + 1]) ** 2).mean(1) * alive).sum() / den_a
        z = out["z"]
        mask = alive
        wsum += w
    loss = (2.0 * tot["consistency"] + tot["reward"] + tot["cont"] + 5.0 * tot["vec"]) / wsum + clone_weight * tot["clone"]
    return loss, {k: float(v.detach()) / (1.0 if k == "clone" else wsum) for k, v in tot.items()}


def train_world_model(net: DreamNet, replay: Replay, steps: int, device: torch.device, gen: torch.Generator, batch: int = 256, length: int = 4,
                      lr: float = 3e-4, teacher: DreamNet | None = None, prefer: str | None = None, until: float | None = None) -> dict:
    """Fit the world model and the stand-in for the frozen driver. `teacher` (the accepted driver) keeps the learned correction steady."""
    params = net.wm_parameters() + list(net.prior.parameters()) + (list(net.delta.parameters()) if teacher is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    net.train()
    first = last = None
    done = 0
    for _ in range(steps):
        if until is not None and time.time() > until:
            break
        b = _to(replay.sample(batch, length, gen, prefer), device)
        td = None
        if teacher is not None:
            with torch.no_grad():
                td = teacher.delta(teacher.encode(b["map"][:, 0], b["vec"][:, 0]))
        loss, parts = wm_loss(net, b, td)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        net.update_targets(0.01)
        done += 1
        last = {"loss": float(loss.detach()), **parts}
        first = first or last
    net.eval()
    return {"steps_done": done, "steps_requested": steps, "first": first, "last": last}


@torch.no_grad()
def model_error(net: DreamNet, chunk: Chunk, device: torch.device, battery_wh: float, max_rows: int = 4096) -> dict:
    """One-step prediction error of the world model on recorded steps it may not have seen: how wrong is the dream about this world?"""
    idx = chunk.valid.reshape(-1).nonzero()[:, 0]
    idx = idx[torch.linspace(0, len(idx) - 1, min(max_rows, len(idx))).long()]
    n = chunk.act.shape[1]
    t, c = torch.div(idx, n, rounding_mode="floor"), idx % n
    z = net.encode(chunk.map[t, c].float().to(device), chunk.vec[t, c].to(device))
    out = net.step(z, chunk.act[t, c].to(device))
    done = chunk.done[t, c].to(device)
    alive = 1 - done
    nxt = chunk.vec[t + 1, c].to(device)
    den = alive.sum().clamp(min=1)
    return {"rows": int(len(idx)),
            "battery_abs_err_wh": float(((out["vec"][:, 0] - nxt[:, 0]).abs() * alive).sum() / den * battery_wh),
            "vec_mse": float((((out["vec"] - nxt) ** 2).mean(1) * alive).sum() / den),
            "reward_abs_err": float((symexp(out["reward"]) - chunk.rew[t, c].to(device)).abs().mean()),
            "episode_end_accuracy": float(((out["cont"] < 0).float() == done).float().mean())}


def lambda_returns(rew: torch.Tensor, disc: torch.Tensor, val: torch.Tensor, lam: float) -> torch.Tensor:
    """rew, disc [H,B] (disc = gamma * P(continue) for that step), val [H+1,B] -> lambda-returns [H,B], bootstrapped from val[H]."""
    out = torch.zeros_like(rew)
    nxt = val[-1]
    for k in reversed(range(rew.shape[0])):
        nxt = rew[k] + disc[k] * ((1 - lam) * val[k + 1] + lam * nxt)
        out[k] = nxt
    return out


def dream_improve(net: DreamNet, replay: Replay, steps: int, device: torch.device, gen: torch.Generator, batch: int = 256, horizon: int = 15,
                  gamma: float = 0.995, lam: float = 0.95, ent: float = 0.003, lr: float = 1e-4, anchor_weight: float = 0.3, critic_warmup: int = 200,
                  prefer: str | None = None, until: float | None = None) -> dict:
    """Actor-critic practice inside the world model. Only the correction (delta) and the critic change; no simulator step is taken.

    anchor_weight ties the driver to what it was before dreaming (KL): a driver that wanders far from recorded behaviour is exploiting model errors.
    The first `critic_warmup` steps train the critic only, so the correction is not pushed by an untrained value.
    """
    anchor = copy.deepcopy(net.delta).requires_grad_(False)
    opt_a = torch.optim.Adam(net.delta.parameters(), lr=lr)
    opt_c = torch.optim.Adam(net.critic.parameters(), lr=3 * lr)
    first = last = None
    done = 0
    for i in range(steps):
        if until is not None and time.time() > until:
            break
        b = _to(replay.sample(batch, 1, gen, prefer), device)
        with torch.no_grad():
            z = net.encode(b["map"][:, 0], b["vec"][:, 0])
        zs, logps, ents, kls, rews, discs = [z], [], [], [], [], []
        for _ in range(horizon):
            with torch.no_grad():
                base = net.prior(z)
            dist = torch.distributions.Categorical(logits=base + net.delta(z))
            a = dist.sample()
            logps.append(dist.log_prob(a)); ents.append(dist.entropy())
            kls.append(torch.distributions.kl_divergence(dist, torch.distributions.Categorical(logits=base + anchor(z))))
            with torch.no_grad():
                out = net.step(z, a)
                z = out["z"]
                rews.append(symexp(out["reward"])); discs.append(gamma * torch.sigmoid(out["cont"]))
            zs.append(z)
        zs_t = torch.stack(zs)
        with torch.no_grad():
            val = symexp(net.target["critic"](zs_t)[..., 0])
            rew, disc = torch.stack(rews), torch.stack(discs)
            ret = lambda_returns(rew, disc, val, lam)
            weight = torch.cat([torch.ones_like(disc[:1]), disc[:-1]]).cumprod(0)
            adv = (ret - val[:-1]) / ret.std().clamp(min=1.0)
        critic_loss = (weight * (net.critic(zs_t[:-1])[..., 0] - symlog(ret)) ** 2).mean()
        opt_c.zero_grad()
        critic_loss.backward()
        opt_c.step()
        net.update_targets(0.02, critic=True)
        kl = torch.stack(kls)
        if i >= critic_warmup:
            actor_loss = -(weight * (torch.stack(logps) * adv + ent * torch.stack(ents) - anchor_weight * kl)).mean()
            opt_a.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.delta.parameters(), 1.0)
            opt_a.step()
        done += 1
        last = {"dream_return": float(ret[0].mean()), "critic_loss": float(critic_loss.detach()), "kl_from_start": float(kl.detach().mean())}
        first = first or last
    return {"steps_done": done, "steps_requested": steps, "first": first, "last": last}


def paired_gain(incumbent: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    """Mean gain in episode return over the current driver on the SAME starts, and its standard error."""
    d = candidate - incumbent
    return float(d.mean()), (float(d.std(unbiased=True) / len(d) ** 0.5) if len(d) > 1 else float("inf"))


def clear_gain(gain: float, se: float, margin_se: float = 2.0) -> bool:
    """A candidate is accepted only if its paired gain is positive and larger than `margin_se` standard errors; otherwise the current driver stays."""
    return gain > 0 and gain > margin_se * se
