"""Small latent world model + the learned correction to the frozen driver (PyTorch; cuda / mps / cpu). About 0.7 M parameters.

encode(map 6x16x16, vec 18) -> z.  step(z, action) -> next z, reward, P(episode continues), next vec.
prior(z)  -> what the frozen driver would do here (learned from its recorded choices; stands in for it inside imagination, where there is no map to show it)
delta(z)  -> the correction the rover has learned since training ended. It starts at exactly zero.
In the real world the rover acts on  frozen_driver(observation) + delta(z):  before any self-learning that IS the frozen driver.
The dynamics are trained on the rover's own recorded steps (learn.train_world_model); delta and the critic practise inside the model (learn.dream_improve).
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from worldmodel.rover.env import MAP_CHANNELS, N_ACTIONS, VEC_DIM

MAP_POOL = 16
NAME_PREFIX = "rover_dream_"


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def pool_map(m: torch.Tensor) -> torch.Tensor:
    """[N,6,crop,crop] -> [N,6,16,16] by block averaging (the model sees the whole 320 m crop at 20 m cells)."""
    k = m.shape[-1] // MAP_POOL
    return F.avg_pool2d(m.float(), k) if k > 1 else m.float()


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class DreamNet(nn.Module):
    def __init__(self, latent: int = 128, hidden: int = 256):
        super().__init__()
        self.latent, self.hidden = latent, hidden
        self.enc_map = nn.Sequential(nn.Conv2d(MAP_CHANNELS, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(), nn.Flatten(),
                                     nn.Linear(64 * (MAP_POOL // 4) ** 2, hidden), nn.SiLU())
        self.enc_vec = nn.Sequential(nn.Linear(VEC_DIM, 128), nn.SiLU())
        self.enc_out = nn.Sequential(nn.Linear(hidden + 128, hidden), nn.SiLU(), nn.Linear(hidden, latent), nn.LayerNorm(latent))
        za = latent + N_ACTIONS
        self.dyn = _mlp(za, hidden, latent)
        self.dyn_norm = nn.LayerNorm(latent)
        self.reward = _mlp(za, hidden, 1)
        self.cont = _mlp(za, hidden, 1)
        self.vec_next = _mlp(za, hidden, VEC_DIM)
        self.prior = _mlp(latent, hidden, N_ACTIONS)
        self.delta = _mlp(latent, hidden, N_ACTIONS)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.critic = _mlp(latent, hidden, 1)
        # Slow copies give stable training targets; they are never trained by gradient.
        self.target = nn.ModuleDict({"enc_map": copy.deepcopy(self.enc_map), "enc_vec": copy.deepcopy(self.enc_vec),
                                     "enc_out": copy.deepcopy(self.enc_out), "critic": copy.deepcopy(self.critic)})
        for p in self.target.parameters():
            p.requires_grad_(False)

    def arch(self) -> dict:
        return {"latent": self.latent, "hidden": self.hidden}

    def wm_parameters(self) -> list[nn.Parameter]:
        mods = (self.enc_map, self.enc_vec, self.enc_out, self.dyn, self.dyn_norm, self.reward, self.cont, self.vec_next)
        return [p for m in mods for p in m.parameters()]

    def encode(self, map16: torch.Tensor, vec: torch.Tensor, target: bool = False) -> torch.Tensor:
        em, ev, eo = (self.target["enc_map"], self.target["enc_vec"], self.target["enc_out"]) if target else (self.enc_map, self.enc_vec, self.enc_out)
        return eo(torch.cat([em(map16), ev(vec)], dim=1))

    def step(self, z: torch.Tensor, action: torch.Tensor) -> dict:
        """One imagined step. reward is in symlog units, cont is a logit."""
        za = torch.cat([z, F.one_hot(action, N_ACTIONS).float()], dim=1)
        return {"z": self.dyn_norm(z + self.dyn(za)), "reward": self.reward(za)[:, 0], "cont": self.cont(za)[:, 0], "vec": self.vec_next(za)}

    def correction(self, obs: dict) -> torch.Tensor:
        """The learned correction for a raw environment observation (added to the frozen driver's logits)."""
        return self.delta(self.encode(pool_map(obs["map"]), obs["vec"]))

    @torch.no_grad()
    def update_targets(self, tau: float, critic: bool = False) -> None:
        names = ("critic",) if critic else ("enc_map", "enc_vec", "enc_out")
        for n in names:
            for pt, p in zip(self.target[n].parameters(), getattr(self, n).parameters()):
                pt.lerp_(p, tau)


def dream_model_path(models_dir: Path, name: str) -> Path:
    """Dream checkpoints can only ever be called rover_dream_*: they can't overwrite a model of the frozen run."""
    if not name.startswith(NAME_PREFIX) or "/" in name or "\\" in name:
        raise ValueError(f"dream checkpoints must be named {NAME_PREFIX}*: got {name!r}")
    return Path(models_dir) / f"{name}.pt"


def save_dream(models_dir: Path, name: str, net: DreamNet, info: dict) -> Path:
    p = dream_model_path(models_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "arch": net.arch(),
                "info": {**info, "experimental_addendum": True}}, tmp)
    tmp.replace(p)
    return p


def load_dream(path: Path, device: torch.device | None = None) -> tuple[DreamNet, dict]:
    st = torch.load(Path(path), map_location="cpu", weights_only=False)
    net = DreamNet(**st["arch"])
    net.load_state_dict(st["state_dict"])
    return net.to(device or torch.device("cpu")).eval(), st["info"]
