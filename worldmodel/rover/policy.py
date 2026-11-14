"""Rover driving network (PyTorch): map crop + vector state -> action logits and a value. Trained with PPO in train.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from worldmodel.nn_blocks import TransformerStack, sincos_1d
from worldmodel.rover.env import MAP_CHANNELS, N_ACTIONS, VEC_DIM


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DriverNet(nn.Module):
    """Conv encoder over the egocentric map (64 x 64) + MLP over the vector (battery, sunlight forecast, goal, lander, link) -> 9 actions + value."""

    def __init__(self, width: int = 32, hidden: int = 256, crop: int = 64):
        super().__init__()
        self.width, self.hidden, self.crop = width, hidden, crop
        w = width
        self.conv = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, w, 3, 2, 1), nn.GroupNorm(4, w), nn.SiLU(),
            nn.Conv2d(w, 2 * w, 3, 2, 1), nn.GroupNorm(4, 2 * w), nn.SiLU(),
            nn.Conv2d(2 * w, 2 * w, 3, 2, 1), nn.GroupNorm(4, 2 * w), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.map_fc = nn.Sequential(nn.Linear(2 * w * 16, hidden), nn.SiLU())
        self.vec_fc = nn.Sequential(nn.Linear(VEC_DIM, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.trunk = nn.Sequential(nn.Linear(hidden + 128, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def arch(self) -> dict:
        return {"kind": "conv", "width": self.width, "hidden": self.hidden, "crop": self.crop}

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([self.map_fc(self.conv(obs["map"])), self.vec_fc(obs["vec"])], dim=1))
        return self.pi(h), self.v(h)[:, 0]


class DriverFormer(nn.Module):
    """Vision-transformer rover driver: a conv stem turns the egocentric map (terrain, slope, hazard, shadow, sunlight, Earth) into 8 x 8 patch tokens,
    the battery / sunlight forecast / goal / lander / link vector becomes one more token, and a transformer (self-attention across all tokens) plans
    the next move. A learned CLS token feeds the policy and value heads. d=512, layers=10 is about 32 M parameters."""

    def __init__(self, d: int = 512, layers: int = 10, heads: int = 8, crop: int = 64, stem: int = 64):
        super().__init__()
        self.d, self.layers, self.heads, self.crop, self.stem_ch = d, layers, heads, crop, stem
        self.stem = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, stem, 3, 2, 1), nn.GroupNorm(8, stem), nn.SiLU(),
            nn.Conv2d(stem, 2 * stem, 3, 2, 1), nn.GroupNorm(8, 2 * stem), nn.SiLU(),
            nn.Conv2d(2 * stem, d, 2, 2))                                   # 64 -> 32 -> 16 -> 8 : 64 patch tokens
        side = crop // 8
        self.register_buffer("pe", sincos_1d(side * side, d), persistent=False)
        self.vec = nn.Sequential(nn.Linear(VEC_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder = TransformerStack(d, layers, heads)
        self.pi = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, N_ACTIONS))
        self.v = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.zeros_(self.pi[-1].bias)

    def arch(self) -> dict:
        return {"kind": "former", "d": self.d, "layers": self.layers, "heads": self.heads, "crop": self.crop, "stem": self.stem_ch}

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.stem(obs["map"]).flatten(2).transpose(1, 2) + self.pe        # [N, 64, d]
        seq = torch.cat([self.cls.expand(tok.shape[0], -1, -1), self.vec(obs["vec"])[:, None], tok], dim=1)
        h = self.encoder(seq)[:, 0]
        return self.pi(h), self.v(h)[:, 0]


def build_model(kind: str = "conv", **kw) -> nn.Module:
    """Rover driver by kind: 'conv' (DriverNet, 0.5 M) or 'former' (DriverFormer vision transformer, ~32 M at d=512, layers=10)."""
    return {"conv": DriverNet, "former": DriverFormer}[kind](**kw)


def checkpoint_id(model: nn.Module) -> str:
    h = hashlib.sha1()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def save_checkpoint(path: str | Path, model: nn.Module, info: dict, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "arch": model.arch(),
                "info": {**info, "checkpoint_id": checkpoint_id(model)}, **(extra or {})}, p)
    return p


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    arch = dict(state["arch"])
    model = build_model(arch.pop("kind", "conv"), **arch)             # old checkpoints have no 'kind': the conv driver
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model.to(device or torch.device("cpu")), state["info"]
