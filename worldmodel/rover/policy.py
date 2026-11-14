"""Rover driving network (PyTorch): map crop + vector state -> action logits and a value. Trained with PPO in train.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

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

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([self.map_fc(self.conv(obs["map"])), self.vec_fc(obs["vec"])], dim=1))
        return self.pi(h), self.v(h)[:, 0]


def checkpoint_id(model: nn.Module) -> str:
    h = hashlib.sha1()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def save_checkpoint(path: str | Path, model: DriverNet, info: dict, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "arch": {"width": model.width, "hidden": model.hidden, "crop": model.crop},
                "info": {**info, "checkpoint_id": checkpoint_id(model)}, **(extra or {})}, p)
    return p


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[DriverNet, dict]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = DriverNet(**state["arch"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model.to(device or torch.device("cpu")), state["info"]
