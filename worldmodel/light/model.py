"""Skyline network (needs torch): learns a CORRECTION on top of the physics ray-cast of the coarse terrain.

Input per cell: heights along every compass ray of the coarse grid, shape [2, range, azimuth] (relative height, elevation angle).
prediction = baseline + delta, where baseline = exact ray-cast of the coarse grid (the physics backup) and delta is learned.
The last layer starts at zero, so an UNTRAINED model equals the physics backup exactly. Circular padding along azimuth: the
network is shift-equivariant in compass direction, like the physics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from contracts.mission import MissionConfig
from contracts.tiles import Tile
from worldmodel.light.data import coarsen, fine_to_coarse
from worldmodel.light.teacher import height_at_cells_torch, sample_rays_torch

REL_H_SCALE_M = 100.0     # model scalings (not mission numbers)
ANGLE_SCALE_DEG = 10.0


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ray_inputs(coarse_t: torch.Tensor, cells_coarse: torch.Tensor, mpp_coarse: float, k_steps: int, n_az: int,
               moon_radius_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(x [N,2,K,n_az], baseline [N,n_az]) from a coarse grid [1,1,H,W] and cells [N,2] in coarse pixels."""
    h = sample_rays_torch(coarse_t, cells_coarse, n_az, k_steps, 1.0)                       # [N,n_az,K]
    h0 = height_at_cells_torch(coarse_t, cells_coarse)[:, None, None]
    d = (torch.arange(1, k_steps + 1, device=h.device, dtype=h.dtype) * mpp_coarse)[None, None, :]
    dh = h - h0 - d * d / (2 * moon_radius_m)
    ang = torch.rad2deg(torch.atan2(dh, d.expand_as(dh)))
    x = torch.stack([dh / REL_H_SCALE_M, ang / ANGLE_SCALE_DEG], dim=1).permute(0, 1, 3, 2).contiguous()  # [N,2,K,n_az]
    return x, ang.amax(dim=2)


class _CircConv(nn.Module):
    """3 x kw conv, zero padding along range, circular padding along azimuth, optional range stride."""

    def __init__(self, cin: int, cout: int, kw: int = 5, stride_range: int = 1):
        super().__init__()
        self.pad = kw // 2
        self.conv = nn.Conv2d(cin, cout, (3, kw), stride=(stride_range, 1), padding=(1, 0))
        self.norm = nn.GroupNorm(4 if cout % 4 == 0 else 1, cout)

    def forward(self, x):
        return F.relu(self.norm(self.conv(F.pad(x, (self.pad, self.pad, 0, 0), mode="circular"))))


class SkylineNet(nn.Module):
    def __init__(self, width: int = 24, n_down: int = 3):
        super().__init__()
        self.width, self.n_down = width, n_down
        layers = [_CircConv(2, width)]
        layers += [_CircConv(width, width, stride_range=2) for _ in range(n_down)]
        self.body = nn.Sequential(*layers)
        self.head1 = nn.Conv1d(2 * width, width, 5, padding=0)
        self.head2 = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.head2.weight)
        nn.init.zeros_(self.head2.bias)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(x)                                       # [N,w,K',n_az]
        g = torch.cat([f.amax(dim=2), f.mean(dim=2)], dim=1)   # collapse range -> [N,2w,n_az]
        g = F.relu(self.head1(F.pad(g, (2, 2), mode="circular")))
        return self.head2(g)[:, 0]                             # degrees

    def forward(self, x: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        return baseline + self.delta(x)


def checkpoint_id(model: nn.Module) -> str:
    h = hashlib.sha1()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def save_checkpoint(path: str | Path, model: SkylineNet, info: dict, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "arch": {"width": model.width, "n_down": model.n_down},
                "info": {**info, "checkpoint_id": checkpoint_id(model)}, **(extra or {})}, p)
    return p


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[SkylineNet, dict]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = SkylineNet(**state["arch"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model.to(device or torch.device("cpu")), state["info"]


class SkylineStudent:
    """Fast skyline for any cells of a tile: coarsens the tile, builds ray inputs, runs the network (or just the baseline)."""

    def __init__(self, model: SkylineNet | None, cfg: MissionConfig, device: torch.device | None = None, info: dict | None = None):
        self.model, self.cfg, self.info = model, cfg, info or {}
        self.device = device or (next(model.parameters()).device if model is not None else torch.device("cpu"))

    @torch.no_grad()
    def skyline(self, tile: Tile, cells_rc: np.ndarray, use_model: bool = True, batch: int = 128) -> np.ndarray:
        lm = self.cfg.light_model
        coarse = torch.from_numpy(coarsen(tile.height, lm.coarse_factor).astype(np.float32))[None, None].to(self.device)
        cells_c = torch.from_numpy(fine_to_coarse(np.atleast_2d(cells_rc), lm.coarse_factor).astype(np.float32)).to(self.device)
        mpp_c = tile.meters_per_pixel * lm.coarse_factor
        out = []
        for s in range(0, len(cells_c), batch):
            x, base = ray_inputs(coarse, cells_c[s:s + batch], mpp_c, lm.range_steps, lm.n_azimuth, self.cfg.moon.radius_m)
            out.append((self.model(x, base) if (use_model and self.model is not None) else base).cpu().numpy())
        return np.concatenate(out).astype(np.float64)
