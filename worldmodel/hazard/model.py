"""Small U-Net for hazard probability (needs torch). Input: features from worldmodel.hazard.features."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldmodel.hazard.features import N_CHANNELS


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        g = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Plain U-Net. Output is a LOGIT per pixel (sigmoid -> probability the pixel is unsafe)."""

    def __init__(self, in_ch: int = N_CHANNELS, base: int = 16, depth: int = 4):
        super().__init__()
        self.in_ch, self.base, self.depth = in_ch, base, depth
        chs = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_Block(in_ch if i == 0 else chs[i - 1], chs[i]) for i in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([_Block(chs[i] * 2, chs[i]) for i in range(depth)])
        self.head = nn.Conv2d(base, 1, 1)

    @property
    def multiple(self) -> int:
        return 2**self.depth

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.enc[self.depth](x)
        for i in reversed(range(self.depth)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)[:, 0]


@torch.no_grad()
def _forward_padded(model: UNet, x: torch.Tensor) -> torch.Tensor:
    _, h, w = x.shape
    m = model.multiple
    ph, pw = (-h) % m, (-w) % m
    xp = F.pad(x[None], (0, pw, 0, ph), mode="reflect") if (ph or pw) else x[None]
    return torch.sigmoid(model(xp))[0, :h, :w]


@torch.no_grad()
def predict_probability(model: UNet, features: np.ndarray, device: torch.device | None = None,
                        window: int = 768, margin: int = 64) -> np.ndarray:
    """Probability map [H,W] in 0..1. Big tiles are run in overlapping windows."""
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.from_numpy(features).to(device)
    _, h, w = x.shape
    if h <= window + 2 * margin and w <= window + 2 * margin:
        return _forward_padded(model, x).cpu().numpy()
    out = np.zeros((h, w), dtype=np.float32)
    for r0 in range(0, h, window):
        for c0 in range(0, w, window):
            r1, c1 = min(r0 + window, h), min(c0 + window, w)
            a, b = max(r0 - margin, 0), max(c0 - margin, 0)
            e, f = min(r1 + margin, h), min(c1 + margin, w)
            y = _forward_padded(model, x[:, a:e, b:f]).cpu().numpy()
            out[r0:r1, c0:c1] = y[r0 - a:r1 - a, c0 - b:c1 - b]
    return out


def checkpoint_id(model: UNet) -> str:
    h = hashlib.sha1()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def save_checkpoint(path: str | Path, model: UNet, info: dict, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "arch": {"in_ch": model.in_ch, "base": model.base, "depth": model.depth},
             "info": {**info, "checkpoint_id": checkpoint_id(model)}, **(extra or {})}
    torch.save(state, p)
    return p


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[UNet, dict]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = UNet(**state["arch"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model.to(device or torch.device("cpu")), state["info"]
