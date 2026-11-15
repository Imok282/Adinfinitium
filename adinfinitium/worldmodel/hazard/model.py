"""Small U-Net for hazard probability (needs torch). Input: features from worldmodel.hazard.features."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from worldmodel.hazard.features import N_CHANNELS
from worldmodel.nn_blocks import ResBlock, SelfAttention2d, groups


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

    preferred_window = 768

    def arch(self) -> dict:
        return {"kind": "unet", "in_ch": self.in_ch, "base": self.base, "depth": self.depth}

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


class AttnUNet(nn.Module):
    """Deep residual U-Net with multi-head self-attention at the coarse scales (global terrain context). Output is a LOGIT per pixel.

    Channels grow base, 2b, 4b, 8b (capped at 8b); `blocks` residual blocks per level; self-attention at the deepest `attn_levels` encoder
    levels, in the bottleneck and in the matching decoder levels. base=72, depth=4, blocks=2 is about 60 M parameters."""

    preferred_window = 256      # attention cost grows with the pixel count squared (MPS materialises it): run big tiles in 256 px windows + 64 px margins

    def __init__(self, in_ch: int = N_CHANNELS, base: int = 72, depth: int = 4, blocks: int = 2, heads: int = 8, attn_levels: int = 2):
        super().__init__()
        self.in_ch, self.base, self.depth, self.blocks, self.heads, self.attn_levels = in_ch, base, depth, blocks, heads, attn_levels
        chs = [min(base * 2**i, base * 8) for i in range(depth + 1)]
        use_attn = [i >= depth - attn_levels for i in range(depth)]
        self.stem = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(depth):
            layers = [ResBlock(chs[i], chs[i]) for _ in range(blocks)]
            if use_attn[i]:
                layers.append(SelfAttention2d(chs[i], heads))
            self.enc.append(nn.Sequential(*layers))
            self.down.append(nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1))
        self.mid = nn.Sequential(ResBlock(chs[depth], chs[depth]), SelfAttention2d(chs[depth], heads), ResBlock(chs[depth], chs[depth]))
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList()
        for i in range(depth):
            layers = [ResBlock(2 * chs[i], chs[i])] + [ResBlock(chs[i], chs[i]) for _ in range(blocks - 1)]
            if use_attn[i]:
                layers.append(SelfAttention2d(chs[i], heads))
            self.dec.append(nn.Sequential(*layers))
        self.head = nn.Sequential(nn.GroupNorm(groups(chs[0]), chs[0]), nn.SiLU(), nn.Conv2d(chs[0], 1, 1))

    @property
    def multiple(self) -> int:
        return 2**self.depth

    def arch(self) -> dict:
        return {"kind": "attn", "in_ch": self.in_ch, "base": self.base, "depth": self.depth, "blocks": self.blocks, "heads": self.heads,
                "attn_levels": self.attn_levels}

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = self.down[i](x)
        x = self.mid(x)
        for i in reversed(range(self.depth)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)[:, 0]


def build_model(kind: str = "unet", **kw) -> nn.Module:
    """Hazard network by kind: 'unet' (plain, 1.9 M at base 16) or 'attn' (deep residual attention U-Net, ~60 M at base 72)."""
    return {"unet": UNet, "attn": AttnUNet}[kind](**kw)


@torch.no_grad()
def _forward_padded(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    _, h, w = x.shape
    m = model.multiple
    ph, pw = (-h) % m, (-w) % m
    xp = F.pad(x[None], (0, pw, 0, ph), mode="reflect") if (ph or pw) else x[None]
    return torch.sigmoid(model(xp))[0, :h, :w]


@torch.no_grad()
def predict_probability(model: nn.Module, features: np.ndarray, device: torch.device | None = None,
                        window: int | None = None, margin: int = 64) -> np.ndarray:
    """Probability map [H,W] in 0..1. Big tiles are run in overlapping windows (`model.preferred_window` px by default)."""
    window = window or getattr(model, "preferred_window", 768)
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


def checkpoint_id(model: nn.Module) -> str:
    h = hashlib.sha1()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def save_checkpoint(path: str | Path, model: nn.Module, info: dict, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")     # a kill mid-write must not leave a truncated checkpoint
    state = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "arch": model.arch(),
             "info": {**info, "checkpoint_id": checkpoint_id(model)}, **(extra or {})}
    torch.save(state, tmp)
    tmp.replace(p)
    return p


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    arch = dict(state["arch"])
    model = build_model(arch.pop("kind", "unet"), **arch)             # checkpoints from before the attention model have no 'kind': plain U-Net
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model.to(device or torch.device("cpu")), state["info"]
