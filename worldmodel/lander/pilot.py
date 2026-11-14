"""Learned lander pilot (PyTorch): a RESIDUAL on top of classic ZEM/ZEV guidance.

  action = clip(classic(obs) + residual_scale * clamp(r), -1, 1)       r ~ Normal(mu(obs, camera image), sigma)

The network sees the 21-number observation (13 sensors + 8-direction hazard ring), the classic command itself, and an egocentric camera image of
the ground below (slope, hazard belief, relative height). The last layer is zero-initialised, so an UNTRAINED pilot flies exactly like the physics
backup (tested). `PilotFormer` is the large model: a conv stem cuts the image into 8 x 8 = 64 patch tokens, the state becomes one more token,
a transformer reasons over all of them (where is the safe ground, how fast must I divert). d=384, layers=8 is about 15 M parameters;
d=512, layers=12 about 38 M. `PilotMLP` is the small CPU-test version.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from worldmodel.nn_blocks import TransformerStack, sincos_1d

STATE_DIM = 21 + 3
RESIDUAL_SCALE = 0.3


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PilotMLP(nn.Module):
    def __init__(self, hidden: int = 256, crop: int = 32, width: int = 24):
        super().__init__()
        self.hidden, self.crop, self.width = hidden, crop, width
        w = width
        self.conv = nn.Sequential(nn.Conv2d(3, w, 3, 2, 1), nn.GroupNorm(4, w), nn.SiLU(), nn.Conv2d(w, 2 * w, 3, 2, 1), nn.GroupNorm(4, 2 * w), nn.SiLU(),
                                  nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.trunk = nn.Sequential(nn.Linear(2 * w * 16 + STATE_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, 3)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((3,), -1.2))
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def arch(self) -> dict:
        return {"kind": "mlp", "hidden": self.hidden, "crop": self.crop, "width": self.width}

    def forward(self, state: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([self.conv(img), state], dim=1))
        return torch.tanh(self.mu(h)), self.v(h)[:, 0]


class PilotFormer(nn.Module):
    def __init__(self, d: int = 384, layers: int = 8, heads: int = 8, crop: int = 32, stem: int = 48):
        super().__init__()
        self.d, self.layers, self.heads, self.crop, self.stem_ch = d, layers, heads, crop, stem
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1), nn.GroupNorm(8, stem), nn.SiLU(), nn.Conv2d(stem, 2 * stem, 3, 2, 1), nn.GroupNorm(8, 2 * stem), nn.SiLU(),
                                  nn.Conv2d(2 * stem, d, 2, 2))                      # 32 -> 32 -> 16 -> 8: 64 patch tokens
        side = crop // 4
        self.register_buffer("pe", sincos_1d(side * side, d), persistent=False)
        self.state = nn.Sequential(nn.Linear(STATE_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder = TransformerStack(d, layers, heads)
        self.mu = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
        self.v = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.full((3,), -1.2))
        nn.init.zeros_(self.mu[-1].weight)
        nn.init.zeros_(self.mu[-1].bias)

    def arch(self) -> dict:
        return {"kind": "former", "d": self.d, "layers": self.layers, "heads": self.heads, "crop": self.crop, "stem": self.stem_ch}

    def forward(self, state: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.stem(img).flatten(2).transpose(1, 2) + self.pe
        seq = torch.cat([self.cls.expand(tok.shape[0], -1, -1), self.state(state)[:, None], tok], dim=1)
        h = self.encoder(seq)[:, 0]
        return torch.tanh(self.mu(h)), self.v(h)[:, 0]


def build_model(kind: str = "mlp", **kw) -> nn.Module:
    return {"mlp": PilotMLP, "former": PilotFormer}[kind](**kw)


def pilot_features(obs: torch.Tensor, classic: torch.Tensor) -> torch.Tensor:
    return torch.cat([obs, classic], dim=1)


def residual_action(classic: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return (classic + RESIDUAL_SCALE * r.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)


def checkpoint_id(model: nn.Module) -> str:
    h = hashlib.sha1()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def save_checkpoint(path: str | Path, model: nn.Module, info: dict, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "arch": model.arch(),
                "info": {**info, "checkpoint_id": checkpoint_id(model)}, **(extra or {})}, p)
    return p


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    st = torch.load(Path(path), map_location="cpu", weights_only=False)
    arch = dict(st["arch"])
    model = build_model(arch.pop("kind"), **arch)
    model.load_state_dict(st["state_dict"])
    model.eval()
    return model.to(device or torch.device("cpu")), st["info"]
