"""Conditional diffusion for terrain patches (needs torch). Trained only on real laser pixels; outputs are GENERATED.

Network: small U-Net with a time embedding. Input channels [noisy height, known heights, known mask]; output = clean-height estimate (x0).
Sampler: DDIM/DDPM-style from pure noise; at KNOWN (measured) pixels the clean value is replaced by the measurement at every step.
Heights are normalised per patch by mean/scale of the known pixels (data.normalise), so inference can recompute them exactly.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from worldmodel.nn_blocks import ResBlock, SelfAttention2d, groups


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """alpha_bar[t] for t = 0..T (index 0 = no noise)."""
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 0.99999).float()


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, temb: int):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, padding=1), nn.Conv2d(cout, cout, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
        self.t = nn.Linear(temb, cout)

    def forward(self, x, te):
        x = F.silu(self.n1(self.c1(x)))
        x = x + self.t(te)[:, :, None, None]
        return F.silu(self.n2(self.c2(x)))


class DenoiseUNet(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3, T: int = 100):
        super().__init__()
        self.base, self.depth, self.T = base, depth, T
        self.temb = base * 4
        self.t_mlp = nn.Sequential(nn.Linear(base, self.temb), nn.SiLU(), nn.Linear(self.temb, self.temb))
        chs = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_Block(3 if i == 0 else chs[i - 1], chs[i], self.temb) for i in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([_Block(chs[i] * 2, chs[i], self.temb) for i in range(depth)])
        self.head = nn.Conv2d(base, 1, 1)

    def arch(self) -> dict:
        return {"kind": "unet", "base": self.base, "depth": self.depth, "T": self.T}

    def forward(self, x_t, cond_val, cond_mask, t):
        te = self.t_mlp(time_embedding(t, self.base))
        x = torch.cat([x_t, cond_val, cond_mask], dim=1)
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x, te)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.enc[self.depth](x, te)
        for i in reversed(range(self.depth)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1), te)
        return self.head(x)


class _Seq(nn.ModuleList):
    """Sequence of blocks that all take (x, embedding)."""

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        for m in self:
            x = m(x, e)
        return x


class DenoiseAttnUNet(nn.Module):
    """Large conditional denoiser: residual blocks with a time embedding and multi-head self-attention at the coarse scales (global terrain context
    over the 320 m patch). Same interface as DenoiseUNet. base=192, depth=3, blocks=2, attn_levels=2 is about 112 M parameters."""

    def __init__(self, base: int = 128, depth: int = 3, T: int = 100, blocks: int = 2, heads: int = 8, attn_levels: int = 2):
        super().__init__()
        self.base, self.depth, self.T, self.blocks, self.heads, self.attn_levels = base, depth, T, blocks, heads, attn_levels
        emb = base * 4
        self.t_mlp = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        chs = [min(base * 2**i, base * 4) for i in range(depth + 1)]
        use_attn = [i >= depth - attn_levels for i in range(depth)]
        self.stem = nn.Conv2d(3, chs[0], 3, padding=1)
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        for i in range(depth):
            layers = [ResBlock(chs[i], chs[i], emb) for _ in range(blocks)] + ([SelfAttention2d(chs[i], heads)] if use_attn[i] else [])
            self.enc.append(_Seq(layers))
            self.down.append(nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1))
        self.mid = _Seq([ResBlock(chs[depth], chs[depth], emb), SelfAttention2d(chs[depth], heads), ResBlock(chs[depth], chs[depth], emb)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList()
        for i in range(depth):
            layers = [ResBlock(2 * chs[i], chs[i], emb)] + [ResBlock(chs[i], chs[i], emb) for _ in range(blocks - 1)]
            self.dec.append(_Seq(layers + ([SelfAttention2d(chs[i], heads)] if use_attn[i] else [])))
        self.head = nn.Sequential(nn.GroupNorm(groups(chs[0]), chs[0]), nn.SiLU(), nn.Conv2d(chs[0], 1, 1))

    def arch(self) -> dict:
        return {"kind": "attn", "base": self.base, "depth": self.depth, "T": self.T, "blocks": self.blocks, "heads": self.heads, "attn_levels": self.attn_levels}

    def forward(self, x_t, cond_val, cond_mask, t):
        e = self.t_mlp(time_embedding(t, self.base))
        x = self.stem(torch.cat([x_t, cond_val, cond_mask], dim=1))
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x, e)
            skips.append(x)
            x = self.down[i](x)
        x = self.mid(x, e)
        for i in reversed(range(self.depth)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1), e)
        return self.head(x)


def build_model(kind: str = "unet", **kw) -> nn.Module:
    """Terrain denoiser by kind: 'unet' (2 M at base 32) or 'attn' (attention U-Net, ~100 M at base 128)."""
    return {"unet": DenoiseUNet, "attn": DenoiseAttnUNet}[kind](**kw)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    a = ab[t][:, None, None, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


@torch.no_grad()
def sample(model: nn.Module, known_val: torch.Tensor, known_mask: torch.Tensor, steps: int = 25, eta: float = 1.0,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """known_val/known_mask [B,1,H,W] (normalised heights, 0/1 mask). Returns clean-height samples [B,1,H,W] (normalised)."""
    dev = known_val.device
    T = model.T
    ab = cosine_alpha_bar(T).to(dev)
    ts = torch.linspace(T, 1, steps).round().long().tolist()
    x = torch.randn(known_val.shape, device=dev, generator=generator)
    for i, t in enumerate(ts):
        tt = torch.full((x.shape[0],), t, device=dev, dtype=torch.long)
        x0 = model(x, known_val, known_mask, tt).clamp(-8, 8)
        x0 = torch.where(known_mask > 0.5, known_val, x0)
        a_t = ab[t]
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=dev)
        eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
        sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
        noise = torch.randn(x.shape, device=dev, generator=generator)
        x = a_prev.sqrt() * x0 + (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps + sigma * noise
    return torch.where(known_mask > 0.5, known_val, x0)


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
    model = build_model(arch.pop("kind", "unet"), **arch)             # old checkpoints have no 'kind': the small U-Net
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model.to(device or torch.device("cpu")), state["info"]
