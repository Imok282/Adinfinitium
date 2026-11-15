"""Shared neural building blocks for the larger architectures (PyTorch only; runs on cuda / mps / cpu, native Windows fine)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv x2 with a skip; optional time/conditioning embedding added after the first conv (diffusion)."""

    def __init__(self, cin: int, cout: int, emb: int = 0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(groups(cin), cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(groups(cout), cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)
        self.emb = nn.Linear(emb, cout) if emb else None

    def forward(self, x: torch.Tensor, e: torch.Tensor | None = None) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        if self.emb is not None and e is not None:
            h = h + self.emb(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over the pixels of a feature map (residual). Uses fused scaled-dot-product attention."""

    def __init__(self, ch: int, heads: int = 8):
        super().__init__()
        assert ch % heads == 0, f"channels {ch} not divisible by heads {heads}"
        self.heads = heads
        self.norm = nn.GroupNorm(groups(ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).permute(1, 0, 2, 4, 3)
        o = F.scaled_dot_product_attention(q, k, v)                              # [B, heads, HW, C/heads]
        return x + self.proj(o.permute(0, 1, 3, 2).reshape(b, c, h, w))


class MLP(nn.Module):
    def __init__(self, d: int, ratio: int = 4, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, ratio * d), nn.GELU(), nn.Dropout(drop), nn.Linear(ratio * d, d))

    def forward(self, x):
        return self.net(x)


class TransformerStack(nn.Module):
    """Pre-norm transformer encoder (self-attention + MLP) over tokens [B, N, d]."""

    def __init__(self, d: int, layers: int, heads: int, ratio: int = 4, drop: float = 0.0):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d, heads, ratio * d, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.enc(x))


def sincos_1d(n: int, d: int, period: float | None = None) -> torch.Tensor:
    """[n, d] sinusoidal positions; period = n gives a circular (compass) encoding."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    if period is not None:
        pos = pos * (2 * math.pi / period)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2, dtype=torch.float32) / d)[None]
    ang = pos * (freqs if period is None else torch.arange(1, d // 2 + 1, dtype=torch.float32)[None])
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
