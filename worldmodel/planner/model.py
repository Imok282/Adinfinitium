"""Mission planner networks: P(mission success) for each candidate (landing cell x landing hour) of one world.

PlannerSetNet: each candidate's features become a token; a transformer lets candidates of the same world be compared with each other (a listwise ranker),
then a head gives a calibrated probability. LogisticPlanner: the honest small baseline on the same features. Both are trained on simulated outcomes only.
Data are small (hundreds to a few thousand simulated candidates), so the planner is deliberately small (~0.4 M at d=128, layers=3): a bigger network would
only memorise. Backup: the exact Navigator ranking (features.navigator_score).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from worldmodel.nn_blocks import TransformerStack
from worldmodel.planner.features import NF


class PlannerSetNet(nn.Module):
    def __init__(self, d: int = 128, layers: int = 3, heads: int = 4, nf: int = NF):
        super().__init__()
        self.d, self.layers, self.heads, self.nf = d, layers, heads, nf
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.SiLU(), nn.Linear(d, d))
        self.encoder = TransformerStack(d, layers, heads)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 1))
        self.register_buffer("mu", torch.zeros(nf))
        self.register_buffer("sd", torch.ones(nf))

    def arch(self) -> dict:
        return {"kind": "set", "d": self.d, "layers": self.layers, "heads": self.heads, "nf": self.nf}

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, K, F] candidates of B worlds (padded); mask [B, K] True = real. Returns logits [B, K]."""
        h = self.inp((x - self.mu) / self.sd)
        ctx = self.encoder(h)                                       # every candidate attends to the others of its world
        if mask is not None:
            pooled = (ctx * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            pooled = ctx.mean(1)
        return self.head(torch.cat([h, ctx - pooled[:, None]], dim=-1))[..., 0]


class LogisticPlanner(nn.Module):
    def __init__(self, nf: int = NF):
        super().__init__()
        self.nf = nf
        self.lin = nn.Linear(nf, 1)
        self.register_buffer("mu", torch.zeros(nf))
        self.register_buffer("sd", torch.ones(nf))

    def arch(self) -> dict:
        return {"kind": "logistic", "nf": self.nf}

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.lin((x - self.mu) / self.sd)[..., 0]


def build_model(kind: str = "set", **kw) -> nn.Module:
    return {"set": PlannerSetNet, "logistic": LogisticPlanner}[kind](**kw)


def checkpoint_id(model: nn.Module) -> str:
    h = hashlib.sha1()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def save_checkpoint(path: str | Path, model: nn.Module, info: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "arch": model.arch(),
                "info": {**info, "checkpoint_id": checkpoint_id(model)}}, p)
    return p


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict]:
    st = torch.load(Path(path), map_location="cpu", weights_only=False)
    arch = dict(st["arch"])
    model = build_model(arch.pop("kind"), **arch)
    model.load_state_dict(st["state_dict"])
    model.eval()
    return model, st["info"]
