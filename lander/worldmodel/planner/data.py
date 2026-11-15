"""Load the simulated-outcome label files into per-world groups, split by whole site (train / val Site11 / TEST Site23 + Site42)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from contracts.paths import data_root
from worldmodel.hazard.lola import HOLDOUT_SITES, VAL_SITES


@dataclass
class Group:
    region: str
    date: str
    x: np.ndarray          # [K, F]
    y: np.ndarray          # [K] simulated p_mission
    n_eff: np.ndarray      # [K] effective number of simulated missions behind y
    backup: np.ndarray     # [K] Navigator score
    extra: dict

    @property
    def site(self) -> str:
        return self.region.split("lola5m_")[1].split("_")[0].replace("site", "Site")


def role(g: Group) -> str:
    s = g.site
    if s in HOLDOUT_SITES:
        return "test"
    if s in VAL_SITES:
        return "val"
    return "train"


def load_groups() -> list[Group]:
    out = []
    for p in sorted((data_root() / "planner").glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        n = np.full(len(z["p_mission"]), float(min(int(z["n_land"]), int(z["n_rover"]))), np.float32)
        out.append(Group(str(z["region"]), str(z["date"]), z["features"], z["p_mission"], n, z["backup"],
                         {k: z[k] for k in ("p_land", "p_ice", "p_return", "scatter_m", "cell_rc", "hour")}))
    return out


def batch(groups: list[Group], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k = max(len(g.y) for g in groups)
    f = groups[0].x.shape[1]
    x = np.zeros((len(groups), k, f), np.float32)
    y = np.zeros((len(groups), k), np.float32)
    w = np.zeros((len(groups), k), np.float32)
    m = np.zeros((len(groups), k), bool)
    for i, g in enumerate(groups):
        n = len(g.y)
        x[i, :n], y[i, :n], w[i, :n], m[i, :n] = g.x, g.y, g.n_eff, True
    t = lambda a: torch.as_tensor(a, device=device)
    return t(x), t(y), t(w), t(m)
