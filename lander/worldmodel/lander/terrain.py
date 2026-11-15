"""Terrain lookups on a tile for the PyTorch lander environment (same maths as sim_jax/terrain.py: bilinear height and slope, nearest hazard).

Frame: tile ENU metres centred on the tile (x east, y north). `hazard` is the TRUTH used by the touchdown check (the official LOLA slope rule);
`belief` (optional) is what the camera/hazard model thinks (the hazard U-Net or its backup): the pilot's image crop uses it, so the gap between
belief and truth is part of the honest exam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from contracts.tiles import Tile


@dataclass
class TerrainT:
    height: torch.Tensor      # [H,W] m
    slope: torch.Tensor       # [H,W] deg
    hazard: torch.Tensor      # [H,W] 0..1 truth
    belief: torch.Tensor      # [H,W] 0..1 what the camera model believes
    mpp: float
    region: str = "?"

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.height.shape)

    @property
    def half_w_m(self) -> float:
        return (self.height.shape[1] - 1) / 2 * self.mpp

    @property
    def half_h_m(self) -> float:
        return (self.height.shape[0] - 1) / 2 * self.mpp

    @property
    def device(self) -> torch.device:
        return self.height.device

    def to_rc(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = self.height.shape
        col = x / self.mpp + (w - 1) / 2.0
        row = (h - 1) / 2.0 - y / self.mpp
        return row.clamp(0.0, h - 1.0), col.clamp(0.0, w - 1.0)

    def in_bounds(self, x, y):
        return (x.abs() <= self.half_w_m) & (y.abs() <= self.half_h_m)

    def bilinear(self, grid: torch.Tensor, x, y):
        h, w = grid.shape
        r, c = self.to_rc(x, y)
        r0, c0 = r.floor().long(), c.floor().long()
        r1, c1 = (r0 + 1).clamp(max=h - 1), (c0 + 1).clamp(max=w - 1)
        fr, fc = r - r0, c - c0
        flat = grid.reshape(-1)
        g = lambda rr, cc: flat[rr * w + cc]
        top = g(r0, c0) * (1 - fc) + g(r0, c1) * fc
        bot = g(r1, c0) * (1 - fc) + g(r1, c1) * fc
        return top * (1 - fr) + bot * fr

    def nearest(self, grid: torch.Tensor, x, y):
        _, w = grid.shape
        r, c = self.to_rc(x, y)
        return grid.reshape(-1)[r.round().long() * w + c.round().long()]

    def height_at(self, x, y):
        return self.bilinear(self.height, x, y)

    def slope_at(self, x, y):
        return self.bilinear(self.slope, x, y)

    def hazard_at(self, x, y):
        return self.nearest(self.hazard, x, y)

    def cell_to_xy(self, rc: np.ndarray) -> np.ndarray:
        h, w = self.shape
        return np.stack([(rc[:, 1] - (w - 1) / 2.0) * self.mpp, ((h - 1) / 2.0 - rc[:, 0]) * self.mpp], axis=1)

    def crop(self, x: torch.Tensor, y: torch.Tensor, span_m: torch.Tensor, px: int, slope_limit: float) -> torch.Tensor:
        """Egocentric north-up image [N,3,px,px] centred on (x,y): slope/limit, hazard belief, height relative to the centre (/50 m).
        `span_m` [N] is the side length of the square seen (a nadir camera sees more ground from higher up)."""
        n = x.shape[0]
        h, w = self.shape
        stack = torch.stack([self.slope / slope_limit, self.belief, self.height / 50.0])[None]           # [1,3,H,W]
        u = torch.linspace(-0.5, 0.5, px, device=x.device)
        gx = x[:, None, None] + span_m[:, None, None] * u[None, None, :]
        gy = y[:, None, None] + span_m[:, None, None] * (-u)[None, :, None]
        gx, gy = gx.expand(n, px, px), gy.expand(n, px, px)
        cn = (gx / self.mpp + (w - 1) / 2.0) / (w - 1) * 2 - 1
        rn = ((h - 1) / 2.0 - gy / self.mpp) / (h - 1) * 2 - 1
        img = torch.nn.functional.grid_sample(stack.expand(n, -1, -1, -1), torch.stack([cn, rn], dim=-1), mode="bilinear", padding_mode="border", align_corners=True)
        ctr = self.bilinear(self.height, x, y)[:, None, None] / 50.0
        img[:, 2] = img[:, 2] - ctr
        return img.clamp(-4.0, 4.0)


def safe_target_cells(tile: Tile, margin_px: int = 3) -> np.ndarray:
    """(row, col) of cells whose (2m+1)^2 neighbourhood has no hazard in the backup map (numpy, no jax)."""
    from numpy.lib.stride_tricks import sliding_window_view

    haz = tile.hazard >= 0.5
    k = 2 * margin_px + 1
    ok = ~sliding_window_view(np.pad(haz, margin_px, constant_values=True), (k, k)).any(axis=(-1, -2))
    inner = np.zeros_like(ok)
    m = 120                                              # keep targets away from the tile edge (start offsets reach 600 m)
    inner[m:-m, m:-m] = True
    return np.argwhere(ok & inner)


def terrain_from_tile(tile: Tile, device: torch.device | None = None, belief: np.ndarray | None = None) -> TerrainT:
    dev = device or torch.device("cpu")
    f = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
    haz = f(tile.hazard)
    return TerrainT(f(tile.height), f(tile.slope), haz, f(belief) if belief is not None else haz, float(tile.meters_per_pixel), tile.region)


def flat_terrain(size: int = 64, mpp: float = 50.0, device: torch.device | None = None) -> TerrainT:
    z = torch.zeros(size, size, device=device)
    return TerrainT(z, z.clone(), z.clone(), z.clone(), mpp, "flat")
