"""Exact skyline teacher: ray-cast a height grid in every compass direction. This is the physics backup.

For a cell and azimuth az the skyline elevation is  max over range d of  atan2(h(d) - h0 - d^2/(2R), d)  (terrain + Moon curvature,
no refraction). Grid azimuth: 0 = up on the map (+y, row decreasing), 90 = +x (column increasing), clockwise.

Two implementations that must agree (tested): a numpy one (no torch, small jobs) and a torch one (GPU/MPS/CPU, batched).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def n_steps(max_range_m: float, mpp: float, step_px: float = 1.0) -> int:
    return int(max_range_m // (mpp * step_px))


def valid_cells_mask(shape: tuple[int, int], k_px: float) -> tuple[int, int, int, int]:
    """(row_min, row_max, col_min, col_max) inclusive range of cells whose rays stay inside the grid."""
    m = int(np.ceil(k_px))
    return m, shape[0] - 1 - m, m, shape[1] - 1 - m


def _check_inside(shape, cells: np.ndarray, k_px: float) -> None:
    r0, r1, c0, c1 = valid_cells_mask(shape, k_px)
    if np.any(cells[:, 0] < r0) or np.any(cells[:, 0] > r1) or np.any(cells[:, 1] < c0) or np.any(cells[:, 1] > c1):
        raise ValueError(f"a cell is closer than {k_px:.0f} px to the grid edge: its rays would leave the grid")


def raycast_skyline_numpy(height: np.ndarray, mpp: float, cells_rc: np.ndarray, moon_radius_m: float, max_range_m: float,
                          n_az: int = 360, step_px: float = 1.0) -> np.ndarray:
    """Skyline [N, n_az] in degrees. cells_rc [N,2] float (row, col) in this grid's pixels."""
    cells = np.atleast_2d(np.asarray(cells_rc, dtype=np.float64))
    k = np.arange(1, n_steps(max_range_m, mpp, step_px) + 1) * step_px            # px
    _check_inside(height.shape, cells, k[-1])
    az = np.radians(np.arange(n_az) * (360.0 / n_az))
    d = k * mpp
    out = np.empty((len(cells), n_az))
    for i, (r, c) in enumerate(cells):
        h0 = map_coordinates(height.astype(np.float64), [[r], [c]], order=1, mode="nearest")[0]
        rr = r - np.cos(az)[:, None] * k[None, :]
        cc = c + np.sin(az)[:, None] * k[None, :]
        hh = map_coordinates(height.astype(np.float64), [rr.ravel(), cc.ravel()], order=1, mode="nearest").reshape(rr.shape)
        out[i] = np.degrees(np.arctan2(hh - h0 - d[None, :] ** 2 / (2 * moon_radius_m), d[None, :])).max(axis=1)
    return out


def sample_rays_torch(height_t, cells_t, n_az: int, k_steps: int, step_px: float):
    """Heights along every ray: [N, n_az, K]. height_t [1,1,H,W]; cells_t [N,2] float (row, col)."""
    import torch
    import torch.nn.functional as F

    _, _, H, W = height_t.shape
    n = cells_t.shape[0]
    az = torch.arange(n_az, device=cells_t.device, dtype=cells_t.dtype) * (2 * torch.pi / n_az)
    k = torch.arange(1, k_steps + 1, device=cells_t.device, dtype=cells_t.dtype) * step_px
    rows = cells_t[:, 0, None, None] - torch.cos(az)[None, :, None] * k[None, None, :]
    cols = cells_t[:, 1, None, None] + torch.sin(az)[None, :, None] * k[None, None, :]
    grid = torch.stack([2 * cols / (W - 1) - 1, 2 * rows / (H - 1) - 1], dim=-1).reshape(1, n * n_az, k_steps, 2)
    h = F.grid_sample(height_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return h.reshape(n, n_az, k_steps)


def height_at_cells_torch(height_t, cells_t):
    import torch.nn.functional as F

    _, _, H, W = height_t.shape
    grid = torch_stack_xy(cells_t, H, W).reshape(1, -1, 1, 2)
    return F.grid_sample(height_t, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)


def torch_stack_xy(cells_t, H: int, W: int):
    import torch

    return torch.stack([2 * cells_t[:, 1] / (W - 1) - 1, 2 * cells_t[:, 0] / (H - 1) - 1], dim=-1)


def raycast_skyline_torch(height: np.ndarray, mpp: float, cells_rc: np.ndarray, moon_radius_m: float, max_range_m: float,
                          n_az: int = 360, step_px: float = 1.0, device=None, batch: int = 64) -> np.ndarray:
    """Same result as raycast_skyline_numpy, batched on a GPU/MPS/CPU device."""
    import torch

    cells = np.atleast_2d(np.asarray(cells_rc, dtype=np.float64))
    k_steps = n_steps(max_range_m, mpp, step_px)
    _check_inside(height.shape, cells, k_steps * step_px)
    dev = device or torch.device("cpu")
    ht = torch.from_numpy(np.ascontiguousarray(height, dtype=np.float32))[None, None].to(dev)
    d = (torch.arange(1, k_steps + 1, device=dev, dtype=torch.float32) * step_px * mpp)[None, None, :]
    out = []
    for s in range(0, len(cells), batch):
        c = torch.from_numpy(cells[s:s + batch].astype(np.float32)).to(dev)
        h = sample_rays_torch(ht, c, n_az, k_steps, step_px)
        h0 = height_at_cells_torch(ht, c)[:, None, None]
        ang = torch.rad2deg(torch.atan2(h - h0 - d * d / (2 * moon_radius_m), d.expand_as(h)))
        out.append(ang.amax(dim=2).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
