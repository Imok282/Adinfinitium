"""Training data for the skyline model, built from real LOLA tiles (no pretrained anything).

Per tile: the model sees a COARSE grid (the 5 m height block-averaged `coarse_factor` x, a stand-in for LOLA's coarser
product); the target is the exact skyline ray-cast on the fine 5 m grid. Splits are whole tiles, same as the hazard detector:
train Site01 04 06 07 20 · validation Site11 · TEST Site23 (never loaded by train.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.mission import MissionConfig
from contracts.paths import data_root, tile_path
from contracts.tiles import load_tile
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, VAL_SITES, regions_for_site
from worldmodel.hazard.lola import region_name as _region_name
from worldmodel.light.teacher import n_steps, raycast_skyline_torch, valid_cells_mask

TEST_SITES = HOLDOUT_SITES     # one source of truth for the splits (worldmodel/hazard/lola.py)
__all__ = ["TEST_SITES", "TRAIN_SITES", "VAL_SITES", "SiteSkylines", "build_region", "build_site", "build_site_windows", "regions_for_site"]


def region_name(site: str) -> str:
    return _region_name(site)


def coarsen(height: np.ndarray, factor: int) -> np.ndarray:
    """Block-average by `factor` (cropped to a multiple). Coarse pixel j covers fine pixels [f*j, f*j + f - 1]."""
    h, w = (height.shape[0] // factor) * factor, (height.shape[1] // factor) * factor
    return height[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def fine_to_coarse(cells_fine: np.ndarray, factor: int) -> np.ndarray:
    """Fine-grid (row, col) -> coarse-grid coordinates (a coarse pixel centre sits at f*j + (f-1)/2 in fine pixels)."""
    return (np.asarray(cells_fine, dtype=np.float64) - (factor - 1) / 2.0) / factor


@dataclass
class SiteSkylines:
    region: str
    coarse: np.ndarray        # [Hc, Wc] float32 height in m
    factor: int               # fine pixels per coarse pixel
    mpp_coarse: float
    cells_fine: np.ndarray    # [N,2] float (row, col) on the 5 m grid
    truth: np.ndarray         # [N,n_az] exact skyline on the 5 m grid, degrees

    @property
    def cells_coarse(self) -> np.ndarray:
        return fine_to_coarse(self.cells_fine, self.factor)


def sample_cells(shape: tuple[int, int], margin_px: float, n: int, rng: np.random.Generator) -> np.ndarray:
    r0, r1, c0, c1 = valid_cells_mask(shape, margin_px)
    return np.stack([rng.uniform(r0, r1, n), rng.uniform(c0, c1, n)], axis=1)


def build_site(site: str, cfg: MissionConfig, n_cells: int, seed: int, device=None, cache: bool = True) -> SiteSkylines:
    """The centre window of a site (see build_site_windows for all windows)."""
    return build_region(region_name(site), cfg, n_cells, seed, device, cache)


def build_site_windows(site: str, cfg: MissionConfig, n_cells: int, seed: int, device=None, cache: bool = True) -> list[SiteSkylines]:
    """Every built window of a site (centre + grid windows). A site with no tiles on this machine is skipped with a note."""
    regions = regions_for_site(site)
    if not regions:
        print(f"note: no tiles for {site} on this machine; skipped", flush=True)
    return [build_region(r, cfg, n_cells, seed + 7 * i, device, cache) for i, r in enumerate(regions)]


def build_region(region: str, cfg: MissionConfig, n_cells: int, seed: int, device=None, cache: bool = True) -> SiteSkylines:
    """Fine tile -> (coarse grid, random cells, exact fine skyline). The truth is cached under <ADI_DATA>/light_train/."""
    lm = cfg.light_model
    tile = load_tile(tile_path(region))
    mpp = tile.meters_per_pixel
    k_fine = n_steps(lm.max_ray_range_m, mpp)
    cache_p = data_root() / "light_train" / f"{region}_n{n_cells}_s{seed}_r{int(lm.max_ray_range_m)}_a{lm.n_azimuth}.npz"
    cells = sample_cells(tile.shape, k_fine + 1, n_cells, np.random.default_rng(seed))
    if cache and cache_p.exists():
        truth = np.load(cache_p)["truth"]
    else:
        truth = raycast_skyline_torch(tile.height, mpp, cells, cfg.moon.radius_m, lm.max_ray_range_m, lm.n_azimuth, device=device)
        if cache:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_p, truth=truth.astype(np.float32))
    return SiteSkylines(region, coarsen(tile.height, lm.coarse_factor).astype(np.float32), lm.coarse_factor,
                        mpp * lm.coarse_factor, cells, truth.astype(np.float32))
