"""Landing scenes from real LOLA tiles: terrain + a list of safe target cells, with whole-tile splits (train / val / TEST never loaded for training)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from contracts.mission import MissionConfig
from contracts.paths import tile_path
from contracts.tiles import load_tile
from worldmodel.hazard.lola import regions_for_site
from worldmodel.lander.terrain import TerrainT, safe_target_cells, terrain_from_tile


@dataclass
class Scene:
    terrain: TerrainT
    safe_rc: np.ndarray
    region: str

    def sample_targets(self, n: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        rc = self.safe_rc[rng.integers(len(self.safe_rc), size=n)]
        return torch.as_tensor(self.terrain.cell_to_xy(rc), dtype=torch.float32, device=device)


def scene_for_region(region: str, cfg: MissionConfig, device: torch.device, belief: str = "backup") -> Scene | None:
    tile = load_tile(tile_path(region))
    if belief == "model":
        from worldmodel.rover.world import hazard_probability

        b = hazard_probability(tile, cfg)[0]
    else:
        from worldmodel.hazard.backup import backup_hazard

        b = backup_hazard(tile, cfg)
    safe = safe_target_cells(tile)
    if len(safe) < 50:
        print(f"note: {region} has too few safe target cells; skipped", flush=True)
        return None
    return Scene(terrain_from_tile(tile, device, b), safe, region)


def load_scenes(sites: list[str], cfg: MissionConfig, device: torch.device, belief: str = "backup", windows_per_site: int = 1) -> list[Scene]:
    """belief: 'backup' = the slope-rule hazard map from the sparse laser pixels (cheap), 'model' = the hazard U-Net if a checkpoint exists."""
    out = []
    for site in sites:
        for region in regions_for_site(site)[:windows_per_site]:
            sc = scene_for_region(region, cfg, device, belief)
            if sc is not None:
                out.append(sc)
    return out
