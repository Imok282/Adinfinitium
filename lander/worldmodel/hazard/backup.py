"""Physics backup for the hazard detector (runs first, needs no AI, no torch).

backup_hazard: slope of the gap-filled surface >= the lander's slope limit (config).
label_hazard : the OFFICIAL LOLA slope >= the same limit (the "right answer" the models are scored on).
"""

from __future__ import annotations

import numpy as np

from contracts.mission import MissionConfig, load_mission
from contracts.tiles import Tile, hazard_from_slope
from worldmodel.hazard.features import build_features


def slope_limit_deg(cfg: MissionConfig) -> float:
    return cfg.lander.touchdown_limits.max_slope_deg


def backup_hazard(tile: Tile, cfg: MissionConfig | None = None) -> np.ndarray:
    cfg = cfg or load_mission()
    _, slope = build_features(tile, slope_limit_deg(cfg), cfg.hazard_detector.gap_fill_sigma_px)
    return hazard_from_slope(slope, slope_limit_deg(cfg))


def label_hazard(tile: Tile, cfg: MissionConfig | None = None) -> np.ndarray:
    cfg = cfg or load_mission()
    return hazard_from_slope(tile.slope, slope_limit_deg(cfg))
