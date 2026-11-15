"""Write hazard probabilities into a tile's `hazard` grid (the file contract).

    python -I -m worldmodel.hazard.apply --region lola5m_site23 [--out other_region] [--backup-only]

Physics backup (slope rule on gap-filled input) runs when no checkpoint exists, torch is missing, or
--backup-only is given. The tile's meta says which model made the hazard grid.
Labelling: real input -> MODELLED. GENERATED input stays GENERATED (a fake tile is never relabelled).
"""

from __future__ import annotations

import argparse

import numpy as np

from contracts.mission import MissionConfig, load_mission
from contracts.paths import model_path, tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, load_tile, save_tile
from worldmodel.hazard.backup import backup_hazard, slope_limit_deg
from worldmodel.hazard.features import build_features


def apply_hazard(tile: Tile, cfg: MissionConfig | None = None, checkpoint=None, backup_only: bool = False) -> Tile:
    """Return a copy of `tile` with the hazard grid filled (U-Net if available, else the slope-rule backup)."""
    cfg = cfg or load_mission()
    model_name, ckpt_id = "slope_rule_backup", None
    hazard = None
    ckpt = checkpoint if checkpoint is not None else model_path("hazard_unet")
    if not backup_only and ckpt.exists():
        try:
            from worldmodel.hazard.model import load_checkpoint, pick_device, predict_probability

            device = pick_device()
            model, info = load_checkpoint(ckpt, device)
            feats, _ = build_features(tile, slope_limit_deg(cfg), cfg.hazard_detector.gap_fill_sigma_px)
            hazard = predict_probability(model, feats, device)
            model_name, ckpt_id = "hazard_unet", info["checkpoint_id"]
        except ImportError:
            hazard = None  # no torch on this machine: fall back to the backup
    if hazard is None:
        hazard = backup_hazard(tile, cfg)
    input_prov = tile.meta.provenance
    prov = Provenance.GENERATED if input_prov == Provenance.GENERATED else Provenance.MODELLED
    meta = Meta(provenance=prov,
                source=f"{tile.meta.source} | hazard grid by {model_name}" + (f" {ckpt_id}" if ckpt_id else ""),
                frame=tile.meta.frame, units=tile.meta.units, created_by="worldmodel.hazard.apply")
    extra = {**tile.extra, "hazard_model": model_name, "hazard_checkpoint_id": ckpt_id,
             "hazard_input_provenance": input_prov.value, "hazard_slope_limit_deg": slope_limit_deg(cfg),
             "hazard_note": "probability the pixel is unsafe; 0..1; never use alone for a safety claim"}
    return Tile(region=tile.region, meters_per_pixel=tile.meters_per_pixel, height=tile.height, slope=tile.slope,
                hazard=np.clip(hazard, 0, 1).astype(np.float32), uncertainty=tile.uncertainty,
                psr_mask=tile.psr_mask, meta=meta, center_lat_deg=tile.center_lat_deg,
                center_lon_deg=tile.center_lon_deg, real_mask=tile.real_mask, extra=extra)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", default=None, help="output region name (default: overwrite the input tile)")
    ap.add_argument("--backup-only", action="store_true")
    a = ap.parse_args(argv)
    t = apply_hazard(load_tile(tile_path(a.region)), backup_only=a.backup_only)
    t.region = a.out or t.region
    p = save_tile(t, tile_path(t.region))
    print(f"wrote {p}  hazard by {t.extra['hazard_model']}  mean {t.hazard.mean():.1%}  [{t.meta.provenance.banner}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
