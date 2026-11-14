"""Generate plausible versions of a whole tile (GENERATED — NOT OBSERVED).

    python -I -m worldmodel.terrain.generate --region lola5m_site23 --versions 4 [--backup-only] [--crop 512]

Writes data/tiles/<region>_gen<k>.npz in the official tile format: height = one generated version, uncertainty = the spread (std) of the
versions, real_mask kept, provenance GENERATED. Measured (laser) pixels are kept exactly as measured. Patches are blended with a
smooth window so there are no seams. --backup-only uses the physics backup (gap-fill + LOLA-error noise) and needs no torch.
Slope and hazard are recomputed from the new heights by the contract helpers; hazard is the slope-rule backup, not a safety claim.
"""

from __future__ import annotations

import argparse

import numpy as np

from contracts.mission import MissionConfig, load_mission
from contracts.paths import model_path, tile_path
from contracts.provenance import Meta, Provenance
from contracts.tiles import Tile, hazard_from_slope, save_tile, slope_from_height
from worldmodel.terrain.backup import backup_versions
from worldmodel.terrain.data import PATCH, SD_FLOOR_M

MIN_KNOWN = 8


def _window(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w) + 1e-3


def generate_versions(tile: Tile, cfg: MissionConfig, n_versions: int, model=None, device=None, steps: int = 25, stride: int = 32,
                      seed: int = 0, batch: int = 32) -> np.ndarray:
    """[n_versions, H, W] heights in metres. Model-based if `model` is given, else the physics backup."""
    rng = np.random.default_rng(seed)
    measured = tile.real_mask if tile.real_mask is not None else np.ones(tile.shape, bool)
    h = tile.height.astype(np.float64)
    sigmas = cfg.hazard_detector.gap_fill_sigma_px
    backup = backup_versions(h, measured, tile.uncertainty, sigmas, n_versions, rng)
    if model is None:
        out = backup
    else:
        import torch

        from worldmodel.terrain.model import sample

        H, W = tile.shape
        acc, wsum = np.zeros((n_versions, H, W)), np.zeros((H, W))
        win = _window(PATCH)
        origins = [(r, c) for r in range(0, H - PATCH + 1, stride) for c in range(0, W - PATCH + 1, stride)]
        origins = [(r, c) for r, c in origins if measured[r:r + PATCH, c:c + PATCH].sum() >= MIN_KNOWN]
        for s in range(0, len(origins), batch):
            chunk = origins[s:s + batch]
            kv, km, mus, sds = [], [], [], []
            for r, c in chunk:
                m = measured[r:r + PATCH, c:c + PATCH]
                hp = h[r:r + PATCH, c:c + PATCH]
                mu, sd = hp[m].mean(), max(hp[m].std(), SD_FLOOR_M)
                kv.append(((hp - mu) / sd * m).astype(np.float32))
                km.append(m.astype(np.float32))
                mus.append(mu)
                sds.append(sd)
            kvt = torch.from_numpy(np.stack(kv))[:, None].repeat_interleave(n_versions, 0).to(device)
            kmt = torch.from_numpy(np.stack(km))[:, None].repeat_interleave(n_versions, 0).to(device)
            gen = sample(model, kvt, kmt, steps=steps)[:, 0].cpu().numpy().reshape(len(chunk), n_versions, PATCH, PATCH)
            for (r, c), g, mu, sd in zip(chunk, gen, mus, sds):
                acc[:, r:r + PATCH, c:c + PATCH] += (g * sd + mu) * win
                wsum[r:r + PATCH, c:c + PATCH] += win
        covered = wsum > 5e-3
        out = backup.copy()                                    # patches with too few laser pixels fall back to the physics backup
        out[:, covered] = acc[:, covered] / wsum[covered]
    out = np.where(measured[None], h[None], out)               # measured pixels stay exactly as measured
    return out


def to_tiles(tile: Tile, versions: np.ndarray, cfg: MissionConfig, source: str, region_suffix: str = "gen") -> list[Tile]:
    limit = cfg.lander.touchdown_limits.max_slope_deg
    spread = versions.std(axis=0).astype(np.float32)
    tiles = []
    for k, v in enumerate(versions):
        slope = slope_from_height(v, tile.meters_per_pixel)
        meta = Meta(provenance=Provenance.GENERATED, source=f"{source} (version {k}; from {tile.meta.source})", frame=tile.meta.frame,
                    units=tile.meta.units, created_by="worldmodel.terrain.generate")
        tiles.append(Tile(region=f"{tile.region}_{region_suffix}{k}", meters_per_pixel=tile.meters_per_pixel, height=v.astype(np.float32),
                          slope=slope, hazard=hazard_from_slope(slope, limit), uncertainty=np.maximum(spread, tile.uncertainty * 0.0 + 1e-3),
                          psr_mask=tile.psr_mask, meta=meta, center_lat_deg=tile.center_lat_deg, center_lon_deg=tile.center_lon_deg,
                          real_mask=tile.real_mask, extra={**tile.extra, "generated": "GENERATED — NOT OBSERVED", "version": k,
                                                           "uncertainty_note": "std across generated versions (m); 0 at measured pixels"}))
    return tiles


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True)
    ap.add_argument("--versions", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--crop", type=int, default=0, help="generate only the central crop x crop pixels (faster)")
    ap.add_argument("--backup-only", action="store_true")
    ap.add_argument("--name", default="terrain_diffusion")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_mission()
    from contracts.tiles import load_tile
    tile = load_tile(tile_path(a.region))
    if a.crop:
        r0, c0 = (tile.shape[0] - a.crop) // 2, (tile.shape[1] - a.crop) // 2
        sl = (slice(r0, r0 + a.crop), slice(c0, c0 + a.crop))
        tile = Tile(region=tile.region, meters_per_pixel=tile.meters_per_pixel, height=tile.height[sl], slope=tile.slope[sl], hazard=tile.hazard[sl],
                    uncertainty=tile.uncertainty[sl], psr_mask=tile.psr_mask[sl], meta=tile.meta, center_lat_deg=tile.center_lat_deg,
                    center_lon_deg=tile.center_lon_deg, real_mask=tile.real_mask[sl], extra=tile.extra)
    model, device, source = None, None, "physics backup: gap-fill + LOLA-error noise"
    if not a.backup_only and model_path(a.name).exists():
        from worldmodel.terrain.model import load_checkpoint, pick_device
        device = pick_device(a.device)
        model, info = load_checkpoint(model_path(a.name), device)
        source = f"terrain diffusion model {info['checkpoint_id']}"
    versions = generate_versions(tile, cfg, a.versions, model, device, a.steps)
    for t in to_tiles(tile, versions, cfg, source):
        p = save_tile(t, tile_path(t.region))
        print(f"wrote {p.name}: mean spread {t.uncertainty.mean():.2f} m [{t.meta.provenance.banner}] via {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
