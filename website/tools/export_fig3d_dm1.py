"""3D figure data for site DM1 (research/rover.html and research/end-to-end.html, js/fig3d/dm1.js).

Usage (needs numpy + Pillow; rasterio only when the DEM window is fetched):
    ~/venvs/adi/bin/python -I website/tools/export_fig3d_dm1.py website/assets/data/rover_route_dm1_v31.json \
        website/assets/fig3d/dm1 [--dem dm1_surf.npy]

Inputs
- Terrain: NASA LRO LOLA PGDA 5 m/px DM1 (Barker et al. 2021), the 1024 x 1024 centre window of
  https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/DM1/DM1_final_adj_5mpp_surf.tif, read exactly as worldmodel.hazard.lola
  reads it (centre window, float32). Without --dem the window is fetched over HTTP (about 16 MB of range reads).
  Check: min/max/mean must equal the tile stats stored in records/final/e2e_result_final_v31.json.
- Rover world = the tile's 512 px centre crop (world.origin_rc [256, 256], lander_rc [255.5, 255.5] in that record).
- Route, hazard cells, PSR-edge targets, battery and fleet counts: rover_route_dm1_v31.json (tools/export_rover_e2e.py,
  from e2e_arrays_final_v31.npz of scripts/e2e_test_v31.py --site DM1 --date 2027-06-01 --n 512 --seed 1234).
  Check: the route's stored heights (relative to the lander cell) must match the DEM at the lander = world centre.

Outputs (in <out>): height.bin (256 x 256 uint16, 10 m/px, models3d encoding), relief.jpg, hazard.jpg (512 px), dm1.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PX = 5.0
TILE_STATS = {"min": 51.972877502441406, "max": 672.5637817382812, "mean": 374.7706286245957}  # e2e_result_final_v31.json tile.height
URL = "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/DM1/DM1_final_adj_5mpp_surf.tif"
E2E_CMD = "python scripts/e2e_test_v31.py --site DM1 --date 2027-06-01 --n 512 --seed 1234"


def fetch_window():
    import rasterio
    from rasterio.windows import Window
    last = None
    for att in range(6):
        try:
            env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                               GDAL_HTTP_TIMEOUT="120", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")
            with env, rasterio.open("/vsicurl/" + URL) as r:
                s = 1024
                c0, r0 = (r.width - s) // 2, (r.height - s) // 2
                return r.read(1, window=Window(c0, r0, s, s)).astype(np.float32)
        except Exception as e:  # noqa: BLE001 - flaky range reads: retry
            last = e
            time.sleep(4 * (att + 1))
    raise RuntimeError(f"could not read DM1 window: {last}")


def hillshade(h, mpp, az=315.0, alt=35.0):
    gy, gx = np.gradient(h, mpp)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, e = np.radians(az), np.radians(alt)
    s = np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(s, 0, 1)


def main(route_json, out, dem=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tile = np.load(dem) if dem else fetch_window()
    tile = tile.astype(np.float64)
    stats = {"min": float(tile.min()), "max": float(tile.max()), "mean": float(tile.mean())}
    for k, v in TILE_STATS.items():
        assert abs(stats[k] - v) < 1e-3, f"DEM window differs from the e2e tile ({k} {stats[k]} vs {v})"
    W = tile[256:768, 256:768]                       # rover world (512 px, 5 m)
    lr = lc = 255.5
    h0 = float(W[int(round(lr)), int(round(lc))])
    R = json.load(open(route_json))

    # alignment check: stored route heights vs DEM with the lander at the world centre
    P = np.array(R["points"])
    rr = np.clip(np.round(lr + P[:, 2] / PX).astype(int), 0, 511); cc = np.clip(np.round(lc + P[:, 0] / PX).astype(int), 0, 511)
    route_err = float(np.abs(W[rr, cc] - h0 - P[:, 1]).mean())
    assert route_err < 0.2, f"route heights do not match the DEM (mean abs err {route_err:.3f} m)"

    # heights: 2 x 2 block mean -> 256 x 256 at 10 m, row 0 = north (world row 0 is north: z south = +row)
    Hd = W.reshape(256, 2, 256, 2).mean(axis=(1, 3))
    lo, hi = float(Hd.min()), float(Hd.max())
    q = np.round((Hd - lo) / (hi - lo) * 65535).astype("<u2")
    (out / "height.bin").write_bytes(q.tobytes())

    # textures (512 px, 5 m): warm-grey hillshade; hazard = same with U-Net >= 0.5 blocks in red
    hs = hillshade(W, PX)
    lo_s, hi_s = np.percentile(hs, [1, 99]); hs = np.clip((hs - lo_s) / (hi_s - lo_s), 0, 1)   # flat site: stretch contrast
    base = 0.22 + 0.70 * hs
    grey = np.stack([base * 0.98, base * 0.95, base * 0.88], -1)
    Image.fromarray((grey * 255).astype(np.uint8)).save(out / "relief.jpg", quality=84, optimize=True)
    haz = np.zeros((512, 512), bool)
    for x, z in R["hazard_cells_xz"]:                 # 10 m block centres, metres from the lander (x east, z south)
        c = int(round((x / PX + lc - 0.5) / 2)); r = int(round((z / PX + lr - 0.5) / 2))
        haz[2 * r:2 * r + 2, 2 * c:2 * c + 2] = True
    red = np.array([0.90, 0.42, 0.38])
    hz = grey.copy(); hz[haz] = grey[haz] * 0.25 + red * (0.55 + 0.35 * hs[haz, None])
    Image.fromarray((np.clip(hz, 0, 1) * 255).astype(np.uint8)).save(out / "hazard.jpg", quality=84, optimize=True)

    # route + targets in tile ENU metres (x east, y north) with absolute heights
    T = np.array(R["ice_targets"])
    dT = np.hypot(T[:, 0], T[:, 2])
    ti = int(np.argmin(np.abs(dT - R["target_distance_m"])))
    meta = {
        "meta": {
            "created_by": "website/tools/export_fig3d_dm1.py",
            "provenance": {"terrain": "observed", "hazard": "modelled", "route": "modelled (simulated)"},
            "terrain_source": f"NASA LRO LOLA PGDA 5 m/px DM1 (Barker et al. 2021), centre 1024 px window of {URL}; rover world = its 512 px centre crop (2.56 km), shown at 10 m/px",
            "terrain_check": {"tile_stats_ours": stats, "tile_stats_e2e_record": TILE_STATS, "route_height_mean_abs_err_m": round(route_err, 3)},
            "hazard_source": "hazard_unet_big (60 M U-Net, sha256 91393b30...) P(unsafe) >= 0.5, any 5 m pixel in a 10 m block, from the e2e arrays via rover_route_dm1_v31.json",
            "route_source": R["meta"]["source"],
            "e2e_command": E2E_CMD,
            "frame": "tile ENU metres: x east, y north (origin = lander = world centre), z = LOLA height (m)",
            "note": R["meta"]["note"],
        },
        "terrain": {"n": 256, "m_per_px": 10.0, "size_m": 2560.0, "min_m": lo, "max_m": hi,
                    "encoding": "uint16 little-endian, row 0 = north, h = min + v/65535*(max-min)"},
        "lander": [0.0, 0.0, h0],
        "route": {
            "xy": [[p[0], -p[2]] for p in R["points"]],
            "battery_wh": R["battery_wh"], "battery_cap_wh": R["battery_cap_wh"],
            "dist_m": R["dist_m"], "sun_frac": R["sun_frac"], "dt_s": R["dt_s"],
            "total_dist_m": R["total_dist_m"], "ice_reach_index": R["ice_reach_index"],
            "min_distance_to_target_m": R["min_distance_to_target_m"], "end_distance_from_lander_m": R["end_distance_from_lander_m"],
            "outcome": R["outcome"], "run_id": R["run_id"], "policy": R["policy"], "date": R["date"],
        },
        "targets": [[t[0], -t[2], h0 + t[1]] for t in R["ice_targets"]],
        "target_index": ti,
        "ice_source": R["ice_source"],
        "hazard_rule": R["hazard_rule"],
        "hazard_cells": int(haz.sum() // 4),
        "fleet": R["fleet"],
    }
    (out / "dm1.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"route err {route_err:.3f} m · target {ti} at {dT[ti]:.1f} m · heights {lo:.1f}..{hi:.1f} m · hazard cells {meta['hazard_cells']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    dem = None
    if "--dem" in a:
        i = a.index("--dem"); dem = a[i + 1]; del a[i:i + 2]
    main(a[0], a[1], dem)
