"""Export the rover's real planned route + the landing terrain around it for the website's #rover beat.

Source run: runs/ghosts/rover_lola5m_site23_2027-09-01.json, the run flagged "best" (cautious_15: reached an
ice target (PSR edge) and returned to the lander). Route is in tile ENU metres (origin = tile centre = lander).
Writes (all small):
  website/assets/data/rover_route_site23.json     route points (scene frame: x east, z south, y up, metres from lander),
                                                   times, per-step battery + sunlight (light re-simulation of the same
                                                   seeded batch, checked point by point against the exported path),
                                                   ice target cells, hazard cells near the route.
  website/assets/img/landing_near_site23.png       1280 m (256 px, 5 m/px) LOLA heights centred on the lander, 16 bit
                                                   in R (high byte) + G (low byte), official slope in B (deg * 8).
  website/assets/img/landing_wide_site23.png       5120 m (whole tile, 256 px at 20 m/px), same encoding.
  website/assets/data/landing_site23.json          ranges + meta for both.
  website/assets/img/rover_route_site23.png        static top-down route image (reduced motion / ?static).
Run:  ~/venvs/adi/bin/python website/tools/export_rover_route.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "website"
DATA, IMG = WEB / "assets/data", WEB / "assets/img"
GHOST = REPO / "runs/ghosts/rover_lola5m_site23_2027-09-01.json"
TILE = REPO / "data/tiles/lola5m_site23.npz"
HAZ_DEG = 12.0          # same rule as the tile's backup hazard layer (official slope >= 12 deg)


def enc16(h: np.ndarray, s: np.ndarray, lo: float, hi: float) -> Image.Image:
    q = np.round((h - lo) / max(hi - lo, 1e-6) * 65535).astype(np.uint32)
    rgb = np.stack([(q >> 8).astype(np.uint8), (q & 255).astype(np.uint8), np.clip(s * 8, 0, 255).astype(np.uint8)], -1)
    return Image.fromarray(rgb, "RGB")


def resim(run_id: str, n: int, xyz: np.ndarray):
    """Re-run the same seeded batch (cautious planner) to recover per-step battery and sunlight. Cheap: light cache."""
    try:
        import torch
        from contracts.mission import load_mission
        from worldmodel.rover.env import RoverEnv
        from worldmodel.rover.world import build_world
        cfg = load_mission()
        w = build_world("lola5m_site23", "2027-09-01", cfg, device=torch.device("cpu"))
        env = RoverEnv(w, n, torch.device("cpu"), cfg, seed=7)
        env.reset()
        i = int(run_id.split("_")[-1])
        bat, sun, pos = [float(env.battery[i])], [], [env.pos[i].clone().numpy()]
        for _ in range(len(xyz) - 1):
            with torch.no_grad():
                a = env.planner_action(turn_back=True)
            s = env._sample_coarse(env._slice(env.sun), env.pos)
            sun.append(float(s[i]))
            env.step(a)
            bat.append(float(env.battery[i])); pos.append(env.pos[i].clone().numpy())
        r0, c0 = w.origin_rc
        P = np.array(pos)
        return bat, sun + [sun[-1] if sun else 0.0], P, (r0, c0), cfg.rover.battery_wh
    except Exception as e:  # noqa: BLE001
        print("re-simulation skipped:", e)
        return None


def main() -> None:
    g = json.loads(GHOST.read_text())
    run = next(r for r in g["runs"] if r["best"])
    xyz = np.array(run["xyz_m"], dtype=np.float64)
    t = np.load(TILE, allow_pickle=True)
    hdr = json.loads(str(t["header"]))
    H, S = t["height"].astype(np.float32), t["slope"].astype(np.float32)
    n = H.shape[0]
    c = n // 2
    lander = np.array(g["lander_xyz_m"])
    # near patch: 256 px (1280 m) centred on the tile centre (= lander, ENU origin)
    k = 128
    near_h, near_s = H[c - k:c + k, c - k:c + k], S[c - k:c + k, c - k:c + k]
    wide_h = H.reshape(256, 4, 256, 4).mean((1, 3)); wide_s = S.reshape(256, 4, 256, 4).max((1, 3))
    lo, hi = float(min(near_h.min(), wide_h.min())), float(max(near_h.max(), wide_h.max()))
    enc16(near_h, near_s, lo, hi).save(IMG / "landing_near_site23.png", optimize=True)
    enc16(wide_h, wide_s, lo, hi).save(IMG / "landing_wide_site23.png", optimize=True)
    meta = {"provenance": "observed", "source": "data/tiles/lola5m_site23.npz (NASA LOLA PGDA 5 m/px Site 23, Malapert massif; partly interpolated by LOLA)",
            "frame": "scene: origin = lander (tile centre), x east, z south, y up; PNG row 0 = north", "units": "m"}
    (DATA / "landing_site23.json").write_text(json.dumps({
        "meta": meta, "center_lat_deg": hdr["center_lat_deg"], "center_lon_deg": hdr["center_lon_deg"],
        "height_lo_m": lo, "height_hi_m": hi, "lander_height_m": float(H[c, c]),
        "encoding": "R*256+G -> 0..65535 maps to height_lo..height_hi; B = slope deg * 8",
        "near": {"png": "assets/img/landing_near_site23.png", "size_px": 256, "m_per_px": 5.0},
        "wide": {"png": "assets/img/landing_wide_site23.png", "size_px": 256, "m_per_px": 20.0},
        "hazard_rule": f"official LOLA slope >= {HAZ_DEG} deg (modelled hazard backup)",
    }, indent=1))

    # route -> scene frame (x east, z south, y = LOLA height - lander height)
    pts = np.stack([xyz[:, 0], xyz[:, 2] - H[c, c], -xyz[:, 1]], 1)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 2]))
    dist = np.concatenate([[0], np.cumsum(seg)])
    ice = np.array(g["ice_targets_xyz_m"])
    ice_s = np.stack([ice[:, 0], ice[:, 2] - H[c, c], -ice[:, 1]], 1)
    d_ice = np.min(np.hypot(pts[:, None, 0] - ice_s[None, :, 0], pts[:, None, 2] - ice_s[None, :, 2]), 1)
    reach_i = int(np.argmin(d_ice))
    # hazard cells near the route (inside the near patch), 10 m cells, for the steering overlay
    rr, cc = np.where(near_s >= HAZ_DEG)
    hx, hz = (cc - k + 0.5) * 5.0, (rr - k + 0.5) * 5.0
    near_route = np.min(np.hypot(hx[:, None] - pts[None, :, 0], hz[:, None] - pts[None, :, 2]), 1) < 60
    haz = np.stack([hx[near_route], hz[near_route]], 1)

    out = {"meta": {"provenance": "modelled", "source": f"{GHOST.relative_to(REPO)} run {run['id']} (policy {run['policy']}, outcome {run['outcome']})",
                    "frame": meta["frame"], "units": "m, s, Wh",
                    "note": "MODELLED rover environment on NASA LOLA terrain + SPICE/skyline light; ice targets = PSR edges (dark and cold, NOT confirmed ice)"},
           "run_id": run["id"], "policy": run["policy"], "outcome": run["outcome"], "date": g["date"], "dt_s": g["dt_s"],
           "t_s": run["t_s"], "points": np.round(pts, 2).tolist(), "dist_m": np.round(dist, 1).tolist(),
           "total_dist_m": round(float(dist[-1]), 1), "duration_h": round(run["t_s"][-1] / 3600, 1),
           "max_range_m": round(float(np.hypot(pts[:, 0], pts[:, 2]).max()), 1),
           "ice_reach_index": reach_i, "ice_reach_t_s": run["t_s"][reach_i], "ice_targets": np.round(ice_s, 1).tolist(),
           "ice_source": g["ice_source"], "hazard_cells_xz": np.round(haz, 1).tolist(), "hazard_rule": f"slope >= {HAZ_DEG} deg"}
    rs = resim(run["id"], 16, xyz)
    if rs:
        bat, sun, P, (r0, c0), cap = rs
        # check the re-simulated path against the exported one (tile ENU metres, same formula as worldmodel/ghosts.py)
        ex = np.stack([(P[:, 1] + c0) - (n - 1) / 2.0, (n - 1) / 2.0 - (P[:, 0] + r0)], 1) * 5.0
        err = float(np.abs(ex - xyz[:, :2]).max()) if P.shape[0] == len(xyz) else 1e9
        print("re-sim path max deviation (m):", err)
        if err < 1.0:
            out["battery_wh"] = np.round(bat, 2).tolist(); out["battery_cap_wh"] = cap
            out["sun_frac"] = np.round(sun, 3).tolist()
            out["battery_source"] = "light re-simulation of the same seeded batch (worldmodel.rover.env, cautious planner); path matches the exported run"
    (DATA / "rover_route_site23.json").write_text(json.dumps(out, separators=(",", ":")))

    # static route image: wide hillshade-ish crop of the near patch + route + targets
    gy, gx = np.gradient(near_h, 5.0)
    shade = np.clip(0.55 + (gx * 0.7 - gy * 0.4) * 0.9, 0, 1)
    im = Image.fromarray((shade * 200 + 30).astype(np.uint8), "L").convert("RGB").resize((768, 768), Image.BILINEAR)
    hazm = Image.fromarray(((near_s >= HAZ_DEG) * 150).astype(np.uint8), "L").resize((768, 768))
    im.paste(Image.new("RGB", im.size, (230, 80, 70)), (0, 0), hazm.point(lambda v: int(v * 0.55)))
    dr = ImageDraw.Draw(im)
    to = lambda x, zz: ((x / 5.0 + k) * 3, (zz / 5.0 + k) * 3)
    for x, _, zz in ice_s:
        if abs(x) < 640 and abs(zz) < 640:
            a, b = to(x, zz); dr.ellipse([a - 7, b - 7, a + 7, b + 7], outline=(110, 200, 255), width=3)
    dr.line([to(p[0], p[2]) for p in pts], fill=(255, 181, 71), width=3)
    a, b = to(0, 0); dr.rectangle([a - 6, b - 6, a + 6, b + 6], fill=(255, 255, 255))
    im.save(IMG / "rover_route_site23.png", optimize=True)
    print(run["id"], run["outcome"], "dist", out["total_dist_m"], "h", out["duration_h"], "range", out["max_range_m"], "haz cells", len(haz))


if __name__ == "__main__":
    main()
