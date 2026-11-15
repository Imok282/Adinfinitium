"""Export compact assets for the four interactive 3D views in the "World models" section.

Run from the repo root (READ-ONLY on repo data, writes only website/assets/models3d/):
    python website/tools/export_models3d.py

Views
  a  elevation   data/tiles/lola5m_site23.npz                     observed (NASA LOLA 5 m)
  b  hazard      data/viewer_cache/lola5m_site23_hazard_<mtime>.npy  modelled (U-Net P(unsafe), cached by
                 model_viewer for data/models/hazard_unet.pt; checked against runs/hazard/eval.json below)
                 fallback if the cache is missing/stale: tile['hazard'] = backup slope rule >= 12 deg
  c  light       data/light/lola5m_site23_2027-09-01_points.npz  (SPICE DE440 Sun az / elevation)
                 + our own ray-cast of the observed DEM for each frame (modelled; window-limited)
  d  ghosts      data/tiles/lola5m_site06.npz + runs/ghosts/lander_lola5m_site06.json (modelled)

Heights are written as little-endian uint16 .bin (256 x 256), h = min + v / 65535 * (max - min).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "website" / "assets" / "models3d"
OUT.mkdir(parents=True, exist_ok=True)
N = 256              # mesh grid
TEX = 1024           # colour texture size
LIGHT_TEX = 512      # light frames
R_MOON = 1737400.0
SLOPE_LIMIT = 12.0   # config/mission.yaml lander slope limit (read below if possible)

try:
    import sys
    sys.path.insert(0, str(REPO))
    from contracts.mission import load_mission  # noqa: E402
    _cfg = load_mission()
    SLOPE_LIMIT = float(_cfg.lander.max_slope_deg) if hasattr(_cfg.lander, "max_slope_deg") else SLOPE_LIMIT
except Exception as e:  # pragma: no cover - only affects the label
    print("mission.yaml not loaded, using 12 deg:", e)


def load_tile(name: str):
    d = np.load(REPO / f"data/tiles/{name}.npz", allow_pickle=True)
    hdr = json.loads(str(d["header"]))
    return d, hdr


def block_mean(a: np.ndarray, n: int) -> np.ndarray:
    f = a.shape[0] // n
    return a[: n * f, : n * f].reshape(n, f, n, f).mean(axis=(1, 3))


def hillshade(h: np.ndarray, mpp: float, az_deg=315.0, alt_deg=35.0) -> np.ndarray:
    gy, gx = np.gradient(h, mpp)          # gy: d/drow (south), gx: d/dcol (east)
    gy = -gy                               # north-positive
    nx, ny, nz = -gx, -gy, np.ones_like(h)
    norm = np.sqrt(nx * nx + ny * ny + 1.0)
    az, alt = np.deg2rad(az_deg), np.deg2rad(alt_deg)
    sx, sy, sz = np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)
    return np.clip((nx * sx + ny * sy + nz * sz) / norm, 0, 1)


def resize(a: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((n, n), Image.BILINEAR))


def write_height(tag: str, h: np.ndarray, mpp_src: float, meta: dict) -> dict:
    hs = block_mean(h.astype(np.float64), N)
    lo, hi = float(hs.min()), float(hs.max())
    v = np.round((hs - lo) / max(hi - lo, 1e-6) * 65535).astype("<u2")
    (OUT / f"{tag}_height.bin").write_bytes(v.tobytes())
    m = {
        "n": N, "m_per_px": mpp_src * h.shape[0] / N, "size_m": mpp_src * h.shape[0],
        "min_m": lo, "max_m": hi, "encoding": "uint16 little-endian, row 0 = north, h = min + v/65535*(max-min)",
    }
    m.update(meta)
    return m


def save_jpg(name: str, rgb: np.ndarray, q=84):
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), "RGB").save(OUT / name, quality=q, optimize=True, progressive=True)


def cmap(name: str):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    return colormaps[name]


def grey_shade(shade: np.ndarray) -> np.ndarray:
    g = 0.30 + 0.62 * shade
    return np.stack([g * 0.97, g * 0.98, g * 1.02], -1)


def inset_mask(n: int, kind: str, frac: float) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n] / (n - 1) - 0.5
    if kind == "square":
        return (np.abs(x) <= frac / 2) & (np.abs(y) <= frac / 2)
    return np.hypot(x, y) <= frac / 2


def edge(mask: np.ndarray, w=2) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion
    return binary_dilation(mask, iterations=w) & ~binary_erosion(mask, iterations=w)


# ---------------------------------------------------------------- a: elevation
def view_elevation(meta_all: dict):
    d, hdr = load_tile("lola5m_site23")
    h = d["height"].astype(np.float32)
    mpp = hdr["meters_per_pixel"]
    ht = resize(h, TEX) if h.shape[0] != TEX else h
    shade = hillshade(ht, mpp * h.shape[0] / TEX)
    frac = 0.62
    m = inset_mask(TEX, "square", frac)
    inner = ht[m]
    lo, hi = np.percentile(inner, [1, 99])
    t = np.clip((ht - lo) / (hi - lo), 0, 1)
    col = cmap("rainbow")(t)[..., :3]
    colour = col * (0.45 + 0.65 * shade[..., None])
    rgb = grey_shade(shade)
    rgb[m] = np.clip(colour[m], 0, 1)
    rgb[edge(m)] = (0.95, 0.97, 1.0)
    save_jpg("a_elev_tex.jpg", rgb)
    meta_all["a"] = write_height("site23", h, mpp, {
        "provenance": "observed",
        "source": "data/tiles/lola5m_site23.npz (NASA LRO LOLA PGDA 5 m/px, Site23 Malapert massif, Barker et al. 2021; height partly interpolated by LOLA)",
        "frame": hdr["meta"]["frame"], "units": "m",
        "texture": "a_elev_tex.jpg", "inset": {"kind": "square", "frac": frac},
        "colour_range_m": [float(lo), float(hi)], "colormap": "rainbow (purple low -> red high) x hillshade",
        "laser_coverage": hdr["extra"]["laser_coverage"],
    })
    print("a: relief", float(h.min()), float(h.max()), "colour range", lo, hi)


# ---------------------------------------------------------------- b: hazard
def view_hazard(meta_all: dict):
    d, hdr = load_tile("lola5m_site23")
    h = d["height"].astype(np.float32)
    mpp = hdr["meters_per_pixel"]
    label = d["slope"] >= SLOPE_LIMIT
    ckpt = REPO / "data/models/hazard_unet.pt"
    cache = REPO / "data/viewer_cache" / f"lola5m_site23_hazard_{ckpt.stat().st_mtime_ns}.npy" if ckpt.exists() else None
    if cache is not None and cache.exists():
        p = np.load(cache).astype(np.float32)
        pred = p >= 0.5
        tp = int((pred & label).sum()); fp = int((pred & ~label).sum()); fn = int((~pred & label).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        missed = fn / max(1, tp + fn)
        ev = json.loads((REPO / "runs/hazard/eval.json").read_text())["regions"]["lola5m_site23"]["unet"]
        print(f"b: U-Net cache F1 {f1:.4f} (eval.json {ev['f1']:.4f}), missed {missed:.4f} (eval.json {ev['missed_hazard_rate']:.4f})")
        kind = "unet"
        src = (f"U-Net P(unsafe) for held-out Site 23, checkpoint data/models/hazard_unet.pt, cached by model_viewer at "
               f"data/viewer_cache/{cache.name}; recomputed F1 {f1:.3f} vs label slope >= {SLOPE_LIMIT:g} deg")
        check = {"f1": round(f1, 4), "missed_hazard_rate": round(missed, 4), "eval_json_f1": ev["f1"]}
    else:
        p = d["hazard"].astype(np.float32)
        kind = "backup"
        src = f"backup slope rule (official LOLA slope >= {SLOPE_LIMIT:g} deg), tile['hazard']"
        check = None
    ht = resize(h, TEX)
    shade = hillshade(ht, mpp * h.shape[0] / TEX)
    pt = np.clip(resize(p, TEX), 0, 1)
    frac = 0.70
    m = inset_mask(TEX, "circle", frac)
    col = cmap("RdYlBu_r")(pt)[..., :3]
    colour = col * (0.55 + 0.55 * shade[..., None])
    rgb = grey_shade(shade)
    rgb[m] = np.clip(colour[m], 0, 1)
    rgb[edge(m)] = (0.95, 0.97, 1.0)
    save_jpg("b_hazard_tex.jpg", rgb)
    meta_all["b"] = {
        "height": "site23", "texture": "b_hazard_tex.jpg", "provenance": "modelled", "kind": kind,
        "source": src, "frame": hdr["meta"]["frame"], "units": "probability 0..1 (blue 0 safe -> red 1 unsafe)",
        "inset": {"kind": "circle", "frac": frac}, "check": check, "slope_limit_deg": SLOPE_LIMIT,
    }


# ---------------------------------------------------------------- c: light
def horizon_angle_map(h: np.ndarray, mpp: float, az_deg: float, max_dist_m: float) -> np.ndarray:
    """Max elevation angle (deg) of the terrain toward compass azimuth az for every pixel (curvature included).
    Rays leaving the tile are treated as unblocked (window-limited)."""
    n = h.shape[0]
    dc, dr = np.sin(np.deg2rad(az_deg)), -np.cos(np.deg2rad(az_deg))   # col east, row south
    best = np.full(h.shape, -90.0, np.float32)
    steps = int(max_dist_m / mpp)
    for k in range(1, steps + 1):
        oc, orow = int(round(k * dc)), int(round(k * dr))
        if abs(oc) >= n or abs(orow) >= n:
            break
        dist = np.hypot(oc, orow) * mpp
        if dist <= 0:
            continue
        sh = np.full(h.shape, -1e9, np.float32)
        rs, rd = (slice(orow, n), slice(0, n - orow)) if orow >= 0 else (slice(0, n + orow), slice(-orow, n))
        cs, cd = (slice(oc, n), slice(0, n - oc)) if oc >= 0 else (slice(0, n + oc), slice(-oc, n))
        sh[rd, cd] = h[rs, cs]
        ang = np.degrees(np.arctan((sh - dist * dist / (2 * R_MOON) - h) / dist))
        np.maximum(best, ang, out=best)
    return best


def view_light(meta_all: dict):
    d, hdr = load_tile("lola5m_site23")
    h = d["height"].astype(np.float32)
    mpp = hdr["meters_per_pixel"]
    L = np.load(REPO / "data/light/lola5m_site23_2027-09-01_points.npz", allow_pickle=True)
    lh = json.loads(str(L["header"]))
    times = lh["times_utc"]
    az = L["sun_az_deg"].astype(np.float64)
    hor0 = L["horizon_elev_deg"][0].astype(np.float64)
    cell0 = L["cells"][0]
    # true Sun-centre elevation above the local horizontal at cell 0 = (elevation relative to skyline) + skyline
    hor_at = np.interp(az % 360, np.arange(361), np.append(hor0, hor0[0]))
    e_true = L["sun_elev_deg"][:, 0].astype(np.float64) + hor_at

    # check our ray-cast skyline at cell 0 (full 5 m grid) against the stored one
    my_hor = []
    for a in range(0, 360, 10):
        dc, dr = np.sin(np.deg2rad(a)), -np.cos(np.deg2rad(a))
        best = -90.0
        r0, c0 = int(cell0[0]), int(cell0[1])
        for k in range(1, 1024):
            r, c = int(round(r0 + k * dr)), int(round(c0 + k * dc))
            if not (0 <= r < 1024 and 0 <= c < 1024):
                break
            dist = np.hypot(r - r0, c - c0) * mpp
            best = max(best, np.degrees(np.arctan((h[r, c] - dist * dist / (2 * R_MOON) - h[r0, c0]) / dist)))
        my_hor.append(best)
    hor_err = float(np.mean(np.abs(np.array(my_hor) - hor0[::10])))
    print(f"c: skyline check at cell {cell0.tolist()}: mean |ours - stored| = {hor_err:.3f} deg")

    hl = resize(h, LIGHT_TEX)
    mppl = mpp * h.shape[0] / LIGHT_TEX
    gy, gx = np.gradient(hl, mppl)
    gy = -gy
    nrm = np.sqrt(gx * gx + gy * gy + 1)
    relief = hillshade(hl, mppl, 315, 35)
    n_frames = 16
    idx = np.linspace(0, len(times) - 1, n_frames + 1).round().astype(int)[:-1]
    frames = []
    disc = 0.533
    atlas = Image.new("RGB", (LIGHT_TEX * 4, LIGHT_TEX * 4))
    for fi, ti in enumerate(idx):
        a, e = float(az[ti]), float(e_true[ti])
        hmap = horizon_angle_map(hl, mppl, a, 5200.0)
        lit = np.clip((e - hmap) / disc + 0.5, 0, 1)
        sx, sy, sz = np.sin(np.deg2rad(a)) * np.cos(np.deg2rad(e)), np.cos(np.deg2rad(a)) * np.cos(np.deg2rad(e)), np.sin(np.deg2rad(e))
        lam = np.clip((-gx * sx - gy * sy + sz) / nrm, 0, 1)
        lamn = np.clip(lam / 0.6, 0, 1) ** 0.8
        litc = np.stack([0.20 + 0.80 * lamn, 0.17 + 0.74 * lamn, 0.12 + 0.62 * lamn], -1)
        shadc = np.stack([0.11 + 0.12 * relief, 0.15 + 0.15 * relief, 0.30 + 0.20 * relief], -1)
        rgb = shadc + (litc - shadc) * lit[..., None]
        im = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), "RGB")
        atlas.paste(im, ((fi % 4) * LIGHT_TEX, (fi // 4) * LIGHT_TEX))
        c_lit = float(lit[int(cell0[0] * LIGHT_TEX / 1024), int(cell0[1] * LIGHT_TEX / 1024)])
        frames.append({
            "utc": times[ti], "sun_az_deg": round(a, 2), "sun_elev_deg": round(e, 3),
            "lit_fraction_tile": round(float((lit > 0.5).mean()), 4),
            "centre_sun_fraction_ours": round(c_lit, 3), "centre_sun_fraction_stored": round(float(L["sun_fraction"][ti, 0]), 3),
        })
        print(f"c: frame {fi} {times[ti]} az {a:6.1f} el {e:5.2f} lit {frames[-1]['lit_fraction_tile']:.2f}"
              f" centre ours {c_lit:.2f} stored {frames[-1]['centre_sun_fraction_stored']:.2f}")
    atlas.save(OUT / "c_light_atlas.jpg", quality=80, optimize=True, progressive=True)
    agree = np.mean([abs(f["centre_sun_fraction_ours"] - f["centre_sun_fraction_stored"]) < 0.5 for f in frames])
    meta_all["c"] = {
        "height": "site23", "atlas": "c_light_atlas.jpg", "grid": [4, 4], "frame_px": LIGHT_TEX, "frames": frames,
        "provenance": "modelled",
        "source": ("Sun azimuth/elevation: data/light/lola5m_site23_2027-09-01_points.npz (NASA SPICE DE440, topocentric at the "
                   "tile centre cell); shadows: our ray-cast of the observed LOLA 5 m DEM resampled to 10 m, Moon curvature included"),
        "approximation": ("one Sun direction for the whole 5 km tile; terrain outside the tile is ignored (window-limited), "
                          "so some distant-ridge shadows are missing; shadow drawn dark blue with faint relief for readability"),
        "check": {"skyline_mean_abs_err_deg_vs_stored": round(hor_err, 3), "centre_lit_state_agreement": float(agree)},
        "frame": lh["meta"]["frame"], "units": "lit fraction of the Sun disc (0..1), rendered as colour",
    }


# ---------------------------------------------------------------- d: ghosts
def view_ghosts(meta_all: dict):
    d, hdr = load_tile("lola5m_site06")
    h = d["height"].astype(np.float32)
    mpp = hdr["meters_per_pixel"]
    g = json.loads((REPO / "runs/ghosts/lander_lola5m_site06.json").read_text())
    tx, ty, tz = g["target_xyz_m"]
    ht = resize(h, TEX)
    shade = hillshade(ht, mpp * h.shape[0] / TEX)
    rgb = grey_shade(shade)
    save_jpg("d_site06_tex.jpg", rgb)
    m = write_height("site06", h, mpp, {
        "provenance": "observed",
        "source": "data/tiles/lola5m_site06.npz (NASA LRO LOLA PGDA 5 m/px, Site06 Nobile rim 1, Barker et al. 2021)",
        "frame": hdr["meta"]["frame"], "units": "m",
    })
    m.update({
        "texture": "d_site06_tex.jpg",
        "texture_note": "grey hillshade of the observed DEM",
        "target_tile_xyz_m": [tx, ty, tz],
        "ghost_source": "runs/ghosts/lander_lola5m_site06.json -> website/assets/data/ghosts_site06.json (relative to target)",
        "ghost_provenance": g["provenance"],
        "failure_rates": g["failure_rates"],
        "clip_above_target_m": 3000,
    })
    meta_all["d"] = m
    print("d: target", tx, ty, tz)


if __name__ == "__main__":
    meta: dict = {"meta": {"created_by": "website/tools/export_models3d.py", "note": "see per-view provenance/source"}}
    view_elevation(meta)
    view_hazard(meta)
    view_ghosts(meta)
    view_light(meta)
    (OUT / "models3d.json").write_text(json.dumps(meta, indent=1))
    tot = sum(p.stat().st_size for p in OUT.iterdir())
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:28s} {p.stat().st_size / 1024:8.1f} KB")
    print(f"total {tot / 1e6:.2f} MB")
