# ruff: noqa: C408  (plotly kwargs are idiomatic as dict(...))
"""Pure functions for the model viewer: load predictions, build the 3D figures. No Streamlit here, so they are testable.

Display frame: x = east (column * mpp), y = north (up the terrain map, row decreasing), z = height. Azimuth 0 = +y, 90 = +x, clockwise,
the same grid azimuth as the light contract. Everything is computed from files in the contract formats plus our own trained checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from contracts.mission import MissionConfig, load_mission
from contracts.paths import data_root, model_path, tile_path
from contracts.tiles import Tile, load_tile

CROP = 512                                    # central crop shown in 3D (also the zone where skyline rays stay inside the tile)
SUN, EARTH, EXACT, MODEL, BACKUP = "#E69F00", "#0072B2", "#009E73", "#D55E00", "#6b6b6b"


def available_regions() -> list[str]:
    d = data_root() / "tiles"
    return sorted(p.stem for p in d.glob("lola5m_site*.npz") if "_gen" not in p.stem) if d.exists() else []


def crop_slices(shape: tuple[int, int], size: int = CROP) -> tuple[slice, slice]:
    r0, c0 = (shape[0] - size) // 2, (shape[1] - size) // 2
    return slice(r0, r0 + size), slice(c0, c0 + size)


def hazard_predictions(tile: Tile, cfg: MissionConfig, use_model: bool = True) -> dict:
    """{'unet': P(unsafe) | None, 'backup': 0/1 slope rule on sparse input, 'label': official slope >= limit}. U-Net runs on CPU, cached on disk."""
    from worldmodel.hazard.backup import backup_hazard, label_hazard

    out = {"backup": backup_hazard(tile, cfg), "label": label_hazard(tile, cfg), "unet": None}
    ckpt = model_path("hazard_unet")
    if use_model and ckpt.exists():
        cache = data_root() / "viewer_cache" / f"{tile.region}_hazard_{ckpt.stat().st_mtime_ns}.npy"
        if cache.exists():
            out["unet"] = np.load(cache)
        else:
            import torch

            from worldmodel.hazard.backup import slope_limit_deg
            from worldmodel.hazard.features import build_features
            from worldmodel.hazard.model import load_checkpoint, predict_probability

            model, _ = load_checkpoint(ckpt, torch.device("cpu"))
            feats, _ = build_features(tile, slope_limit_deg(cfg), cfg.hazard_detector.gap_fill_sigma_px)
            out["unet"] = predict_probability(model, feats, torch.device("cpu"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, out["unet"].astype(np.float32))
    return out


def layer_array(name: str, tile: Tile, hz: dict) -> tuple[np.ndarray, str, str, tuple[float, float] | None]:
    """(array, title, colorscale, color range) for a surface colour layer."""
    if name == "Height":
        return tile.height, "height (m)", "Gray", None
    if name == "Slope (official LOLA)":
        return tile.slope, "slope (deg)", "Viridis", (0, 40)
    if name == "Hazard: U-Net probability":
        if hz["unet"] is None:
            raise ValueError("no hazard checkpoint: train with  python -I -m worldmodel.hazard.train")
        return hz["unet"], "P(unsafe) U-Net", "RdYlGn_r", (0, 1)
    if name == "Hazard: slope-rule backup":
        return hz["backup"], "unsafe (slope rule, sparse input)", "RdYlGn_r", (0, 1)
    if name == "Hazard: official label":
        return hz["label"], "unsafe (official slope map)", "RdYlGn_r", (0, 1)
    if name == "U-Net vs label (errors)":
        if hz["unet"] is None:
            raise ValueError("no hazard checkpoint")
        err = (hz["unet"] >= 0.5).astype(float) - (hz["label"] > 0.5).astype(float)      # +1 false alarm, -1 missed hazard
        return err, "U-Net error: red = missed hazard, blue = false alarm", "RdBu", (-1, 1)
    if name == "Backup vs label (errors)":
        err = (hz["backup"] > 0.5).astype(float) - (hz["label"] > 0.5).astype(float)
        return err, "backup error: red = missed hazard, blue = false alarm", "RdBu", (-1, 1)
    raise ValueError(name)


def to_xy(rc: np.ndarray, shape: tuple[int, int], mpp: float, rows_off: int = 0, cols_off: int = 0) -> np.ndarray:
    """(row, col) in the full tile -> display (x east, y north) in metres of the cropped scene (crop origin at the south-west corner)."""
    rc = np.atleast_2d(np.asarray(rc, dtype=float))
    return np.stack([(rc[:, 1] - cols_off) * mpp, (shape[0] - 1 - (rc[:, 0] - rows_off)) * mpp], axis=1)


def surface_figure(tile: Tile, layer: tuple[np.ndarray, str, str, tuple[float, float] | None], z_exag: float = 2.0, ds: int = 2,
                   marks: list[dict] | None = None, title: str = "") -> go.Figure:
    """3D terrain surface of the central crop coloured by `layer`. `marks` = extra Scatter3d traces (rings, rays, cells)."""
    rs, cs = crop_slices(tile.shape, min(CROP, *tile.shape))
    arr, ttl, scale, rng = layer
    h = tile.height[rs, cs][::ds, ::ds].astype(float)
    colour = arr[rs, cs][::ds, ::ds].astype(float)
    mpp = tile.meters_per_pixel * ds
    ny, nx = h.shape
    x = np.arange(nx) * mpp
    y = (ny - 1 - np.arange(ny)) * mpp
    z0 = float(np.nanmean(h))
    fig = go.Figure(go.Surface(x=x, y=y, z=(h - z0) * z_exag, surfacecolor=colour, colorscale=scale, cmin=rng[0] if rng else None,
                               cmax=rng[1] if rng else None, colorbar=dict(title=ttl, len=0.6), showscale=True,
                               lighting=dict(ambient=0.55, diffuse=0.9, specular=0.1, roughness=0.9), hovertemplate="x %{x:.0f} m<br>y %{y:.0f} m<extra></extra>"))
    for m in marks or []:
        fig.add_trace(go.Scatter3d(**m))
    fig.update_layout(title=title, height=620, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=0.02),
                      scene=dict(aspectmode="data", xaxis_title="east (m)", yaxis_title="north (m)", zaxis_title=f"height x{z_exag:g}",
                                 camera=dict(eye=dict(x=1.2, y=-1.5, z=1.0))))
    return fig


def skyline_marks(tile: Tile, cell_rc: np.ndarray, skylines: dict[str, np.ndarray], z_exag: float, ring_m: float = 350.0,
                  directions: dict | None = None) -> list[dict]:
    """3D traces: the candidate cell, one skyline ring per method, and (optionally) Sun / Earth direction rays."""
    rs, cs = crop_slices(tile.shape, min(CROP, *tile.shape))
    h_crop = tile.height[rs, cs]
    z0 = float(np.nanmean(h_crop[::2, ::2]))
    xy = to_xy(cell_rc, h_crop.shape, tile.meters_per_pixel, rs.start, cs.start)[0]
    from scipy.ndimage import map_coordinates
    hc = float(map_coordinates(tile.height.astype(float), [[cell_rc[0]], [cell_rc[1]]], order=1, mode="nearest")[0])
    zc = (hc - z0) * z_exag
    out = [dict(x=[xy[0]], y=[xy[1]], z=[zc], mode="markers", marker=dict(size=7, color="#fff", line=dict(width=2, color="#000")), name="cell")]
    az = np.radians(np.arange(360))
    for name, sky, color in ((k, v, {"exact (5 m ray-cast)": EXACT, "model": MODEL, "physics backup (coarse)": BACKUP}.get(k, "#999")) for k, v in skylines.items()):
        zr = zc + ring_m * np.tan(np.radians(sky)) * z_exag
        xr, yr = xy[0] + ring_m * np.sin(az), xy[1] + ring_m * np.cos(az)
        out.append(dict(x=np.append(xr, xr[0]), y=np.append(yr, yr[0]), z=np.append(zr, zr[0]), mode="lines", line=dict(color=color, width=5), name=f"skyline: {name}"))
    if directions:
        for name, (azd, eld), color in (("Sun", directions["sun"], SUN), ("Earth", directions["earth"], EARTH)):
            d = np.array([np.sin(np.radians(azd)) * np.cos(np.radians(eld)), np.cos(np.radians(azd)) * np.cos(np.radians(eld)), np.sin(np.radians(eld))])
            tip = np.array([xy[0], xy[1], zc]) + 450.0 * d * np.array([1, 1, z_exag])
            out.append(dict(x=[xy[0], tip[0]], y=[xy[1], tip[1]], z=[zc, tip[2]], mode="lines+markers", line=dict(color=color, width=7), marker=dict(size=[0, 9], color=color),
                            name=f"{name} direction"))
    return out


def skyline_figure(skylines: dict[str, np.ndarray], directions: dict | None = None) -> go.Figure:
    colors = {"exact (5 m ray-cast)": EXACT, "model": MODEL, "physics backup (coarse)": BACKUP}
    fig = go.Figure()
    for name, sky in skylines.items():
        fig.add_trace(go.Scatter(x=np.arange(len(sky)), y=sky, mode="lines", name=name, line=dict(color=colors.get(name, "#999"), width=2.5 if name != "model" else 2)))
    if directions:
        for name, (azd, eld), color in (("Sun", directions["sun"], SUN), ("Earth", directions["earth"], EARTH)):
            fig.add_trace(go.Scatter(x=[azd % 360], y=[eld], mode="markers", name=name, marker=dict(size=14, color=color, line=dict(width=2, color="#222"))))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=30), xaxis=dict(title="grid azimuth (0 = up the map, clockwise)", range=[0, 360], dtick=45),
                      yaxis_title="elevation of the skyline (deg)", legend=dict(orientation="h", y=-0.3))
    return fig


def load_directions(hour: int) -> tuple[dict | None, str]:
    """Sun and Earth (grid azimuth, absolute elevation) at an hour of the first light point-series file found (GENERATED stand-in until KAN-23)."""
    from contracts.light import load_points

    files = sorted((data_root() / "light").glob("*_points.npz")) if (data_root() / "light").exists() else []
    if not files:
        return None, "no light file in data/light"
    ps = load_points(files[0])
    i = min(max(int(hour), 0), len(ps.times_utc) - 1)
    sun, earth = ps.absolute_elev_deg("sun")[i, 0], ps.absolute_elev_deg("earth")[i, 0]
    return ({"sun": (float(ps.sun_az_deg[i]), float(sun)), "earth": (float(ps.earth_az_deg[i]), float(earth))},
            f"{ps.times_utc[i]}  [{ps.meta.provenance.banner}] from {files[0].name}")


def load_tile_cached(region: str) -> Tile:
    return load_tile(tile_path(region))


def default_cfg() -> MissionConfig:
    return load_mission()


def light_checkpoint_exists() -> bool:
    return Path(model_path("light_skyline")).exists()


def crop_tile(tile: Tile, size: int) -> Tile:
    """Central size x size crop of a tile (same contract object)."""
    rs, cs = crop_slices(tile.shape, size)
    return Tile(region=tile.region, meters_per_pixel=tile.meters_per_pixel, height=tile.height[rs, cs], slope=tile.slope[rs, cs], hazard=tile.hazard[rs, cs],
                uncertainty=tile.uncertainty[rs, cs], psr_mask=tile.psr_mask[rs, cs], meta=tile.meta, center_lat_deg=tile.center_lat_deg,
                center_lon_deg=tile.center_lon_deg, real_mask=None if tile.real_mask is None else tile.real_mask[rs, cs], extra=tile.extra)
