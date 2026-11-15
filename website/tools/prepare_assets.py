"""Build the website's web-sized assets from the repo (read-only on the repo).

Run from the repo root:
    python website/tools/prepare_assets.py

Writes only into website/assets/. Every output is listed in website/README.md with its source.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "website" / "assets"
IMG = OUT / "img"
DATA = OUT / "data"
MODELS = OUT / "models"
for d in (IMG, DATA, MODELS):
    d.mkdir(parents=True, exist_ok=True)

BG = "#05070d"
FG = "#c9d3e6"


def moon_textures() -> None:
    """LROC colour mosaic -> 2048x1024 JPEG; LOLA LDEM -> 1024x512 bump map."""
    src = REPO / "data/textures/moon/lroc_color_poles_4k.tif"
    im = Image.open(src).convert("RGB").resize((2048, 1024), Image.LANCZOS)
    im.save(IMG / "moon_color_2k.jpg", quality=80, optimize=True, progressive=True)

    dem = Image.open(REPO / "data/textures/moon/ldem_16_uint.tif")
    a = np.asarray(dem.resize((1024, 512), Image.BILINEAR), dtype=np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    Image.fromarray((a * 255).astype(np.uint8), "L").save(IMG / "moon_bump_1k.jpg", quality=85, optimize=True)


def hazard_compare() -> None:
    """Real hazard evaluation figure (held-out Site 23) -> web JPEG."""
    src = REPO / "runs/hazard/compare_lola5m_site23.png"
    im = Image.open(src)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
    w = 1600
    bg = bg.resize((w, round(im.size[1] * w / im.size[0])), Image.LANCZOS)
    bg.save(IMG / "hazard_compare_site23.jpg", quality=82, optimize=True, progressive=True)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
        "axes.edgecolor": "#3a4660", "font.size": 10,
    })
    return plt


def terrain_site23() -> None:
    """Observed LOLA 5 m Site 23 tile: hillshade + PSR + laser coverage."""
    plt = _mpl()
    d = np.load(REPO / "data/tiles/lola5m_site23.npz", allow_pickle=True)
    h = d["height"].astype(np.float32)
    gy, gx = np.gradient(h, 5.0)
    az, alt = np.deg2rad(315), np.deg2rad(20)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3), dpi=110)
    axs[0].imshow(shade, cmap="gray", origin="upper")
    axs[0].set_title("height (hillshade), LOLA 5 m", fontsize=10)
    hz = d["slope"] >= 12.0
    rgb = np.zeros(h.shape + (3,), np.float32)
    rgb[...] = (0.10, 0.55, 0.35)
    rgb[hz] = (0.75, 0.12, 0.20)
    rgb = 0.55 * rgb + 0.45 * shade[..., None]
    psr = d["psr_mask"]
    rgb[psr] = (0.25, 0.45, 1.0)
    axs[1].imshow(rgb, origin="upper")
    axs[1].set_title("slope >= 12 deg (red), PSR (blue)", fontsize=10)
    real = d["real_mask"]
    axs[2].imshow(shade * 0.35, cmap="gray", vmin=0, vmax=1, origin="upper")
    ys, xs = np.nonzero(real[::2, ::2])
    axs[2].scatter(xs * 2, ys * 2, s=0.05, c="#ff5a3c", linewidths=0)
    axs[2].set_title(f"real laser pixels ({real.mean() * 100:.1f}% coverage)", fontsize=10)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Site 23 (Malapert massif) 5.12 x 5.12 km  -  OBSERVED: NASA LOLA PGDA 5 m/px", fontsize=10, color=FG)
    fig.tight_layout()
    fig.savefig(IMG / "terrain_site23.jpg", pil_kwargs={"quality": 82})
    plt.close(fig)


def light_site23() -> None:
    """SPICE + skyline light series for Site 23, Sept 2027 (MODELLED)."""
    plt = _mpl()
    d = np.load(REPO / "data/light/lola5m_site23_2027-09-01_points.npz", allow_pickle=True)
    hor = d["horizon_elev_deg"][0]
    t_h = (d["et_s"] - d["et_s"][0]) / 3600.0
    fig = plt.figure(figsize=(12, 4.3), dpi=110)
    ax1 = fig.add_subplot(1, 2, 1)
    az = np.arange(360)
    ax1.fill_between(az, -5, hor, color="#2b3550")
    ax1.plot(az, hor, color="#9fb3d9", lw=1, label="terrain skyline (ray-cast, LOLA)")
    ax1.plot(d["sun_az_deg"], d["sun_elev_deg"][:, 0], ".", ms=1.6, color="#ffcc55", label="Sun path (SPICE DE440)")
    ax1.plot(d["earth_az_deg"], d["earth_elev_deg"][:, 0], ".", ms=1.6, color="#5ab0ff", label="Earth path")
    ax1.set_xlim(0, 360); ax1.set_ylim(-3, max(8, float(hor.max()) + 2))
    ax1.set_xlabel("azimuth (deg)"); ax1.set_ylabel("elevation (deg)")
    ax1.set_title("Sky from one landing cell", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8, frameon=False)
    ax2 = fig.add_subplot(1, 2, 2)
    sf = d["sun_fraction"]
    im = ax2.imshow(sf.T, aspect="auto", cmap="inferno", vmin=0, vmax=1,
                    extent=[t_h[0] / 24, t_h[-1] / 24, sf.shape[1] - 0.5, -0.5])
    ax2.set_xlabel("days from 2027-09-01"); ax2.set_ylabel("candidate cell")
    ax2.set_title("Fraction of the Sun disc visible, 9 cells", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.04)
    fig.suptitle("Site 23, Sept 2027  -  MODELLED: SPICE ephemeris + skyline from observed LOLA terrain", fontsize=10, color=FG)
    fig.tight_layout()
    fig.savefig(IMG / "light_site23.jpg", pil_kwargs={"quality": 82})
    plt.close(fig)


def ghosts() -> None:
    """Downsample lander ghost trajectories (Site 06) for the finale trails."""
    src = REPO / "runs/ghosts/lander_lola5m_site06.json"
    d = json.loads(src.read_text())
    tx, ty, tz = d["target_xyz_m"]
    runs = []
    for r in d["runs"]:
        xyz = np.asarray(r["xyz_m"], dtype=np.float64)
        step = max(1, len(xyz) // 60)
        pts = xyz[::step]
        if not np.allclose(pts[-1], xyz[-1]):
            pts = np.vstack([pts, xyz[-1]])
        pts = pts - np.array([tx, ty, tz])
        runs.append({
            "policy": r["policy"], "outcome": r["outcome"],
            "miss_m": r.get("target_dist_m"),
            "p": np.round(pts, 1).tolist(),
        })
    out = {
        "meta": {
            "provenance": "modelled",
            "source": "runs/ghosts/lander_lola5m_site06.json (PyTorch lander twin, parity-tested vs sim_jax, NASA LOLA terrain)",
            "frame": "tile ENU metres relative to the target (x east, y north, z up)",
            "units": "m",
            "failure_rates": d["failure_rates"],
            "note": "failures injected at raised rates for the visual; open_loop = no guidance, shown for contrast",
        },
        "runs": runs,
    }
    (DATA / "ghosts_site06.json").write_text(json.dumps(out, separators=(",", ":")))


def numbers() -> None:
    """Collect the headline numbers the page shows into one JSON with sources."""
    bench = json.loads((REPO / "runs/lander/benchmark.json").read_text())
    hazard = json.loads((REPO / "runs/hazard/eval.json").read_text())
    light = json.loads((REPO / "runs/light/eval.json").read_text())
    out = {
        "meta": {"provenance": "modelled", "source": "runs/lander/benchmark.json, runs/hazard/eval.json, runs/light/eval.json, results_pc/extracted/runs/planner/eval.json"},
        "pilot_parameters": bench["parameters"],
        "lander_summary": bench["summary"],
        "lander_rows": bench["rows"],
        "hazard_site23": hazard["regions"]["lola5m_site23"],
        "light_site23": light["regions"]["lola5m_site23"],
    }
    p = REPO / "results_pc/extracted/runs/planner/eval.json"
    if p.exists():
        out["planner_test"] = json.loads(p.read_text())["test"]
    (DATA / "numbers.json").write_text(json.dumps(out, indent=1, default=str))


def landing_patch() -> None:
    """Flattest 128 x 128 px (640 m) window of observed Site 23 -> 8-bit heightmap PNG + range JSON."""
    d = np.load(REPO / "data/tiles/lola5m_site23.npz", allow_pickle=True)
    h, s = d["height"].astype(np.float32), d["slope"].astype(np.float32)
    n, best = 128, None
    for i in range(0, h.shape[0] - n + 1, 32):
        for j in range(0, h.shape[1] - n + 1, 32):
            c = s[i + 48:i + 80, j + 48:j + 80].mean() * 3 + s[i:i + n, j:j + n].mean()
            if best is None or c < best[0]:
                best = (c, i, j)
    _, i, j = best
    w = h[i:i + n, j:j + n]
    lo, hi = float(w.min()), float(w.max())
    img = np.round((w - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    Image.fromarray(img, "L").save(IMG / "landing_patch_site23.png", optimize=True)
    (DATA / "landing_patch_site23.json").write_text(json.dumps({
        "meta": {"provenance": "observed", "source": "data/tiles/lola5m_site23.npz (NASA LOLA PGDA 5 m/px, partly interpolated by LOLA)",
                 "frame": "tile pixel window, row 0 = north", "units": "m"},
        "row": i, "col": j, "size_px": n, "m_per_px": 5.0, "min_m": lo, "max_m": hi,
        "mean_slope_deg": float(s[i:i + n, j:j + n].mean()),
    }, indent=1))


def models() -> None:
    for name in ("lander_B.glb", "rover_B.glb"):
        shutil.copy2(REPO / "vizard/assets/blender" / name, MODELS / name.replace("_B", ""))


if __name__ == "__main__":
    moon_textures(); print("moon")
    hazard_compare(); print("hazard")
    terrain_site23(); print("terrain")
    light_site23(); print("light")
    ghosts(); print("ghosts")
    numbers(); print("numbers")
    landing_patch(); print("landing patch")
    models(); print("models")
