"""Model viewer (trial): what our models predict, in 3D, on real NASA LOLA terrain.

    streamlit run model_viewer/app.py --server.port 8504

Runs the models on CPU (so it does not fight GPU training). Tabs: terrain + hazard (U-Net vs slope-rule backup vs the official map),
skyline (light model vs exact ray-cast vs physics backup, with Sun / Earth directions), generated terrain versions (GENERATED).
This is a harness for looking at predictions, not a safety tool: hazard maps and skylines here are MODELLED, generated terrain is GENERATED.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from model_viewer import scene as sc

st.set_page_config(page_title="Model viewer (trial)", page_icon="🛰️", layout="wide")
cfg = sc.default_cfg()


@st.cache_resource(show_spinner="Loading terrain…")
def tile_for(region: str):
    return sc.load_tile_cached(region)


@st.cache_resource(show_spinner="Running the hazard U-Net on the tile (CPU, cached after the first time)…")
def hazard_for(region: str):
    return sc.hazard_predictions(tile_for(region), cfg)


@st.cache_resource(show_spinner=False)
def student():
    import torch

    from contracts.paths import model_path
    from worldmodel.light.model import SkylineStudent, load_checkpoint
    if not sc.light_checkpoint_exists():
        return SkylineStudent(None, cfg, torch.device("cpu"))
    model, info = load_checkpoint(model_path("light_skyline"), torch.device("cpu"))
    return SkylineStudent(model, cfg, torch.device("cpu"), info)


regions = sc.available_regions()
st.title("🛰️ Model viewer (trial)")
st.caption("Real NASA LOLA terrain, our models' predictions next to their physics baselines. "
           "Hazard and skyline maps are **MODELLED**; terrain versions are **GENERATED — NOT OBSERVED**. Not a safety tool.")
if not regions:
    st.error("No terrain tiles in `data/tiles/`. Create them with `python -I -m worldmodel.hazard.lola all`.")
    st.stop()

with st.sidebar:
    st.header("Scene")
    region = st.selectbox("Site (real LOLA 5 m window)", regions, index=regions.index("lola5m_site23") if "lola5m_site23" in regions else 0)
    z_exag = st.slider("Vertical exaggeration", 1.0, 6.0, 2.0, 0.5)
    ds = st.select_slider("Mesh detail", options=[4, 2, 1], value=2, format_func=lambda v: {4: "low", 2: "medium", 1: "high (slow)"}[v])
    tile = tile_for(region)
    st.caption(f"{tile.shape[0]}x{tile.shape[1]} px at {tile.meters_per_pixel:g} m, laser coverage {tile.extra.get('laser_coverage', float('nan')):.0%}. "
               f"Shown: the central {sc.CROP} px ({sc.CROP * tile.meters_per_pixel / 1000:.1f} km).")
    if region == "lola5m_site23":
        st.success("Site 23 is the HELD-OUT test region: neither the hazard U-Net nor the light model ever trained on it.")
    else:
        st.warning("This site was used for training or validation. Judge the models on Site 23.")

tab_h, tab_l, tab_g, tab_a = st.tabs(["Terrain + hazard", "Skyline (light model)", "Generated terrain", "About"])

# ------------------------------------------------------------------------------------------------------ hazard
with tab_h:
    hz = hazard_for(region)
    names = ["Height", "Slope (official LOLA)", "Hazard: U-Net probability", "Hazard: slope-rule backup", "Hazard: official label",
             "U-Net vs label (errors)", "Backup vs label (errors)"]
    layer_name = st.radio("Colour the terrain by", names, index=2 if hz["unet"] is not None else 3, horizontal=True)
    try:
        layer = sc.layer_array(layer_name, tile, hz)
        st.plotly_chart(sc.surface_figure(tile, layer, z_exag, ds, title=f"{region}: {layer_name}"), width="stretch")
    except ValueError as e:
        st.error(str(e))
    from worldmodel.hazard.metrics import confusion
    rs, cs = sc.crop_slices(tile.shape)
    lab = hz["label"][rs, cs] > 0.5
    cols = st.columns(3 if hz["unet"] is not None else 2)
    for col, (nm, pred) in zip(cols, [("Slope-rule backup", hz["backup"][rs, cs] > 0.5)] + ([("U-Net", hz["unet"][rs, cs] >= 0.5)] if hz["unet"] is not None else [])):
        m = confusion(pred, lab)
        col.metric(nm, f"F1 {m['f1']:.3f}", f"misses {m['missed_hazard_rate']:.1%} of hazards", delta_color="off")
        col.caption(f"precision {m['precision']:.3f} · recall {m['recall']:.3f}")
    cols[-1].caption("Scored on the central crop against NASA's official slope >= 12 deg, from sparse laser pixels only. "
                     "Label is NASA's slope map (made from interpolated heights): 'recovers the official hazard map', not proof of safety.")

# ------------------------------------------------------------------------------------------------------ skyline
with tab_l:
    from worldmodel.light.teacher import raycast_skyline_numpy
    half = sc.CROP // 2 - 12
    c1, c2, c3 = st.columns(3)
    dr = c1.slider("Cell: north-south offset from centre (px)", -half, half, 0, 8)
    dc = c2.slider("Cell: east-west offset from centre (px)", -half, half, 0, 8)
    hour = c3.slider("Hour in the 29.5-day window (Sun/Earth direction)", 0, 708, 120, 6)
    cell = np.array([(tile.shape[0] - 1) / 2 + dr, (tile.shape[1] - 1) / 2 + dc])
    lm = cfg.light_model
    exact = raycast_skyline_numpy(tile.height, tile.meters_per_pixel, cell[None], cfg.moon.radius_m, lm.max_ray_range_m, lm.n_azimuth)[0]
    st_ = student()
    base = st_.skyline(tile, cell[None], use_model=False)[0]
    skylines = {"exact (5 m ray-cast)": exact, "physics backup (coarse)": base}
    if st_.model is not None:
        skylines["model"] = st_.skyline(tile, cell[None], use_model=True)[0]
    directions, dir_label = sc.load_directions(hour)
    m1, m2, m3 = st.columns(3)
    m1.metric("Backup error vs exact", f"{np.abs(base - exact).mean():.2f}° mean", f"max {np.abs(base - exact).max():.1f}°", delta_color="off")
    if "model" in skylines:
        e = np.abs(skylines["model"] - exact)
        m2.metric("Model error vs exact", f"{e.mean():.2f}° mean", f"max {e.max():.1f}°", delta_color="off")
    else:
        m2.info("No light checkpoint yet (training).")
    m3.caption(f"Sun / Earth: {dir_label}. Stand-in directions are analytic, NOT SPICE.")
    marks = sc.skyline_marks(tile, cell, skylines, z_exag, directions=directions)
    left, right = st.columns([3, 2])
    left.plotly_chart(sc.surface_figure(tile, sc.layer_array("Height", tile, {}), z_exag, ds, marks, title="Skyline rings drawn 350 m around the cell"), width="stretch")
    right.plotly_chart(sc.skyline_figure(skylines, directions), width="stretch")
    right.caption("The model predicts the terrain skyline only. Time enters through exact Sun/Earth directions; the geometry that turns skyline + "
                  "direction into sunlight and Earth contact is plain physics. A body above the skyline curve is visible.")

# ------------------------------------------------------------------------------------------------------ generated
with tab_g:
    from contracts.paths import model_path
    from worldmodel.terrain.generate import generate_versions
    st.error("**GENERATED — NOT OBSERVED.** These are imagined versions of terrain, drawn only where LOLA has no laser measurement. "
             "Measured pixels are kept exactly. Never use them alone for a safety claim.", icon="🚫")
    have_model = model_path("terrain_diffusion").exists()
    method = st.radio("Generator", ["Physics backup (gap-fill + LOLA error noise)"] + (["Diffusion model"] if have_model else []), horizontal=True)
    if not have_model:
        st.caption("The diffusion model is not trained yet; showing the physics backup only.")
    nver = st.slider("Versions", 2, 6, 3)
    size = st.select_slider("Generated crop (px)", options=[128, 192, 256], value=192)

    @st.cache_resource(show_spinner="Generating versions…")
    def versions_for(region_: str, method_: str, n: int, size_: int):
        t = sc.crop_tile(tile_for(region_), size_)
        if method_.startswith("Diffusion"):
            import torch

            from worldmodel.terrain.model import load_checkpoint
            model, _ = load_checkpoint(model_path("terrain_diffusion"), torch.device("cpu"))
            return t, generate_versions(t, cfg, n, model, torch.device("cpu"), steps=20)
        return t, generate_versions(t, cfg, n, None)

    if st.button("Generate", type="primary") or st.session_state.get("gen_done"):
        st.session_state["gen_done"] = True
        gt, vers = versions_for(region, method, nver, size)
        k = st.slider("Show version", 0, nver - 1, 0)
        what = st.radio("Colour by", ["Version height", "Uncertainty: spread across versions (m)", "Version minus real gap-fill (m)"], horizontal=True)
        spread = vers.std(axis=0)
        arr = {"Version height": (vers[k], "height (m)", "Gray", None), "Uncertainty: spread across versions (m)": (spread, "std (m)", "Magma", (0, max(float(spread.max()), 0.1))),
               "Version minus real gap-fill (m)": (vers[k] - vers.mean(0), "m", "RdBu", (-2, 2))}[what]
        fake = sc.crop_tile(gt, size)
        fake.height = vers[k].astype(np.float32)
        st.plotly_chart(sc.surface_figure(fake, arr, z_exag, 1 if size <= 192 else 2, title=f"GENERATED version {k} ({method.split(' (')[0]}), {what}"), width="stretch")
        st.caption(f"Mean spread across versions: {spread.mean():.2f} m; it is 0 at pixels the laser measured ({100 * float((gt.real_mask).mean()):.0f}% here).")

# ------------------------------------------------------------------------------------------------------ about
with tab_a:
    st.markdown("""
**What you are looking at**
- **Terrain + hazard:** NASA LOLA 5 m heights. The hazard U-Net sees only the real laser pixels (about 10% of the ground) and predicts where the
  official slope map says "unsafe" (slope >= 12 deg). Compare it with the plain slope rule on the same sparse input.
- **Skyline:** the light model predicts the horizon seen from a point, as a correction on top of a physics ray-cast of coarse terrain; compare with the
  exact ray-cast of the 5 m terrain. Sun and Earth come from exact directions (stand-in: analytic, NOT SPICE until KAN-23).
- **Generated terrain:** a physics backup now, a diffusion model when trained. Always GENERATED.

**Honest limits.** Small models (tens of thousands to ~2 million parameters) trained from scratch on a few real tiles; one held-out site; hazard labels
come from NASA's slope map, which was computed from interpolated heights; rocks and permanently shadowed regions are UNKNOWN; no result here proves a landing is safe.
""")
